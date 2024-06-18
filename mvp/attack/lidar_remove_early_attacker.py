import copy
import os
import open3d as o3d
import numpy as np
import time
import random
import pickle

from .attacker import Attacker
from .adv_shape_attacker import AdvShapeAttacker
from mvp.config import model_3d_path, model_3d_examples
from mvp.data.util import rotation_matrix, get_point_indices_in_bbox, get_open3d_bbox, get_distance, bbox_sensor_to_map, pcd_sensor_to_map, pcd_map_to_sensor, sort_lidar_points, numpy_to_open3d, bbox_to_corners
from mvp.tools.ray_tracing import get_model_mesh, ray_intersection, get_wall_mesh
from mvp.tools.ground_detection import get_ground_plane, get_ground_mesh


class LidarRemoveEarlyAttacker(Attacker):
    def __init__(self, dataset=None, advshape=1, dense=3, sync=1):
        super().__init__()
        self.name = "lidar_remove"
        self.dataset = dataset
        self.load_benchmark_meta()
        self.name = "lidar_remove_early"
        if advshape > 0:
            self.name += "_AS"
        if dense == 1:
            self.name += "_DenseA"
        elif dense == 2:
            self.name += "_DenseAll"
        elif dense == 3:
            self.name += "_Sampled"
        if sync > 0:
            self.name += "_Async"
        self.advshape = advshape
        self.dense = dense
        self.sync = sync

        # A 3D car model used for constructing the 3D ray casting scene (not used if adv-shape is applied).
        self.default_car_model = "car_0200"
        self.default_car_mesh = o3d.io.read_triangle_mesh(os.path.join(model_3d_path, "{}.ply".format(self.default_car_model)))
        # Apply pretrained perturbation on a hardcoded mesh. Use the perturbed mesh as the adversarial shape.
        perturb = np.load(os.path.join(model_3d_path, "remove/mesh_perturb.npy"))
        adv_attacker = AdvShapeAttacker()
        self.mesh = adv_attacker.perturb_mesh(adv_attacker.mesh, perturb)
        self.meshes = [self.mesh.select_by_index(vertex_indices) for vertex_indices in adv_attacker.mesh_divide]

        # In the mode of injecting dense points, fix the distance between the target and the sensor.
        self.dense_distance = 10 # (m)

    def run(self, multi_frame_case, attack_opts):
        new_case = copy.deepcopy(multi_frame_case)
        attack_info = []
        frame_ids = attack_opts["frame_ids"]
        if self.sync != 0:
            frame_ids = [frame_ids[0] - 1] + frame_ids
        
        last_info = None
        last_bbox_to_remove = None
        for frame_id in frame_ids:
            single_vehicle_case = multi_frame_case[frame_id][attack_opts["attacker_vehicle_id"]]
            object_index = single_vehicle_case["object_ids"].index(attack_opts["object_id"])
            bbox_to_remove = single_vehicle_case["gt_bboxes"][object_index]
            lidar_poses = {vehicle_id: multi_frame_case[frame_id][vehicle_id]["lidar_pose"] for vehicle_id in multi_frame_case[frame_id]}
            core_function = self.run_core_sample if self.dense == 3 else self.run_core
            x, info = core_function(single_vehicle_case, {
                "attacker_vehicle_id": attack_opts["attacker_vehicle_id"],
                "object_id": attack_opts["object_id"],
                "bbox": attack_opts["bboxes"][frame_id] if "bboxes" in attack_opts else None,
                "lidar_poses": lidar_poses,
                "attacker_vehicle_id": attack_opts["attacker_vehicle_id"]
            })

            if self.sync != 0 and last_info is not None:
                x = copy.deepcopy(multi_frame_case[frame_id][attack_opts["attacker_vehicle_id"]])
                shift_vector = bbox_to_remove[:3] - last_bbox_to_remove[:3]
                
                append_data = []
                if last_info["replace_data"] is not None and last_info["replace_data"].shape[0] > 0:
                    append_data.append(last_info["replace_data"] + shift_vector)
                if last_info["append_data"] is not None and last_info["append_data"].shape[0] > 0:
                    append_data.append(last_info["append_data"] + shift_vector)
                if len(append_data) > 1:
                    append_data = np.vstack(append_data)
                else:
                    append_data = None

                ignore_indices = []
                if info["replace_indices"] is not None and last_info["replace_indices"].shape[0] > 0:
                    ignore_indices.append(info["replace_indices"])
                if info["ignore_indices"] is not None and last_info["ignore_indices"].shape[0] > 0:
                    ignore_indices.append(info["ignore_indices"])
                if len(append_data) > 1:
                    ignore_indices = np.hstack(ignore_indices).reshape(-1)
                else:
                    ignore_indices = None

                info = {
                    "replace_indices": None,
                    "replace_data": None,
                    "ignore_indices": ignore_indices,
                    "append_data": append_data,
                }
                x["lidar"] = self.apply_ray_tracing(x["lidar"], **info)

            if frame_id in attack_opts["frame_ids"]:
                attack_info.append(info)
                new_case[frame_id][attack_opts["attacker_vehicle_id"]] = x

            last_info = info
            last_bbox_to_remove = bbox_to_remove
        return new_case, attack_info

    def get_meshes(self, bbox):
        wall_bboxes = []
        extend_distance = 0.3

        wall_bbox1 = copy.deepcopy(bbox)
        wall_bbox1[0] -= np.sin(wall_bbox1[6]) * (wall_bbox1[4] / 2 + extend_distance)
        wall_bbox1[1] += np.cos(wall_bbox1[6]) * (wall_bbox1[4] / 2 + extend_distance)
        wall_bbox1[4] = 0.01
        wall_bboxes.append(wall_bbox1)

        wall_bbox2 = copy.deepcopy(bbox)
        wall_bbox2[0] += np.cos(wall_bbox2[6]) * (wall_bbox2[3] / 2 + extend_distance)
        wall_bbox2[1] += np.sin(wall_bbox2[6]) * (wall_bbox2[3] / 2 + extend_distance)
        wall_bbox2[3] = 0.01
        wall_bboxes.append(wall_bbox2)

        wall_bbox3 = copy.deepcopy(bbox)
        wall_bbox3[0] += np.sin(wall_bbox3[6]) * (wall_bbox3[4] / 2 + extend_distance)
        wall_bbox3[1] -= np.cos(wall_bbox3[6]) * (wall_bbox3[4] / 2 + extend_distance)
        wall_bbox3[4] = 0.01
        wall_bboxes.append(wall_bbox3)

        wall_bbox4 = copy.deepcopy(bbox)
        wall_bbox4[0] -= np.cos(wall_bbox4[6]) * (wall_bbox4[3] / 2 + extend_distance)
        wall_bbox4[1] -= np.sin(wall_bbox4[6]) * (wall_bbox4[3] / 2 + extend_distance)
        wall_bbox4[3] = 0.01
        wall_bboxes.append(wall_bbox4)

        wall_meshs = []
        for wall_bbox in wall_bboxes:
            wall_meshs.append(get_wall_mesh(wall_bbox))
        return wall_meshs

    def post_process_meshes(self, meshes, bbox):
        new_meshes = []
        for mesh in meshes:
            x = copy.deepcopy(mesh)
            scale = np.max((bbox[3:6] + 0.6) / np.array([4.9, 2.5, 2.0])) 
            x = x.scale(scale, np.array([0, 0, 0]).T)
            x = x.rotate(rotation_matrix(0, bbox[6], 0), np.zeros(3).T)
            x = x.translate(bbox[:3])
            new_meshes.append(x)
        return new_meshes

    def run_core(self, single_vehicle_case, attack_opts):
        """ attack_opts: {
                "frame_ids": [-1],
                "attacker_vehicle_id": int,
                "object_id": int
            }
        """
        new_case = copy.deepcopy(single_vehicle_case)
        attacker_pcd = new_case["lidar"][:, :3]
        try:
            if "object_index" in attack_opts:
                object_index = attack_opts["object_index"]
            elif "object_id" in attack_opts:
                object_index = new_case["object_ids"].index(attack_opts["object_id"])
            bbox_to_remove = new_case["gt_bboxes"][object_index]
        except:
            print("The target object is not available")
            bbox_to_remove = attack_opts["bbox"]

        extended_bbox_to_remove = copy.deepcopy(bbox_to_remove)
        extended_bbox_to_remove[2] -= 5
        extended_bbox_to_remove[3:5] += 0.6
        extended_bbox_to_remove[5] = 10
        bbox_to_remove_o3d = get_open3d_bbox(extended_bbox_to_remove)
        
        point_indices = bbox_to_remove_o3d.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(attacker_pcd))
        point_indices = np.array(point_indices)
        if point_indices.shape[0] == 0:
            return new_case, {}

        points = attacker_pcd
        distance = np.sum(points ** 2, axis=1) ** 0.5
        direction = points / np.tile(distance.reshape((-1, 1)), (1, 3))
        direction = []
        for i in np.arange(-15, 5, 0.2):
            for j in np.arange(-180, 180, 0.2):
                direction.append(np.array([np.sin(np.radians(j)), np.cos(np.radians(j)), np.sin(np.radians(i))]))
        direction = np.array(direction)
        rays = np.hstack([np.zeros((direction.shape[0], 3)), direction])
        selected_rays = rays[point_indices]

        extra_rays_list = []
        target_offset = bbox_to_remove[:2]

        if self.dense in [1, 2]:
            target_distance = get_distance(target_offset)
            lidar_offset = target_offset / target_distance * (target_distance - self.dense_distance)
            extra_rays = copy.deepcopy(rays)
            extra_rays[:, :2] = lidar_offset
            extra_rays_list.append(extra_rays)
        if self.dense == 2:
            attacker_lidar_pose = attack_opts["lidar_poses"][attack_opts["attacker_vehicle_id"]]
            for vehicle_id, lidar_pose in attack_opts["lidar_poses"].items():
                if vehicle_id == attack_opts["attacker_vehicle_id"]:
                    continue
                lidar_offset = pcd_map_to_sensor(lidar_pose[np.newaxis, :3], attacker_lidar_pose)[0, :2]
                target_distance = get_distance(target_offset, lidar_offset)
                lidar_offset = target_offset + (lidar_offset - target_offset) / target_distance * self.dense_distance
                extra_rays = copy.deepcopy(rays)
                extra_rays[:, :2] = lidar_offset
                extra_rays_list.append(extra_rays)

        if self.advshape:
            meshes = self.post_process_meshes([self.mesh], bbox_to_remove)
        else:
            meshes = []
            plane_model, _ = get_ground_plane(attacker_pcd, method="ransac")
            ground_mesh = get_ground_mesh(plane_model)
            meshes.append(ground_mesh)
            default_car_model = "car_0200"
            for i in range(new_case["gt_bboxes"].shape[0]):
                if i == object_index:
                    continue
                bbox = new_case["gt_bboxes"][i]
                meshes.append(
                    get_model_mesh(default_car_model, bbox)
                )
        
        intersect_points = ray_intersection(meshes, selected_rays)

        extra_points_list = []
        for extra_rays in extra_rays_list:
            extra_intersect_points = ray_intersection(meshes, extra_rays)
            extra_index_mask = (extra_intersect_points[:,0] ** 2 < 10000)
            extra_intersect_points = extra_intersect_points[extra_index_mask]
            extra_point_indices = bbox_to_remove_o3d.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(extra_intersect_points))
            extra_intersect_points = extra_intersect_points[extra_point_indices]
            extra_points_list.append(extra_intersect_points)

        ignore_indices = None
        append_data = None
        replace_indices = None
        replace_data = None

        if self.dense > 0:
            ignore_indices = point_indices
            if len(extra_points_list) > 0:
                append_data = np.vstack(extra_points_list)
            else:
                append_data = extra_points_list[0]
            attacker_pcd = np.delete(attacker_pcd, ignore_indices, axis=0)
            attacker_pcd = np.vstack([attacker_pcd, append_data])
            attacker_pcd, _ = sort_lidar_points(attacker_pcd)
        else:
            in_range_mask = (intersect_points[:,0] ** 2 < 10000)
            replace_indices = point_indices[in_range_mask]
            replace_data = intersect_points[in_range_mask]
            ignore_indices = point_indices[np.logical_not(in_range_mask)]
            attacker_pcd[replace_indices, :3] = replace_data

        info = {
            "replace_indices": replace_indices,
            "replace_data": replace_data,
            "ignore_indices": ignore_indices,
            "append_data": append_data
        }

        new_case["lidar"] = np.hstack([attacker_pcd, np.ones((attacker_pcd.shape[0], 1))])

        return new_case, info

    def run_core_sample(self, single_vehicle_case, attack_opts):
        """ attack_opts: {
                "frame_ids": [-1],
                "attacker_vehicle_id": int,
                "object_id": int
            }
        """
        new_case = copy.deepcopy(single_vehicle_case)
        attacker_pcd = new_case["lidar"][:, :3]
        try:
            if "object_index" in attack_opts:
                object_index = attack_opts["object_index"]
            elif "object_id" in attack_opts:
                object_index = new_case["object_ids"].index(attack_opts["object_id"])
            bbox_to_remove = new_case["gt_bboxes"][object_index]
        except:
            print("The target object is not available")
            bbox_to_remove = attack_opts["bbox"]

        points = attacker_pcd
        distance = np.sum(points ** 2, axis=1) ** 0.5
        direction = points / np.tile(distance.reshape((-1, 1)), (1, 3))
        rays = np.hstack([np.zeros((direction.shape[0], 3)), direction])

        # Gets meshes of four edges.
        if self.meshes is None:
            meshes = self.get_meshes(bbox_to_remove)
        else:
            meshes = self.post_process_meshes(self.meshes, bbox_to_remove)
        
        # Gets casted points on edges.
        replace_mask_list = []
        replace_data_list = []
        for i in range(len(meshes)):
            intersect_points = ray_intersection([meshes[i]], rays)
            in_range_mask = (intersect_points[:,0] ** 2 < 10000)
            replace_mask_list.append(in_range_mask)
            replace_data_list.append(intersect_points)

        # Estimate weights of four edges.
        mesh_weight = np.zeros(len(meshes))
        attacker_lidar_pose = attack_opts["lidar_poses"][attack_opts["attacker_vehicle_id"]]
        for vehicle_id, lidar_pose in attack_opts["lidar_poses"].items():
            if vehicle_id == attack_opts["attacker_vehicle_id"]:
                continue
            lidar_offset = pcd_map_to_sensor(lidar_pose[np.newaxis, :3], attacker_lidar_pose)[0, :3]
            for i, mesh in enumerate(meshes):
                vertices = np.asarray(mesh.vertices)
                h_angle = np.arctan2(vertices[:, 1] - lidar_offset[1], vertices[:, 0] - lidar_offset[0])
                v_angle = (vertices[:, 2] - lidar_offset[2]) / get_distance(vertices[:, :2], lidar_offset[:2])
                mesh_weight[i] += ((h_angle.max() - h_angle.min()) / 0.005) * ((v_angle.max() - v_angle.min()) / 0.01)

        # Point sampling
        replace_data = []
        # append_data = []
        point_sampling_weight = np.vstack(replace_mask_list).T * mesh_weight
        replace_indices = np.argwhere(np.logical_or.reduce(replace_mask_list)).reshape(-1).astype(np.int32)
        for i in replace_indices:
            replace_data.append(
                replace_data_list[
                    np.random.choice(mesh_weight.shape[0], p=point_sampling_weight[i]/np.sum(point_sampling_weight[i]))
                ][i]
            )
        replace_data = np.array(replace_data)

        ignore_indices = None
        append_data = None

        info = {
            "replace_indices": replace_indices,
            "replace_data": replace_data,
            "ignore_indices": ignore_indices,
            "append_data": append_data
        }

        new_case["lidar"] = self.apply_ray_tracing(new_case["lidar"], **info)

        return new_case, info