import numpy as np
import open3d as o3d
from mvp.data.util import rotation_matrix
import cv2
import matplotlib.pyplot as plt
import matplotlib

from .general import get_xylims
from mvp.config import model_3d_examples
from mvp.data.util import bbox_shift, bbox_rotate, pcd_sensor_to_map, bbox_sensor_to_map
from .general import draw_bbox_2d


def draw_attack(attack, normal_case, attack_case, mode="multi_frame", show=False, save=None):
    if mode == "multi_frame":
        frame_ids = attack["attack_meta"]["attack_frame_ids"]
        frame_num = len(frame_ids)
        fig, axes = plt.subplots(frame_num, 2, figsize=(40, 20 * frame_num))

        # draw normal case first
        for case_id, case in enumerate([normal_case, attack_case]):
            for frame_id in frame_ids:
                if frame_num <= 1:
                    ax = axes[case_id]
                else:
                    ax = axes[frame_ids.index(frame_id)][case_id]

                # draw point clouds
                # pointcloud_all = pcd_sensor_to_map(case[frame_id][attack["attack_opts"]["attacker_vehicle_id"]]["lidar"], case[frame_id][attack["attack_opts"]["attacker_vehicle_id"]]["lidar_pose"])[:,:3]
                pointcloud_all = np.vstack([pcd_sensor_to_map(vehicle_data["lidar"], vehicle_data["lidar_pose"])[:,:3] for vehicle_id, vehicle_data in case[frame_id].items()])
                xlim, ylim = get_xylims(pointcloud_all)
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                # ax.set_aspect('equal', adjustable='box')
                ax.scatter(pointcloud_all[:,0], pointcloud_all[:,1], s=0.01, c="black")

                # label the location of attacker and victim
                attacker_vehicle_id = attack["attack_meta"]["attacker_vehicle_id"]
                attacker_vehicle_data = case[frame_id][attacker_vehicle_id]
                victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
                victim_vehicle_data = case[frame_id][victim_vehicle_id]
                ax.scatter(*victim_vehicle_data["lidar_pose"][:2].tolist(), s=100, c="green")
                ax.scatter(*attacker_vehicle_data["lidar_pose"][:2].tolist(), s=100, c="red")

                # draw gt/result bboxes
                total_bboxes = []
                if "gt_bboxes" in victim_vehicle_data:
                    total_bboxes.append((bbox_sensor_to_map(victim_vehicle_data["gt_bboxes"], victim_vehicle_data["lidar_pose"]), victim_vehicle_data["object_ids"], "g"))
                if "pred_bboxes" in victim_vehicle_data:
                    total_bboxes.append((bbox_sensor_to_map(victim_vehicle_data["pred_bboxes"], victim_vehicle_data["lidar_pose"]), None, "r"))
                # label the position of spoofing/removal
                bbox = attack["attack_meta"]["bboxes"][frame_ids.index(frame_id)]
                bbox = bbox_sensor_to_map(bbox, attacker_vehicle_data["lidar_pose"])
                total_bboxes.append((bbox[None,:], None, 'red'))

                draw_bbox_2d(ax, total_bboxes)
    else:
        raise NotImplementedError()

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()


def draw_attack_trace(trace, show=False, save=None):
    pass
