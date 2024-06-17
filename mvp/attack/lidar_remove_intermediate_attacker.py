import random
import pickle
import numpy as np
import copy

from .attacker import Attacker
from mvp.data.util import bbox_map_to_sensor, bbox_sensor_to_map, pcd_sensor_to_map
from mvp.tools.iou import iou3d


class LidarRemoveIntermediateAttacker(Attacker):
    def __init__ (self, perception, dataset=None, step=100, sync=1, init=True, online=True):
        super().__init__()
        self.dataset = dataset
        self.name = "lidar_remove"
        self.load_benchmark_meta()
        self.name = "lidar_remove_intermediate"
        self.max_perturb = 10

        self.name += "_Step{}".format(step - 1)
        if sync > 0:
            self.name += "_Async"
        if init:
            self.name += "_Init"
        if online:
            self.name += "_Online"
        self.step = step
        self.sync = sync
        self.init = init
        self.online = online

        if perception.model_name != "pointpillar":
            self.name += "_{}".format(perception.model_name)

        if step <=  2:
            self.learn_rate = 2
        else:
            self.learn_rate = 0.05
        self.perception = perception

    def run(self, multi_frame_case, attack_opts):
        case = copy.deepcopy(multi_frame_case)
        info = [{} for i in range(10)]
        init_perturbation = None

        for frame_index, frame_id in enumerate(attack_opts["frame_ids"]):
            attacker_id = attack_opts["attacker_vehicle_id"]
            ego_id = attack_opts["victim_vehicle_id"]
            info[frame_id][ego_id] = {}

            if self.sync == 0:
                optimize_case = copy.deepcopy(case[frame_id])
                real_case = None
            else:
                optimize_case = copy.deepcopy(case[frame_id - 1])
                real_case = copy.deepcopy(case[frame_id])
            original_case = copy.deepcopy(optimize_case)
            real_original_case = copy.deepcopy(real_case)

            try:
                object_index = optimize_case[attacker_id]["object_ids"].index(attack_opts["object_id"])
                bbox_to_remove = optimize_case[attacker_id]["gt_bboxes"][object_index]
            except:
                bbox_to_remove = attack_opts["bboxes"][frame_id if self.sync == 0 else frame_id - 1]
            bbox_to_remove_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(bbox_to_remove, optimize_case[attacker_id]["lidar_pose"]),
                optimize_case[ego_id]["lidar_pose"])
            try:
                real_object_index = case[frame_id][attacker_id]["object_ids"].index(attack_opts["object_id"])
                real_bbox_to_remove = case[frame_id][attacker_id]["gt_bboxes"][real_object_index]
            except:
                real_bbox_to_remove = attack_opts["bboxes"][frame_id]
            real_bbox_to_remove_ego = bbox_map_to_sensor(
                bbox_sensor_to_map(real_bbox_to_remove, case[frame_id][attacker_id]["lidar_pose"]),
                case[frame_id][ego_id]["lidar_pose"])
            
            if self.init:
                optimize_case[attacker_id]["lidar"] = self.apply_ray_tracing(optimize_case[attacker_id]["lidar"], **attack_opts["attack_info"][frame_id if self.sync == 0 else frame_id - 1])
                if real_case is not None:
                    real_case[attacker_id]["lidar"] = self.apply_ray_tracing(real_case[attacker_id]["lidar"], **attack_opts["attack_info"][frame_id])

            result = self.perception.attack_intermediate(optimize_case, ego_id, attacker_id, max_perturb=self.max_perturb, mode="remove", bbox=bbox_to_remove_ego, max_iteration=self.step, lr=self.learn_rate, real_case=real_case, original_case=original_case, real_original_case=real_original_case, real_bbox=real_bbox_to_remove_ego, init_perturbation=init_perturbation, feature_size=10)

            if self.online:
                init_perturbation = result["perturbation"]
            
            case[frame_id][ego_id]["pred_bboxes"] = result["pred_bboxes"]
            case[frame_id][ego_id]["pred_scores"] = result["pred_scores"]
            info[frame_id][ego_id] = {"pred_bboxes": result["pred_bboxes"], "pred_scores": result["pred_scores"]}

        return case, info
