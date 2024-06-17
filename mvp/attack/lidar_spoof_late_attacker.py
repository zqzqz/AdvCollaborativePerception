import random
import pickle
import numpy as np
import copy

from .attacker import Attacker
from mvp.data.util import bbox_map_to_sensor, bbox_sensor_to_map, pcd_sensor_to_map


class LidarSpoofLateAttacker(Attacker):
    def __init__ (self, perception, dataset=None):
        super().__init__()
        self.name = "lidar_spoof"
        self.dataset = dataset
        self.load_benchmark_meta()
        self.perception = perception
        self.name = "lidar_spoof_late"

    def run(self, multi_frame_case, attack_opts):
        case = copy.deepcopy(multi_frame_case)
        attack_results = []
        for frame_id in range(10):
            attack_results.append({})
            if frame_id == 9:
                multi_vehicle_case = multi_frame_case[frame_id]
                attacker_id = attack_opts["attacker_vehicle_id"]
                ego_id = attack_opts["victim_vehicle_id"]
                bbox_to_spoof = attack_opts["positions"][frame_id]
                result = self.perception.attack_late(multi_vehicle_case, ego_id, attacker_id, mode="spoof", bbox=bbox_to_spoof)
                case[frame_id][ego_id]["pred_bboxes"] = result["pred_bboxes"]
                case[frame_id][ego_id]["pred_scores"] = result["pred_scores"]
                attack_results[-1][ego_id] = {
                    "pred_bboxes": result["pred_bboxes"],
                    "pred_scores": result["pred_scores"]
                }
        return case, attack_results

    def build_benchmark_meta(self, write=False, max_cnt=500):
        raise NotImplementedError("Should use the same benchmark as LidarSpoofAttacker.")