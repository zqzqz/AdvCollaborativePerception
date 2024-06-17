import os
import numpy as np

from test_base import *
from test_occupancy_map import *
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.defense.perception_defender import PerceptionDefender
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.tools.squeezeseg.interface import SqueezeSegInterface


def run_defender(case, defender, perception_feature, occupancy_feature, defend_opts):
    case = dataset.load_feature(case, perception_feature)
    case = dataset.load_feature(case, occupancy_feature)
    case, score, metrics = defender.run(case, defend_opts)
    return case, score, metrics


if __name__ == "__main__":
    from mvp.data.opv2v_dataset import OPV2VDataset
    dataset = OPV2VDataset(root_path=os.path.join(root, "data/OPV2V"), mode="test")
    defender = PerceptionDefender()
    perception = OpencoodPerception(fusion_method="intermediate", model_name="pointpillar")
    lidar_seg_api = SqueezeSegInterface()

    case = dataset.get_case(0, tag="multi_frame", use_lidar=True)
    perception_feature = [{} for _ in range(10)]
    occupancy_feature = [{} for _ in range(10)]
    for vehicle_id, vehicle_data in case[9].items():
        pred_bboxes, pred_scores = perception.run(case[9], vehicle_id)
        perception_feature[9][vehicle_id] = {"pred_bboxes": pred_bboxes}
        occupancy_feature[9][vehicle_id] = test_occupancy_map(vehicle_data, lidar_seg_api)
    case, score, metrics = run_defender(case, defender, perception_feature, occupancy_feature, {"frame_ids": [9]})
    print(metrics)