import os
import pickle
import logging
import copy
import numpy as np
import traceback

from test_base import *
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_spoof_early_attacker import LidarSpoofEarlyAttacker
from mvp.attack.lidar_spoof_intermediate_attacker import LidarSpoofIntermediateAttacker
from mvp.attack.lidar_spoof_late_attacker import LidarSpoofLateAttacker
from mvp.attack.lidar_remove_early_attacker import LidarRemoveEarlyAttacker
from mvp.attack.lidar_remove_intermediate_attacker import LidarRemoveIntermediateAttacker
from mvp.attack.lidar_remove_late_attacker import LidarRemoveLateAttacker


dataset = OPV2VDataset(root_path=os.path.join(root, "data/OPV2V"), mode="test")
perception = {
    "early": OpencoodPerception(fusion_method="early", model_name="pointpillar"),
    "intermediate": OpencoodPerception(fusion_method="intermediate", model_name="pointpillar"),
    "late": OpencoodPerception(fusion_method="late", model_name="pointpillar"),
}
attackers = [
    LidarSpoofEarlyAttacker(dataset, dense=0, sync=0),
    LidarSpoofEarlyAttacker(dataset, dense=1, sync=0),
    LidarSpoofEarlyAttacker(dataset, dense=2, sync=0),
    LidarSpoofEarlyAttacker(dataset, dense=3, sync=1),
    LidarRemoveEarlyAttacker(dataset, advshape=0, dense=0, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=0, dense=1, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=0, dense=2, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=0, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=1, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=2, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=3, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=3, sync=1),
    LidarSpoofIntermediateAttacker(perception["intermediate"], dataset, step=100, sync=0, init=False, online=False),
    LidarSpoofIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=0, init=False, online=False),
    LidarSpoofIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=1, init=False, online=False),
    LidarSpoofIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=1, init=False, online=True),
    LidarSpoofIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=0, init=True, online=False),
    LidarSpoofIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=1, init=True, online=False),
    LidarSpoofIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=1, init=True, online=True),
    LidarRemoveIntermediateAttacker(perception["intermediate"], dataset, step=100, sync=0, init=False, online=False),
    LidarRemoveIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=0, init=False, online=False),
    LidarRemoveIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=1, init=False, online=False),
    LidarRemoveIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=1, init=False, online=True),
    LidarRemoveIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=0, init=True, online=False),
    LidarRemoveIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=1, init=True, online=False),
    LidarRemoveIntermediateAttacker(perception["intermediate"], dataset, step=2, sync=1, init=True, online=True),
    LidarSpoofLateAttacker(perception["late"], dataset),
    LidarRemoveLateAttacker(perception["late"], dataset),
]


def test_attack_sample(attacker):
    print("########################## {} #########################".format(attacker.name))
    attack_id = 0
    attack = attacker.attack_list[attack_id]
    attack_opts = attack["attack_opts"]
    case_id = attack["attack_meta"]["case_id"]
    ego_id = attack["attack_meta"]["victim_vehicle_id"]
    attack_opts["victim_vehicle_id"] = ego_id
    attack_opts["frame_ids"] = [9]

    normal_case = dataset.get_case(case_id, tag="multi_frame", use_lidar=True)
    if (isinstance(attacker, LidarSpoofIntermediateAttacker) or isinstance(attacker, LidarRemoveIntermediateAttacker)):
        # The information of ray tracing results. Here it is just a placeholder.
        attack_opts["attack_info"] = [{} for _ in range(10)]
        if attacker.online:
            attack_opts["frame_ids"] = [i for i in range(1, 10)]

    attack_case, attack_info = attacker.run(normal_case, attack_opts)
    print(attack_info)


if __name__ == "__main__":
    for attacker in attackers:
        test_attack_sample(attacker)