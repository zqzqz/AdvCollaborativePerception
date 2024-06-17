import os
import numpy as np

from test_base import *
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.attack.adv_shape_attacker import AdvShapeAttacker
from mvp.attack.lidar_remove_early_attacker import LidarRemoveEarlyAttacker
from mvp.perception.opencood_perception import OpencoodPerception

dataset = OPV2VDataset(root_path=os.path.join(root, "data/OPV2V"), mode="test")
perception = OpencoodPerception(fusion_method="early", model_name="pointpillar")
attacker = LidarRemoveEarlyAttacker(dataset)

attack = attacker.attack_list[0]
case_id = attack["attack_meta"]["case_id"]
case = dataset.get_case(case_id)
attack_opts = {
    "victim_vehicle_id": attack["attack_meta"]["victim_vehicle_id"],
    "bboxes": attack["attack_meta"]["bboxes"],
    **attack["attack_opts"]
}
attack_opts["frame_ids"] = [9]

attacker = AdvShapeAttacker(dataset=dataset, perception=perception, attacker=attacker)
attacker.run()