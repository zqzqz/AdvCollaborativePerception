import os

from test_base import *
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.visualize.general import draw_multi_vehicle_case


def test_sample():
    dataset = OPV2VDataset(root_path=os.path.join(root, "data/OPV2V"), mode="test")

    case = dataset.get_case(100, tag="multi_vehicle")
    ego_id = max(list(case.keys()))

    perception = OpencoodPerception(fusion_method="intermediate", model_name="pointpillar")

    result = perception.run_multi_vehicle(case, ego_id=ego_id)
    print(result[ego_id]["pred_bboxes"])

    draw_multi_vehicle_case(result, ego_id, mode="matplotlib", gt_bboxes=result[ego_id]["gt_bboxes"], pred_bboxes=result[ego_id]["pred_bboxes"], show=True)


if __name__ == "__main__":
    test_sample()