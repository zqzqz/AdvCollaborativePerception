import os
from mmdet3d.apis import inference_detector, init_model

from .base import Detector
from mvp.data.util import write_bin
from mvp.config import third_party_root, tmp_root


class PointPillarsDetector(Detector):
    def __init__(self):
        config = os.path.join(third_party_root, "mmdetection3d/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_opv2v-3d-car.py")
        # checkpoint = os.path.join(third_party_root, "mmdetection3d/checkpoints/pointpillars/hv_pointpillars_secfpn_6x8_160e_opv2v-3d-car.pth")
        checkpoint = os.path.join(third_party_root, "mmdetection3d/work_dirs/hv_pointpillars_secfpn_6x8_160e_opv2v-3d-car/latest.pth")
        device = "cuda:0"
        self.model = init_model(config, checkpoint, device=device)
        self.bin_path = os.path.join(tmp_root, "pointcloud.bin.tmp")

    def run(self, pointcloud, bboxes=True, scores=True, labels=True):
        output = {}
        write_bin(pointcloud, self.bin_path)
        result, _ = inference_detector(self.model, self.bin_path)
        if bboxes:
            b = result[0]['boxes_3d'].tensor.detach().numpy()
            output["bboxes"] = b
        if scores:
            s = result[0]['scores_3d'].detach().numpy()
            output["scores"] = s
        if scores:
            l = result[0]['labels_3d'].detach().numpy()
            output["labels"] = l
        return output