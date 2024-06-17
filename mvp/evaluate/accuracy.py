import numpy as np
import pickle
import os

from mvp.data.util import get_point_indices_in_bbox
from mvp.tools.iou import iou3d


def get_accuracy(dataset, detector, iou_threshold=0.5):
    report = {"TP": 0, "FP": 0, "P": 0, "PP": 0}

    for case_id, case in dataset.case_generator(index=True, tag="multi_frame"):
        result_case = detector.run_multi_frame(case)
        for frame_id, frame_data in enumerate(result_case):
            for vehicle_id, vehicle_data in frame_data.items():
                gt_bboxes = vehicle_data["gt_bboxes"]
                result_bboxes = vehicle_data["result_bboxes"]

                # filter gt boxes
                select_indices = []
                for i in range(gt_bboxes.shape[0]):
                    point_number = len(get_point_indices_in_bbox(gt_bboxes[i], vehicle_data["lidar"]))
                    if point_number > 5:
                        select_indices.append(i)
                gt_bboxes = gt_bboxes[select_indices]

                report["P"] += gt_bboxes.shape[0]
                report["PP"] += result_bboxes.shape[0]

                for i in range(result_bboxes.shape[0]):
                    max_iou = 0
                    for j in range(gt_bboxes.shape[0]):
                        iou = iou3d(result_bboxes[i], gt_bboxes[j])
                        max_iou = max(max_iou, iou)
                    if max_iou >= iou_threshold:
                        report["TP"] += 1
                    else:
                        report["FP"] += 1
        print("case {} done".format(case_id))
        print(report)
    return report