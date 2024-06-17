import cv2
import numpy as np
import scipy

from mvp.data.util import pcd_sensor_to_map
from mvp.tools.polygon_space import points_to_polygon


def filter_segmentation(pcd, lidar_seg, lidar_pose, in_lane_mask=None, point_height=None, max_range=50):
    object_segments = []
    for info in lidar_seg["info"]:
        object_points = pcd[info["indices"]]
        if np.min(np.sum(object_points[:,:2] ** 2, axis=1)) > max_range ** 2:
            continue
        object_points = pcd_sensor_to_map(object_points, lidar_pose)
        if point_height is not None:
            if point_height[info["indices"]].min() < -0.5 or point_height[info["indices"]].max() > 3 or point_height[info["indices"]].max() < 0.6:
                continue
        if scipy.spatial.distance.cdist(object_points[:, :2], object_points[:, :2]).max() > 8:
            continue
        occupied_area = points_to_polygon(object_points[:,:2])
        if occupied_area.area > 20:
            continue
        if occupied_area.area < 0.5:
            continue
        if in_lane_mask is not None:
            if in_lane_mask[info["indices"]].sum() <= 0.2 * len(info["indices"]):
                continue
        object_segments.append(info["indices"])
    return object_segments


def get_detection_from_segmentation(pcd, object_segments):
    detections = []
    for object_segment in object_segments:
        object_points = pcd[object_segment]
        cnt = object_points[:,:2].reshape((-1,1,2)).astype(np.float32)
        box = cv2.minAreaRect(cnt)
        detections.append(list(box[0]))
    detections = np.array(detections)

    return detections