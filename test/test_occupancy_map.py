import os, sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)
import numpy as np
import matplotlib.pyplot as plt
import pickle

from test_base import *
from mvp.config import data_root
from mvp.data.util import pcd_sensor_to_map
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.tools.squeezeseg.interface import SqueezeSegInterface
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.lidar_seg import lidar_segmentation
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.polygon_space import get_occupied_space, get_free_space, bbox_to_polygon
from mvp.tools.squeezeseg.interface import SqueezeSegInterface


def test_occupancy_map(case, lidar_seg_api):
    lidar, lidar_pose = case["lidar"], case["lidar_pose"]
    pcd = pcd_sensor_to_map(lidar, lidar_pose)

    lane_info = pickle.load(open(os.path.join(data_root, "carla/{}_lane_info.pkl".format(case["map"])), "rb"))
    lane_areas = pickle.load(open(os.path.join(data_root, "carla/{}_lane_areas.pkl".format(case["map"])), "rb"))
    lane_planes = pickle.load(open(os.path.join(data_root, "carla/{}_ground_planes.pkl".format(case["map"])), "rb"))

    ground_indices, in_lane_mask, point_height = get_ground_plane(pcd, lane_info=lane_info, lane_areas=lane_areas, lane_planes=lane_planes, method="map")
    lidar_seg = lidar_segmentation(lidar, method="squeezeseq", interface=lidar_seg_api)
    
    object_segments = filter_segmentation(lidar, lidar_seg, lidar_pose, in_lane_mask=in_lane_mask, point_height=point_height, max_range=50)
    object_mask = np.zeros(pcd.shape[0]).astype(bool)
    if len(object_segments) > 0:
        object_indices = np.hstack(object_segments)
        object_mask[object_indices] = True

    ego_bbox = case["ego_bbox"]
    ego_area = bbox_to_polygon(ego_bbox)
    ego_area_height = ego_bbox[5]

    ret = {
        "ego_area": ego_area,
        "ego_area_height": ego_area_height,
        "plane": None,
        "ground_indices": ground_indices,
        "point_height": point_height,
        "object_segments": object_segments,
    }

    height_thres = 0
    occupied_areas, occupied_areas_height = get_occupied_space(pcd, object_segments, point_height=point_height, height_thres=height_thres)
    free_areas = get_free_space(lidar, lidar_pose, object_mask, in_lane_mask=in_lane_mask, point_height=point_height, max_range=50, height_thres=height_thres, height_tolerance=0.2)
    ret["occupied_areas"] = occupied_areas
    ret["occupied_areas_height"] = occupied_areas_height
    ret["free_areas"] = free_areas

    return ret


if __name__ == "__main__":
    from mvp.data.opv2v_dataset import OPV2VDataset
    dataset = OPV2VDataset(root_path=os.path.join(root, "data/OPV2V"), mode="test")
    lidar_seg_api = SqueezeSegInterface()

    case = dataset.get_case(0, tag="multi_vehicle", use_lidar=True)
    vehicle_ids = list(case.keys())
    omap = test_occupancy_map(case[vehicle_ids[0]], lidar_seg_api)
    print(omap)
