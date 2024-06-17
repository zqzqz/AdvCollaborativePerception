import numpy as np
import cv2
import shapely
import copy

from mvp.tools.object_tracking import Tracker
from mvp.data.util import pcd_sensor_to_map, pcd_map_to_sensor, get_point_indices_in_bbox, bbox_sensor_to_map
from mvp.tools.squeezeseg.interface import SqueezeSegInterface
from mvp.tools.polygon_space import get_occupied_space, get_free_space
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.lidar_seg import lidar_segmentation
from mvp.defense.detection_util import filter_segmentation, get_detection_from_segmentation


class SyncObjectState:
    def __init__(self, object_id, location=None, velocity=None, acceleration=None, timestamp=0):
        self.object_id = object_id
        self.location = np.zeros(2) if location is None else location
        self.velocity = np.zeros(2) if velocity is None else velocity
        self.acceleration = np.zeros(2) if acceleration is None else acceleration
        self.timestamp = timestamp

    def update(self, location, timestamp):
        assert(timestamp > self.timestamp)
        velocity = (location - self.location) / (timestamp - self.timestamp)
        self.acceleration = (velocity - self.velocity) / (timestamp - self.timestamp)
        self.velocity = velocity
        self.location = location
        self.timestamp = timestamp

    def predict(self, timestamp):
        velocity = self.velocity + self.acceleration * (timestamp - self.timestamp)
        delta_location = np.zeros(2)
        for i in range(2):
            if self.acceleration[i] == 0:
                delta_location[i] = velocity[i] * (timestamp - self.timestamp)
            else:
                delta_location[i] = (velocity[i] ** 2 - self.velocity[i] ** 2) / (2 * self.acceleration[i])
        return delta_location


class SyncHandler:
    def __init__(self):
        self.tracker = Tracker(10, 1, 10)
        self.states = {}

    def update_object(self, object_id, location, timestamp):
        if object_id not in self.states:
            self.states[object_id] = SyncObjectState(object_id, location, timestamp)
        else:
            self.states[object_id].update(location, timestamp)

    def predict_object(self, object_id, timestamp):
        assert(object_id in self.states)
        return self.states[object_id].predict(timestamp)

    def update_pcd_gt(self, lidar_pose, gt_bboxes, object_ids, timestamp):
        gt_bboxes2 = bbox_sensor_to_map(gt_bboxes, lidar_pose)
        for object_index, object_id in enumerate(object_ids):
            self.update_object(object_id, gt_bboxes2[object_index][:2], timestamp)

    def update_pcd(self, detections, timestamp):
        _, assignment, _, _ = self.tracker.update(detections)
        for i, track in enumerate(self.tracker.tracks):
            if assignment[i] < 0:
                continue
            self.update_object(track.trackId, detections[assignment[i]], timestamp)

    def predict_pcd_gt(self, pcd, lidar_pose, new_lidar_pose, gt_bboxes, object_ids, timestamp):
        pcd2 = pcd_sensor_to_map(pcd, lidar_pose)
        non_object_mask = np.ones(pcd.shape[0]).astype(bool)
        for object_index, object_id in enumerate(object_ids):
            gt_bbox = gt_bboxes[object_index]
            point_indices = get_point_indices_in_bbox(gt_bbox, pcd)
            if object_id not in self.states:
                continue
            delta_location = self.states[object_id].predict(timestamp)
            pcd2[point_indices, :2] += delta_location
            non_object_mask[point_indices] = 0
        return pcd_map_to_sensor(pcd2, new_lidar_pose)

    def predict_pcd(self, pcd, lidar_pose, new_lidar_pose, object_segments, detections, timestamp):
        pcd2 = pcd_sensor_to_map(pcd, lidar_pose)
        for _, object_state in self.states.items():
            match = np.argwhere(np.sum(detections - object_state.location, axis=1) == 0).reshape(-1)
            if len(match) != 1:
                continue
            index = match[0]
            delta_location = object_state.predict(timestamp)
            pcd2[object_segments[index], :2] += delta_location
        return pcd_map_to_sensor(pcd2, new_lidar_pose)

    def predict_area(self, free_areas, occupied_areas, detections, timestamp):
        assert(detections.shape[0] == len(occupied_areas))
        new_occupied_areas = []
        for _, object_state in self.states.items():
            match = np.argwhere(np.sum(detections - object_state.location, axis=1) == 0).reshape(-1)
            if len(match) != 1:
                continue
            index = match[0]
            delta_location = object_state.predict(timestamp)
            area = occupied_areas[index]
            new_area = shapely.affinity.translate(area, xoff=delta_location[0], yoff=delta_location[1])
            new_occupied_areas.append(new_area)

            new_free_areas = []
            for free_area in free_areas:
                new_free_areas.append(free_area.difference(new_area))
            free_areas = new_free_areas
        
        return free_areas, occupied_areas


def preprocess_sync_gt(case, vehicle_id, lidar_seg_api):
    sync = SyncHandler()
    for frame_id in range(9):
        vehicle_data = case[frame_id][vehicle_id]
        lidar_pose = vehicle_data["lidar_pose"]
        gt_bboxes = vehicle_data["gt_bboxes"]
        object_ids = vehicle_data["object_ids"]

        sync.update_pcd_gt(lidar_pose, gt_bboxes, object_ids, frame_id * 0.1)

    pcd_sensor = case[8][vehicle_id]["lidar"]
    lidar_pose = case[8][vehicle_id]["lidar_pose"]
    gt_bboxes = vehicle_data["gt_bboxes"]
    object_ids = vehicle_data["object_ids"]

    pcd = pcd_sensor_to_map(pcd_sensor, lidar_pose)
    lidar_seg = lidar_segmentation(pcd_sensor, method="squeezeseq", interface=lidar_seg_api)
    object_mask = (lidar_seg["class"] == 1)
    object_segments = filter_segmentation(pcd_sensor, lidar_seg)
    detections = get_detection_from_segmentation(pcd, object_segments)
    occupied_areas = get_occupied_space(pcd, object_segments)
    plane, _ = get_ground_plane(pcd, method="ransac")
    free_areas = get_free_space(pcd, lidar_pose[:3], plane, object_mask, max_range=50)

    new_free_areas, new_occupied_areas = sync.predict_area(free_areas, occupied_areas, detections, 0.9)

    new_case = copy.deepcopy(case)
    new_case[9][vehicle_id]["free_areas"] = new_free_areas
    new_case[9][vehicle_id]["occupied_areas"] = new_occupied_areas
    return new_case


def preprocess_sync(case, vehicle_id, lidar_seg_api):
    sync = SyncHandler()
    for frame_id in range(9):
        vehicle_data = case[frame_id][vehicle_id]
        lidar_pose = vehicle_data["lidar_pose"]
        pcd_sensor = vehicle_data["lidar"]
        pcd = pcd_sensor_to_map(pcd_sensor, lidar_pose)
        lidar_seg = lidar_segmentation(pcd_sensor, method="squeezeseq", interface=lidar_seg_api)
        object_segments = filter_segmentation(pcd_sensor, lidar_seg)
        detections = get_detection_from_segmentation(pcd, object_segments)

        sync.update_pcd(detections, frame_id * 0.1)

    object_mask = (lidar_seg["class"] == 1)
    occupied_areas = get_occupied_space(pcd, object_segments)
    plane, _ = get_ground_plane(pcd, method="ransac")
    free_areas = get_free_space(pcd, lidar_pose[:3], plane, object_mask, max_range=50)

    new_free_areas, new_occupied_areas = sync.predict_area(free_areas, occupied_areas, detections, 0.9)

    new_case = copy.deepcopy(case)
    new_case[9][vehicle_id]["free_areas"] = new_free_areas
    new_case[9][vehicle_id]["occupied_areas"] = new_occupied_areas
    return new_case