import open3d as o3d
import numpy as np
import math
import copy
from shapely.geometry import Polygon


def numpy_to_open3d(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:,:3])
    pcd.paint_uniform_color([0, 0, 1])
    return pcd


def numpy_to_open3d_t(data, device="CPU:0"):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = o3d.core.Tensor(data[:,:3], o3d.core.float64, o3d.core.Device(device))
    return pcd


def read_pcd(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    xyz = np.asarray(pcd.points)
    try:
        intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    except:
        intensity = np.ones((xyz.shape[0], 1)) * 0.1
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def read_bin(filepath):
    return np.fromfile(filepath, dtype=np.float32, count=-1).reshape([-1, 4])


def write_pcd(pcd_data: np.ndarray, filepath):
    pcd = numpy_to_open3d(pcd_data)
    o3d.io.write_point_cloud(filepath, pcd)


def write_bin(pcd_data: np.ndarray, filepath):
    pcd_data.tofile(filepath)


def pcd_to_bin(pcd_path, bin_path):
    pcd_data = read_pcd(pcd_path)
    write_bin(pcd_data, bin_path)


def rotation_matrix(roll, yaw, pitch):
    R = np.array([[np.cos(yaw)*np.cos(pitch), 
                   np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), 
                   np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                  [np.sin(yaw)*np.cos(pitch), 
                   np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), 
                   np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                  [-np.sin(pitch), 
                   np.cos(pitch)*np.sin(roll), 
                   np.cos(pitch)*np.cos(roll)]])
    return R


def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular :
        pitch = math.atan2(R[2,1] , R[2,2])
        roll = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0])
    else :
        pitch = math.atan2(-R[1,2], R[1,1])
        roll = math.atan2(-R[2,0], sy)
        yaw = 0
    
    return roll, yaw, pitch


def transformation_to_pose(R):
    x, y, z = R[0,3], R[1,3], R[2,3]

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
 
    if not singular :
        pitch = math.atan2(R[2,1] , R[2,2])
        roll = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0])
    else :
        pitch = math.atan2(-R[1,2], R[1,1])
        roll = math.atan2(-R[2,0], sy)
        yaw = 0
 
    return np.array([x, y, z, roll / np.pi * 180, yaw / np.pi * 180, pitch / np.pi * 180])


def pose_to_transformation(pose):
    x, y, z, roll, yaw, pitch = pose[0], pose[1], pose[2], np.radians(pose[3]), np.radians(pose[4]), np.radians(pose[5])
    R = np.array([[np.cos(yaw)*np.cos(pitch), 
                   np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), 
                   np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll),
                   x],
                  [np.sin(yaw)*np.cos(pitch), 
                   np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), 
                   np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll),
                   y],
                  [-np.sin(pitch), 
                   np.cos(pitch)*np.sin(roll), 
                   np.cos(pitch)*np.cos(roll),
                   z],
                  [0, 0, 0, 1]])
    return R


def bbox_map_to_sensor(bbox, sensor_calib):
    sensor_location = sensor_calib[:3]
    sensor_rotation = sensor_calib[3:] * np.pi / 180
    new_bbox = np.copy(bbox)
    if bbox.ndim == 1:
        new_bbox[:3] -= sensor_location
        new_bbox[:3] = np.dot(
                        np.linalg.inv(rotation_matrix(*(sensor_rotation))),
                        new_bbox[:3].T).T
        new_bbox[6] -= sensor_rotation[1]
    elif bbox.ndim == 2:
        new_bbox[:,:3] -= sensor_location
        new_bbox[:,:3] = np.dot(
                        np.linalg.inv(rotation_matrix(*(sensor_rotation))),
                        new_bbox[:,:3].T).T
        new_bbox[:,6] -= sensor_rotation[1]
    else:
        raise Exception("Wrong dimension of bbox")
    return new_bbox


def bbox_sensor_to_map(bbox, sensor_calib):
    sensor_location = sensor_calib[:3]
    sensor_rotation = sensor_calib[3:] * np.pi / 180
    new_bbox = np.copy(bbox)
    if bbox.ndim == 1:
        new_bbox[6] += sensor_rotation[1]
        new_bbox[:3] = np.dot(
                        rotation_matrix(*(sensor_rotation)),
                        new_bbox[:3].T).T
        new_bbox[:3] += sensor_location
    elif bbox.ndim == 2:
        new_bbox[:,6] += sensor_rotation[1]
        new_bbox[:,:3] = np.dot(
                        rotation_matrix(*(sensor_rotation)),
                        new_bbox[:,:3].T).T
        new_bbox[:,:3] += sensor_location
    else:
        raise Exception("Wrong dimension of bbox")
    return new_bbox


def pcd_sensor_to_map(pcd, sensor_calib):
    sensor_location = sensor_calib[:3]
    sensor_rotation = sensor_calib[3:] * np.pi / 180
    new_pcd = np.copy(pcd)
    new_pcd[:, :3] = np.dot(
                    rotation_matrix(*(sensor_rotation)),
                    new_pcd[:, :3].T).T
    new_pcd[:, :3] += sensor_location
    return new_pcd


def pcd_map_to_sensor(pcd, sensor_calib):
    sensor_location = sensor_calib[:3]
    sensor_rotation = sensor_calib[3:] * np.pi / 180
    new_pcd = np.copy(pcd)
    new_pcd[:, :3] -= sensor_location
    new_pcd[:, :3] = np.dot(
                        np.linalg.inv(rotation_matrix(*(sensor_rotation))),
                        new_pcd[:, :3].T).T
    return new_pcd


def bbox_shift(bbox, location):
    new_bbox = np.copy(bbox)
    if bbox.ndim == 1:
        new_bbox[:3] += location
    elif bbox.ndim == 2:
        new_bbox[:,:3] += location
    else:
        raise Exception("Wrong dimension of bbox")
    return new_bbox


def bbox_rotate(bbox, rotation):
    new_bbox = np.copy(bbox)
    if bbox.ndim == 1:
        new_bbox[:3] = np.dot(
                        rotation_matrix(*rotation),
                        new_bbox[:3].T).T
        new_bbox[6] += rotation[1]
    elif bbox.ndim == 2:
        new_bbox[:,:3] = np.dot(
                        rotation_matrix(*rotation),
                        new_bbox[:,:3].T).T
        new_bbox[:,6] += rotation[1]
    else:
        raise Exception("Wrong dimension of bbox")
    return new_bbox


def bbox_transform(bbox, location, rotation):
    return bbox_shift(bbox_rotate(bbox, rotation), location)


def point_shift(pcd, shift):
    new_pcd = pcd.copy()
    new_pcd[:,:3] += shift
    return new_pcd


def point_rotate(pcd, rotation):
    new_pcd = pcd.copy()
    m = rotation_matrix(*rotation)
    new_pcd = np.dot(m, new_pcd.T).T
    return new_pcd


def get_open3d_bbox(bbox):
    # KITTI format to open3d
    bbox_new = bbox.copy()
    bbox_new[2] += bbox_new[5] / 2
    # call open3d api
    o3d_bbox = o3d.geometry.OrientedBoundingBox(center=bbox_new[:3], R=rotation_matrix(0, bbox_new[6], 0), extent=bbox_new[3:6])
    return o3d_bbox


def get_point_indices_in_bbox(bbox: np.ndarray, points: np.ndarray):
    o3d_bbox = get_open3d_bbox(bbox)
    indices = o3d_bbox.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points[:,:3]))
    indices = np.array(indices).reshape(-1).astype(np.int32)
    return indices


def get_bbox_vertices(bbox: np.ndarray):
    points = np.array([
        [-bbox[3]/2, -bbox[4]/2, bbox[2]],
        [-bbox[3]/2, bbox[4]/2, bbox[2]],
        [bbox[3]/2, -bbox[4]/2, bbox[2]],
        [bbox[3]/2, bbox[4]/2, bbox[2]],
        [-bbox[3]/2, -bbox[4]/2, bbox[2]+bbox[5]],
        [-bbox[3]/2, bbox[4]/2, bbox[2]+bbox[5]],
        [bbox[3]/2, -bbox[4]/2, bbox[2]+bbox[5]],
        [bbox[3]/2, bbox[4]/2, bbox[2]+bbox[5]]
    ])

    points = np.dot(rotation_matrix(0, bbox[6], 0), points.T).T
    points[:,:2] += bbox[:2]
    return points


def get_distance(p1, p2=None):
    if p2 is not None:
        return np.sqrt(np.sum((p1 - p2) ** 2, axis=p1.ndim-1))
    else:
        return np.sqrt(np.sum(p1 ** 2, axis=p1.ndim-1))


def merge_pointclouds(case, ego_id=None):
    pointcloud_all = []
    if ego_id is not None:
        ego_lidar_pose = case[ego_id]["lidar_pose"]

    for vehicle_id, vehicle_data in case.items():
        pointcloud = vehicle_data["lidar"]
        lidar_pose = vehicle_data["lidar_pose"]
        pointcloud_map = pcd_sensor_to_map(pointcloud, lidar_pose)
        if ego_id is None:
            pointcloud_all.append(pointcloud_map[:, :3])
        else:
            pointcloud_all.append(pcd_map_to_sensor(pointcloud_map, ego_lidar_pose)[:, :3])
    pointcloud_all = np.vstack(pointcloud_all)

    return pointcloud_all


def lane_to_polygon(lane_data):
    lane_area = np.zeros((lane_data["left_boundary"].shape[0] + lane_data["right_boundary"].shape[0], 2))
    lane_area[:lane_data["left_boundary"].shape[0]] = lane_data["left_boundary"][:, :2]
    lane_area[lane_data["left_boundary"].shape[0]:] = lane_data["right_boundary"][::-1, :2]
    lane_area = Polygon(lane_area).simplify(tolerance=0.1)
    return lane_area


def sort_lidar_points(pcd, angle_gap=0.2):
    distance = np.sqrt(np.sum(pcd[:,:2] ** 2, axis=1))
    # Vertical angle
    v_angle = np.arctan2(distance, -pcd[:,2])
    v_angle_sort_indices = np.argsort(v_angle)[::-1]
    sorted_v_angle = v_angle[v_angle_sort_indices]
    angle_delta = sorted_v_angle - np.concatenate((np.array([sorted_v_angle[0]]), sorted_v_angle[:-1]), axis=None)
    rings = np.cumsum((angle_delta < -angle_gap / 180 * np.pi).astype(np.int8))
    # Horizontal angle
    h_angle = np.arctan2(pcd[:, 0], pcd[:, 1])

    points_stack = []
    sort_indices = []

    for ring_id in np.sort(np.unique(rings)):
        ring_indices = np.argwhere(rings == ring_id).reshape(-1)
        ring_h_angle = h_angle[v_angle_sort_indices][ring_indices]
        ring_shuffle_indices = np.argsort(ring_h_angle)
        points_stack.append(pcd[v_angle_sort_indices][ring_indices][ring_shuffle_indices])
        sort_indices.append(v_angle_sort_indices[ring_indices][ring_shuffle_indices])

    return np.vstack(points_stack), np.hstack(sort_indices).reshape(-1)


def bbox_to_corners_batch(bbox):
    assert corners.ndim == 3
    corners = []
    dx = np.vstack([bbox[:, 3] / 2 * np.cos(bbox[:, 6]), bbox[:, 3] / 2 * np.sin(bbox[:, 6])]).T
    dy = np.vstack([-bbox[:, 4] / 2 * np.sin(bbox[:, 6]), bbox[:, 4] / 2 * np.cos(bbox[:, 6])]).T
    center = bbox[:, :3]
    for x in [1, -1]:
        for y in [1, -1]:
            for z in [0, 1]:
                corner = copy.deepcopy(center)
                corner[:, :2] += x * dx
                corner[:, :2] += y * dy
                corner[:, 2] += z * bbox[:, 5]
                corners.append(corner)
    corners = np.array(corners)
    return corners


def bbox_to_corners(bbox):
    return bbox_to_corners_batch(bbox[np.newaxis, :])[0]


def corners_to_bbox_batch(corners):
    assert corners.ndim == 3
    batch_size = corners.shape[0]

    xyz = np.mean(corners[:, [0, 3, 5, 6], :], axis=1)
    h = abs(np.mean(corners[:, 4:, 2] - corners[:, :4, 2], axis=1,
                    keepdims=True))
    l = (np.sqrt(np.sum((corners[:, 0, [0, 1]] - corners[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corners[:, 2, [0, 1]] - corners[:, 1, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corners[:, 4, [0, 1]] - corners[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corners[:, 5, [0, 1]] - corners[:, 6, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    w = (np.sqrt(
        np.sum((corners[:, 0, [0, 1]] - corners[:, 1, [0, 1]]) ** 2, axis=1,
               keepdims=True)) +
         np.sqrt(np.sum((corners[:, 2, [0, 1]] - corners[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corners[:, 4, [0, 1]] - corners[:, 5, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corners[:, 6, [0, 1]] - corners[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    theta = (np.arctan2(corners[:, 1, 1] - corners[:, 2, 1],
                        corners[:, 1, 0] - corners[:, 2, 0]) +
             np.arctan2(corners[:, 0, 1] - corners[:, 3, 1],
                        corners[:, 0, 0] - corners[:, 3, 0]) +
             np.arctan2(corners[:, 5, 1] - corners[:, 6, 1],
                        corners[:, 5, 0] - corners[:, 6, 0]) +
             np.arctan2(corners[:, 4, 1] - corners[:, 7, 1],
                        corners[:, 4, 0] - corners[:, 7, 0]))[:,
            np.newaxis] / 4

    bbox = np.concatenate([xyz, l, w, h, theta], axis=1).reshape(batch_size, 7)
    bbox[:, 2] -= 0.5 * bbox[:, 5]
    return bbox


def corners_to_bbox(corners):
    return corners_to_bbox_batch(corners[np.newaxis, :])[0]