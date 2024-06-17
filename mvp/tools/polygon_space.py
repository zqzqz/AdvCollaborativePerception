import numpy as np
from shapely.geometry import Point, shape, MultiPoint, Polygon
from shapely.ops import unary_union
from matplotlib.path import Path

from mvp.data.util import point_rotate, point_shift, pcd_sensor_to_map


def points_in_polygon(points, polygon):
    points = points[:, :2]
    x, y = polygon.exterior.coords.xy
    vertices = np.vstack([np.array(x), np.array(y)]).T
    p = Path(vertices)
    mask = p.contains_points(points)
    return mask


def points_to_polygon(points):
    area = MultiPoint(points).convex_hull
    return area


def bbox_to_polygon(bbox):
    points = np.array([
        [bbox[3]/2, bbox[4]/2, 0],
        [-bbox[3]/2, bbox[4]/2, 0],
        [-bbox[3]/2, -bbox[4]/2, 0],
        [bbox[3]/2, -bbox[4]/2, 0]
    ])
    points = point_rotate(points, [0, bbox[6], 0])
    points = point_shift(points, [bbox[0], bbox[1], 0])
    points = points[:,:2].tolist()
    return Polygon(points)


def get_occupied_space(pcd_data, object_segments, point_height=None, height_thres=0):
    occupied_areas = []
    occupied_areas_height = []
    for object_segment in object_segments:
        object_points = pcd_data[object_segment]
        object_point_height = point_height[object_segment]
        # object_points = object_points[object_point_height > height_thres]
        # if object_points.shape[0] <= 2:
        #     continue
        occupied_area = points_to_polygon(object_points[:,:2])
        occupied_areas.append(occupied_area)
        occupied_areas_height.append(object_point_height.max())

    return occupied_areas, occupied_areas_height


def get_free_space_fast(pcd_data, center, object_mask, in_lane_mask, point_height, height_thres=0, max_range=50, angle_split=360):
    distance = np.sqrt(np.sum((pcd_data[:,:2] - center[:2]) ** 2, axis=1))
    range_mask = distance < max_range
    pcd = pcd_data[range_mask]
    point_height = point_height[range_mask]
    in_lane_mask = in_lane_mask[range_mask]
    object_mask = object_mask[range_mask]

    rays = (pcd[:,:3] - center[:3]) / np.tile(np.sqrt(np.sum((pcd[:,:3] - center[:3]) ** 2, axis=1)), (3, 1)).T
    height_mask = point_height < height_thres
    pcd[height_mask,:3] -= np.tile(point_height[height_mask] - height_thres, (3, 1)).T * rays[height_mask]
    centered_pcd = pcd[:,:2] - center[:2]
    distance = np.sqrt(np.sum((pcd[:,:2] - center[:2]) ** 2, axis=1))
    is_object = np.logical_or(in_lane_mask == 0,
                    np.logical_or(
                        np.logical_and(point_height > height_thres, point_height < 3), 
                        np.logical_and(object_mask,
                            np.logical_not(height_mask))))

    angles = np.floor(np.arctan2(centered_pcd[:,1], centered_pcd[:,0]) / np.pi / 2 * angle_split).astype(np.int32)
    angles += (angles < 0) * angle_split

    angle_to_point = lambda a, d : (d * np.cos(a * 2 * np.pi / angle_split) + center[0], d * np.sin(a * 2 * np.pi / angle_split) + center[1])

    inner_area = []
    for angle in range(angle_split):
        angle_point_mask = (angles == angle)
        if np.sum(angle_point_mask) == 0:
            inner_area.append(angle_to_point(angle, max_range))
            inner_area.append(angle_to_point(angle + 1, max_range))
            continue
        angle_object_mask = is_object[angle_point_mask]
        if np.sum(angle_object_mask) == 0:
            inner_area.append(angle_to_point(angle, max_range))
            inner_area.append(angle_to_point(angle + 1, max_range))
            continue
        min_dist = np.min(distance[angle_point_mask][angle_object_mask])
        inner_area.append(angle_to_point(angle, min_dist))
        inner_area.append(angle_to_point(angle + 1, min_dist))
    
    return [Polygon(inner_area).simplify(tolerance=0.1)]


def get_free_space(lidar, lidar_pose, object_mask, in_lane_mask, point_height, height_thres=0, height_tolerance=0.1, angle_split=360, max_range=50, ray_count=64):
    distance = np.sqrt(np.sum(lidar[:,:2] ** 2, axis=1))
    range_mask = distance < max_range
    pcd = lidar[range_mask]
    point_height = point_height[range_mask]
    in_lane_mask = in_lane_mask[range_mask]
    object_mask = object_mask[range_mask]
    distance = distance[range_mask]

    rays = pcd[:,:3] / np.tile(np.sqrt(np.sum(pcd[:,:3] ** 2, axis=1)), (3, 1)).T
    height_mask = point_height < height_thres
    pcd[height_mask,:3] -= np.tile(point_height[height_mask] - height_thres, (3, 1)).T * rays[height_mask]
    is_object = np.logical_or(in_lane_mask == 0,
                    np.logical_or(point_height > height_thres + height_tolerance, 
                        np.logical_and(object_mask,
                            np.logical_not(height_mask))))

    theta = np.arctan2(distance, -pcd[:,2])
    theta_delta = theta - np.concatenate((np.array([theta[0]]), theta[:-1]), axis=None)
    rings = np.cumsum((theta_delta < - 0.2 / 180 * np.pi).astype(np.int32))
    rings = ray_count - 1 - rings

    angles = np.floor(np.arctan2(pcd[:,1], pcd[:,0]) / np.pi / 2 * angle_split).astype(np.int32)
    angles += (angles < 0) * angle_split
    unique_angles = np.unique(angles)

    free_space_map = np.zeros((angle_split, ray_count))
    distance_map = np.zeros((angle_split, ray_count, 2))
    for angle in unique_angles:
        for ring in range(ray_count):
            point_mask = np.logical_and(angles == angle, rings == ring)
            point_number = np.sum(point_mask)
            if point_number == 0:
                free_space_map[angle, ring] = -1
            else:
                distance_map[angle, ring, 0] = distance[point_mask].min()
                distance_map[angle, ring, 1] = distance[point_mask].max()
                if np.sum(np.logical_and(is_object, point_mask)) == 0:
                    free_space_map[angle, ring] = 1

    polygons = []
    angle_to_point = lambda a, d : [d * np.cos(a * 2 * np.pi / angle_split), d * np.sin(a * 2 * np.pi / angle_split)]
    sensor_points_to_map_polygon = lambda L : Polygon(pcd_sensor_to_map(np.hstack([np.array(L), -1.7 * np.ones((len(L), 1))]), lidar_pose)[:, :2])

    inner_area = []
    for angle in range(angle_split):
        if (free_space_map[angle, :] == 0).sum() == 0:
            farest_distance = distance_map[angle, :, 1].max()
            inner_area.append(angle_to_point(angle, farest_distance))
            inner_area.append(angle_to_point(angle + 1, farest_distance))
            free_space_map[angle, :] = 0
        else:
            min_dist = distance_map[angle, np.argwhere(free_space_map[angle, :] == 0).reshape(-1), 0].min()
            inner_area.append(angle_to_point(angle, min_dist))
            inner_area.append(angle_to_point(angle + 1, min_dist))
            removed_indices = np.argwhere(np.logical_and(free_space_map[angle, :] > 0, distance_map[angle, :, 1] <= min_dist)).reshape(-1)
            if len(removed_indices) > 0:
                free_space_map[angle, removed_indices] = 0
    
    polygons.append(sensor_points_to_map_polygon(inner_area))

    for angle in range(angle_split):
        ring_start = None
        for ring in range(ray_count):
            further_object_indices = np.argwhere(free_space_map[angle, ring:] == 0).reshape(-1)
            if free_space_map[angle, ring] > 0 and (len(further_object_indices) == 0 or \
                distance_map[angle, further_object_indices + ring, 0].min() > distance_map[angle, ring, 0]):
                if ring_start is None:
                    ring_start = ring
                elif ring == ray_count - 1 and ring_start < ray_count - 1:
                    polygons.append(sensor_points_to_map_polygon([
                        angle_to_point(angle, distance_map[angle, ring_start, 1]),
                        angle_to_point((angle + 1) % angle_split, distance_map[angle, ring_start, 1]),
                        angle_to_point((angle + 1) % angle_split, distance_map[angle, ring, 0]),
                        angle_to_point(angle, distance_map[angle, ring, 0]),
                    ]))
            else:
                if ring_start is not None and ring_start + 1 < ring:
                    polygons.append(sensor_points_to_map_polygon([
                        angle_to_point(angle, distance_map[angle, ring_start, 1]),
                        angle_to_point((angle + 1) % angle_split, distance_map[angle, ring_start, 1]),
                        angle_to_point((angle + 1) % angle_split, distance_map[angle, ring - 1, 0]),
                        angle_to_point(angle, distance_map[angle, ring - 1, 0]),
                    ]))
                ring_start = None

    free_area = unary_union(polygons)
    if isinstance(free_area, Polygon):
        free_area = [free_area]
    else:
        free_area = list(free_area.geoms)
    return free_area
