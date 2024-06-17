import open3d as o3d
import numpy as np
import math
import scipy

from mvp.data.util import numpy_to_open3d
from mvp.tools.polygon_space import points_in_polygon


def fit_plane(points):
    A = np.copy(points)
    if A[:,2].max() - A[:,2].min() < 0.1:
        return np.array([0, 0, 1, -A[:,2].mean()])
    else:
        b = np.ones(3)
        x = np.linalg.solve(A, b)
        return np.array([x[0]/x[2], x[1]/x[2], 1, -1/x[2]])


def get_inliers(pcd_data, plane_model, point_err_thres=0.2):
    [a, b, c, d] = plane_model
    x = np.sum(pcd_data[:,:3] * np.array([a, b, c]), axis=1) + d
    inliers = np.argwhere(np.absolute(x) < point_err_thres).reshape(-1)
    return inliers


def get_ground_plane_map(pcd_data, lane_info, lane_areas, lane_planes):
    ground_mask = np.zeros(pcd_data.shape[0])
    in_lane_mask = np.zeros(pcd_data.shape[0])
    point_height = np.ones(pcd_data.shape[0]) * 100
    for i, lane_area in enumerate(lane_areas):
        point_mask = points_in_polygon(pcd_data[:, :2], lane_area)
        if np.sum(point_mask) == 0:
            continue
        point_indices = np.argwhere(point_mask > 0).reshape(-1)
        in_lane_mask[point_indices] = 1
        points = pcd_data[point_indices]
        ref_points_list = [np.array(lane_info[i][k]) for k in ["xyz_left", "xyz_right"]]
        ref_indices_list = []
        for ref_points in ref_points_list:
            D = scipy.spatial.distance.cdist(points[:, :2], ref_points[:, :2])
            ref_indices_list.append(np.argmin(D, axis=1))
        for pi in range(points.shape[0]):
            plane_points = [ref_points_list[0][ref_indices_list[0][pi]],
                            ref_points_list[0][ref_indices_list[0][pi] + 1 if ref_indices_list[0][pi] < ref_points_list[0].shape[0] - 1 else ref_indices_list[0][pi] - 1],
                            ref_points_list[1][ref_indices_list[1][pi]]]
            plane = fit_plane(plane_points)
            point_height[point_indices[pi]] = points[pi, 2] + (plane[0] * points[pi, 0] + plane[1] * points[pi, 1] + plane[3]) / plane[2]
        inliers = np.argwhere(np.absolute(point_height[point_indices]) < 0.2).reshape(-1)
        ground_mask[point_indices[inliers]] = 1
    inliers = np.argwhere(ground_mask > 0).reshape(-1)
    return inliers, in_lane_mask.astype(bool), point_height


def get_ground_plane_ransac(pcd_data, mask=None, err_thres=0.2):
    if mask is not None:
        clipped_pcd = pcd_data[mask]
    else:
        clipped_pcd = pcd_data
    pcd = numpy_to_open3d(clipped_pcd)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                            ransac_n=3,
                                            num_iterations=1000)
    inliers = np.array(get_inliers(clipped_pcd, plane_model, point_err_thres=err_thres)).reshape(-1)
    if mask is not None:
        inliers = np.argwhere(mask > 0).reshape(-1)[inliers]
    return plane_model, inliers


def get_ground_plane_naive(pcd_data, ground_z=0, point_err_thres=0.1):
    plane_model = [0, 0, 1, -ground_z]
    inliers = get_inliers(pcd_data, plane_model, point_err_thres=point_err_thres)
    return plane_model, inliers


def get_ground_plane(pcd_data, **kwargs):
    default_method = "ransac"
    func_map = {
        "naive": get_ground_plane_naive,
        "ransac": get_ground_plane_ransac,
        "map": get_ground_plane_map
    }

    if "method" in kwargs:
        method = kwargs["method"]
        if method not in list(func_map.keys()):
            method = default_method
        del kwargs["method"]
    else:
        method = default_method
    
    return func_map[method](pcd_data, **kwargs)


def get_ground_mesh(plane_model, center=(0,0), margin=100):
    # ax + by + cz + d = 0
    [a, b, c, d] = plane_model
    corners = [[center[0]+margin, center[1]+margin], [center[0]+margin, center[1]-margin], [center[0]-margin, center[1]-margin], [center[0]-margin, center[1]+margin]]
    for corner in corners:
        [x, y] = corner
        z = (a * x + b * y + d) / (-c)
        corner.append(z)
    corners.append([center[0],center[1],-margin])
    
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(np.array(corners)),
        triangles=o3d.utility.Vector3iVector(np.array([[2,1,0],[3,2,0],[0,1,4],[1,2,4],[2,3,4],[3,0,4]]))
    )
    return mesh


def get_point_height(points, plane_model):
    x, y, z = points[:,0], points[:,1], points[:,2]
    a, b, c, d = plane_model[0], plane_model[1], plane_model[2], plane_model[3]
    return z + (a * x + b * y + d) / c
