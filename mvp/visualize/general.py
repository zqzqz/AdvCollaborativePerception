import numpy as np
import open3d as o3d
from mvp.data.util import rotation_matrix
import cv2
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Polygon

from mvp.config import model_3d_examples, color_map
from mvp.data.util import bbox_shift, bbox_rotate, pcd_sensor_to_map, pcd_map_to_sensor, bbox_sensor_to_map, get_open3d_bbox


def set_equal_axis_scale(ax):
    ax.set_aspect('equal', adjustable='box')


def show_or_save(show, save):
    if show:
        plt.show()
    if save:
        plt.savefig(save)
    if show or save:
        plt.close()
    

def get_xylims(points):
    xlim, ylim = [points[:,0].min(), points[:,0].max()], [points[:,1].min(), points[:,1].max()]
    lim = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    xlim = [sum(xlim)/2 - lim/2, sum(xlim)/2 + lim/2]
    ylim = [sum(ylim)/2 - lim/2, sum(ylim)/2 + lim/2]
    return xlim, ylim


def draw_bboxes_2d(ax, bboxes, bboxes_ids, color, linewidth=1):
    for i in range(bboxes.shape[0]):
        boxp = cv2.boxPoints(((bboxes[i][0], bboxes[i][1]), (bboxes[i][3], bboxes[i][4]), bboxes[i][6]/np.pi*180))
        boxp = np.insert(boxp, boxp.shape[0], boxp[0,:], 0)
        xs, ys = zip(*boxp)
        ax.plot(xs, ys, linewidth=linewidth, color=color)
        if bboxes_ids is not None:
            ax.text(xs[0], ys[0], str(bboxes_ids[i]), fontsize='xx-small')


def draw_bbox_2d(ax, bboxes_id_color):
    for bboxes, bboxes_ids, color in bboxes_id_color:
        for i in range(bboxes.shape[0]):
            boxp = cv2.boxPoints(((bboxes[i][0], bboxes[i][1]), (bboxes[i][3], bboxes[i][4]), bboxes[i][6]/np.pi*180))
            boxp = np.insert(boxp, boxp.shape[0], boxp[0,:], 0)
            xs, ys = zip(*boxp)
            ax.plot(xs, ys, linewidth=1, color=color)
            if bboxes_ids is not None:
                ax.text(xs[0], ys[0], str(bboxes_ids[i]), fontsize='xx-small')


def draw_polygons(ax, polygons, fill=True, border=True, color="r", alpha=1, linewidth=1):
    if isinstance(polygons, Polygon):
        polygons = [polygons]
    polygons = sum([[p] if isinstance(p, Polygon) else [pp for pp in p] for p in polygons], [])
    for polygon in polygons:
        x, y = polygon.exterior.coords.xy
        if fill:
            plt.fill(x, y, color=color, alpha=alpha)
        if border:
            x.append(x[0])
            y.append(y[0])
            plt.plot(x, y, color=color, linewidth=linewidth)


def draw_trajectories(ax, trajectories, trajectory_ids=None, color="b", markersize=10, **kwargs):
    if not isinstance(trajectories, list):
        trajectories = [trajectories]
    if trajectory_ids is not None and not isinstance(trajectory_ids, list):
        trajectory_ids = [trajectory_ids]

    for index, trajectory in enumerate(trajectories):
        ax.plot(trajectory[:, 0], trajectory[:, 1], "o-", color=color, markersize=markersize, **kwargs)
        ax.plot(trajectory[0, 0], trajectory[0, 1], "o", color=color, markersize=markersize * 2, **kwargs)
        if trajectory_ids is not None and trajectory_ids[index] is not None:
            ax.text(trajectory[0, 0] + 0.5, trajectory[0, 1] + 0.5, str(trajectory_ids[index]), fontsize='xx-small')


def draw_pointclouds(ax, pointclouds, color="b"):
    if isinstance(pointclouds, list):
        pointcloud_all = np.vstack([p[:,:3] for p in pointclouds])
    else:
        pointcloud_all = pointclouds[:,:3]
    ax.scatter(pointcloud_all[:,0], pointcloud_all[:,1], s=0.1, c=color)


def draw_matplotlib(pointclouds, gt_bboxes=None, pred_bboxes=None, gt_bboxes_ids=None, pred_bboxes_ids=None, show=False, save=None):
    if isinstance(pointclouds, list):
        pointcloud_all = np.vstack([p[:,:3] for p in pointclouds])
    else:
        pointcloud_all = pointclouds[:,:3]

    fig, ax = plt.subplots(figsize=(40,40))
    xlim, ylim = get_xylims(pointcloud_all)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.scatter(pointcloud_all[:,0], pointcloud_all[:,1], s=0.1, c="blue")

    total_bboxes = []
    if gt_bboxes is not None:
        total_bboxes.append((gt_bboxes, gt_bboxes_ids, "g"))
    if pred_bboxes is not None:
        total_bboxes.append((pred_bboxes, pred_bboxes_ids, "r"))

    draw_bbox_2d(ax, total_bboxes)
    set_equal_axis_scale(ax)

    show_or_save(show, save)
    return ax


def draw_open3d(pointclouds, gt_bboxes=None, pred_bboxes=None, show=True, save=""):
    # TODO: paint different pcd in various colors?
    pointcloud_all = o3d.geometry.PointCloud()
    if isinstance(pointclouds, list):
        pointcloud_all.points = o3d.utility.Vector3dVector(np.vstack([p[:,:3] for p in pointclouds]))
    else:
        pointcloud_all.points = o3d.utility.Vector3dVector(pointclouds[:,:3])
    
    pointcloud_all.paint_uniform_color([0, 0, 1])

    objects_to_draw = [pointcloud_all]

    total_bboxes = []
    if gt_bboxes is not None:
        total_bboxes.append((gt_bboxes, [0, 1, 0]))
    if pred_bboxes is not None:
        total_bboxes.append((pred_bboxes, [1, 0, 0]))

    for bboxes, color in total_bboxes:
        for i in range(bboxes.shape[0]):
            bbox = get_open3d_bbox(bboxes[i])
            bbox.color = color
            objects_to_draw.append(bbox)

    o3d.visualization.draw_geometries(objects_to_draw)


def draw_multi_vehicle_case(case, ego_id=None, mode="matplotlib", gt_bboxes=None, pred_bboxes=None, system="map", show=False, save=None, center=np.zeros(3)):
    pointcloud_all = []

    for vehicle_id, vehicle_data in case.items():
        pointcloud = vehicle_data["lidar"].astype(np.float64)
        if system == "map":
            lidar_pose = vehicle_data["lidar_pose"]
            pointcloud_map = pcd_sensor_to_map(pointcloud, lidar_pose)
            pointcloud_all.append(pointcloud_map)
        elif system == "ego":
            lidar_pose = vehicle_data["lidar_pose"]
            pointcloud_map = pcd_sensor_to_map(pointcloud, lidar_pose)
            pointcloud_ego = pcd_map_to_sensor(pointcloud_map, case[ego_id]["lidar_pose"])
            pointcloud_all.append(pointcloud_ego)
    pointcloud_all = np.vstack(pointcloud_all)
    pointcloud_all[:, :3] -= center

    if gt_bboxes is not None:
        if system == "map":
            gt_bboxes = bbox_sensor_to_map(gt_bboxes, case[ego_id]["lidar_pose"])
        gt_bboxes[:, :3] -= center
    
    if pred_bboxes is not None:
        if system == "map":
            pred_bboxes = bbox_sensor_to_map(pred_bboxes, case[ego_id]["lidar_pose"])
        pred_bboxes[:, :3] -= center
    
    if mode == "matplotlib":
        return draw_matplotlib(pointcloud_all, gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes, show=show, save=save)
    elif mode == "open3d":
        return draw_open3d(pointcloud_all, gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes, show=show, save=save)
    else:
        raise NotImplementedError()
