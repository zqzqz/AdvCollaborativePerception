import os
import sys
import argparse
import pickle
import yaml
import numpy as np
import open3d as o3d


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


def draw_open3d(pointclouds, labels, show=True, save=""):
    # TODO: paint different pcd in various colors?
    pointcloud_all = o3d.geometry.PointCloud()
    pointcloud_all.points = o3d.utility.Vector3dVector(np.vstack([p.points for p in pointclouds]))
    pointcloud_all.paint_uniform_color([0, 0, 1])

    bboxes = []
    for i, label in labels.items():
        center = np.array(label["location"])+np.array([0, 0, label["extent"][2]])
        R = rotation_matrix(*(np.array(label["angle"])*np.pi/180))
        extent = np.array(label["extent"])*2
        bbox = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
        bbox.color = [1, 0, 0]
        bboxes.append(bbox)

    o3d.visualization.draw_geometries(bboxes + [pointcloud_all])


def main(args):
    if args.dataset not in ["train", "validate", "test"]:
        raise Exception("Wrong dataset")
    
    with open(os.path.join(args.datadir, "{}.pkl".format(args.dataset)), 'rb') as f:
        meta = pickle.load(f)

    if args.scenario not in meta:
        raise Exception("Wrong scenario")

    if args.frame not in meta[args.scenario]["data"]:
        raise Exception("Wrong frame")

    pointclouds = []
    for lidar_id in meta[args.scenario]["data"][args.frame]:
        pcd_path = meta[args.scenario]["data"][args.frame][lidar_id]["lidar"]
        pcd = o3d.io.read_point_cloud(pcd_path)
        calib_path = meta[args.scenario]["data"][args.frame][lidar_id]["calib"]
        with open(calib_path, 'r') as f:
            calib = yaml.load(f, Loader=yaml.Loader)
        print(np.mean(np.asarray(pcd.points), axis=0))
        pcd.points = o3d.utility.Vector3dVector(np.dot(
            rotation_matrix(*(np.array(calib["lidar_pose"][3:])*np.pi/180)), 
            np.asarray(pcd.points).T).T + np.array(calib["lidar_pose"][:3]))
        pointclouds.append(pcd)
    labels = meta[args.scenario]["label"][args.frame]

    draw_open3d(pointclouds, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to visualize point clouds")
    parser.add_argument("--datadir", type=str, default="data/OPV2V", help="Path to the dataset")
    parser.add_argument("--dataset", type=str, default="", help="Dataset train/validate/test")
    parser.add_argument("--scenario", type=str, default="", help="Scenario id")
    parser.add_argument("--frame", type=int, default=-1, help="Frame id")
    args = parser.parse_args()

    main(args)