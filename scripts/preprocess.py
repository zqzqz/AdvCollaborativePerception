import os, sys
import numpy as np
import pickle
from collections import OrderedDict
import yaml
import open3d as o3d
mvp_root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

load_camera = False
load_lidar = False
load_calib = False
datadir = os.path.join(mvp_root, "data/OPV2V")

def process_scenario(dataset, scenario):
    scenario_path = os.path.join(datadir, dataset, scenario)
    print(scenario_path)
    scenario_dict = {}

    vehicle_ids = []
    for x in os.listdir(scenario_path):
        if os.path.isdir(os.path.join(scenario_path, x)):
            vehicle_ids.append(int(x))
    vehicle_ids.sort()
    assert(len(vehicle_ids) > 0)
    
    sample_lidar_path = os.path.join(scenario_path, "{}".format(vehicle_ids[0]))
    frame_ids = []
    for x in os.listdir(sample_lidar_path):
        if x.split('.')[1] == "yaml":
            frame_ids.append(int(x.split('.')[0]))
    frame_ids.sort()

    scenario_dict["vehicle_ids"] = vehicle_ids
    scenario_dict["frame_ids"] = frame_ids
    scenario_dict["data"] = OrderedDict()
    scenario_dict["label"] = OrderedDict()
    for frame_id in frame_ids:
        scenario_dict["data"][frame_id] = OrderedDict()
        scenario_dict["label"][frame_id] = {}
        for vehicle_id in vehicle_ids:
            vehicle_data = {
                "lidar": os.path.join(dataset, scenario, "{}".format(vehicle_id), "{:06d}.pcd".format(frame_id)),
                "camera0": os.path.join(dataset, scenario, "{}".format(vehicle_id), "{:06d}_camera0.png".format(frame_id)),
                "camera1": os.path.join(dataset, scenario, "{}".format(vehicle_id), "{:06d}_camera1.png".format(frame_id)),
                "camera2": os.path.join(dataset, scenario, "{}".format(vehicle_id), "{:06d}_camera2.png".format(frame_id)),
                "camera3": os.path.join(dataset, scenario, "{}".format(vehicle_id), "{:06d}_camera3.png".format(frame_id)),
                "calib": os.path.join(dataset, scenario, "{}".format(vehicle_id), "{:06d}.yaml".format(frame_id)),
            }

            with open(os.path.join(datadir, vehicle_data["calib"]), 'r') as f:
                label_data = yaml.load(f, Loader=yaml.Loader)
                for object_id, label in label_data["vehicles"].items():
                    if object_id not in scenario_dict["label"][frame_id]:
                        scenario_dict["label"][frame_id][object_id] = label

            if load_camera:
                pass
            if load_lidar:
                pcd = o3d.io.read_point_cloud(os.path.join(datadir, vehicle_data["lidar"]))
                vehicle_data["lidar"] = np.asarray(pcd.points)
            if load_calib:
                with open(os.path.join(datadir, vehicle_data["calib"]), 'r') as f:
                    calib = yaml.load(f, Loader=yaml.Loader)
                vehicle_data["calib"] = calib

            scenario_dict["data"][frame_id][vehicle_id] = vehicle_data
        
    return scenario_dict


def process_dataset(dataset):
    dataset_path = os.path.join(datadir, dataset)
    print(dataset_path)
    dataset_dict = {}
    for scenario in os.listdir(dataset_path):
        scenario_path = os.path.join(dataset_path, scenario)
        if not os.path.isdir(scenario_path):
            continue
        scenario_dict = process_scenario(dataset, scenario)
        dataset_dict[scenario] = scenario_dict
    return dataset_dict


def preprocess():
    print(datadir)
    for dataset in ["test"]:
        dataset_dict = process_dataset(dataset)
        with open(os.path.join(datadir, "{}.pkl".format(dataset)), 'wb') as f:
            pickle.dump(dataset_dict, f)


preprocess()
