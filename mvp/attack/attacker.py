import pickle
import os
import copy
import numpy as np

from mvp.config import data_root
from mvp.data.util import write_pcd, read_pcd, sort_lidar_points
from mvp.visualize.attack import draw_attack


class Attacker():
    def __init__(self):
        self.attack_list = None
        self.name = "base"
        self.dataset = None
        self.attack_list = []

    def set_dataset(self, dataset):
        self.dataset = dataset

    def build_benchmark_meta(self, write=False):
        raise NotImplementedError

    def save_benchmark_meta(self):
        with open(os.path.join(self.dataset.root_path, "attack", "{}.pkl".format(self.name)), 'wb') as f:
            pickle.dump(self.attack_list, f)

    def load_benchmark_meta(self):
        try:
            with open(os.path.join(self.dataset.root_path, "attack", "{}.pkl".format(self.name)), 'rb') as f:
                self.attack_list = pickle.load(f)
        except Exception as e:
            print("no benchmark found", e)

    def build_benchmark(self, write=False, resume=True):
        for attack_id, attack in enumerate(self.attack_list):
            print(attack_id)
            pcd_dir = os.path.join(self.dataset.root_path, "attack", self.name, "{:06d}".format(attack_id))
            if resume:
                if os.path.exists(os.path.join(pcd_dir, "attack_info.pkl")):
                    continue
            attacker = attack["attack_meta"]["attacker_vehicle_id"]
            case_id = attack["attack_meta"]["case_id"]
            case = self.dataset.get_case(case_id, "multi_frame")
            new_case = copy.deepcopy(case)
            attack_opts = copy.deepcopy(attack["attack_opts"])
            attack_opts.update({"bboxes": attack["attack_meta"]["bboxes"]})
            new_case, info = self.run(new_case, attack_opts)
            if write:
                os.makedirs(pcd_dir, exist_ok=True)
                pickle.dump(info, open(os.path.join(pcd_dir, "attack_info.pkl"), "wb"))

    def load_benchmark(self, index=True, frame_ids=[9], use_lidar=True, use_camera=False):
        for idx in range(len(self.attack_list)):
            attack, case = self.load_benchmark_by_id(idx, frame_ids=frame_ids, use_camera=use_camera)
            if index:
                yield idx, attack, case
            else:
                yield attack, case

    def load_benchmark_by_id(self, idx, frame_ids=[9], use_lidar=True, use_camera=False):
        attack = self.attack_list[idx]
        if frame_ids is None:
            frame_ids = [i for i in range(len(attack["attack_meta"]["frame_ids"]))]
        case_id, attacker = attack["attack_meta"]["case_id"], attack["attack_meta"]["attacker_vehicle_id"]
        case = self.dataset.get_case(case_id, tag="multi_frame", use_camera=use_camera, use_lidar=use_lidar)

        if use_lidar:
            attack_info = pickle.load(open(os.path.join(self.dataset.root_path, "attack", self.name, "{:06d}".format(idx), "attack_info.pkl"), "rb"))
            for i in frame_ids:
                try:
                    case[i][attacker]["lidar"] = self.apply_ray_tracing(case[i][attacker]["lidar"], **attack_info[i])
                except:
                    pass
        return attack, case

    @staticmethod
    def apply_ray_tracing(lidar, replace_indices=None, replace_data=None, ignore_indices=None, append_data=None):
        if replace_indices is not None and replace_indices.shape[0] > 0:
            lidar[replace_indices, :3] = replace_data
        if ignore_indices is not None and ignore_indices.shape[0] > 0:
            try:
                lidar = np.delete(lidar, ignore_indices, axis=0)
            except:
                pass
        if append_data is not None and append_data.shape[0] > 0:
            tmp_pcd = np.vstack([lidar[:, :3], append_data])
            tmp_pcd, _ = sort_lidar_points(tmp_pcd)
            lidar = np.hstack([tmp_pcd, np.ones((tmp_pcd.shape[0], 1))])
        return lidar
