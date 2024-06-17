import os, sys
from mvp.config import third_party_root, model_root
squeezeseg_root = os.path.join(third_party_root, "SqueezeSegV3")
sys.path.append(os.path.join(squeezeseg_root, "src"))
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from tasks.semantic.modules.segmentator import *
from tasks.semantic.postproc.KNN import KNN
from common.laserscan import LaserScan

from mvp.config import class_id_map, class_id_inv_map
from mvp.tools.cluster import get_clusters
from mvp.tools.ground_detection import get_ground_plane


class SqueezeSegInterface():
    def __init__(self):
        self.modeldir = os.path.join(model_root, "SqueezeSegV3")  
        arch_cfg_path = os.path.join(self.modeldir, "arch_cfg.yaml")
        data_cfg_path = os.path.join(self.modeldir, "data_cfg.yaml")
        self.ARCH = yaml.safe_load(open(arch_cfg_path, 'r'))
        self.DATA = yaml.safe_load(open(data_cfg_path, 'r'))
        self.n_classes = len(self.DATA["learning_map_inv"])
        self.sensor = self.ARCH["dataset"]["sensor"]
        self.max_points = self.ARCH["dataset"]["max_points"]
        self.sensor_img_means = torch.tensor(self.sensor["img_means"],
                                            dtype=torch.float)
        self.sensor_img_stds = torch.tensor(self.sensor["img_stds"],
                                            dtype=torch.float)
        self.batch_size = 1

        # concatenate the encoder and the head
        with torch.no_grad():
            self.model = Segmentator(self.ARCH,
                                    self.n_classes,
                                    self.modeldir)

        # use knn post processing?
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                            self.n_classes)

        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

    def preprocess(self, lidar):
        scan = LaserScan(project=True,
                        H=self.sensor["img_prop"]["height"],
                        W=self.sensor["img_prop"]["width"],
                        fov_up=self.sensor["fov_up"],
                        fov_down=self.sensor["fov_down"])
        scan.set_points(lidar[:,:3], lidar[:,3])

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                        proj_xyz.clone().permute(2,0,1),
                        proj_remission.unsqueeze(0).clone()])

        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        return proj, proj_mask, proj_x, proj_y, proj_range, unproj_range, unproj_n_points

    def run(self, lidar):
        # switch to evaluate mode
        self.model.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            proj_in, proj_mask, p_x, p_y, proj_range, unproj_range, npoints = self.preprocess(lidar)
            p_x = p_x[:npoints]
            p_y = p_y[:npoints]
            proj_range = proj_range[:npoints]
            unproj_range = unproj_range[:npoints]
            proj_in = proj_in.unsqueeze(0)
            proj_mask = proj_mask.unsqueeze(0)

            if self.gpu:
                proj_in = proj_in.cuda()
                proj_mask = proj_mask.cuda()
                p_x = p_x.cuda()
                p_y = p_y.cuda()

                if self.post:
                    proj_range = proj_range.cuda()
                    unproj_range = unproj_range.cuda()

            proj_output, _, _, _, _ = self.model(proj_in, proj_mask)
            proj_argmax = proj_output[0].argmax(dim=0)

            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                            unproj_range,
                                            proj_argmax,
                                            p_x,
                                            p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

        new_pred_np = np.zeros(pred_np.shape)
        new_pred_np[pred_np == 1] = class_id_inv_map["car"]
        # new_pred_np[pred_np == 4] = class_id_inv_map["truck"]
        # new_pred_np[pred_np == 6] = class_id_inv_map["person"]
        # new_pred_np[pred_np == 2] = class_id_inv_map["bicycle"]

        _, ground_indices = get_ground_plane(lidar)
        new_pred_np[ground_indices] = 0

        object_indices = np.argwhere(new_pred_np == class_id_inv_map["car"]).reshape(-1)
        object_points = lidar[object_indices]
        cluster_indices_list = get_clusters(object_points)
        info = []
        for cluster_indices in cluster_indices_list:
            info.append({"indices": object_indices[cluster_indices], "category_id": class_id_inv_map["car"]})

        return {
            "class": new_pred_np,
            "info": info
        }