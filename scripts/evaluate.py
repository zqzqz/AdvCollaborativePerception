import os, sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
sys.path.append(root)
import pickle
import logging
import copy
import numpy as np
import traceback
from collections import OrderedDict

from mvp.config import data_root
from mvp.data.util import bbox_sensor_to_map, bbox_map_to_sensor, pcd_sensor_to_map, pcd_map_to_sensor, get_distance
from mvp.tools.iou import iou3d
from mvp.tools.polygon_space import bbox_to_polygon
from mvp.tools.squeezeseg.interface import SqueezeSegInterface
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.lidar_seg import lidar_segmentation
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.polygon_space import get_occupied_space, get_free_space, bbox_to_polygon
from mvp.visualize.attack import draw_attack
from mvp.visualize.defense import visualize_defense, draw_roc
from mvp.data.opv2v_dataset import OPV2VDataset
from mvp.perception.opencood_perception import OpencoodPerception
from mvp.attack.lidar_spoof_early_attacker import LidarSpoofEarlyAttacker
from mvp.attack.lidar_spoof_intermediate_attacker import LidarSpoofIntermediateAttacker
from mvp.attack.lidar_spoof_late_attacker import LidarSpoofLateAttacker
from mvp.attack.lidar_remove_early_attacker import LidarRemoveEarlyAttacker
from mvp.attack.lidar_remove_intermediate_attacker import LidarRemoveIntermediateAttacker
from mvp.attack.lidar_remove_late_attacker import LidarRemoveLateAttacker
from mvp.defense.perception_defender import PerceptionDefender


logging.basicConfig(level=logging.INFO)
result_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../result")
attack_frame_ids = [9]
total_frames = 10

dataset = OPV2VDataset(root_path=os.path.join(data_root, "OPV2V"), mode="test")

perception_list = [
    OpencoodPerception(fusion_method="early", model_name="pointpillar"),
    OpencoodPerception(fusion_method="intermediate", model_name="pointpillar"),
    OpencoodPerception(fusion_method="late", model_name="pointpillar"),
]
perception_dict = OrderedDict([(x.name, x) for x in perception_list])

attacker_list = [
    LidarSpoofEarlyAttacker(dataset, dense=0, sync=0),
    LidarSpoofEarlyAttacker(dataset, dense=1, sync=0),
    LidarSpoofEarlyAttacker(dataset, dense=2, sync=0),
    LidarSpoofEarlyAttacker(dataset, dense=2, sync=1),
    LidarSpoofEarlyAttacker(dataset, dense=3, sync=0),
    LidarSpoofEarlyAttacker(dataset, dense=3, sync=1),
    LidarRemoveEarlyAttacker(dataset, advshape=0, dense=0, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=0, dense=1, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=0, dense=2, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=0, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=1, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=2, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=2, sync=1),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=3, sync=0),
    LidarRemoveEarlyAttacker(dataset, advshape=1, dense=3, sync=1),
    LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=100, sync=0, init=False, online=False),
    LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=0, init=False, online=False),
    LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=False, online=False),
    LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=False, online=True),
    LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=0, init=True, online=False),
    LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=True, online=False),
    LidarSpoofIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=True, online=True),
    LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=100, sync=0, init=False, online=False),
    LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=0, init=False, online=False),
    LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=False, online=False),
    LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=False, online=True),
    LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=0, init=True, online=False),
    LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=True, online=False),
    LidarRemoveIntermediateAttacker(perception_dict["pointpillar_intermediate"], dataset, step=2, sync=1, init=True, online=True),
    LidarSpoofLateAttacker(perception_dict["pointpillar_late"], dataset),
    LidarRemoveLateAttacker(perception_dict["pointpillar_late"], dataset),
]
attacker_dict = OrderedDict([(x.name, x) for x in attacker_list])

defender_list = [
    PerceptionDefender(),
]
defender_dict = OrderedDict([(x.name, x) for x in defender_list])


pickle_cache = OrderedDict()
pickle_cache_size = 600


def pickle_cache_load(file_path):
    file_path = os.path.normpath(file_path)
    if file_path in pickle_cache:
        return pickle_cache[file_path]
    else:
        data = pickle.load(open(file_path, 'rb'))
        if len(pickle_cache) >= pickle_cache_size:
            pickle_cache.popitem(last=False)
        pickle_cache[file_path] = data
        return data
    

def pickle_cache_dump(data, file_path):
    file_path = os.path.normpath(file_path)
    if file_path in pickle_cache:
        pickle_cache[file_path] = data
    pickle.dump(data, open(file_path, 'wb'))


def normal_case_iterator(f):
    def wrapper(*args, **kwargs):
        for case_id, case in dataset.case_generator(tag="multi_frame", index=True, use_lidar=True, use_camera=False):
            data_dir = os.path.join(result_dir, "normal/{:06d}".format(case_id))
            os.makedirs(data_dir, exist_ok=True)

            kwargs.update({
                "case_id": case_id,
                "case": case,
                "data_dir": data_dir,
            })
            f(*args, **kwargs)
    return wrapper


def attack_case_iterator(f):
    def wrapper(*args, **kwargs):
        attacker = args[0]
        for attack_id, attack in enumerate(attacker.attack_list):
            data_dir = os.path.join(result_dir, "attack/{}/{:06d}".format(attacker.name, attack_id))
            os.makedirs(data_dir, exist_ok=True)
            case_id = attack["attack_meta"]["case_id"]
            case = dataset.get_case(case_id, tag="multi_frame", use_lidar=True, use_camera=False)

            kwargs.update({
                "case_id": case_id,
                "case": case,
                "data_dir": data_dir,
                "attack_id": attack_id,
                "attack": attack,
            })
            f(*args, **kwargs)
    return wrapper


@normal_case_iterator
def normal_perception(case_id=None, case=None, data_dir=None):
    for perception_name, perception in perception_dict.items():
        save_file = os.path.join(data_dir, "{}.pkl".format(perception_name))
        if os.path.isfile(save_file):
            logging.info("Skipped case_id {}".format(case_id))
            continue
        else:
            logging.info("Processing perception {} on normal case {}".format(perception.name, case_id))

        perception_feature = [{} for _ in range(total_frames)]
        for frame_id in attack_frame_ids:
            for vehicle_id in list(case[frame_id].keys()):
                pred_bboxes, pred_scores = perception.run(case[frame_id], ego_id=vehicle_id)
                perception_feature[frame_id][vehicle_id] = {"pred_bboxes": pred_bboxes, "pred_scores": pred_scores}

        pickle_cache_dump(perception_feature, save_file)
        

@attack_case_iterator
def attack_perception(attacker, case_id=None, case=None, data_dir=None, attack_id=None, attack=None):
    save_file = os.path.join(data_dir, "attack_info.pkl")
    if os.path.isfile(save_file):
        logging.info("Skipped attack {} and attack case {}".format(attacker.name, attack_id))
        return
    else:
        logging.info("Processing attack {} and attack case {}".format(attacker.name, attack_id))

    attack_opts = attack["attack_opts"]
    attack_opts["victim_vehicle_id"] = attack["attack_meta"]["victim_vehicle_id"]
    attack_opts["frame_ids"] = [9]
    attack["attack_meta"]["attack_frame_ids"] = [9]

    if (isinstance(attacker, LidarSpoofEarlyAttacker) or isinstance(attacker, LidarRemoveEarlyAttacker)) and attacker.dense == 2 and attacker.sync == 1:
        # Need to attack all frames here as the data is used for online intermediate-fusion attack.
        attack_opts["frame_ids"] = [i for i in range(10)]

    if (isinstance(attacker, LidarSpoofIntermediateAttacker) or isinstance(attacker, LidarRemoveIntermediateAttacker)):
        # Intermediate-fusion attacks need the result of ray casting.
        if attacker.init:
            attack_category = attacker.name.split('_')[1]
            attack_opts["attack_info"] = pickle_cache_load(os.path.join(data_dir, "../../lidar_{}_early_AS_DenseAll_Async/{:06d}/attack_info.pkl".format(attack_category, attack_id)))
        else:
            attack_opts["attack_info"] = [{} for _ in range(10)]
        if attacker.online:
            attack_opts["frame_ids"] = [i for i in range(1, 10)]

    new_case, attack_info = attacker.run(case, attack_opts)
    pickle_cache_dump(attack_info, save_file)

    if isinstance(attacker, LidarSpoofEarlyAttacker) or isinstance(attacker, LidarRemoveEarlyAttacker):
        # Early-fusion attacks are block box attacks. We need to apply certain models to evaluate their performance.
        for perception_name in ["pointpillar_early", "pointpillar_intermediate"]:
            perception_save_file = os.path.join(data_dir, "{}.pkl".format(perception_name))
            perception = perception_dict[perception_name]
            perception_feature = [{} for _ in range(total_frames)]
            for frame_id in attack_frame_ids:
                pred_bboxes, pred_scores = perception.run(new_case[frame_id], ego_id=attack_opts["victim_vehicle_id"])
                perception_feature[frame_id][attack_opts["victim_vehicle_id"]] = {"pred_bboxes": pred_bboxes, "pred_scores": pred_scores}
            pickle_cache_dump(perception_feature, perception_save_file)
            
            # Visualization
            dataset.load_feature(new_case, perception_feature)
            draw_attack(attack, case, new_case, mode="multi_frame", show=False, save=os.path.join(data_dir, "{}.png".format(perception_name)))
    else:
        # Visualization
        dataset.load_feature(new_case, attack_info)
        draw_attack(attack, case, new_case, mode="multi_frame", show=False, save=os.path.join(data_dir, "visualization.png"))


def attack_evaluation(attacker, perception_name):
    logging.info("Evaluating attack {} at perception {}".format(attacker.name, perception_name))
    case_number = len(attacker.attack_list)
    success_log = np.zeros(case_number).astype(bool)
    max_iou = np.zeros((case_number, 2)).astype(np.float32)
    best_score = np.zeros((case_number, 2)).astype(np.float32)

    save_dir = os.path.join(result_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)

    @attack_case_iterator
    def attack_evaluation_processor(attacker, perception_name, case_id=None, case=None, data_dir=None, attack_id=None, attack=None):
        ego_id = attack["attack_meta"]["victim_vehicle_id"]
        attacker_id = attack["attack_meta"]["attacker_vehicle_id"]
        case_id = attack["attack_meta"]["case_id"]
        attack_bbox = bbox_sensor_to_map(attack["attack_meta"]["bbox"][-1], case[9][attacker_id]["lidar_pose"])
        attack_bbox = bbox_map_to_sensor(attack_bbox, case[-1][ego_id]["lidar_pose"])

        feature_data = pickle_cache_load(os.path.join(result_dir, "normal/{:06d}/{}.pkl".format(case_id, perception_name)))
        
        pred_bboxes = feature_data[-1][ego_id]["pred_bboxes"]
        pred_scores = feature_data[-1][ego_id]["pred_scores"]
        for j, pred_bbox in enumerate(pred_bboxes):
            iou = iou3d(pred_bbox, attack_bbox)
            if iou > max_iou[attack_id, 0]:
                max_iou[attack_id, 0] = iou
                best_score[attack_id, 0] = pred_scores[j]

        if "early" in attacker.name:
            feature_data = pickle_cache_load(os.path.join(data_dir, "{}.pkl".format(perception_name)))
        else:
            feature_data = pickle_cache_load(os.path.join(data_dir, "attack_info.pkl"))

        pred_bboxes = feature_data[-1][ego_id]["pred_bboxes"]
        pred_scores = feature_data[-1][ego_id]["pred_scores"]
        for j, pred_bbox in enumerate(pred_bboxes):
            iou = iou3d(pred_bbox, attack_bbox)
            if iou > max_iou[attack_id, 1]:
                max_iou[attack_id, 1] = iou
                best_score[attack_id, 1] = pred_scores[j]

        if attacker.name.startswith("lidar_spoof") and max_iou[attack_id, 1] > 0:
            success_log[attack_id] = True
        if attacker.name.startswith("lidar_remove") and max_iou[attack_id, 1] == 0:
            success_log[attack_id] = True

    attack_evaluation_processor(attacker, perception_name)

    pickle_cache_dump({"success": success_log, "iou": max_iou, "score": best_score},
                      os.path.join(save_dir, "attack_result_{}_{}.pkl".format(attacker.name, perception_name)))

    logging.info("Evaluation of attack {} at perception {}, total case number {:.2f}, success number {:.2f}, success rate {:.2f}, average IoU {:.2f}, average score {:.2f},".format(
        attacker.name, perception_name, success_log.shape[0], np.sum(success_log > 0), np.mean(success_log), np.mean(max_iou[:, 1]), np.mean(best_score[:, 1])))


@normal_case_iterator
def occupancy_map(lidar_seg_api, case_id=None, case=None, data_dir=None):
    save_file = os.path.join(data_dir, "occupancy_map.pkl")
    if os.path.isfile(save_file):
        logging.info("Skipped case {}".format(case_id))
        return
    else:
        logging.info("Processing occupancy map of case {}".format(case_id))

    occupancy_feature = [{} for _ in range(total_frames)]
    for frame_id in attack_frame_ids:
        for vehicle_id, vehicle_data in case[frame_id].items():
            lidar, lidar_pose = vehicle_data["lidar"], vehicle_data["lidar_pose"]
            pcd = pcd_sensor_to_map(lidar, lidar_pose)

            lane_info = pickle_cache_load(os.path.join(data_root, "carla/{}_lane_info.pkl".format(vehicle_data["map"])))
            lane_areas = pickle_cache_load(os.path.join(data_root, "carla/{}_lane_areas.pkl".format(vehicle_data["map"])))
            lane_planes = pickle_cache_load(os.path.join(data_root, "carla/{}_ground_planes.pkl".format(vehicle_data["map"])))

            ground_indices, in_lane_mask, point_height = get_ground_plane(pcd, lane_info=lane_info, lane_areas=lane_areas, lane_planes=lane_planes, method="map")
            lidar_seg = lidar_segmentation(lidar, method="squeezeseq", interface=lidar_seg_api)
            
            object_segments = filter_segmentation(lidar, lidar_seg, lidar_pose, in_lane_mask=in_lane_mask, point_height=point_height, max_range=50)
            object_mask = np.zeros(pcd.shape[0]).astype(bool)
            if len(object_segments) > 0:
                object_indices = np.hstack(object_segments)
                object_mask[object_indices] = True

            ego_bbox = vehicle_data["ego_bbox"]
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
            
            occupancy_feature[frame_id][vehicle_id] = ret

    pickle_cache_dump(occupancy_feature, save_file)


@attack_case_iterator
def defense(attacker, defender, perception_name, case_id=None, case=None, data_dir=None, attack_id=None, attack=None):
    if "early" in attacker.name:
        save_file = os.path.join(data_dir, "{}_{}.pkl".format(defender.name, perception_name))
        vis_file = os.path.join(data_dir, "{}_{}.png".format(defender.name, perception_name))
    else:
        save_file = os.path.join(data_dir, "{}.pkl".format(defender.name))
        vis_file = os.path.join(data_dir, "{}.png".format(defender.name))
    # if os.path.isfile(save_file):
    #     logging.info("Skipped case {}".format(case_id))
    #     return
    # else:
    #     logging.info("Processing defense {} against attack {} on attack case {}".format(defender.name, attacker.name, attack_id))
    logging.info("Processing defense {} against attack {} on attack case {}".format(defender.name, attacker.name, attack_id))

    if "early" in attacker.name:
        perception_feature = pickle_cache_load(os.path.join(data_dir, "{}.pkl".format(perception_name)))
    else:
        perception_feature = pickle_cache_load(os.path.join(data_dir, "attack_info.pkl"))
    case = dataset.load_feature(case, perception_feature)

    occupancy_feature = pickle_cache_load(os.path.join(result_dir, "normal/{:06d}/occupancy_map.pkl".format(case_id)))
    case = dataset.load_feature(case, occupancy_feature)

    defend_opts = {"frame_ids": [9]}
    new_case, score, metrics = defender.run(case, defend_opts)

    pickle_cache_dump(metrics, save_file)
    visualize_defense(case, metrics, show=False, save=vis_file)


def defense_evaluation(attacker, defender, perception_name):
    save_dir = os.path.join(result_dir, "evaluation")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "defense_result_{}_{}_{}.pkl".format(attacker.name, defender.name, perception_name))

    defense_results = {
        "spoof_error": [],
        "spoof_label": [],
        "spoof_location": [],
        "remove_error": [],
        "remove_label": [],
        "remove_location": [],
        "success": [],
    }

    @attack_case_iterator
    def defense_evaluation_processor(attacker, defender, perception_name, case_id=None, case=None, data_dir=None, attack_id=None, attack=None, iou_thres=0.7, dist_thres=40):
        if "early" in attacker.name:
            defense_file = os.path.join(data_dir, "{}_{}.pkl".format(defender.name, perception_name))
        else:
            defense_file = os.path.join(data_dir, "{}.pkl".format(defender.name))
        metrics = pickle_cache_load(defense_file)

        attacker_vehicle_id = attack["attack_meta"]["attacker_vehicle_id"]
        victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
        attack_mode =  "spoof" if "spoof" in attacker.name else "remove"
        attack_bbox = bbox_sensor_to_map(attack["attack_meta"]["bboxes"][-1], case[-1][attacker_vehicle_id]["lidar_pose"])

        victim_vehicle_id = attack["attack_meta"]["victim_vehicle_id"]
        for frame_id in attack_frame_ids:
            vehicle_metrics = metrics[frame_id][victim_vehicle_id]

        gt_bboxes = vehicle_metrics["gt_bboxes"]
        pred_bboxes = vehicle_metrics["pred_bboxes"]
        lidar_pose = vehicle_metrics["lidar_pose"]

        # iou 2d
        gt_bboxes[:, 2] = 0
        gt_bboxes[:, 5] = 1
        pred_bboxes[:, 2] = 0
        pred_bboxes[:, 5] = 1

        iou = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
        for i, gt_bbox in enumerate(gt_bboxes):
            for j, pred_bbox in enumerate(pred_bboxes):
                iou[i, j] = iou3d(gt_bbox, pred_bbox)

        spoof_label = np.max(iou, axis=0) <= iou_thres
        spoof_mask = np.logical_and(get_distance(pred_bboxes[:, :2], lidar_pose[:2]) > 1, get_distance(pred_bboxes[:, :2], lidar_pose[:2]) <= dist_thres)
        remove_label = np.max(iou, axis=1) <= iou_thres
        remove_mask = get_distance(gt_bboxes[:, :2], lidar_pose[:2]) <= dist_thres

        spoof_error = np.zeros(pred_bboxes.shape[0])
        spoof_location = np.zeros((pred_bboxes.shape[0], 2))
        for error_area, error, gt_error, bbox_index in vehicle_metrics["spoof"]:
            if error > spoof_error[bbox_index]:
                spoof_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
                spoof_error[bbox_index] = error

        remove_error = np.zeros(gt_bboxes.shape[0])
        remove_location = np.zeros((gt_bboxes.shape[0], 2))
        for error_area, error, gt_error, bbox_index in vehicle_metrics["remove"]:
            if bbox_index < 0:
                continue
            if error > remove_error[bbox_index]:
                remove_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
                remove_error[bbox_index] = error

        detected_location = spoof_location if attack_mode == "spoof" else remove_location
        is_success = np.min(get_distance(detected_location, attack_bbox[:2])) < 2

        defense_results["spoof_error"].append(spoof_error[spoof_mask])
        defense_results["spoof_label"].append(spoof_label[spoof_mask])
        defense_results["spoof_location"].append(spoof_location[spoof_mask])
        defense_results["remove_error"].append(remove_error[remove_mask])
        defense_results["remove_label"].append(remove_label[remove_mask])
        defense_results["remove_location"].append(remove_location[remove_mask])
        defense_results["success"].append(np.array([is_success]).astype(np.int8))

    defense_evaluation_processor(attacker, defender, perception_name)

    for key, data in defense_results.items():
        defense_results[key] = np.concatenate(data).reshape(-1)

    pickle_cache_dump(defense_results, save_file)
    spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres = draw_roc(defense_results["spoof_error"], defense_results["spoof_label"],
            save=os.path.join(save_dir, "roc_lidar_spoof_{}_{}_{}.png".format(attacker.name, defender.name, perception_name)))
    remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres = draw_roc(defense_results["remove_error"], defense_results["remove_label"],
            save=os.path.join(save_dir, "roc_lidar_remove_{}_{}_{}.png".format(attacker.name, defender.name, perception_name)))
    
    attack_result = pickle_cache_load(os.path.join(save_dir, "attack_result_{}_{}.pkl".format(attacker.name, perception_name)))
    success_rate = np.mean(attack_result["success"] * defense_results["success"])
    
    logging.info("Evaluation of defense {} against attack {} on perception {} success rate {:.2f}: For spoofing attack, best TPR {:.2f}, best FPR {:.2f}, ROC AUC {:.2f}, best threshold {:.2f}; For removal attack, best TPR {:.2f}, best FPR {:.2f}, ROC AUC {:.2f}, best threshold {:.2f}." .format(
        defender.name, attacker.name, perception_name, success_rate,
        spoof_best_TPR, spoof_best_FPR, spoof_roc_auc, spoof_best_thres, remove_best_TPR, remove_best_FPR, remove_roc_auc, remove_best_thres
    ))


def main():
    normal_perception()

    for attacker_name, attacker in attacker_dict.items():
        attack_perception(attacker)
        if "early" in attacker_name:
            for perception_name in ["pointpillar_early", "pointpillar_intermediate"]:
                attack_evaluation(attacker, perception_name)
        else:
            attack_evaluation(attacker, attacker.perception.name)
        
    lidar_seg_api = SqueezeSegInterface()
    occupancy_map(lidar_seg_api)

    for attacker_name, attacker in attacker_dict.items():
        for defender_name, defender in defender_dict.items():
            if "early" in attacker_name:
                for perception_name in ["pointpillar_early", "pointpillar_intermediate"]:
                    defense(attacker, defender, perception_name)
                    defense_evaluation(attacker, defender, perception_name)
            else:
                defense(attacker, defender, attacker.perception.name)
                defense_evaluation(attacker, defender, perception_name)


if __name__ == "__main__":
    main()
