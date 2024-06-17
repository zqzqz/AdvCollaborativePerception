import numpy as np

from mvp.tools.iou import iou2d
from mvp.visualize.general import draw_matplotlib


def lidar_to_camera(lidar, extrinsic, intrinsic):
    lidar_in_camera_3d = lidar.copy()
    lidar_in_camera_3d[:,3] = 1
    lidar_in_camera_3d = np.dot(lidar_in_camera_3d, extrinsic[:3,:].T)
    lidar_in_camera_3d = np.vstack([lidar_in_camera_3d[:,1], -lidar_in_camera_3d[:,2], lidar_in_camera_3d[:,0]]).T
    lidar_in_camera_2d = np.dot(lidar_in_camera_3d, intrinsic.T)
    scale = lidar_in_camera_2d[:,2]
    lidar_in_camera_2d[:,0] /= scale
    lidar_in_camera_2d[:,1] /= scale
    return lidar_in_camera_2d


def get_lidar_2d_bbox(pcd_on_camera, indices, image_shape):
    points = pcd_on_camera[indices]
    depth = np.mean(points[:,2])
    if depth < 0:
        return None
    x = points[:,0].min()
    y = points[:,1].min()
    width = points[:,0].max() - x
    height = points[:,1].max() - y
    if x + width < 0 or y + height < 0 or x >= image_shape[0] or y >= image_shape[1]:
        return None
    return [x, y, width, height]


def get_image_depth(pcd_on_camera, image_shape):
    pcd_2d = np.floor(pcd_on_camera[:,:2]).astype(np.int32)
    mask = (pcd_2d[:,0] >= 0) * (pcd_2d[:,1] >= 0) * (pcd_2d[:,0] < image_shape[0]) * (pcd_2d[:,1] < image_shape[1])
    indices = np.argwhere((mask > 0)).astype(np.int32).reshape(-1)
    pcd_2d = pcd_2d[indices]
    depth = pcd_on_camera[indices,2]
    order = np.argsort(depth).reshape(-1)
    
    image_depth = np.ones(image_shape) * (-1)
    for i in order:
        image_depth[pcd_2d[i,0], pcd_2d[i,1]] = depth[i]

    return image_depth


def parse_lidar_bboxes(pcd_on_camera, lidar_seg, image_shape):
    depth_bboxes = {}
    for info in lidar_seg["info"]:
        if info["category_id"] != 1:
            continue
        bbox = get_lidar_2d_bbox(pcd_on_camera, info["indices"], image_shape=image_shape)
        if bbox is None:
            continue
        bbox[0] = max(bbox[0], 0)
        bbox[1] = max(bbox[1], 0)
        bbox[2] = min(bbox[2], image_shape[0] - bbox[0])
        bbox[3] = min(bbox[3], image_shape[1] - bbox[1])
        depth = np.mean(pcd_on_camera[info["indices"]][:,2])
        if depth < 0:
            continue

        depth_bboxes[int(depth * 1000)] = bbox
    
    bboxes = []
    depth_list = list(depth_bboxes.keys())
    depth_list.sort()
    for depth in depth_list:
        bbox = depth_bboxes[depth]
        occluded = False
        # for _bbox in bboxes:
        #     if bbox[0] >= _bbox[0] and bbox[1] >= _bbox[1] and bbox[0] + bbox[2] <= _bbox[0] + _bbox[2] and bbox[1] + bbox[3] <= _bbox[1] + _bbox[3]:
        #         occluded = True
        #         break
        if not occluded:
            bboxes.append(bbox)

    return bboxes


def parse_camera_bboxes(camera_seg):
    bboxes = []
    for info in camera_seg["info"]:
        if info["category_id"] != 1:
            continue
        bboxes.append(info["bbox"])
    return bboxes