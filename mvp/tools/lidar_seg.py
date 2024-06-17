import numpy as np

from mvp.data.util import numpy_to_open3d


def lidar_segmentation_dbscan(full_pcd, ground_indices, cluster_thres=0.5, min_point_num=8):
    non_ground_mask = np.ones(full_pcd.shape[0]).astype(bool)
    non_ground_mask[ground_indices] = False
    non_ground_indices = np.argwhere(non_ground_mask > 0).reshape(-1)
    pcd = full_pcd[non_ground_mask]

    open3d_pcd = numpy_to_open3d(pcd)
    labels = np.array(open3d_pcd.cluster_dbscan(eps=cluster_thres, min_points=min_point_num, print_progress=False))

    info = []
    for label in np.unique(labels):
        indices = np.argwhere(labels == label).reshape(-1)
        info.append({
            "indices": non_ground_indices[indices]
        })

    return {"info": info}


def lidar_segmentation_slr(full_pcd, ground_indices, cluster_thres=0.5):
    non_ground_mask = np.ones(full_pcd.shape[0]).astype(bool)
    non_ground_mask[ground_indices] = False
    non_ground_indices = np.argwhere(non_ground_mask > 0).reshape(-1)
    pcd = full_pcd[non_ground_mask]

    distance = np.sqrt(np.sum(pcd[:,:2] ** 2, axis=1))
    angle = np.arctan2(distance, -pcd[:,2])
    angle_delta = angle - np.concatenate((np.array([angle[0]]), angle[:-1]), axis=None)
    rings = np.cumsum((angle_delta < - 0.2 / 180 * np.pi).astype(np.int8))
    prev_pcd = np.vstack((np.array([pcd[0,:]]), pcd[:-1,:]))
    dist_delta = np.sqrt(np.sum((pcd - prev_pcd) ** 2, axis=1))
    breaks = dist_delta > cluster_thres

    instance_label = np.zeros(pcd.shape[0]).astype(np.int32)
    instance_count = 0
    ring_start = None
    for i in range(pcd.shape[0]):
        if i == 0:
            # The first point.
            instance_count += 1
            ring_start = i
            instance_label[i] = instance_count
            continue

        # Finds cluster in the same ring.
        if rings[i] != rings[i-1]:
            ring_start = i
        else:
            # The last point is the same ring.
            if breaks[i] == 0:
                instance_label[i] = instance_label[i-1]
            if i < pcd.shape[0] - 1 and rings[i] != rings[i+1]:
                # It is the last point in this ring.
                if np.sqrt(np.sum((pcd[i] - pcd[ring_start]) ** 2)) <= cluster_thres:
                    if instance_label[i] == 0:
                        instance_label[i] = instance_label[ring_start]
                    else:
                        instance_label[instance_label == instance_label[i]] = instance_label[ring_start]
        if instance_label[i] > 0:
            continue

        # Finds cluster in the last ring.
        last_ring_indices = np.argwhere((rings >= rings[i] - 1) * (rings < rings[i])).reshape(-1)
        if len(last_ring_indices) == 0:
            instance_count += 1
            instance_label[i] = instance_count
            continue
        last_ring_distance = np.sqrt(np.sum((pcd[i] - pcd[last_ring_indices]) ** 2, axis=1))
        if np.min(last_ring_distance) <= cluster_thres:
            instance_label[i] = instance_label[last_ring_indices[np.argmin(last_ring_distance)]]
        if instance_label[i] > 0:
            continue

        instance_count += 1
        instance_label[i] = instance_count

    # TODO: class of instances.
    point_class = np.zeros(pcd.shape[0])
    info = []

    for label in range(1, instance_count + 1):
        indices = np.argwhere(instance_label == label).reshape(-1)
        if len(indices) < 10:
            continue
        if len(np.unique(rings[indices])) <= 3:
            continue
        info.append({"indices": non_ground_indices[indices]})
    
    return {"class": point_class, "info": info}


def lidar_segmentation(pcd, method="cluster", **kwargs):
    if method == "squeezeseq":
        if "interface" not in kwargs or kwargs["interface"] is None:
            from .squeezeseg.interface import SqueezeSegInterface
            interface = SqueezeSegInterface()
        else:
            interface = kwargs["interface"]
        return interface.run(pcd)
    elif method == "dbscan":
        return lidar_segmentation_dbscan(pcd, **kwargs)
    elif method == "slr":
        return lidar_segmentation_slr(pcd, **kwargs)
    else:
        raise NotImplementedError("Unknown method")


def preprocess_lidar_seg(multi_frame_data, frame_ids=None, vehicle_ids=None):
    if frame_ids is None:
        frame_ids = range(len(multi_frame_data))
    if vehicle_ids is None:
        vehicle_ids = list(multi_frame_data[0].keys())
    for frame_id in frame_ids:
        frame_data = multi_frame_data[frame_id]
        for vehicle_id in vehicle_ids:
            vehicle_data = frame_data[vehicle_id]
            lidar_data = vehicle_data["lidar"]
            lidar_seg = lidar_segmentation(lidar_data)
            vehicle_data["lidar_seg"] = lidar_seg
    return multi_frame_data