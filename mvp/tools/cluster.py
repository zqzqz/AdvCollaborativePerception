import open3d as o3d
import numpy as np

from mvp.data.util import numpy_to_open3d


def get_clusters(pcd, **kwargs):
    if pcd.shape[0] == 0:
        return []
    cloud = numpy_to_open3d(pcd)
    if "eps" not in kwargs:
        kwargs["eps"] = 1
    if "min_points" not in kwargs:
        kwargs["min_points"] = 5
    label = np.asarray(cloud.cluster_dbscan(**kwargs))
    result = []
    for i in range(np.max(label) + 1):
        indices = np.argwhere(label == i).reshape(-1)
        if indices.shape[0] <= 0:
            continue
        result.append(indices)
    return result


def points_to_bbox(pcd):
    cloud = numpy_to_open3d(pcd)
    print(cloud.get_oriented_bounding_box())