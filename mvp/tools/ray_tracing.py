import os
import open3d as o3d
import numpy as np 

from mvp.data.util import rotation_matrix, bbox_shift, bbox_rotate
from mvp.config import model_3d_path, model_3d_examples


def get_model_mesh(model_3d_name, bbox):
    mesh = o3d.io.read_triangle_mesh(os.path.join(model_3d_path, "{}.ply".format(model_3d_name)))
    model_bbox = np.array(model_3d_examples[model_3d_name])
    translate = bbox[:3]
    rotate = bbox[6]
    scale = np.min(bbox[3:6] / model_bbox[3:6]) 
    if scale is not None:
        mesh.scale(scale, np.array([0, 0, 0]).T)
    if rotate is not None:
        mesh.rotate(rotation_matrix(0, rotate, 0), np.zeros(3).T)
    if translate is not None:
        mesh.translate(np.array([translate[0], translate[1], translate[2]]).T)
    return mesh


def get_wall_mesh(bbox):
    wall = o3d.geometry.TriangleMesh.create_box(width=bbox[3], height=bbox[4], depth=bbox[5])
    wall.translate(np.array([-bbox[3]/2, -bbox[4]/2, 0]).T)
    wall.rotate(rotation_matrix(0, bbox[6], 0), np.zeros(3).T)
    wall.translate(np.array([bbox[0], bbox[1], bbox[2]]).T)
    return wall


def get_model_bbox(model_3d_name, bbox):
    model_bbox = np.array(model_3d_examples[model_3d_name])
    translate = bbox[:3]
    rotate = bbox[6]
    scale = np.min(bbox[3:6] / model_bbox[3:6])
    bbox = model_bbox
    if scale is not None:
        bbox[3:6] *= scale
    if rotate is not None:
        bbox = bbox_rotate(bbox, np.array([0, float(rotate), 0]))
    if translate is not None:
        bbox = bbox_shift(bbox, np.array([translate[0], translate[1], translate[2]]))
    return bbox


def ray_intersection(meshes, rays):
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id_map = {}
    for mesh in meshes:
        mesh_cuda = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_id = scene.add_triangles(mesh_cuda)
        mesh_id_map[mesh_id] = mesh

    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ray_size = rays.shape[0]

    ans_raw = scene.cast_rays(rays)

    ans = {}
    for key in ans_raw:
        ans[key] = ans_raw[key].numpy()

    intersection = np.zeros((ray_size, 3))
    for i in range(ray_size):
        data = {key:value[i] for key, value in ans.items()}
        # default filter: no intersection
        if data["t_hit"] > 10000:
            intersect_point = np.array([np.inf, np.inf, np.inf])
        else:
            mesh = mesh_id_map[data["geometry_ids"]]
            triangle_vertices = mesh.triangles[data["primitive_ids"]]
            intersect_point = (1 - np.sum(data["primitive_uvs"])) * mesh.vertices[triangle_vertices[0]] + data["primitive_uvs"][0] * mesh.vertices[triangle_vertices[1]] + data["primitive_uvs"][1] * mesh.vertices[triangle_vertices[2]]
        intersection[i] = intersect_point
    
    return intersection
