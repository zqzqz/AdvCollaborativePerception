import numpy as np
import open3d as o3d
import copy
import random
import logging
import pickle

from .attacker import Attacker
from mvp.config import model_3d_path, model_3d_examples
from mvp.data.util import rotation_matrix, get_point_indices_in_bbox, get_open3d_bbox, get_distance, bbox_sensor_to_map, pcd_sensor_to_map, pcd_map_to_sensor, sort_lidar_points, get_distance, numpy_to_open3d, bbox_map_to_sensor
from mvp.tools.ray_tracing import get_model_mesh, ray_intersection, get_wall_mesh
from mvp.tools.ground_detection import get_ground_plane, get_ground_mesh


class AdvShapeAttacker(Attacker):
    def __init__(self, dataset=None, perception=None, attacker=None):
        super().__init__()
        self.name = "adv_shape"
        self.dataset = dataset
        self.perception = perception
        self.attacker = attacker
        self.mesh, self.mesh_divide = self.init_mesh()

    def run(self):
        import pygad
        num_genes = np.asarray(self.mesh.vertices).reshape(-1).shape[0]
        attacks = self.attacker.attack_list[::12]

        correct = lambda x : x.clip(-0.2, 0.2).reshape((-1, 3))

        def fitness_func(solution, solution_idx):
            x = correct(solution)
            mesh = self.perturb_mesh(self.mesh, x)
            fitness_list = []
            for attack in attacks:
                attack_opts = {
                    "victim_vehicle_id": attack["attack_meta"]["victim_vehicle_id"],
                    "bboxes": attack["attack_meta"]["bboxes"],
                    **attack["attack_opts"]
                }
                attack_opts["frame_ids"] = [9]
                case = self.dataset.get_case(attack["attack_meta"]["case_id"], tag="multi_frame")
                fitness = self.attack(mesh, self.mesh_divide, case, attack_opts)
                logging.warn("Fitness: {:.3f}".format(fitness))
                fitness_list.append(fitness)
            average_fitness = sum(fitness_list) / len(fitness_list)
            logging.warn("Average fitness: {:.3f}".format(average_fitness))
            return average_fitness
        
        def checkpoint_func(solver):
            logging.warn("Generation done; saving checkpoint.")
            solution, _, _ = solver.best_solution()
            solution = correct(solution)
            np.save("mesh_perturb.npy", solution)
            o3d.io.write_triangle_mesh("mesh.obj", self.perturb_mesh(self.mesh, solution))
            with open("mesh_divide.pkl", "wb") as f:
                pickle.dump(self.mesh_divide, f)

        solver = pygad.GA(
            gene_type=float,
            num_generations=5,
            on_generation=checkpoint_func,
            num_parents_mating=4,
            fitness_func=fitness_func,
            initial_population=None,
            num_genes=num_genes,
            sol_per_pop=10,
            init_range_low=-0.2,
            init_range_high=0.2,
            parent_selection_type="sss",
            keep_parents=1,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10,
        )

        solver.run()

    @staticmethod
    def init_mesh():
        bbox = np.array([0, 0, 0, 4.9, 2.5, 2.0, 0])
        mesh = get_wall_mesh(bbox)
        mesh = mesh.subdivide_midpoint(2)

        vertex_indices_list = []
        vertices = np.asarray(mesh.vertices)
        vertex_indices_list.append(np.argwhere(vertices[:, 0] > bbox[3]/2-0.01).reshape(-1))
        vertex_indices_list.append(np.argwhere(vertices[:, 0] < -bbox[3]/2+0.01).reshape(-1))
        vertex_indices_list.append(np.argwhere(np.logical_and(vertices[:, 0] >= 0, vertices[:, 1] > bbox[4]/2-0.01)).reshape(-1))
        vertex_indices_list.append(np.argwhere(np.logical_and(vertices[:, 0] <= 0, vertices[:, 1] > bbox[4]/2-0.01)).reshape(-1))
        vertex_indices_list.append(np.argwhere(np.logical_and(vertices[:, 0] >= 0, vertices[:, 1] < -bbox[4]/2+0.01)).reshape(-1))
        vertex_indices_list.append(np.argwhere(np.logical_and(vertices[:, 0] <= 0, vertices[:, 1] < -bbox[4]/2+0.01)).reshape(-1))
        vertex_indices_list.append(np.argwhere(np.logical_and(vertices[:, 0] >= 0, vertices[:, 2] > bbox[5]-0.01)).reshape(-1))
        vertex_indices_list.append(np.argwhere(np.logical_and(vertices[:, 0] <= 0, vertices[:, 2] > bbox[5]-0.01)).reshape(-1))
        vertex_indices_list.append(np.argwhere(np.logical_and(vertices[:, 0] >= 0, vertices[:, 2] < 0.01)).reshape(-1))
        vertex_indices_list.append(np.argwhere(np.logical_and(vertices[:, 0] <= 0, vertices[:, 2] < 0.01)).reshape(-1))

        return mesh, vertex_indices_list
        
    @staticmethod
    def perturb_mesh(mesh, perturbation):
        new_mesh = copy.deepcopy(mesh)
        vertices = np.asarray(mesh.vertices)
        vertices += perturbation
        new_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        return new_mesh

    def attack(self, mesh, mesh_divide, multi_frame_case, attack_opts):
        case = copy.deepcopy(multi_frame_case)
        attacker_id = attack_opts["attacker_vehicle_id"]
        ego_id = attack_opts["victim_vehicle_id"]
        
        self.attacker.meshes = [mesh.select_by_index(vertex_indices) for vertex_indices in mesh_divide]
        case, _ = self.attacker.run(case, attack_opts)

        object_index = case[9][attacker_id]["object_ids"].index(attack_opts["object_id"])
        bbox_to_remove = case[9][attacker_id]["gt_bboxes"][object_index]
        bbox_to_remove_ego = bbox_map_to_sensor(
            bbox_sensor_to_map(bbox_to_remove, case[9][attacker_id]["lidar_pose"]),
            case[9][ego_id]["lidar_pose"])
        fitness = self.perception.black_box_attack_fitness(case[9], ego_id, bbox_to_remove_ego, mode="remove")
        
        return fitness