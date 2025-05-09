import numpy as np
from typing import List
from functools import lru_cache

from src.models.structure import Structure


class DistributedLoad:
    # NOTE: distributed load is implemented just for plate problems,
    # and is uniform for all of the plate structure surface
    def __init__(self, magnitude):
        self.magnitude = magnitude


class DynamicLoad:
    def __init__(self, node, dof, time, magnitude):
        self.node = node
        self.dof = dof
        self.magnitude = magnitude
        self.time = time


class JointLoad:
    def __init__(self, node, dof, magnitude):
        self.node = node
        self.dof = dof
        self.magnitude = magnitude


class Loads:
    def __init__(self, input):
        self.joint_loads: List[JointLoad] = input.get("joint_loads")
        self.dynamic_loads: List[DynamicLoad] = input.get("dynamic_loads")
        self.distributed_load: DistributedLoad = input.get("distributed_load")

    def assemble_joint_load(self, structure, loads, time_step=None):
        f_total = np.zeros((structure.dofs_count, 1))
        node_dofs_count = structure.node_dofs_count
        for load in loads:
            load_magnitude = load.magnitude[time_step, 0] if time_step is not None else load.magnitude
            f_total[node_dofs_count * load.node + load.dof] = f_total[node_dofs_count * load.node + load.dof] + load_magnitude
        return f_total

    def get_joint_loads_from_distributed_load(self, structure, distributed_load):
        joint_loads_from_distributed_load: List[JointLoad] = []
        for member in structure.members:
            equivalent_load_vector = member.get_distributed_equivalent_load_vector(q=distributed_load.magnitude)
            for i, node_joint_load_magnitude in enumerate(equivalent_load_vector):
                joint_loads_from_distributed_load.append(
                    JointLoad(
                        node=member.nodes[i].num,
                        dof=0,
                        magnitude=node_joint_load_magnitude,
                    )
                )
        return joint_loads_from_distributed_load

    def get_total_load(self, structure, loads, time_step=None):
        f_total = np.zeros((structure.dofs_count, 1))
        # f_total = np.zeros((9, 1))
        loads_dict = vars(loads)
        for load in loads_dict:
            if loads_dict[load]:
                if load == "joint_loads":
                    f_total = f_total + self.assemble_joint_load(structure, loads_dict[load])
                elif load == "distributed_load":
                    joint_loads_from_distributed_load = self.get_joint_loads_from_distributed_load(structure, loads_dict[load])
                    f_total = f_total + self.assemble_joint_load(structure, joint_loads_from_distributed_load)
                elif load == "dynamic_loads":
                    f_total = f_total + self.assemble_joint_load(structure, loads_dict[load], time_step)
        return f_total

    def apply_boundary_conditions(self, boundaries_dof_mask, load):
        reduced_load = load[boundaries_dof_mask][:, 0]
        return reduced_load

    def get_zero_and_nonzero_mass_load(self, structure: Structure, load):
        pt = load.copy()
        p0 = load.copy()
        mass_i = 0
        zero_i = 0
        zero_mass_dofs_i = 0
        for dof in range(structure.dofs_count):
            if structure.zero_mass_dofs.any() and dof == structure.zero_mass_dofs[zero_mass_dofs_i]:
                pt = np.delete(pt, dof - zero_i, 0)
                zero_i += 1
                zero_mass_dofs_i += 1
            else:
                p0 = np.delete(p0, dof - mass_i, 0)
                mass_i += 1
        return pt, p0

    def apply_static_condensation(self, structure: Structure, load):
        reduced_pt = self.apply_boundary_conditions(structure.zero_mass_boundaries_mask, load)
        reduced_p0 = self.apply_boundary_conditions(structure.mass_boundaries_mask, load)
        condensed_load = reduced_pt - np.transpose(structure.reduced_k0t) @ (structure.reduced_k00_inv @ reduced_p0)
        return condensed_load, reduced_p0

    def get_modal_load(self, load, modes):
        return np.transpose(modes) @ load
