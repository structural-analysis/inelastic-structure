import numpy as np
from typing import List

from src.models.structure import Structure


class Concentrated:
    def __init__(self):
        pass


class Distributed:
    def __init__(self, element, magnitude):
        self.element = element
        self.magnitude = magnitude


class Dynamic:
    def __init__(self, node, dof, time, magnitude):
        self.node = node
        self.dof = dof
        self.magnitude = magnitude
        self.time = time


class Joint:
    def __init__(self, node, dof, magnitude):
        self.node = node
        self.dof = dof
        self.magnitude = magnitude


class Loads:
    def __init__(self, input):
        self.joint: List[Joint] = input.get("joint")
        self.dynamic: List[Dynamic] = input.get("dynamic")
        self.distributed: List[Distributed] = input.get("distributed")
        self.concentrated: List[Concentrated] = input.get("concentrated")

    def assemble_joint_load(self, structure, loads, time_step=None):
        # f_total = np.zeros((9, 1))
        f_total = np.zeros((structure.dofs_count, 1))
        f_total = np.matrix(f_total)
        # node_dofs_count = 3
        node_dofs_count = structure.node_dofs_count
        for load in loads:
            load_magnitude = load.magnitude[time_step, 0] if time_step is not None else load.magnitude
            f_total[node_dofs_count * load.node + load.dof] = f_total[node_dofs_count * load.node + load.dof] + load_magnitude
        return f_total

    def get_total_load(self, structure, loads, time_step=None):
        f_total = np.zeros((structure.dofs_count, 1))
        # f_total = np.zeros((9, 1))
        f_total = np.matrix(f_total)
        loads_dict = vars(loads)
        for load in loads_dict:
            if loads_dict[load]:
                if load == "joint":
                    f_total = f_total + self.assemble_joint_load(structure, loads_dict[load])
                elif load == "dynamic":
                    f_total = f_total + self.assemble_joint_load(structure, loads_dict[load], time_step)
        return f_total

    def apply_boundary_conditions(self, boundaries_dof, load):
        reduced_load = load
        row_deleted_counter = 0
        for boundary in boundaries_dof:
            reduced_load = np.delete(reduced_load, boundary - row_deleted_counter, 0)
            row_deleted_counter += 1
        return reduced_load

    def get_zero_and_nonzero_mass_load(self, structure: Structure, load):
        pt = load.copy()
        p0 = load.copy()
        mass_i = 0
        zero_i = 0
        zero_mass_dofs_i = 0
        for dof in range(structure.dofs_count):
            if dof == structure.zero_mass_dofs[zero_mass_dofs_i]:
                pt = np.delete(pt, dof - zero_i, 0)
                zero_i += 1
                zero_mass_dofs_i += 1
            else:
                p0 = np.delete(p0, dof - mass_i, 0)
                mass_i += 1
        return pt, p0

    def apply_static_condensation(self, structure: Structure, load):
        pt, p0 = self.get_zero_and_nonzero_mass_load(structure, load)
        mass_bounds = structure.mass_bounds
        zero_mass_bounds = structure.zero_mass_bounds
        reduced_pt = self.apply_boundary_conditions(mass_bounds, pt)
        reduced_p0 = self.apply_boundary_conditions(zero_mass_bounds, p0)
        condensed_load = reduced_pt - np.dot(np.transpose(structure.reduced_k0t), np.dot(structure.reduced_k00_inv, reduced_p0))
        return condensed_load, reduced_p0

    def get_modal_load(self, load, modes):
        return np.dot(np.transpose(modes), load)
