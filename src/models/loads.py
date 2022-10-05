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

    def _assemble_joint_load(self, structure, loads, time_step=None):
        # f_total = np.zeros((9, 1))
        f_total = np.zeros((structure.total_dofs_num, 1))
        f_total = np.matrix(f_total)
        # node_dofs_num = 3
        node_dofs_num = structure.node_dofs_num
        for load in loads:
            load_magnitude = load.magnitude[time_step, 0] if time_step else load.magnitude
            f_total[node_dofs_num * load.node + load.dof] = f_total[node_dofs_num * load.node + load.dof] + load_magnitude
        return f_total

    def get_load_vector(self, structure, loads, time_step=None):
        f_total = np.zeros((structure.total_dofs_num, 1))
        # f_total = np.zeros((9, 1))
        f_total = np.matrix(f_total)
        loads_dict = vars(loads)
        for load in loads_dict:
            if loads_dict[load]:
                if load == "joint":
                    f_total = f_total + self._assemble_joint_load(structure, loads_dict[load])
                elif load == "dynamic":
                    f_total = f_total + self._assemble_joint_load(structure, loads_dict[load], time_step)
        return f_total

    def apply_load_boundry_conditions(self, structure: Structure, force):
        reduced_f = force
        deleted_counter = 0
        for i in range(len(structure.boundaries_dof)):
            reduced_f = np.delete(
                reduced_f, structure.boundaries_dof[i] - deleted_counter, 0
            )
            deleted_counter += 1
        return reduced_f
