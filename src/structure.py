import numpy as np


class Structure:
    def __init__(self, n_nodes, node_n_dof, elements, boundaries, loads):
        self.n_nodes = n_nodes
        self.node_n_dof = node_n_dof
        self.elements = elements
        self.boundaries = boundaries
        self.loads = loads
        self.k = self.assemble()
        self.reduced_k = self.apply_boundry_conditions()
        self.f = self.apply_loading()
        self.reduced_f = self.apply_load_boundry_conditions()

    def _transform_loc_2d_matrix_to_glob(self, element_transform, element_stiffness):
        element_global_stiffness = np.dot(np.dot(np.transpose(element_transform), element_stiffness), element_transform)
        return element_global_stiffness

    def assemble(self):
        structure_stiffness = np.zeros((self.node_n_dof * self.n_nodes, self.node_n_dof * self.n_nodes))
        for eln in range(len(self.elements)):
            element_n_nodes = len(self.elements[eln].nodes)
            element_n_dof = self.elements[eln].k.shape[0]
            element_node_n_dof = element_n_dof / element_n_nodes
            element_global_stiffness = self._transform_loc_2d_matrix_to_glob(self.elements[eln].t, self.elements[eln].k)
            for i in range(element_n_dof):
                for j in range(element_n_dof):
                    local_element_node_row = int(j // element_node_n_dof)
                    p = int(element_node_n_dof * self.elements[eln].nodes[local_element_node_row] + j % element_node_n_dof)
                    local_element_node_column = int(i // element_node_n_dof)
                    q = int(element_node_n_dof * self.elements[eln].nodes[local_element_node_column] + i % element_node_n_dof)
                    structure_stiffness[p, q] = structure_stiffness[p, q] + element_global_stiffness[j, i]
        return structure_stiffness

    def apply_boundry_conditions(self):
        reduced_matrix = self.k
        deleted_counter = 0
        for i in range(len(self.boundaries)):
            # delete column
            reduced_matrix = np.delete(
                reduced_matrix, 3 * self.boundaries[i, 0] + self.boundaries[i, 1] - deleted_counter, 1
            )
            # delete row
            reduced_matrix = np.delete(
                reduced_matrix, 3 * self.boundaries[i, 0] + self.boundaries[i, 1] - deleted_counter, 0
            )
            deleted_counter += 1
        return reduced_matrix

    def _assemble_join_load(self):
        f_total = np.zeros((self.n_nodes * self.node_n_dof, 1))
        f_total = np.matrix(f_total)
        for joint_load in self.loads["joint_loads"]:
            f_total[self.node_n_dof * int(joint_load[0]) + int(joint_load[1])] = f_total[self.node_n_dof * int(joint_load[0]) + int(joint_load[1])] + joint_load[2]
        return f_total

    def apply_loading(self):
        f_total = np.zeros((self.n_nodes * self.node_n_dof, 1))
        f_total = np.matrix(f_total)
        for load in self.loads:
            if load == "joint_loads":
                f_total = f_total + self._assemble_join_load()
        return f_total

    def apply_load_boundry_conditions(self):
        reduced_f = self.f
        deleted_counter = 0
        for i in range(len(self.boundaries)):
            reduced_f = np.delete(
                reduced_f, 3 * self.boundaries[i, 0] + self.boundaries[i, 1] - deleted_counter, 0
            )
            deleted_counter += 1
        return reduced_f
