import numpy as np


class Structure:
    def __init__(self, n_nodes, node_n_dof, elements, boundaries):
        self.n_nodes = n_nodes
        self.node_n_dof = node_n_dof
        self.elements = elements
        self.boundaries = boundaries
        self.k = self.assemble()
        self.reduced_k = self.apply_boundry_conditions()

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
        # Ft.shape[1]
        # if M.shape[1] == 1:
        #     for BC in range(len(JTR)):
        #         MR=np.delete(MR,3*JTR[BC,0]+JTR[BC,1]-jj,0) #delete row 1
        #         jj+=1
        # elif M.shape[1] != 1:
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
