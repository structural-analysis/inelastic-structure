from scipy.linalg import cho_factor, cho_solve
import numpy as np


class Structure:
    # TODO: can't solve truss, fix reduced matrix to model trusses.
    # ycn: yield components num
    def __init__(self, nodes_num, dim, elements, boundaries, loads, limits):
        self.nodes_num = nodes_num
        self.node_dof_num = 3 if dim.lower() == "2d" else 6
        self.total_dofs_num = self.node_dof_num * self.nodes_num
        self.elements = elements
        self.ycn = self._ycn()
        self.boundaries = boundaries
        self.loads = loads
        self.limits = limits
        self.k = self.assemble()
        self.reduced_k = self.apply_boundry_conditions()
        self.f = self.apply_loading()
        self.ck = cho_factor(self.reduced_k)
        self.elastic_nodal_disp = self.compute_structure_displacement(self.f)
        self.elastic_elements_disps = self.get_elements_disps(self.elastic_nodal_disp)
        self.elastic_elements_forces = self._elastic_internal_forces()["elements_forces"]
        self.p0 = self._elastic_internal_forces()["p0"]
        self.d0 = self._elastic_nodal_disp_limits()
        self.pv = self._sensitivity_matrices()["pv"]
        self.elements_forces_sensitivity_matrix = self._sensitivity_matrices()["elements_forces_sensitivity_matrix"]
        self.elements_disps_sensitivity_matrix = self._sensitivity_matrices()["elements_disps_sensitivity_matrix"]
        self.nodal_disps_sensitivity_matrix = self._sensitivity_matrices()["nodal_disps_sensitivity_matrix"]
        self.dv = self._nodal_disp_limits_sensitivity_rows()
        self.phi = self._create_phi()
        self.yield_points_pieces = self._get_yield_points_pieces()

    def _ycn(self):
        ycn = 0
        for element in self.elements:
            ycn = ycn + element.total_ycn
        return ycn

    def _transform_loc_2d_matrix_to_glob(self, element_transform, element_stiffness):
        element_global_stiffness = np.dot(np.dot(np.transpose(element_transform), element_stiffness), element_transform)
        return element_global_stiffness

    def assemble(self):
        empty_stiffness = np.zeros((self.node_dof_num * self.nodes_num, self.node_dof_num * self.nodes_num))
        structure_stiffness = np.matrix(empty_stiffness)
        for eln in range(len(self.elements)):
            element_nodes_num = len(self.elements[eln].nodes)
            element_dof_num = self.elements[eln].k.shape[0]
            element_node_dof_num = element_dof_num / element_nodes_num
            element_global_stiffness = self._transform_loc_2d_matrix_to_glob(self.elements[eln].t, self.elements[eln].k)
            for i in range(element_dof_num):
                for j in range(element_dof_num):
                    local_element_node_row = int(j // element_node_dof_num)
                    p = int(element_node_dof_num * self.elements[eln].nodes[local_element_node_row].num + j % element_node_dof_num)
                    local_element_node_column = int(i // element_node_dof_num)
                    q = int(element_node_dof_num * self.elements[eln].nodes[local_element_node_column].num + i % element_node_dof_num)
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
        f_total = np.zeros((self.nodes_num * self.node_dof_num, 1))
        f_total = np.matrix(f_total)
        for joint_load in self.loads["joint_loads"]:
            f_total[self.node_dof_num * int(joint_load[0]) + int(joint_load[1])] = f_total[self.node_dof_num * int(joint_load[0]) + int(joint_load[1])] + joint_load[2]
        return f_total

    def apply_loading(self):
        f_total = np.zeros((self.nodes_num * self.node_dof_num, 1))
        f_total = np.matrix(f_total)
        for load in self.loads:
            if load == "joint_loads":
                f_total = f_total + self._assemble_join_load()
        return f_total

    def apply_load_boundry_conditions(self, force):
        reduced_f = force
        deleted_counter = 0
        for i in range(len(self.boundaries)):
            reduced_f = np.delete(
                reduced_f, 3 * self.boundaries[i, 0] + self.boundaries[i, 1] - deleted_counter, 0
            )
            deleted_counter += 1
        return reduced_f

    def get_elements_disps(self, disp):
        total_elements_num = len(self.elements)
        empty_elements_disps = np.zeros((total_elements_num, 1), dtype=object)
        elements_disps = np.matrix(empty_elements_disps)
        for i_element, element in enumerate(self.elements):
            element_dof_num = element.k.shape[0]
            element_nodes_num = len(element.nodes)
            element_node_dof_num = int(element_dof_num / element_nodes_num)
            v = np.zeros((element_dof_num, 1))
            v = np.matrix(v)
            for i in range(element_dof_num):
                element_node = i // element_node_dof_num
                node_dof = i % element_node_dof_num
                v[i, 0] = disp[element_node_dof_num * element.nodes[element_node].num + node_dof, 0]
            u = element.t * v
            elements_disps[i_element, 0] = u
        return elements_disps

    def compute_structure_displacement(self, force):
        j = 0
        o = 0
        boundaries_num = len(self.boundaries)
        reduced_forces = self.apply_load_boundry_conditions(force)
        reduced_disp = cho_solve(self.ck, reduced_forces)
        disp = np.zeros((self.node_dof_num * self.nodes_num, 1))
        disp = np.matrix(disp)
        for i in range(self.node_dof_num * self.nodes_num):
            if (j != boundaries_num and i == self.node_dof_num * self.boundaries[j, 0] + self.boundaries[j, 1]):
                j += 1
            else:
                disp[i, 0] = reduced_disp[o, 0]
                o += 1
        return disp

    def _elastic_internal_forces(self):
        ycn = self.ycn
        elements = self.elements
        elements_disps = self.elastic_elements_disps

        fixed_force = np.zeros((self.node_dof_num * 2, 1))
        fixed_force = np.matrix(fixed_force)

        # calculate p0
        total_elements_num = len(elements)
        empty_elements_forces = np.zeros((total_elements_num, 1), dtype=object)
        elements_forces = np.matrix(empty_elements_forces)
        empty_p0 = np.zeros((ycn, 1))
        p0 = np.matrix(empty_p0)
        current_p0_row = 0

        for i, element in enumerate(elements):
            if element.__class__.__name__ == "FrameElement2D":
                element_force = element.get_nodal_force(elements_disps[i, 0], fixed_force)
                elements_forces[i, 0] = element_force
                if not element.has_axial_yield:
                    p0[current_p0_row] = element_force[2, 0]
                    p0[current_p0_row + 1] = element_force[5, 0]
                else:
                    p0[current_p0_row] = element_force[0, 0]
                    p0[current_p0_row + 1] = element_force[2, 0]
                    p0[current_p0_row + 2] = element_force[3, 0]
                    p0[current_p0_row + 3] = element_force[5, 0]
            current_p0_row = current_p0_row + element.total_ycn
        return {"elements_forces": elements_forces, "p0": p0}

    def _sensitivity_matrices(self):
        # fv: equivalent global force vector for a yield component's udef
        ycn = self.ycn
        elements = self.elements
        empty_pv = np.zeros((ycn, ycn))
        pv = np.matrix(empty_pv)
        pv_column = 0
        total_components_num = 0
        for element in elements:
            total_components_num += element.total_ycn
        total_elements_num = len(elements)

        empty_elements_forces_sensitivity_matrix = np.zeros((total_elements_num, total_components_num), dtype=object)
        empty_elements_disps_sensitivity_matrix = np.zeros((total_elements_num, total_components_num), dtype=object)
        empty_nodal_disps_sensitivity_matrix = np.zeros((1, total_components_num), dtype=object)

        elements_forces_sensitivity_matrix = np.matrix(empty_elements_forces_sensitivity_matrix)
        elements_disps_sensitivity_matrix = np.matrix(empty_elements_disps_sensitivity_matrix)
        nodal_disps_sensitivity_matrix = np.matrix(empty_nodal_disps_sensitivity_matrix)

        for i_element, element in enumerate(elements):
            if element.__class__.__name__ == "FrameElement2D":
                for yield_point_udef in element.udefs:
                    udef_components_num = yield_point_udef.shape[1]
                    for i_component in range(udef_components_num):
                        fv_size = self.node_dof_num * self.nodes_num
                        fv = np.zeros((fv_size, 1))
                        fv = np.matrix(fv)
                        component_udef_global = element.t.T * yield_point_udef[:, i_component]
                        start_dof = self.node_dof_num * element.nodes[0].num
                        end_dof = self.node_dof_num * element.nodes[1].num

                        fv[start_dof] = component_udef_global[0]
                        fv[start_dof + 1] = component_udef_global[1]
                        fv[start_dof + 2] = component_udef_global[2]

                        fv[end_dof] = component_udef_global[3]
                        fv[end_dof + 1] = component_udef_global[4]
                        fv[end_dof + 2] = component_udef_global[5]

                        affected_struc_disp = self.compute_structure_displacement(fv)
                        nodal_disps_sensitivity_matrix[0, pv_column] = affected_struc_disp
                        affected_elem_disps = self.get_elements_disps(affected_struc_disp)
                        current_affected_element_ycns = 0
                        for i_affected_element, affected_elem_disp in enumerate(affected_elem_disps):

                            if i_element == i_affected_element:
                                fixed_force = -yield_point_udef[:, i_component]
                            else:
                                fixed_force = np.zeros((self.node_dof_num * 2, 1))
                                fixed_force = np.matrix(fixed_force)
                            # FIXME: affected_elem_disp[0, 0] is for numpy oskolation when use matrix in matrix and enumerating on it.
                            affected_element_force = self.elements[i_affected_element].get_nodal_force(affected_elem_disp[0, 0], fixed_force)
                            elements_forces_sensitivity_matrix[i_affected_element, pv_column] = affected_element_force
                            elements_disps_sensitivity_matrix[i_affected_element, pv_column] = affected_elem_disp[0, 0]

                            if not element.has_axial_yield:
                                pv[current_affected_element_ycns, pv_column] = affected_element_force[2, 0]
                                pv[current_affected_element_ycns + 1, pv_column] = affected_element_force[5, 0]
                            else:
                                pv[current_affected_element_ycns, pv_column] = affected_element_force[0, 0]
                                pv[current_affected_element_ycns + 1, pv_column] = affected_element_force[2, 0]
                                pv[current_affected_element_ycns + 2, pv_column] = affected_element_force[3, 0]
                                pv[current_affected_element_ycns + 3, pv_column] = affected_element_force[5, 0]
                            current_affected_element_ycns = current_affected_element_ycns + self.elements[i_affected_element].total_ycn

                        pv_column += 1
        results = {
            "pv": pv,
            "nodal_disps_sensitivity_matrix": nodal_disps_sensitivity_matrix,
            "elements_forces_sensitivity_matrix": elements_forces_sensitivity_matrix,
            "elements_disps_sensitivity_matrix": elements_disps_sensitivity_matrix,
        }
        return results

    def get_global_dof(self, node_num, dof_num):
        glob_dof = int(self.node_dof_num * node_num + dof_num)
        return glob_dof

    def _elastic_nodal_disp_limits(self):
        disp_limits = self.limits["disp_limits"]
        disp_limits_num = disp_limits.shape[0]
        empty_d0 = np.zeros((disp_limits_num, 1))
        d0 = np.matrix(empty_d0)
        for i, disp_limit in enumerate(disp_limits):
            node_num = disp_limit[0]
            dof_num = disp_limit[1]
            dof = self.get_global_dof(node_num, dof_num)
            d0[i, 0] = self.elastic_nodal_disp[dof, 0]
        return d0

    def _nodal_disp_limits_sensitivity_rows(self):
        ycn = self.ycn
        disp_limits = self.limits["disp_limits"]
        disp_limits_num = disp_limits.shape[0]
        empty_dv = np.zeros((disp_limits_num, ycn))
        dv = np.matrix(empty_dv)
        for i, disp_limit in enumerate(disp_limits):
            node_num = disp_limit[0]
            dof_num = disp_limit[1]
            dof = self.get_global_dof(node_num, dof_num)
            for j in range(ycn):
                dv[i, j] = self.nodal_disps_sensitivity_matrix[0, j][dof, 0]
        return dv

    def _create_phi(self):
        phi_row_size = 0
        phi_column_size = 0
        for element in self.elements:
            phi_row_size = phi_row_size + element.section.phi.shape[0] * len(element.yield_points)
            phi_column_size = phi_column_size + element.section.phi.shape[1] * len(element.yield_points)

        empty_phi = np.zeros((phi_row_size, phi_column_size))
        phi = np.matrix(empty_phi)
        current_row = 0
        current_column = 0
        for element in self.elements:
            for _ in range(len(element.yield_points)):
                for yield_section_row in range(element.section.phi.shape[0]):
                    for yield_section_column in range(element.section.phi.shape[1]):
                        phi[current_row + yield_section_row, current_column + yield_section_column] = element.section.phi[yield_section_row, yield_section_column]
                current_column = current_column + element.section.phi.shape[1]
                current_row = current_row + element.section.phi.shape[0]
        return phi

    def _get_yield_points_pieces(self):
        piece_counter = 0
        yield_points_pieces = []
        for element in self.elements:
            for yield_point in element.yield_points:
                start_piece_num = piece_counter
                piece_counter += yield_point.pieces_num
                end_piece_num = piece_counter
                yield_point_pieces = tuple(range(start_piece_num, end_piece_num))
                yield_points_pieces.append(yield_point_pieces)
        return yield_points_pieces