from scipy.linalg import cho_factor, cho_solve
import numpy as np


class General:
    def __init__(self, input_general):
        self.nodes_num = input_general["nodes_num"]
        self.dim = input_general["structure_dim"]
        self.include_softening = input_general["include_softening"]
        self.node_dofs_num = 3 if self.dim.lower() == "2d" else 6
        self.total_dofs_num = self.node_dofs_num * self.nodes_num


class YieldSpecs:
    def __init__(self, yield_specs_dict):
        self.points_num = yield_specs_dict["points_num"]
        self.components_num = yield_specs_dict["components_num"]
        self.pieces_num = yield_specs_dict["pieces_num"]


class Elements:
    def __init__(self, elements_list):
        self.list = elements_list
        self.num = len(elements_list)
        self.yield_specs = YieldSpecs(self.get_yield_specs_dict())

    def get_yield_specs_dict(self):
        points_num = 0
        components_num = 0
        pieces_num = 0

        for element in self.list:
            points_num += element.yield_specs.points_num
            components_num += element.yield_specs.components_num
            pieces_num += element.yield_specs.pieces_num

        yield_specs_dict = {
            "points_num": points_num,
            "components_num": components_num,
            "pieces_num": pieces_num,
        }
        return yield_specs_dict


class Structure:
    # TODO: can't solve truss, fix reduced matrix to model trusses.
    def __init__(self, input):
        self.general = General(input["general"])
        self.elements = Elements(input["elements_list"])
        self.yield_specs = self.elements.yield_specs
        self.boundaries = input["boundaries"]
        self.boundaries_dof = self._get_boundaries_dof()
        self.loads = input["loads"]
        self.limits = input["limits"]
        self.k = self.get_stiffness()
        self.m = self.get_mass()
        self.zero_mass_dofs = self.get_zero_mass_dofs()
        self.reduced_k = self.apply_boundary_condition(self.boundaries_dof, self.k)
        self.mass_bounds, self.zero_mass_bounds = self.condense_boundary()
        self.f = self.get_load_vector()
        self.yield_points_indices = self.get_yield_points_indices()

        self.elastic_nodal_disp = self.get_nodal_disp(self.f)
        self.elastic_elements_disps = self.get_elements_disps(self.elastic_nodal_disp)
        self.elastic_elements_forces = self.get_internal_forces()["elements_forces"]
        self.p0 = self.get_internal_forces()["p0"]
        self.d0 = self.get_nodal_disp_limits()

        self.pv = self.get_sensitivity()["pv"]
        self.elements_forces_sensitivity = self.get_sensitivity()["elements_forces_sensitivity"]
        self.elements_disps_sensitivity = self.get_sensitivity()["elements_disps_sensitivity"]
        self.nodal_disps_sensitivity = self.get_sensitivity()["nodal_disps_sensitivity"]
        self.dv = self.get_nodal_disp_limits_sensitivity_rows()

        self.phi = self.create_phi()
        self.q = self.create_q()
        self.h = self.create_h()
        self.w = self.create_w()
        self.cs = self.create_cs()

    def _transform_loc_2d_matrix_to_glob(self, element_transform, element_stiffness):
        element_global_stiffness = np.dot(np.dot(np.transpose(element_transform), element_stiffness), element_transform)
        return element_global_stiffness

    def get_stiffness(self):
        empty_stiffness = np.zeros((self.general.total_dofs_num, self.general.total_dofs_num))
        structure_stiffness = np.matrix(empty_stiffness)
        for element in self.elements.list:
            element_global_stiffness = self._transform_loc_2d_matrix_to_glob(element.t, element.k)
            structure_stiffness = self._assemble_elements(element, element_global_stiffness, structure_stiffness)
        return structure_stiffness

    def apply_boundary_condition(self, boundaries_dof, structure_prop):
        reduced_structure_prop = structure_prop
        row_deleted_counter = 0
        col_deleted_counter = 0
        if structure_prop.shape[0] == structure_prop.shape[1]:
            for boundary in boundaries_dof:
                reduced_structure_prop = np.delete(reduced_structure_prop, boundary - row_deleted_counter, 0)
                reduced_structure_prop = np.delete(reduced_structure_prop, boundary - col_deleted_counter, 1)
                row_deleted_counter += 1
                col_deleted_counter += 1

        else:
            for boundary in self.mass_bounds:
                reduced_structure_prop = np.delete(reduced_structure_prop, boundary - col_deleted_counter, 1)
                col_deleted_counter += 1

            for boundary in self.zero_mass_bounds:
                reduced_structure_prop = np.delete(reduced_structure_prop, boundary - row_deleted_counter, 0)
                row_deleted_counter += 1

        return reduced_structure_prop

    def get_mass(self):
        # mass per length is applied in global direction so there is no need to transform.
        empty_mass = np.zeros((self.general.total_dofs_num, self.general.total_dofs_num))
        structure_mass = np.matrix(empty_mass)
        for element in self.elements.list:
            if element.m is not None:
                structure_mass = self._assemble_elements(element, element.m, structure_mass)
        return structure_mass

    def get_zero_mass_dofs(self):
        return np.sort(np.where(~self.m.any(axis=1))[0])

    def _assemble_elements(self, element, element_prop, structure_prop):
        element_nodes_num = len(element.nodes)
        element_dofs_num = element.k.shape[0]
        element_node_dofs_num = element_dofs_num / element_nodes_num
        for i in range(element_dofs_num):
            for j in range(element_dofs_num):
                local_element_node_row = int(j // element_node_dofs_num)
                p = int(element_node_dofs_num * element.nodes[local_element_node_row].num + j % element_node_dofs_num)
                local_element_node_column = int(i // element_node_dofs_num)
                q = int(element_node_dofs_num * element.nodes[local_element_node_column].num + i % element_node_dofs_num)
                structure_prop[p, q] = structure_prop[p, q] + element_prop[j, i]
        return structure_prop

    def _assemble_joint_load(self):
        f_total = np.zeros((self.general.total_dofs_num, 1))
        f_total = np.matrix(f_total)
        for joint_load in self.loads["joint_loads"]:
            f_total[self.general.node_dofs_num * int(joint_load[0]) + int(joint_load[1])] = f_total[self.general.node_dofs_num * int(joint_load[0]) + int(joint_load[1])] + joint_load[2]
        return f_total

    def get_load_vector(self):
        f_total = np.zeros((self.general.total_dofs_num, 1))
        f_total = np.matrix(f_total)
        for load in self.loads:
            if load == "joint_loads":
                f_total = f_total + self._assemble_joint_load()
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
        empty_elements_disps = np.zeros((self.elements.num, 1), dtype=object)
        elements_disps = np.matrix(empty_elements_disps)
        for i_element, element in enumerate(self.elements.list):
            element_dofs_num = element.k.shape[0]
            element_nodes_num = len(element.nodes)
            element_node_dofs_num = int(element_dofs_num / element_nodes_num)
            v = np.zeros((element_dofs_num, 1))
            v = np.matrix(v)
            for i in range(element_dofs_num):
                element_node = i // element_node_dofs_num
                node_dof = i % element_node_dofs_num
                v[i, 0] = disp[element_node_dofs_num * element.nodes[element_node].num + node_dof, 0]
            u = element.t * v
            elements_disps[i_element, 0] = u
        return elements_disps

    def get_nodal_disp(self, force):
        j = 0
        o = 0
        boundaries_num = len(self.boundaries)
        reduced_forces = self.apply_load_boundry_conditions(force)
        reduced_disp = cho_solve(cho_factor(self.reduced_k), reduced_forces)
        empty_nodal_disp = np.zeros((self.general.total_dofs_num, 1))
        nodal_disp = np.matrix(empty_nodal_disp)
        for i in range(self.general.total_dofs_num):
            if (j != boundaries_num and i == self.general.node_dofs_num * self.boundaries[j, 0] + self.boundaries[j, 1]):
                j += 1
            else:
                nodal_disp[i, 0] = reduced_disp[o, 0]
                o += 1
        return nodal_disp

    def get_internal_forces(self):
        elements_disps = self.elastic_elements_disps

        fixed_force = np.zeros((self.general.node_dofs_num * 2, 1))
        fixed_force = np.matrix(fixed_force)

        # calculate p0
        empty_elements_forces = np.zeros((self.elements.num, 1), dtype=object)
        elements_forces = np.matrix(empty_elements_forces)
        empty_p0 = np.zeros((self.yield_specs.components_num, 1))
        p0 = np.matrix(empty_p0)
        current_p0_row = 0

        for i, element in enumerate(self.elements.list):
            if element.__class__.__name__ == "FrameElement2D":
                element_force = element.get_nodal_force(elements_disps[i, 0], fixed_force)
                elements_forces[i, 0] = element_force
                if not element.section.nonlinear.has_axial_yield:
                    p0[current_p0_row] = element_force[2, 0]
                    p0[current_p0_row + 1] = element_force[5, 0]
                else:
                    p0[current_p0_row] = element_force[0, 0]
                    p0[current_p0_row + 1] = element_force[2, 0]
                    p0[current_p0_row + 2] = element_force[3, 0]
                    p0[current_p0_row + 3] = element_force[5, 0]
            current_p0_row = current_p0_row + element.yield_specs.components_num
        return {"elements_forces": elements_forces, "p0": p0}

    def get_sensitivity(self):
        # fv: equivalent global force vector for a yield component's udef
        elements = self.elements.list
        empty_pv = np.zeros((self.yield_specs.components_num, self.yield_specs.components_num))
        pv = np.matrix(empty_pv)
        pv_column = 0

        empty_elements_forces_sensitivity = np.zeros((self.elements.num, self.yield_specs.components_num), dtype=object)
        empty_elements_disps_sensitivity = np.zeros((self.elements.num, self.yield_specs.components_num), dtype=object)
        empty_nodal_disps_sensitivity = np.zeros((1, self.yield_specs.components_num), dtype=object)

        elements_forces_sensitivity = np.matrix(empty_elements_forces_sensitivity)
        elements_disps_sensitivity = np.matrix(empty_elements_disps_sensitivity)
        nodal_disps_sensitivity = np.matrix(empty_nodal_disps_sensitivity)

        for i_element, element in enumerate(elements):
            if element.__class__.__name__ == "FrameElement2D":
                for yield_point_udef in element.udefs:
                    udef_components_num = yield_point_udef.shape[1]
                    for i_component in range(udef_components_num):
                        fv_size = self.general.total_dofs_num
                        fv = np.zeros((fv_size, 1))
                        fv = np.matrix(fv)
                        component_udef_global = element.t.T * yield_point_udef[:, i_component]
                        start_dof = self.general.node_dofs_num * element.nodes[0].num
                        end_dof = self.general.node_dofs_num * element.nodes[1].num

                        fv[start_dof] = component_udef_global[0]
                        fv[start_dof + 1] = component_udef_global[1]
                        fv[start_dof + 2] = component_udef_global[2]

                        fv[end_dof] = component_udef_global[3]
                        fv[end_dof + 1] = component_udef_global[4]
                        fv[end_dof + 2] = component_udef_global[5]

                        affected_struc_disp = self.get_nodal_disp(fv)
                        nodal_disps_sensitivity[0, pv_column] = affected_struc_disp
                        affected_elem_disps = self.get_elements_disps(affected_struc_disp)
                        current_affected_element_ycns = 0
                        for i_affected_element, affected_elem_disp in enumerate(affected_elem_disps):

                            if i_element == i_affected_element:
                                fixed_force = -yield_point_udef[:, i_component]
                            else:
                                fixed_force = np.zeros((self.general.node_dofs_num * 2, 1))
                                fixed_force = np.matrix(fixed_force)
                            # FIXME: affected_elem_disp[0, 0] is for numpy oskolation when use matrix in matrix and enumerating on it.
                            affected_element_force = self.elements.list[i_affected_element].get_nodal_force(affected_elem_disp[0, 0], fixed_force)
                            elements_forces_sensitivity[i_affected_element, pv_column] = affected_element_force
                            elements_disps_sensitivity[i_affected_element, pv_column] = affected_elem_disp[0, 0]

                            if not element.section.nonlinear.has_axial_yield:
                                pv[current_affected_element_ycns, pv_column] = affected_element_force[2, 0]
                                pv[current_affected_element_ycns + 1, pv_column] = affected_element_force[5, 0]
                            else:
                                pv[current_affected_element_ycns, pv_column] = affected_element_force[0, 0]
                                pv[current_affected_element_ycns + 1, pv_column] = affected_element_force[2, 0]
                                pv[current_affected_element_ycns + 2, pv_column] = affected_element_force[3, 0]
                                pv[current_affected_element_ycns + 3, pv_column] = affected_element_force[5, 0]
                            current_affected_element_ycns = current_affected_element_ycns + self.elements.list[i_affected_element].yield_specs.components_num

                        pv_column += 1
        results = {
            "pv": pv,
            "nodal_disps_sensitivity": nodal_disps_sensitivity,
            "elements_forces_sensitivity": elements_forces_sensitivity,
            "elements_disps_sensitivity": elements_disps_sensitivity,
        }
        return results

    def get_global_dof(self, node_num, dof):
        global_dof = int(self.general.node_dofs_num * node_num + dof)
        return global_dof

    def get_nodal_disp_limits(self):
        disp_limits = self.limits["disp_limits"]
        disp_limits_num = disp_limits.shape[0]
        empty_d0 = np.zeros((disp_limits_num, 1))
        d0 = np.matrix(empty_d0)
        for i, disp_limit in enumerate(disp_limits):
            node = disp_limit[0]
            node_dof = disp_limit[1]
            dof = self.get_global_dof(node, node_dof)
            d0[i, 0] = self.elastic_nodal_disp[dof, 0]
        return d0

    def get_nodal_disp_limits_sensitivity_rows(self):
        disp_limits = self.limits["disp_limits"]
        disp_limits_num = disp_limits.shape[0]
        empty_dv = np.zeros((disp_limits_num, self.yield_specs.components_num))
        dv = np.matrix(empty_dv)
        for i, disp_limit in enumerate(disp_limits):
            node = disp_limit[0]
            node_dof = disp_limit[1]
            dof = self.get_global_dof(node, node_dof)
            for j in range(self.yield_specs.components_num):
                dv[i, j] = self.nodal_disps_sensitivity[0, j][dof, 0]
        return dv

    def create_phi(self):
        empty_phi = np.zeros((self.yield_specs.components_num, self.yield_specs.pieces_num))
        phi = np.matrix(empty_phi)
        current_row = 0
        current_column = 0
        for element in self.elements.list:
            for _ in range(element.yield_specs.points_num):
                for yield_section_row in range(element.section.yield_specs.phi.shape[0]):
                    for yield_section_column in range(element.section.yield_specs.phi.shape[1]):
                        phi[current_row + yield_section_row, current_column + yield_section_column] = element.section.yield_specs.phi[yield_section_row, yield_section_column]
                current_column = current_column + element.section.yield_specs.phi.shape[1]
                current_row = current_row + element.section.yield_specs.phi.shape[0]
        return phi

    def create_q(self):
        empty_q = np.zeros((2 * self.yield_specs.points_num, self.yield_specs.pieces_num))
        q = np.matrix(empty_q)
        yield_point_counter = 0
        yield_pieces_num_counter = 0
        for element in self.elements.list:
            for _ in range(element.yield_specs.points_num):
                q[2 * yield_point_counter:2 * yield_point_counter + 2, yield_pieces_num_counter:element.section.yield_specs.pieces_num + yield_pieces_num_counter] = element.section.softening.q
                yield_point_counter += 1
                yield_pieces_num_counter += element.section.yield_specs.pieces_num
        return q

    def create_h(self):
        empty_h = np.zeros((self.yield_specs.pieces_num, 2 * self.yield_specs.points_num))
        h = np.matrix(empty_h)
        yield_point_counter = 0
        yield_pieces_num_counter = 0
        for element in self.elements.list:
            for _ in range(element.yield_specs.points_num):
                h[yield_pieces_num_counter:element.section.yield_specs.pieces_num + yield_pieces_num_counter, 2 * yield_point_counter:2 * yield_point_counter + 2] = element.section.softening.h
                yield_point_counter += 1
                yield_pieces_num_counter += element.section.yield_specs.pieces_num
        return h

    def create_w(self):
        empty_w = np.zeros((2 * self.yield_specs.points_num, 2 * self.yield_specs.points_num))
        w = np.matrix(empty_w)
        yield_point_counter = 0
        for element in self.elements.list:
            for _ in range(element.yield_specs.points_num):
                w[2 * yield_point_counter:2 * yield_point_counter + 2, 2 * yield_point_counter:2 * yield_point_counter + 2] = element.section.softening.w
                yield_point_counter += 1
        return w

    def create_cs(self):
        empty_cs = np.zeros((2 * self.yield_specs.points_num, 1))
        cs = np.matrix(empty_cs)
        yield_point_counter = 0
        for element in self.elements.list:
            for _ in range(element.yield_specs.points_num):
                cs[2 * yield_point_counter:2 * yield_point_counter + 2, 0] = element.section.softening.cs
                yield_point_counter += 1
        return cs

    def get_yield_points_indices(self):
        yield_points_indices = []
        index_counter = 0
        for element in self.elements.list:
            yield_point_pieces = int(element.yield_specs.pieces_num / element.yield_specs.points_num)
            for _ in range(element.yield_specs.points_num):
                yield_points_indices.append(
                    {
                        "begin": index_counter,
                        "end": index_counter + yield_point_pieces - 1,
                    }
                )
                index_counter += yield_point_pieces
        return yield_points_indices

    def _get_boundaries_dof(self):
        boundaries_size = self.boundaries.shape[0]
        boundaries_dof = np.zeros(boundaries_size, dtype=int)
        for i in range(boundaries_size):
            boundaries_dof[i] = int(self.general.node_dofs_num * self.boundaries[i, 0] + self.boundaries[i, 1])
        return np.sort(boundaries_dof)

    def condense_boundary(self):
        zero_mass_dofs = self.zero_mass_dofs

        mass_dof_i = 0
        zero_mass_dof_i = 0

        mass_bounds = self.boundaries_dof.copy()
        zero_mass_bounds = self.boundaries_dof.copy()

        bound_i = 0
        mass_bound_i = 0
        zero_mass_bound_i = 0

        if self.zero_mass_dofs.any():
            for dof in range(self.general.total_dofs_num):
                if dof == zero_mass_dofs[zero_mass_dof_i]:
                    if bound_i < self.boundaries_dof.shape[0]:
                        if dof == self.boundaries_dof[bound_i]:
                            mass_bounds = np.delete(mass_bounds, bound_i - mass_bound_i, 0)
                            mass_bound_i += 1

                            zero_mass_bounds[bound_i - zero_mass_bound_i] = zero_mass_bounds[bound_i - zero_mass_bound_i] - mass_dof_i

                            bound_i += 1
                    zero_mass_dof_i += 1
                else:
                    if dof == self.boundaries_dof[bound_i]:
                        mass_bounds[bound_i - mass_bound_i] = mass_bounds[bound_i - mass_bound_i] - zero_mass_dof_i

                        zero_mass_bounds = np.delete(zero_mass_bounds, bound_i - zero_mass_bound_i, 0)
                        zero_mass_bound_i += 1

                        bound_i += 1
                    mass_dof_i += 1
        return mass_bounds, zero_mass_bounds

    def apply_static_condensation(self):
        mtt, ktt, k00, k0t = self.get_zero_and_nonzero_mass_props()

        mass_bounds = self.mass_bounds
        zero_mass_bounds = self.zero_mass_bounds
        reduced_ktt = self.apply_boundary_condition(mass_bounds, ktt)
        reduced_mtt = self.apply_boundary_condition(mass_bounds, mtt)
        reduced_k00 = self.apply_boundary_condition(zero_mass_bounds, k00)
        reduced_k0t = self.apply_boundary_condition(self.boundaries_dof, k0t)
        reduced_k00_inv = np.linalg.inv(reduced_k00)
        ku0 = -(np.dot(reduced_k00_inv, reduced_k0t))
        khat = reduced_ktt - np.dot(np.dot(np.transpose(reduced_k0t), reduced_k00_inv), reduced_k0t)
        return khat, reduced_mtt, ku0, reduced_k00_inv, reduced_k00

    def get_zero_and_nonzero_mass_props(self):
        mtt = self.m.copy()
        ktt = self.k.copy()
        k00 = self.k.copy()
        k0t = self.k.copy()
        # for zero mass rows and columns
        mass_i = 0
        # for non-zero mass rows and columns
        zero_i = 0
        zero_mass_dofs_i = 0
        for dof in range(self.general.total_dofs_num):
            if dof == self.zero_mass_dofs[zero_mass_dofs_i]:
                mtt = np.delete(mtt, dof - zero_i, 1)
                mtt = np.delete(mtt, dof - zero_i, 0)
                ktt = np.delete(ktt, dof - zero_i, 1)
                ktt = np.delete(ktt, dof - zero_i, 0)
                k0t = np.delete(k0t, dof - zero_i, 1)
                zero_i += 1
                zero_mass_dofs_i += 1
            else:
                k00 = np.delete(k00, dof - mass_i, 1)
                k00 = np.delete(k00, dof - mass_i, 0)
                k0t = np.delete(k0t, dof - mass_i, 0)
                mass_i += 1
        return mtt, ktt, k00, k0t
