import numpy as np
from scipy.linalg import cho_factor, cho_solve


class YieldSpecs:
    def __init__(self, yield_specs_dict):
        self.points_num = yield_specs_dict["points_num"]
        self.components_num = yield_specs_dict["components_num"]
        self.pieces_num = yield_specs_dict["pieces_num"]


class Members:
    def __init__(self, members_list):
        self.list = members_list
        self.num = len(members_list)
        self.yield_specs = YieldSpecs(self.get_yield_specs_dict())

    def get_yield_specs_dict(self):
        points_num = 0
        components_num = 0
        pieces_num = 0

        for member in self.list:
            points_num += member.yield_specs.points_num
            components_num += member.yield_specs.components_num
            pieces_num += member.yield_specs.pieces_num

        yield_specs_dict = {
            "points_num": points_num,
            "components_num": components_num,
            "pieces_num": pieces_num,
        }
        return yield_specs_dict


class Structure:
    # TODO: can't solve truss, fix reduced matrix to model trusses.
    def __init__(self, input):
        self.initial_nodes_num = input["nodes_num"]
        self.members = Members(members_list=input["members"])
        self.dim = input["dim"]
        self.nodes_num = self.get_nodes_num()
        self.node_dofs_num = 3 if self.dim.lower() == "2d" else 3
        self.total_dofs_num = self.node_dofs_num * self.nodes_num
        self.include_softening = input["include_softening"]
        self.yield_specs = self.members.yield_specs
        self.nodal_boundaries = input["nodal_boundaries"]
        self.linear_boundaries = input["linear_boundaries"]
        self.boundaries = self.populate_boundaries()
        self.boundaries_dof = self.get_boundaries_dof()
        self.loads = input["loads"]
        self.limits = input["limits"]
        self.k = self.get_stiffness()
        self.m = self.get_mass()
        self.zero_mass_dofs = self.get_zero_mass_dofs()
        self.reduced_k = self.apply_boundary_condition(self.boundaries_dof, self.k)
        self.kc = cho_factor(self.reduced_k)
        self.mass_bounds, self.zero_mass_bounds = self.condense_boundary()
        # self.condensed_k, self.condensed_m, self.ku0, self.reduced_k00_inv, self.reduced_k00 = self.apply_static_condensation()
        # self.wns, self.wds, self.modes = self.compute_modes_props()
        self.f = self.get_load_vector()
        self.yield_points_indices = self.get_yield_points_indices()

        self.elastic_nodal_disp = self.get_nodal_disp(self.f)
        self.elastic_members_disps = self.get_members_disps(self.elastic_nodal_disp)
        self.elastic_members_forces = self.get_internal_forces()["members_forces"]
        self.p0 = self.get_internal_forces()["p0"]
        self.d0 = self.get_nodal_disp_limits()

        self.pv = self.get_sensitivity()["pv"]
        self.members_forces_sensitivity = self.get_sensitivity()["members_forces_sensitivity"]
        self.members_disps_sensitivity = self.get_sensitivity()["members_disps_sensitivity"]
        self.nodal_disps_sensitivity = self.get_sensitivity()["nodal_disps_sensitivity"]
        self.dv = self.get_nodal_disp_limits_sensitivity_rows()

        self.phi = self.create_phi()
        self.q = self.create_q()
        self.h = self.create_h()
        self.w = self.create_w()
        self.cs = self.create_cs()

    def get_nodes_num(self):
        nodes_num = self.initial_nodes_num
        for member in self.members.list:
            if member.__class__.__name__ == "PlateMember":
                nodes_num = member.nodes_num
        return nodes_num

    def _transform_loc_2d_matrix_to_glob(self, member_transform, member_stiffness):
        member_global_stiffness = np.dot(np.dot(np.transpose(member_transform), member_stiffness), member_transform)
        return member_global_stiffness

    def get_stiffness(self):
        empty_stiffness = np.zeros((self.total_dofs_num, self.total_dofs_num))
        structure_stiffness = np.matrix(empty_stiffness)
        for member in self.members.list:
            member_global_stiffness = self._transform_loc_2d_matrix_to_glob(member.t, member.k)
            structure_stiffness = self._assemble_members(member, member_global_stiffness, structure_stiffness)
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

    def apply_load_boundry_conditions(self, force):
        reduced_f = force
        deleted_counter = 0
        for i in range(len(self.boundaries_dof)):
            reduced_f = np.delete(
                reduced_f, self.boundaries_dof[i] - deleted_counter, 0
            )
            deleted_counter += 1
        return reduced_f

    def get_mass(self):
        # mass per length is applied in global direction so there is no need to transform.
        empty_mass = np.zeros((self.total_dofs_num, self.total_dofs_num))
        structure_mass = np.matrix(empty_mass)
        for member in self.members.list:
            if member.m is not None:
                structure_mass = self._assemble_members(member, member.m, structure_mass)
        return structure_mass

    def get_zero_mass_dofs(self):
        return np.sort(np.where(~self.m.any(axis=1))[0])

    def _assemble_members(self, member, member_prop, structure_prop):
        member_nodes_num = len(member.nodes)
        member_dofs_num = member.k.shape[0]
        member_node_dofs_num = member_dofs_num / member_nodes_num
        for i in range(member_dofs_num):
            for j in range(member_dofs_num):
                local_member_node_row = int(j // member_node_dofs_num)
                p = int(member_node_dofs_num * member.nodes[local_member_node_row].num + j % member_node_dofs_num)
                local_member_node_column = int(i // member_node_dofs_num)
                q = int(member_node_dofs_num * member.nodes[local_member_node_column].num + i % member_node_dofs_num)
                structure_prop[p, q] = structure_prop[p, q] + member_prop[j, i]
        return structure_prop

    def _assemble_joint_load(self, loads, time_step=None):
        f_total = np.zeros((self.total_dofs_num, 1))
        f_total = np.matrix(f_total)
        for load in loads:
            load_magnitude = load.magnitude[time_step, 0] if time_step else load.magnitude
            f_total[self.node_dofs_num * load.node + load.dof] = f_total[self.node_dofs_num * load.node + load.dof] + load_magnitude
        return f_total

    def get_load_vector(self, time_step=None):
        f_total = np.zeros((self.total_dofs_num, 1))
        f_total = np.matrix(f_total)
        for load in self.loads:
            if self.loads[load]:
                if load == "joint":
                    f_total = f_total + self._assemble_joint_load(self.loads[load])
                elif load == "dynamic":
                    f_total = f_total + self._assemble_joint_load(self.loads[load], time_step)
        return f_total

    def apply_load_boundary_conditions(self, force):
        reduced_f = force
        deleted_counter = 0
        for i in range(len(self.boundaries_dof)):
            reduced_f = np.delete(
                reduced_f, self.boundaries_dof[i] - deleted_counter, 0
            )
            deleted_counter += 1
        return reduced_f

    def get_members_disps(self, disp):
        empty_members_disps = np.zeros((self.members.num, 1), dtype=object)
        members_disps = np.matrix(empty_members_disps)
        for i_member, member in enumerate(self.members.list):
            member_dofs_num = member.total_dofs_num
            member_nodes_num = len(member.nodes)
            member_node_dofs_num = int(member_dofs_num / member_nodes_num)
            v = np.zeros((member_dofs_num, 1))
            v = np.matrix(v)
            for i in range(member_dofs_num):
                member_node = i // member_node_dofs_num
                node_dof = i % member_node_dofs_num
                v[i, 0] = disp[member_node_dofs_num * member.nodes[member_node].num + node_dof, 0]
            u = member.t * v
            members_disps[i_member, 0] = u
        return members_disps

    def get_nodal_disp(self, load):
        j = 0
        o = 0
        reduced_forces = self.apply_load_boundry_conditions(load)
        reduced_disp = cho_solve(self.kc, reduced_forces)
        empty_nodal_disp = np.zeros((self.total_dofs_num, 1))
        nodal_disp = np.matrix(empty_nodal_disp)
        for i in range(self.total_dofs_num):
            if (j != self.boundaries_dof.shape[0] and i == self.boundaries_dof[j]):
                j += 1
            else:
                nodal_disp[i, 0] = reduced_disp[o, 0]
                o += 1
        return nodal_disp

    def get_internal_forces(self):
        members_disps = self.elastic_members_disps

        fixed_force = np.zeros((self.node_dofs_num * 2, 1))
        fixed_force = np.matrix(fixed_force)

        # calculate p0
        empty_members_forces = np.zeros((self.members.num, 1), dtype=object)
        members_forces = np.matrix(empty_members_forces)
        empty_p0 = np.zeros((self.yield_specs.components_num, 1))
        p0 = np.matrix(empty_p0)
        current_p0_row = 0

        for i, member in enumerate(self.members.list):
            if member.__class__.__name__ == "FrameMember2D":
                member_force = member.get_nodal_force(members_disps[i, 0], fixed_force)
                members_forces[i, 0] = member_force
                if not member.section.nonlinear.has_axial_yield:
                    p0[current_p0_row] = member_force[2, 0]
                    p0[current_p0_row + 1] = member_force[5, 0]
                else:
                    p0[current_p0_row] = member_force[0, 0]
                    p0[current_p0_row + 1] = member_force[2, 0]
                    p0[current_p0_row + 2] = member_force[3, 0]
                    p0[current_p0_row + 3] = member_force[5, 0]
            current_p0_row = current_p0_row + member.yield_specs.components_num
        return {"members_forces": members_forces, "p0": p0}

    def get_sensitivity(self):
        # fv: equivalent global force vector for a yield component's udef
        members = self.members.list
        empty_pv = np.zeros((self.yield_specs.components_num, self.yield_specs.components_num))
        pv = np.matrix(empty_pv)
        pv_column = 0

        empty_members_forces_sensitivity = np.zeros((self.members.num, self.yield_specs.components_num), dtype=object)
        empty_members_disps_sensitivity = np.zeros((self.members.num, self.yield_specs.components_num), dtype=object)
        empty_nodal_disps_sensitivity = np.zeros((1, self.yield_specs.components_num), dtype=object)

        members_forces_sensitivity = np.matrix(empty_members_forces_sensitivity)
        members_disps_sensitivity = np.matrix(empty_members_disps_sensitivity)
        nodal_disps_sensitivity = np.matrix(empty_nodal_disps_sensitivity)

        for i_member, member in enumerate(members):
            if member.__class__.__name__ == "FrameMember2D":
                for yield_point_udef in member.udefs:
                    udef_components_num = yield_point_udef.shape[1]
                    for i_component in range(udef_components_num):
                        fv_size = self.total_dofs_num
                        fv = np.zeros((fv_size, 1))
                        fv = np.matrix(fv)
                        component_udef_global = member.t.T * yield_point_udef[:, i_component]
                        start_dof = self.node_dofs_num * member.nodes[0].num
                        end_dof = self.node_dofs_num * member.nodes[1].num

                        fv[start_dof] = component_udef_global[0]
                        fv[start_dof + 1] = component_udef_global[1]
                        fv[start_dof + 2] = component_udef_global[2]

                        fv[end_dof] = component_udef_global[3]
                        fv[end_dof + 1] = component_udef_global[4]
                        fv[end_dof + 2] = component_udef_global[5]

                        affected_struc_disp = self.get_nodal_disp(fv)
                        nodal_disps_sensitivity[0, pv_column] = affected_struc_disp
                        affected_member_disps = self.get_members_disps(affected_struc_disp)
                        current_affected_member_ycns = 0
                        for i_affected_member, affected_member_disp in enumerate(affected_member_disps):

                            if i_member == i_affected_member:
                                fixed_force = -yield_point_udef[:, i_component]
                            else:
                                fixed_force = np.zeros((self.node_dofs_num * 2, 1))
                                fixed_force = np.matrix(fixed_force)
                            # FIXME: affected_member_disp[0, 0] is for numpy oskolation when use matrix in matrix and enumerating on it.
                            affected_member_force = self.members.list[i_affected_member].get_nodal_force(affected_member_disp[0, 0], fixed_force)
                            members_forces_sensitivity[i_affected_member, pv_column] = affected_member_force
                            members_disps_sensitivity[i_affected_member, pv_column] = affected_member_disp[0, 0]

                            if not member.section.nonlinear.has_axial_yield:
                                pv[current_affected_member_ycns, pv_column] = affected_member_force[2, 0]
                                pv[current_affected_member_ycns + 1, pv_column] = affected_member_force[5, 0]
                            else:
                                pv[current_affected_member_ycns, pv_column] = affected_member_force[0, 0]
                                pv[current_affected_member_ycns + 1, pv_column] = affected_member_force[2, 0]
                                pv[current_affected_member_ycns + 2, pv_column] = affected_member_force[3, 0]
                                pv[current_affected_member_ycns + 3, pv_column] = affected_member_force[5, 0]
                            current_affected_member_ycns = current_affected_member_ycns + self.members.list[i_affected_member].yield_specs.components_num

                        pv_column += 1
        results = {
            "pv": pv,
            "nodal_disps_sensitivity": nodal_disps_sensitivity,
            "members_forces_sensitivity": members_forces_sensitivity,
            "members_disps_sensitivity": members_disps_sensitivity,
        }
        return results

    def get_global_dof(self, node_num, dof):
        global_dof = int(self.node_dofs_num * node_num + dof)
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
        for member in self.members.list:
            for _ in range(member.yield_specs.points_num):
                for yield_section_row in range(member.section.yield_specs.phi.shape[0]):
                    for yield_section_column in range(member.section.yield_specs.phi.shape[1]):
                        phi[current_row + yield_section_row, current_column + yield_section_column] = member.section.yield_specs.phi[yield_section_row, yield_section_column]
                current_column = current_column + member.section.yield_specs.phi.shape[1]
                current_row = current_row + member.section.yield_specs.phi.shape[0]
        return phi

    def create_q(self):
        empty_q = np.zeros((2 * self.yield_specs.points_num, self.yield_specs.pieces_num))
        q = np.matrix(empty_q)
        yield_point_counter = 0
        yield_pieces_num_counter = 0
        for member in self.members.list:
            for _ in range(member.yield_specs.points_num):
                q[2 * yield_point_counter:2 * yield_point_counter + 2, yield_pieces_num_counter:member.section.yield_specs.pieces_num + yield_pieces_num_counter] = member.section.softening.q
                yield_point_counter += 1
                yield_pieces_num_counter += member.section.yield_specs.pieces_num
        return q

    def create_h(self):
        empty_h = np.zeros((self.yield_specs.pieces_num, 2 * self.yield_specs.points_num))
        h = np.matrix(empty_h)
        yield_point_counter = 0
        yield_pieces_num_counter = 0
        for member in self.members.list:
            for _ in range(member.yield_specs.points_num):
                h[yield_pieces_num_counter:member.section.yield_specs.pieces_num + yield_pieces_num_counter, 2 * yield_point_counter:2 * yield_point_counter + 2] = member.section.softening.h
                yield_point_counter += 1
                yield_pieces_num_counter += member.section.yield_specs.pieces_num
        return h

    def create_w(self):
        empty_w = np.zeros((2 * self.yield_specs.points_num, 2 * self.yield_specs.points_num))
        w = np.matrix(empty_w)
        yield_point_counter = 0
        for member in self.members.list:
            for _ in range(member.yield_specs.points_num):
                w[2 * yield_point_counter:2 * yield_point_counter + 2, 2 * yield_point_counter:2 * yield_point_counter + 2] = member.section.softening.w
                yield_point_counter += 1
        return w

    def create_cs(self):
        empty_cs = np.zeros((2 * self.yield_specs.points_num, 1))
        cs = np.matrix(empty_cs)
        yield_point_counter = 0
        for member in self.members.list:
            for _ in range(member.yield_specs.points_num):
                cs[2 * yield_point_counter:2 * yield_point_counter + 2, 0] = member.section.softening.cs
                yield_point_counter += 1
        return cs

    def get_yield_points_indices(self):
        yield_points_indices = []
        index_counter = 0
        for member in self.members.list:
            yield_point_pieces = int(member.yield_specs.pieces_num / member.yield_specs.points_num)
            for _ in range(member.yield_specs.points_num):
                yield_points_indices.append(
                    {
                        "begin": index_counter,
                        "end": index_counter + yield_point_pieces - 1,
                    }
                )
                index_counter += yield_point_pieces
        return yield_points_indices

    def populate_boundaries(self):
        print(self.linear_boundaries)
        return self.nodal_boundaries

    def get_boundaries_dof(self):
        boundaries_size = self.boundaries.shape[0]
        boundaries_dof = np.zeros(boundaries_size, dtype=int)
        for i in range(boundaries_size):
            boundaries_dof[i] = int(self.node_dofs_num * self.boundaries[i, 0] + self.boundaries[i, 1])
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
            for dof in range(self.total_dofs_num):
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
        condensed_m = self.apply_boundary_condition(mass_bounds, mtt)
        reduced_k00 = self.apply_boundary_condition(zero_mass_bounds, k00)
        reduced_k0t = self.apply_boundary_condition(self.boundaries_dof, k0t)
        reduced_k00_inv = np.linalg.inv(reduced_k00)
        ku0 = -(np.dot(reduced_k00_inv, reduced_k0t))
        condensed_k = reduced_ktt - np.dot(np.dot(np.transpose(reduced_k0t), reduced_k00_inv), reduced_k0t)
        return condensed_k, condensed_m, ku0, reduced_k00_inv, reduced_k00

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
        for dof in range(self.total_dofs_num):
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

    # def compute_modes_props(self):
    #     damping = self.damping
    #     eigvals, modes = eigh(self.condensed_k, self.condensed_m, eigvals_only=False)
    #     wn = np.sqrt(eigvals)
    #     wd = np.sqrt(1 - damping ** 2) * wn
    #     return wn, wd, modes

    # def compute_modal_props(self):
    #     m_modal = np.dot(np.transpose(self.modes), np.dot(self.condensed_m, self.modes))
    #     k_modal = np.dot(np.transpose(self.modes), np.dot(self.condensed_k, self.modes))
    #     return m_modal, k_modal

    # def compute_i_duhamel(self, t1, t2, wn, wd):
    #     damping = self.damping
    #     wd = np.sqrt(1 - damping ** 2) * wn
    #     i11 = (np.exp(damping * wn * t2) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.cos(wd * t2) + wd * np.sin(wd * t2))
    #     i12 = (np.exp(damping * wn * t1) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.cos(wd * t1) + wd * np.sin(wd * t1))
    #     i1 = i11 - i12
    #     i21 = (np.exp(damping * wn * t2) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.sin(wd * t2) - wd * np.cos(wd * t2))
    #     i22 = (np.exp(damping * wn * t1) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.sin(wd * t1) - wd * np.cos(wd * t1))
    #     i2 = i21 - i22
    #     i3 = (t2 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i21 + ((wd) / ((damping * wn) ** 2 + wd ** 2)) * i11 - ((t1 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i22 + ((wd) / ((damping * wn) ** 2 + wd ** 2)) * i12)
    #     i4 = (t2 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i11 - ((wd) / ((damping * wn) ** 2 + wd ** 2)) * i21 - ((t1 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i12 - ((wd) / ((damping * wn) ** 2 + wd ** 2)) * i22)
    #     return i1, i2, i3, i4

    # def compute_abx_duhamel(self, t1, t2, i1, i2, i3, i4, wn, mn, a1, b1, p1, p2):
    #     damping = self.damping
    #     deltat = t2 - t1
    #     deltap = p2 - p1
    #     wd = np.sqrt(1 - damping ** 2) * wn
    #     a2 = a1 + (p1 - t1 * deltap / deltat) * i1 + (deltap / deltat) * i4
    #     b2 = b1 + (p1 - t1 * deltap / deltat) * i2 + (deltap / deltat) * i3
    #     un = (np.exp(-1 * damping * wn * t2) / (mn * wd)) * (a2 * np.sin(wd * t2) - b2 * np.cos(wd * t2))
    #     return a2, b2, un

    # def displacement_unrestrained(U, JTR):
    #     i_restraint = 0
    #     i_free = 0
    #     size_of_U_nonrestraint = U.shape[0]+JTR.shape[0]
    #     U_nonrestraint = np.zeros((size_of_U_nonrestraint, 1))
    #     for i in range(size_of_U_nonrestraint):
    #         if i == 3*JTR[i_restraint, 0]+JTR[i_restraint, 1]:
    #             U_nonrestraint[i, 0] = 0
    #             i_restraint += 1
    #         else:
    #             U_nonrestraint[i, 0] = U[i_free, 0]
    #             i_free += 1
    #     return U_nonrestraint

    # def non_condensed_displacement(Ut, U0):

    #     size_of_U = Ut.shape[0]+U0.shape[0]
    #     U = np.zeros((size_of_U, 1))
    #     i_U0 = 0
    #     i_Ut = 0
    #     for i in range(size_of_U):
    #         if i % 3 == 2:
    #             U[i, 0] = U0[i_U0, 0]
    #             i_U0 += 1
    #         else:
    #             U[i, 0] = Ut[i_Ut, 0]
    #             i_Ut += 1
    #     return U
