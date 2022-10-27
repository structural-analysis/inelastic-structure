import numpy as np
from scipy.linalg import cho_solve

from src.models.loads import Loads
from src.models.structure import Structure


class Analysis:
    def __init__(self, structure_input, loads_input, general_info):
        self.structure = Structure(structure_input)
        self.loads = Loads(loads_input)
        self.general_info = general_info
        self.type = self._get_type()

        if self.type == "static":
            self.total_load = self.loads.get_total_load(self.structure, self.loads)
            self.elastic_nodal_disp = self.get_nodal_disp(self.total_load)
            self.elastic_members_disps = self.get_members_disps(self.elastic_nodal_disp)
            internal_forces = self.get_internal_forces(self.elastic_members_disps)
            self.elastic_members_forces = internal_forces["members_forces"]
            self.p0 = internal_forces["p0"]
            self.d0 = self.get_nodal_disp_limits(self.elastic_nodal_disp)
            sensitivity = self.get_sensitivity()
            self.pv = sensitivity["pv"]
            self.members_forces_sensitivity = sensitivity["members_forces_sensitivity"]
            self.members_disps_sensitivity = sensitivity["members_disps_sensitivity"]
            self.nodal_disps_sensitivity = sensitivity["nodal_disps_sensitivity"]
            self.dv = self.get_nodal_disp_limits_sensitivity_rows()
        elif self.type == "dynamic":
            self.damping = self.general_info["dynamic_analysis"]["damping"]
            structure = self.structure
            loads = self.loads

            modes = np.matrix(structure.modes)
            modes_num = modes.shape[1]
            self.m_modal = structure.get_modal_property(structure.condensed_m, modes)
            self.k_modal = structure.get_modal_property(structure.condensed_k, modes)

            time_steps = loads.dynamic[0].magnitude.shape[0]
            self.time_steps = time_steps
            self.time = loads.dynamic[0].time

            self.modal_loads = np.zeros((time_steps, modes_num, 1))
            self.a_duhamel = np.zeros((time_steps, modes_num, 1))
            self.b_duhamel = np.zeros((time_steps, modes_num, 1))
            self.un = np.zeros((time_steps, modes_num, 1))
            self.elastic_nodal_disp = np.zeros((time_steps, structure.total_dofs_num, 1))
            self.elastic_members_disps = np.zeros((time_steps, structure.members.num, 1), dtype=object)
            print(self.elastic_nodal_disp.shape)
            self.elastic_members_forces = np.zeros((time_steps, structure.members.num, 1), dtype=object)
            self.p0 = np.zeros((time_steps, structure.yield_specs.components_num, 1))
            # self.d0 = np.zeros((time_steps, structure.limits["disp_limits"].shape[0], 1))
            # self.elastic_nodal_disp
            # self.elastic_nodal_disp

            for time_step in range(1, time_steps):
                print(f"{time_step=}")
                self.total_load = loads.get_total_load(structure, loads, time_step)
                self.elastic_nodal_disp[time_step, :, :] = self.get_dynamic_nodal_disp(self.total_load, modes, time_step)

                # with open("disp.txt", "a") as f:
                #     f.write(f"{self.u[time_step, 4, 0]}\n")
                self.elastic_members_disps[time_step, :, :] = self.get_members_disps(self.elastic_nodal_disp[time_step, :, :])
                internal_forces = self.get_internal_forces(self.elastic_members_disps[time_step, :, :])
                self.elastic_members_forces[time_step, :, :] = internal_forces["members_forces"]
                self.p0[time_step, :, :] = internal_forces["p0"]
                self.d0[time_step, :, :] = self.get_nodal_disp_limits(self.elastic_nodal_disp)
                pv = self.get_dynamic_sensitivity(modes, time_step)
                print(f"{pv=}")
                # sensitivity = self.get_sensitivity()
                # self.pv = sensitivity["pv"]
                # self.members_forces_sensitivity = sensitivity["members_forces_sensitivity"]
                # self.members_disps_sensitivity = sensitivity["members_disps_sensitivity"]
                # self.nodal_disps_sensitivity = sensitivity["nodal_disps_sensitivity"]
                # self.dv = self.get_nodal_disp_limits_sensitivity_rows()

    def _get_type(self):
        if self.general_info.get("dynamic_analysis") and self.general_info["dynamic_analysis"]["enabled"]:
            type = "dynamic"
        else:
            type = "static"
        return type

    def get_nodal_disp(self, total_load):
        j = 0
        o = 0
        structure = self.structure
        reduced_total_load = self.loads.apply_boundary_conditions(structure.boundaries_dof, total_load)
        reduced_disp = cho_solve(structure.kc, reduced_total_load)
        empty_nodal_disp = np.zeros((structure.total_dofs_num, 1))
        nodal_disp = np.matrix(empty_nodal_disp)
        for i in range(structure.total_dofs_num):
            if (j != structure.boundaries_dof.shape[0] and i == structure.boundaries_dof[j]):
                j += 1
            else:
                nodal_disp[i, 0] = reduced_disp[o, 0]
                o += 1
        return nodal_disp

    def get_members_disps(self, disp):
        structure = self.structure
        empty_members_disps = np.zeros((structure.members.num, 1), dtype=object)
        members_disps = np.matrix(empty_members_disps)
        for i_member, member in enumerate(structure.members.list):
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

    def get_internal_forces(self, members_disps):
        structure = self.structure

        fixed_force = np.zeros((structure.node_dofs_num * 2, 1))
        fixed_force = np.matrix(fixed_force)

        # calculate p0
        empty_members_forces = np.zeros((structure.members.num, 1), dtype=object)
        members_forces = np.matrix(empty_members_forces)
        empty_p0 = np.zeros((structure.yield_specs.components_num, 1))
        p0 = np.matrix(empty_p0)
        current_p0_row = 0

        for i, member in enumerate(structure.members.list):
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
            elif member.__class__.__name__ == "PlateMember":
                member_force = member.get_yield_components_force(members_disps[i, 0])
                p0[current_p0_row:(current_p0_row + member.yield_specs.components_num)] = member_force
                members_forces[i, 0] = member_force
            current_p0_row = current_p0_row + member.yield_specs.components_num
        return {"members_forces": members_forces, "p0": p0}

    def get_nodal_disp_limits(self, elastic_nodal_disp):
        structure = self.structure
        disp_limits = structure.limits["disp_limits"]
        disp_limits_num = disp_limits.shape[0]
        empty_d0 = np.zeros((disp_limits_num, 1))
        d0 = np.matrix(empty_d0)
        for i, disp_limit in enumerate(disp_limits):
            node = disp_limit[0]
            node_dof = disp_limit[1]
            dof = structure.get_global_dof(node, node_dof)
            d0[i, 0] = elastic_nodal_disp[dof, 0]
        return d0

    def get_sensitivity(self):
        structure = self.structure
        # fv: equivalent global force vector for a yield component's udef
        members = structure.members.list
        empty_pv = np.zeros((structure.yield_specs.components_num, structure.yield_specs.components_num))
        pv = np.matrix(empty_pv)
        pv_column = 0

        empty_members_forces_sensitivity = np.zeros((structure.members.num, structure.yield_specs.components_num), dtype=object)
        empty_members_disps_sensitivity = np.zeros((structure.members.num, structure.yield_specs.components_num), dtype=object)
        empty_nodal_disps_sensitivity = np.zeros((1, structure.yield_specs.components_num), dtype=object)

        members_forces_sensitivity = np.matrix(empty_members_forces_sensitivity)
        members_disps_sensitivity = np.matrix(empty_members_disps_sensitivity)
        nodal_disps_sensitivity = np.matrix(empty_nodal_disps_sensitivity)

        for i_member, member in enumerate(members):
            if member.__class__.__name__ == "FrameMember2D":
                for yield_point_udef in member.udefs:
                    udef_components_num = yield_point_udef.shape[1]
                    for i_component in range(udef_components_num):
                        fv_size = structure.total_dofs_num
                        fv = np.zeros((fv_size, 1))
                        fv = np.matrix(fv)
                        component_udef_global = member.t.T * yield_point_udef[:, i_component]
                        start_dof = structure.node_dofs_num * member.nodes[0].num
                        end_dof = structure.node_dofs_num * member.nodes[1].num

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
                                fixed_force = np.zeros((structure.node_dofs_num * 2, 1))
                                fixed_force = np.matrix(fixed_force)
                            # FIXME: affected_member_disp[0, 0] is for numpy oskolation when use matrix in matrix and enumerating on it.
                            affected_member_force = structure.members.list[i_affected_member].get_nodal_force(affected_member_disp[0, 0], fixed_force)
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
                            current_affected_member_ycns = current_affected_member_ycns + structure.members.list[i_affected_member].yield_specs.components_num

                        pv_column += 1
        results = {
            "pv": pv,
            "nodal_disps_sensitivity": nodal_disps_sensitivity,
            "members_forces_sensitivity": members_forces_sensitivity,
            "members_disps_sensitivity": members_disps_sensitivity,
        }
        return results

    def get_nodal_disp_limits_sensitivity_rows(self):
        structure = self.structure
        disp_limits = structure.limits["disp_limits"]
        disp_limits_num = disp_limits.shape[0]
        empty_dv = np.zeros((disp_limits_num, structure.yield_specs.components_num))
        dv = np.matrix(empty_dv)
        for i, disp_limit in enumerate(disp_limits):
            node = disp_limit[0]
            node_dof = disp_limit[1]
            dof = structure.get_global_dof(node, node_dof)
            for j in range(structure.yield_specs.components_num):
                dv[i, j] = self.nodal_disps_sensitivity[0, j][dof, 0]
        return dv

    def get_i_duhamel(self, t1, t2, wn, wd):
        damping = self.damping
        wd = np.sqrt(1 - damping ** 2) * wn
        i11 = (np.exp(damping * wn * t2) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.cos(wd * t2) + wd * np.sin(wd * t2))
        i12 = (np.exp(damping * wn * t1) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.cos(wd * t1) + wd * np.sin(wd * t1))
        i1 = i11 - i12
        i21 = (np.exp(damping * wn * t2) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.sin(wd * t2) - wd * np.cos(wd * t2))
        i22 = (np.exp(damping * wn * t1) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.sin(wd * t1) - wd * np.cos(wd * t1))
        i2 = i21 - i22
        i3 = (t2 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i21 + ((wd) / ((damping * wn) ** 2 + wd ** 2)) * i11 - ((t1 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i22 + ((wd) / ((damping * wn) ** 2 + wd ** 2)) * i12)
        i4 = (t2 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i11 - ((wd) / ((damping * wn) ** 2 + wd ** 2)) * i21 - ((t1 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i12 - ((wd) / ((damping * wn) ** 2 + wd ** 2)) * i22)
        return i1, i2, i3, i4

    def get_abx_duhamel(self, t1, t2, i1, i2, i3, i4, wn, mn, a1, b1, p1, p2):
        damping = self.damping
        deltat = t2 - t1
        deltap = p2 - p1
        wd = np.sqrt(1 - damping ** 2) * wn
        a2 = a1 + (p1 - t1 * deltap / deltat) * i1 + (deltap / deltat) * i4
        b2 = b1 + (p1 - t1 * deltap / deltat) * i2 + (deltap / deltat) * i3
        un = (np.exp(-1 * damping * wn * t2) / (mn * wd)) * (a2 * np.sin(wd * t2) - b2 * np.cos(wd * t2))
        return a2, b2, un

    def get_dynamic_nodal_disp(self, total_load, modes, time_step):
        structure = self.structure
        loads = self.loads
        modes_num = modes.shape[1]
        condense_load, reduced_p0 = loads.apply_static_condensation(structure, total_load)
        self.modal_load = loads.get_modal_load(condense_load, modes)
        self.modal_loads[time_step, :, :] = self.modal_load

        t1 = self.time[time_step - 1, 0]
        t2 = self.time[time_step, 0]
        for mode_num in range(modes_num):
            wn = structure.wns[mode_num]
            wd = structure.wds[mode_num]
            i1, i2, i3, i4 = self.get_i_duhamel(t1, t2, wn, wd)

            mn = self.m_modal[mode_num, mode_num]
            p1 = self.modal_loads[time_step - 1, mode_num]
            p2 = self.modal_loads[time_step, mode_num]
            a1 = self.a_duhamel[time_step - 1, mode_num]
            b1 = self.b_duhamel[time_step - 1, mode_num]
            a2, b2, un = self.get_abx_duhamel(t1, t2, i1, i2, i3, i4, wn, mn, a1, b1, p1, p2)
            self.a_duhamel[time_step, mode_num] = a2
            self.b_duhamel[time_step, mode_num] = b2
            self.un[time_step, mode_num] = un

        ut = np.dot(modes, self.un[time_step, :, :])
        u0 = np.dot(structure.reduced_k00, reduced_p0) + np.dot(structure.ku0, ut)

        unrestrianed_ut = structure.undo_disp_boundary_condition(ut, structure.mass_bounds)
        unrestrianed_u0 = structure.undo_disp_boundary_condition(u0, structure.zero_mass_bounds)

        u = structure.undo_disp_condensation(unrestrianed_ut, unrestrianed_u0)
        return u

    def get_dynamic_unit_nodal_disp(self, total_load, modes, time_step):
        structure = self.structure
        loads = self.loads
        modes_num = modes.shape[1]
        condense_load, reduced_p0 = loads.apply_static_condensation(structure, total_load)
        modal_load = loads.get_modal_load(condense_load, modes)
        # self.modal_loads[time_step, :, :] = modal_load
        a2_modes = np.zeros((modes_num, 1))
        b2_mdoes = np.zeros((modes_num, 1))
        p2_modes = np.zeros((modes_num, 1))
        t1 = self.time[time_step - 1, 0]
        t2 = self.time[time_step, 0]
        for mode_num in range(modes_num):
            wn = structure.wns[mode_num]
            wd = structure.wds[mode_num]
            i1, i2, i3, i4 = self.get_i_duhamel(t1, t2, wn, wd)

            mn = self.m_modal[mode_num, mode_num]
            p1 = 0
            p2 = modal_load
            a1 = 0
            b1 = 0
            a2, b2, un = self.get_abx_duhamel(t1, t2, i1, i2, i3, i4, wn, mn, a1, b1, p1, p2)
            a2_modes[mode_num, 0] = a2
            b2_mdoes[mode_num, 0] = b2
            p2_modes[mode_num, 0] = p2
            self.un[time_step, mode_num] = un

        ut = np.dot(modes, self.un[time_step, :, :])
        u0 = np.dot(structure.reduced_k00, reduced_p0) + np.dot(structure.ku0, ut)

        unrestrianed_ut = structure.undo_disp_boundary_condition(ut, structure.mass_bounds)
        unrestrianed_u0 = structure.undo_disp_boundary_condition(u0, structure.zero_mass_bounds)

        u = structure.undo_disp_condensation(unrestrianed_ut, unrestrianed_u0)
        return u, p2_modes, a2_modes, b2_mdoes

    def get_dynamic_sensitivity(self, modes, time_step):
        structure = self.structure
        # fv: equivalent global force vector for a yield component's udef
        members = structure.members.list
        empty_pv = np.zeros((structure.yield_specs.components_num, structure.yield_specs.components_num))
        pv = np.matrix(empty_pv)
        pv_column = 0

        empty_members_forces_sensitivity = np.zeros((structure.members.num, structure.yield_specs.components_num), dtype=object)
        empty_members_disps_sensitivity = np.zeros((structure.members.num, structure.yield_specs.components_num), dtype=object)
        empty_nodal_disps_sensitivity = np.zeros((1, structure.yield_specs.components_num), dtype=object)
        empty_p2_sensitivity = np.zeros((1, structure.yield_specs.components_num), dtype=object)
        empty_a2_sensitivity = np.zeros((1, structure.yield_specs.components_num), dtype=object)
        empty_b2_sensitivity = np.zeros((1, structure.yield_specs.components_num), dtype=object)

        members_forces_sensitivity = np.matrix(empty_members_forces_sensitivity)
        members_disps_sensitivity = np.matrix(empty_members_disps_sensitivity)
        nodal_disps_sensitivity = np.matrix(empty_nodal_disps_sensitivity)
        p2_sensitivity = np.matrix(empty_p2_sensitivity)
        a2_sensitivity = np.matrix(empty_a2_sensitivity)
        b2_sensitivity = np.matrix(empty_b2_sensitivity)

        for i_member, member in enumerate(members):
            if member.__class__.__name__ == "FrameMember2D":
                for yield_point_udef in member.udefs:
                    udef_components_num = yield_point_udef.shape[1]
                    for i_component in range(udef_components_num):
                        fv_size = structure.total_dofs_num
                        fv = np.zeros((fv_size, 1))
                        fv = np.matrix(fv)
                        component_udef_global = member.t.T * yield_point_udef[:, i_component]
                        start_dof = structure.node_dofs_num * member.nodes[0].num
                        end_dof = structure.node_dofs_num * member.nodes[1].num

                        fv[start_dof] = component_udef_global[0]
                        fv[start_dof + 1] = component_udef_global[1]
                        fv[start_dof + 2] = component_udef_global[2]

                        fv[end_dof] = component_udef_global[3]
                        fv[end_dof + 1] = component_udef_global[4]
                        fv[end_dof + 2] = component_udef_global[5]

                        affected_struc_disp, p2_modes, a2_modes, b2_mdoes = self.get_dynamic_unit_nodal_disp(fv, modes, time_step)
                        nodal_disps_sensitivity[0, pv_column] = affected_struc_disp
                        p2_sensitivity[0, pv_column] = p2_modes
                        a2_sensitivity[0, pv_column] = a2_modes
                        b2_sensitivity[0, pv_column] = b2_mdoes
                        affected_member_disps = self.get_members_disps(affected_struc_disp)
                        current_affected_member_ycns = 0
                        for i_affected_member, affected_member_disp in enumerate(affected_member_disps):

                            if i_member == i_affected_member:
                                fixed_force = -yield_point_udef[:, i_component]
                            else:
                                fixed_force = np.zeros((structure.node_dofs_num * 2, 1))
                                fixed_force = np.matrix(fixed_force)
                            # FIXME: affected_member_disp[0, 0] is for numpy oskolation when use matrix in matrix and enumerating on it.
                            affected_member_force = structure.members.list[i_affected_member].get_nodal_force(affected_member_disp[0, 0], fixed_force)
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
                            current_affected_member_ycns = current_affected_member_ycns + structure.members.list[i_affected_member].yield_specs.components_num

                        pv_column += 1
        results = {
            "pv": pv,
            "nodal_disps_sensitivity": nodal_disps_sensitivity,
            "members_forces_sensitivity": members_forces_sensitivity,
            "members_disps_sensitivity": members_disps_sensitivity,
        }
        return results
