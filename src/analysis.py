import numpy as np
from dataclasses import dataclass
from scipy.linalg import cho_solve

from src.models.loads import Loads
from src.program.prepare import RawData
from src.program.main import MahiniMethod
from src.models.structure import Structure


@dataclass
class InternalResponses:
    p0: np.matrix
    members_nodal_forces: np.matrix


@dataclass
class StaticSensitivity:
    pv: np.matrix
    nodal_disps: np.matrix
    members_forces: np.matrix
    members_disps: np.matrix


@dataclass
class DynamicSensitivity:
    pv: np.matrix
    nodal_disps: np.matrix
    members_forces: np.matrix
    members_disps: np.matrix
    modal_loads: np.matrix
    a2: np.matrix
    b2: np.matrix


class Analysis:
    def __init__(self, structure_input, loads_input, general_info):
        self.structure = Structure(structure_input)
        self.loads = Loads(loads_input)
        self.general_info = general_info

        if self.type == "static":
            self.total_load = self.loads.get_total_load(self.structure, self.loads)
            self.elastic_nodal_disp = self.get_nodal_disp(self.total_load)
            self.elastic_members_disps = self.get_members_disps(self.elastic_nodal_disp[0, 0])
            internal_responses = self.get_internal_responses(self.elastic_members_disps)
            self.elastic_members_nodal_forces = internal_responses.members_nodal_forces
            self.p0 = internal_responses.p0
            self.d0 = self.get_nodal_disp_limits(self.elastic_nodal_disp[0, 0])
            sensitivity = self.get_sensitivity()
            self.pv = sensitivity.pv
            self.members_forces_sensitivity = sensitivity.members_forces
            self.members_disps_sensitivity = sensitivity.members_disps
            self.nodal_disps_sensitivity = sensitivity.nodal_disps
            self.dv = self.get_nodal_disp_limits_sensitivity_rows()
            raw_data = RawData(self)
            mahini_method = MahiniMethod(raw_data)
            self.plastic_vars = mahini_method.solve()

        elif self.type == "dynamic":
            self.damping = self.general_info["dynamic_analysis"]["damping"]
            structure = self.structure
            loads = self.loads

            modes = np.matrix(structure.modes)
            modes_count = modes.shape[1]
            self.m_modal = structure.get_modal_property(structure.condensed_m, modes)
            self.k_modal = structure.get_modal_property(structure.condensed_k, modes)

            time_steps = loads.dynamic[0].magnitude.shape[0]
            self.time_steps = time_steps
            self.time = loads.dynamic[0].time

            # self.modal_loads = np.zeros((time_steps, modes_count, 1))
            # self.a_duhamel = np.zeros((time_steps, modes_count, 1))
            # self.b_duhamel = np.zeros((time_steps, modes_count, 1))
            # self.un = np.zeros((time_steps, modes_count, 1))

            self.modal_loads = np.matrix(np.zeros((time_steps, 1), dtype=object))
            modal_load = np.matrix(np.zeros((modes_count, 1)))
            modal_loads = np.matrix((1, 1), dtype=object)
            modal_loads[0, 0] = modal_load
            self.modal_loads[0, 0] = modal_loads
            a_duhamel = np.matrix(np.zeros((modes_count, 1)))
            a_duhamels = np.matrix((1, 1), dtype=object)
            self.a_duhamel = np.matrix(np.zeros((time_steps, 1), dtype=object))
            a_duhamels[0, 0] = a_duhamel
            self.a_duhamel[0, 0] = a_duhamels

            b_duhamel = np.matrix(np.zeros((modes_count, 1)))
            b_duhamels = np.matrix((1, 1), dtype=object)
            self.b_duhamel = np.matrix(np.zeros((time_steps, 1), dtype=object))
            b_duhamels[0, 0] = b_duhamel
            self.b_duhamel[0, 0] = b_duhamels

            self.modal_disp_history = np.matrix(np.zeros((time_steps, 1), dtype=object))

            # self.elastic_nodal_disp_history = np.zeros((time_steps, structure.dofs_count, 1))
            self.elastic_nodal_disp_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.elastic_members_disps_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.elastic_members_forces_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.nodal_disps_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.members_forces_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.members_disps_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.modal_loads_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.a2_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.b2_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.p0_history = np.zeros((time_steps, 1), dtype=object)
            self.d0_history = np.zeros((time_steps, 1), dtype=object)
            self.pv_history = np.zeros((time_steps, 1), dtype=object)

            # self.p0_history = np.zeros((time_steps, structure.yield_specs.components_count, 1))
            # self.d0_history = np.zeros((time_steps, structure.limits["disp_limits"].shape[0], 1))
            # self.pv_history = np.zeros((time_steps, structure.yield_specs.components_count, structure.yield_specs.components_count))

            for time_step in range(1, time_steps):
                print(f"{time_step=}")
                self.total_load = loads.get_total_load(structure, loads, time_step)
                a2s, b2s, elastic_modal_loads, elastic_nodal_disp = self.get_dynamic_nodal_disp(
                    time_step=time_step,
                    modes=modes,
                    previous_modal_loads=self.modal_loads[time_step - 1, 0],
                    total_load=self.total_load,
                    a1s=self.a_duhamel[time_step - 1, 0],
                    b1s=self.b_duhamel[time_step - 1, 0],
                )
                self.a_duhamel[time_step, 0] = a2s
                self.b_duhamel[time_step, 0] = b2s

                self.elastic_nodal_disp_history[time_step, 0] = elastic_nodal_disp
                elastic_members_disps = self.get_members_disps(elastic_nodal_disp[0, 0])
                self.elastic_members_disps_history[time_step, 0] = elastic_members_disps

                internal_responses = self.get_internal_responses(elastic_members_disps)
                self.elastic_members_forces_history[time_step, 0] = internal_responses.members_nodal_forces
                # with open("section4-elastic-force.txt", "a") as f:
                #     f.write(f"{internal_forces['members_forces'][1][0, 0][5, 0]}\n")
                self.p0 = internal_responses.p0
                self.p0_history[time_step, 0] = self.p0

                self.d0 = self.get_nodal_disp_limits(elastic_nodal_disp[0, 0])
                self.d0_history[time_step, 0] = self.d0

                sensitivity = self.get_dynamic_sensitivity(modes, time_step)
                self.pv = sensitivity.pv
                self.pv_history[time_step, 0] = self.pv

                self.nodal_disps_sensitivity = sensitivity.nodal_disps
                self.nodal_disps_sensitivity_history[time_step, 0] = self.nodal_disps_sensitivity

                self.members_forces_sensitivity = sensitivity.members_forces
                self.members_forces_sensitivity_history[time_step, 0] = self.members_forces_sensitivity

                self.members_disps_sensitivity = sensitivity.members_disps
                self.members_disps_sensitivity_history[time_step, 0] = self.members_disps_sensitivity

                self.modal_loads_sensitivity = sensitivity.modal_loads
                self.modal_loads_sensitivity_history[time_step, 0] = self.modal_loads_sensitivity

                self.a2_sensitivity = sensitivity.a2
                self.a2_sensitivity_history[time_step, 0] = self.a2_sensitivity

                self.b2_sensitivity = sensitivity.b2
                self.b2_sensitivity_history[time_step, 0] = self.b2_sensitivity

                self.dv = self.get_nodal_disp_limits_sensitivity_rows()
                raw_data = RawData(self)
                mahini_method = MahiniMethod(raw_data)
                self.plastic_vars = mahini_method.solve()

    @property
    def type(self):
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
        nodal_disp = np.matrix(np.zeros((1, 1), dtype=object))
        disp = np.matrix(np.zeros((structure.dofs_count, 1)))
        for i in range(structure.dofs_count):
            if (j != structure.boundaries_dof.shape[0] and i == structure.boundaries_dof[j]):
                j += 1
            else:
                disp[i, 0] = reduced_disp[o, 0]
                o += 1
        nodal_disp[0, 0] = disp
        return nodal_disp

    def get_members_disps(self, disp):
        structure = self.structure
        members_disps = np.matrix(np.zeros((structure.members.num, 1), dtype=object))
        for i_member, member in enumerate(structure.members.list):
            member_dofs_count = member.dofs_count
            member_nodes_count = len(member.nodes)
            member_node_dofs_count = int(member_dofs_count / member_nodes_count)
            v = np.zeros((member_dofs_count, 1))
            v = np.matrix(v)
            for i in range(member_dofs_count):
                member_node = i // member_node_dofs_count
                node_dof = i % member_node_dofs_count
                v[i, 0] = disp[member_node_dofs_count * member.nodes[member_node].num + node_dof, 0]
            u = member.t * v
            members_disps[i_member, 0] = u
        return members_disps

    def get_internal_responses(self, members_disps):
        structure = self.structure
        # calculate p0
        members_nodal_forces = np.matrix(np.zeros((structure.members.num, 1), dtype=object))
        p0 = np.matrix(np.zeros((structure.yield_specs.components_count, 1)))
        base_p0_row = 0

        for i, member in enumerate(structure.members.list):
            member_response = member.get_response(members_disps[i, 0])
            member_nodal_force = member_response.nodal_force
            member_yield_components_force = member_response.yield_components_force
            members_nodal_forces[i, 0] = member_nodal_force
            p0[base_p0_row:(base_p0_row + member.yield_specs.components_count)] = member_yield_components_force
            base_p0_row = base_p0_row + member.yield_specs.components_count
        return InternalResponses(p0=p0, members_nodal_forces=members_nodal_forces)

    def get_nodal_disp_limits(self, elastic_nodal_disp):
        structure = self.structure
        disp_limits = structure.limits["disp_limits"]
        disp_limits_count = disp_limits.shape[0]
        d0 = np.matrix(np.zeros((disp_limits_count, 1)))
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
        pv = np.matrix(np.zeros((structure.yield_specs.components_count, structure.yield_specs.components_count)))
        members_forces_sensitivity = np.matrix(np.zeros((structure.members.num, structure.yield_specs.components_count), dtype=object))
        nodal_disps_sensitivity = np.matrix(np.zeros((1, structure.yield_specs.components_count), dtype=object))
        members_disps_sensitivity = np.matrix(np.zeros((structure.members.num, structure.yield_specs.components_count), dtype=object))
        pv_column = 0

        for member_num, member in enumerate(members):
            for force in member.udefs.T:
                fv = np.matrix(np.zeros((structure.dofs_count, 1)))
                global_force = member.t.T * force.T
                local_node_base_dof = 0
                for node in member.nodes:
                    global_node_base_dof = structure.node_dofs_count * node.num
                    for i in range(structure.node_dofs_count):
                        fv[global_node_base_dof + i] = global_force[local_node_base_dof + i]
                    local_node_base_dof += structure.node_dofs_count

                affected_structure_disp = self.get_nodal_disp(fv)
                nodal_disps_sensitivity[0, pv_column] = affected_structure_disp[0, 0]
                affected_member_disps = self.get_members_disps(affected_structure_disp[0, 0])
                current_affected_member_ycns = 0
                for affected_member_num, affected_member_disp in enumerate(affected_member_disps):
                    fixed_force = -force.T if member_num == affected_member_num else None
                    affected_member_response = structure.members.list[affected_member_num].get_response(affected_member_disp[0, 0], fixed_force)
                    affected_member_nodal_force = affected_member_response.nodal_force
                    affected_member_yield_components_force = affected_member_response.yield_components_force
                    members_forces_sensitivity[affected_member_num, pv_column] = affected_member_nodal_force
                    members_disps_sensitivity[affected_member_num, pv_column] = affected_member_disp[0, 0]
                    pv[current_affected_member_ycns:(current_affected_member_ycns + structure.members.list[affected_member_num].yield_specs.components_count), pv_column] = affected_member_yield_components_force
                    current_affected_member_ycns = current_affected_member_ycns + structure.members.list[affected_member_num].yield_specs.components_count
                pv_column += 1

        sensitivity = StaticSensitivity(
            pv=pv,
            nodal_disps=nodal_disps_sensitivity,
            members_forces=members_forces_sensitivity,
            members_disps=members_disps_sensitivity,
        )
        return sensitivity

    def get_nodal_disp_limits_sensitivity_rows(self):
        structure = self.structure
        disp_limits = structure.limits["disp_limits"]
        disp_limits_count = disp_limits.shape[0]
        dv = np.matrix(np.zeros((disp_limits_count, structure.yield_specs.components_count)))
        for i, disp_limit in enumerate(disp_limits):
            node = disp_limit[0]
            node_dof = disp_limit[1]
            dof = structure.get_global_dof(node, node_dof)
            for j in range(structure.yield_specs.components_count):
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

    def get_modal_disp(self, time_step, modes, previous_modal_loads, modal_loads, a1s, b1s):
        structure = self.structure
        modes_count = modes.shape[1]

        modal_disp = np.matrix(np.zeros((modes_count, 1)))
        modal_disps = np.matrix(np.zeros((1, 1), dtype=object))
        a2 = np.matrix(np.zeros((modes_count, 1)))
        a2s = np.matrix(np.zeros((1, 1), dtype=object))
        b2 = np.matrix(np.zeros((modes_count, 1)))
        b2s = np.matrix(np.zeros((1, 1), dtype=object))

        t1 = self.time[time_step - 1, 0]
        t2 = self.time[time_step, 0]
        for mode_num in range(modes_count):
            wn = structure.wns[mode_num]
            wd = structure.wds[mode_num]
            i1, i2, i3, i4 = self.get_i_duhamel(t1, t2, wn, wd)

            mn = self.m_modal[mode_num, mode_num]
            p1 = previous_modal_loads[0, 0][mode_num, 0]
            p2 = modal_loads[0, 0][mode_num, 0]
            a1 = a1s[0, 0][mode_num, 0]
            b1 = b1s[0, 0][mode_num, 0]

            a2[mode_num, 0], b2[mode_num, 0], modal_disp[mode_num, 0] = self.get_abx_duhamel(t1, t2, i1, i2, i3, i4, wn, mn, a1, b1, p1, p2)

        a2s[0, 0] = a2
        b2s[0, 0] = b2

        modal_disps[0, 0] = modal_disp
        return modal_disps, a2s, b2s

    def get_dynamic_nodal_disp(self, time_step, modes, previous_modal_loads, total_load, a1s, b1s):
        structure = self.structure
        loads = self.loads
        nodal_disp = np.matrix(np.zeros((1, 1), dtype=object))
        modal_loads = np.matrix(np.zeros((1, 1), dtype=object))

        condense_load, reduced_p0 = loads.apply_static_condensation(structure, total_load)
        modal_loads[0, 0] = loads.get_modal_load(condense_load, modes)
        modal_disps, a2s, b2s = self.get_modal_disp(
            time_step=time_step,
            modes=modes,
            # previous_modal_loads=self.modal_loads[time_step - 1, 0],
            previous_modal_loads=previous_modal_loads,
            modal_loads=modal_loads,
            a1s=a1s,
            b1s=b1s,
            # b1s=self.b_duhamel[time_step - 1, 0],
        )
        self.modal_disp_history[time_step, 0] = modal_disps
        ut = np.dot(modes, modal_disps[0, 0])
        u0 = np.dot(structure.reduced_k00, reduced_p0) + np.dot(structure.ku0, ut)

        unrestrianed_ut = structure.undo_disp_boundary_condition(ut, structure.mass_bounds)
        unrestrianed_u0 = structure.undo_disp_boundary_condition(u0, structure.zero_mass_bounds)

        disp = structure.undo_disp_condensation(unrestrianed_ut, unrestrianed_u0)
        nodal_disp[0, 0] = disp
        return a2s, b2s, modal_loads, nodal_disp

    def get_dynamic_sensitivity(self, modes, time_step):
        structure = self.structure
        modes_count = modes.shape[1]
        # fv: equivalent global force vector for a yield component's udef
        members = structure.members.list
        pv = np.matrix(np.zeros((
            structure.yield_specs.components_count, structure.yield_specs.components_count
        )))
        pv_column = 0

        members_forces_sensitivity = np.matrix(np.zeros((
            structure.members.num, structure.yield_specs.components_count), dtype=object
        ))
        members_disps_sensitivity = np.matrix(np.zeros((
            structure.members.num, structure.yield_specs.components_count), dtype=object
        ))
        nodal_disps_sensitivity = np.matrix(np.zeros((
            1, structure.yield_specs.components_count), dtype=object
        ))
        modal_load_sensitivity = np.matrix(np.zeros((
            1, structure.yield_specs.components_count), dtype=object
        ))
        a2_sensitivity = np.matrix(np.zeros((
            1, structure.yield_specs.components_count), dtype=object
        ))
        b2_sensitivity = np.matrix(np.zeros((
            1, structure.yield_specs.components_count), dtype=object
        ))

        for member_num, member in enumerate(members):
            for force in member.udefs.T:
                fv = np.matrix(np.zeros((structure.dofs_count, 1)))
                global_force = member.t.T * force.T
                local_node_base_dof = 0
                for node in member.nodes:
                    global_node_base_dof = structure.node_dofs_count * node.num
                    for i in range(structure.node_dofs_count):
                        fv[global_node_base_dof + i] = global_force[local_node_base_dof + i]
                    local_node_base_dof += structure.node_dofs_count

                    # affected_struc_disp, p2_modes, a2_modes, b2_mdoes = self.get_dynamic_unit_nodal_disp(fv, modes, time_step)
                    a1 = np.matrix(np.zeros((modes_count, 1)))
                    a1s = np.matrix((1, 1), dtype=object)
                    a1s[0, 0] = a1
                    b1 = np.matrix(np.zeros((modes_count, 1)))
                    b1s = np.matrix((1, 1), dtype=object)
                    b1s[0, 0] = b1
                    initial_modal_load = np.matrix(np.zeros((modes_count, 1)))
                    initial_modal_loads = np.matrix((1, 1), dtype=object)
                    initial_modal_loads[0, 0] = initial_modal_load

                affected_a2s, affected_b2s, affected_modal_load, affected_struc_disp = self.get_dynamic_nodal_disp(
                    time_step=time_step,
                    modes=modes,
                    previous_modal_loads=initial_modal_loads,
                    total_load=fv,
                    a1s=a1s,
                    b1s=b1s,
                )

                nodal_disps_sensitivity[0, pv_column] = affected_struc_disp
                modal_load_sensitivity[0, pv_column] = affected_modal_load
                a2_sensitivity[0, pv_column] = affected_a2s
                b2_sensitivity[0, pv_column] = affected_b2s
                affected_member_disps = self.get_members_disps(affected_struc_disp[0, 0])
                current_affected_member_ycns = 0

                for affected_member_num, affected_member_disp in enumerate(affected_member_disps):
                    fixed_force = -force.T if member_num == affected_member_num else None
                    affected_member_response = structure.members.list[affected_member_num].get_response(affected_member_disp[0, 0], fixed_force)
                    affected_member_nodal_force = affected_member_response.nodal_force
                    affected_member_yield_components_force = affected_member_response.yield_components_force
                    members_forces_sensitivity[affected_member_num, pv_column] = affected_member_nodal_force
                    members_disps_sensitivity[affected_member_num, pv_column] = affected_member_disp[0, 0]
                    pv[current_affected_member_ycns:(current_affected_member_ycns + structure.members.list[affected_member_num].yield_specs.components_count), pv_column] = affected_member_yield_components_force
                    current_affected_member_ycns = current_affected_member_ycns + structure.members.list[affected_member_num].yield_specs.components_count
                pv_column += 1

        sensitivity = DynamicSensitivity(
            pv=pv,
            nodal_disps=nodal_disps_sensitivity,
            members_forces=members_forces_sensitivity,
            members_disps=members_disps_sensitivity,
            modal_loads=modal_load_sensitivity,
            a2=a2_sensitivity,
            b2=b2_sensitivity,
        )
        return sensitivity
