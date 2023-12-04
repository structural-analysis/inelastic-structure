import numpy as np
from dataclasses import dataclass

from ..program.main import MahiniMethod
from ..functions import get_elastoplastic_response
from .initial_analysis import InitialAnalysis, AnalysisType


@dataclass
class DynamicSensitivity:
    pv: np.matrix
    nodal_disp: np.matrix
    members_nodal_forces: np.matrix
    members_disps: np.matrix
    modal_loads: np.matrix
    a2s: np.matrix
    b2s: np.matrix


class InelasticAnalysis:
    def __init__(self, initial_analysis: InitialAnalysis):
        self.initial_data = initial_analysis.initial_data
        self.analysis_data = initial_analysis.analysis_data
        self.analysis_type = initial_analysis.analysis_type

        if self.analysis_type is AnalysisType.STATIC:
            mahini_method = MahiniMethod(initial_data=self.initial_data, analysis_data=self.analysis_data)
            self.plastic_vars = mahini_method.solve()

        elif self.analysis_type is AnalysisType.DYNAMIC:
            time_steps = initial_analysis.time_steps
            self.nodal_disp_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.members_nodal_forces_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.members_disps_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.modal_loads_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.plastic_vars_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.a2_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.b2_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.p0_history = np.zeros((time_steps, 1), dtype=object)
            self.p0_history[0, 0] = np.matrix(np.zeros((structure.yield_specs.intact_components_count, 1)))
            self.d0_history = np.zeros((time_steps, 1), dtype=object)
            self.pv_history = np.zeros((time_steps, 1), dtype=object)
            initial_pv = np.matrix(np.zeros((
                structure.yield_specs.intact_components_count, structure.yield_specs.intact_components_count
            )))
            self.pv_history[0, 0] = initial_pv
            self.load_level = 0

            self.plastic_multipliers_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.plastic_multipliers_prev = np.matrix(np.zeros((self.initial_data.plastic_vars_count, 1)))

            for time_step in range(1, time_steps):
                print(f"{time_step=}")
                print(f"{self.time[time_step][0, 0]=}")
                self.total_load = loads.get_total_load(structure, loads, time_step)

                elastic_a2s, elastic_b2s, elastic_modal_loads, elastic_nodal_disp = self.get_dynamic_nodal_disp(
                    time_step=time_step,
                    modes=modes,
                    previous_modal_loads=self.modal_loads[time_step - 1, 0],
                    total_load=self.total_load,
                    a1s=self.a_duhamel[time_step - 1, 0],
                    b1s=self.b_duhamel[time_step - 1, 0],
                )

                self.a_duhamel[time_step, 0] = elastic_a2s
                self.b_duhamel[time_step, 0] = elastic_b2s
                self.modal_loads[time_step, 0] = elastic_modal_loads
                self.elastic_nodal_disp_history[time_step, 0] = elastic_nodal_disp
                elastic_members_disps = self.get_members_disps(elastic_nodal_disp[0, 0])
                self.elastic_members_disps_history[time_step, 0] = elastic_members_disps
                internal_responses = self.get_internal_responses(elastic_members_disps)
                self.elastic_members_nodal_forces_history[time_step, 0] = internal_responses.members_nodal_forces

                if self.structure.is_inelastic:
                    self.p0 = internal_responses.p0
                    self.p0_prev = self.p0_history[time_step - 1, 0]
                    self.p0_history[time_step, 0] = self.p0

                    d0 = self.get_nodal_disp_limits(elastic_nodal_disp[0, 0])
                    self.d0 = d0
                    self.d0_history[time_step, 0] = d0
                    sensitivity = self.get_dynamic_sensitivity(modes, time_step)
                    self.pv = sensitivity.pv
                    self.pv_prev = self.pv_history[time_step - 1, 0]
                    self.pv_history[time_step, 0] = sensitivity.pv
                    self.nodal_disp_sensitivity_history[time_step, 0] = sensitivity.nodal_disp
                    self.members_nodal_forces_sensitivity_history[time_step, 0] = sensitivity.members_nodal_forces
                    self.members_disps_sensitivity_history[time_step, 0] = sensitivity.members_disps
                    self.modal_loads_sensitivity_history[time_step, 0] = sensitivity.modal_loads
                    self.a2_sensitivity_history[time_step, 0] = sensitivity.a2s
                    self.b2_sensitivity_history[time_step, 0] = sensitivity.b2s

                    self.dv = self.get_nodal_disp_limits_sensitivity_rows()
                    self.load_level_prev = self.load_level
                    mahini_method = MahiniMethod(self.initial_data)
                    self.plastic_vars = mahini_method.solve_dynamic()
                    self.plastic_vars_history[time_step, 0] = self.plastic_vars
                    self.delta_plastic_multipliers = self.plastic_vars["pms_history"][-1]
                    self.load_level = self.plastic_vars["load_level_history"][-1]
                    self.plastic_multipliers = self.delta_plastic_multipliers + self.plastic_multipliers_prev
                    self.plastic_multipliers_history[time_step, 0] = self.plastic_multipliers
                    self.plastic_multipliers_prev = self.plastic_multipliers
                    phi_x = structure.yield_specs.intact_phi * self.plastic_multipliers

                    elastoplastic_a2s = get_elastoplastic_response(
                        load_level=self.load_level,
                        phi_x=phi_x,
                        elastic_response=elastic_a2s,
                        sensitivity=sensitivity.a2s,
                    )

                    elastoplastic_b2s = get_elastoplastic_response(
                        load_level=self.load_level,
                        phi_x=phi_x,
                        elastic_response=elastic_b2s,
                        sensitivity=sensitivity.b2s,
                    )

                    elastoplastic_modal_loads = get_elastoplastic_response(
                        load_level=self.load_level,
                        phi_x=phi_x,
                        elastic_response=elastic_modal_loads,
                        sensitivity=sensitivity.modal_loads,
                    )

                    elastoplastic_nodal_disp = get_elastoplastic_response(
                        load_level=self.load_level,
                        phi_x=phi_x,
                        elastic_response=elastic_nodal_disp,
                        sensitivity=sensitivity.nodal_disp,
                    )

                    elastoplastic_members_disps = get_elastoplastic_response(
                        load_level=self.load_level,
                        phi_x=phi_x,
                        elastic_response=elastic_members_disps,
                        sensitivity=sensitivity.members_disps,
                    )

                    elastoplastic_members_nodal_forces = get_elastoplastic_response(
                        load_level=self.load_level,
                        phi_x=phi_x,
                        elastic_response=internal_responses.members_nodal_forces,
                        sensitivity=sensitivity.members_nodal_forces,
                    )

                    self.a_duhamel[time_step, 0] = elastoplastic_a2s
                    self.b_duhamel[time_step, 0] = elastoplastic_b2s
                    self.modal_loads[time_step, 0] = elastoplastic_modal_loads
                    # print(f"{elastoplastic_members_nodal_forces[0, 0]=}")
                    # print(f"{elastoplastic_members_nodal_forces[1, 0]=}")
                    print("///////////////////////////////////////////////////////")

    def get_i_duhamel(self, t1, t2, wn, wd):
        damping = self.damping
        wd = np.sqrt(1 - damping ** 2) * wn
        i12 = (np.exp(damping * wn * t2) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.cos(wd * t2) + wd * np.sin(wd * t2))
        i11 = (np.exp(damping * wn * t1) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.cos(wd * t1) + wd * np.sin(wd * t1))
        i1 = i12 - i11
        i22 = (np.exp(damping * wn * t2) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.sin(wd * t2) - wd * np.cos(wd * t2))
        i21 = (np.exp(damping * wn * t1) / ((damping * wn) ** 2 + wd ** 2)) * (damping * wn * np.sin(wd * t1) - wd * np.cos(wd * t1))
        i2 = i22 - i21
        i3 = (t2 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i22 + (wd / ((damping * wn) ** 2 + wd ** 2)) * i12 - ((t1 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i21 + (wd / ((damping * wn) ** 2 + wd ** 2)) * i11)
        i4 = (t2 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i12 - (wd / ((damping * wn) ** 2 + wd ** 2)) * i22 - ((t1 - (damping * wn / ((damping * wn) ** 2 + wd ** 2))) * i11 - (wd / ((damping * wn) ** 2 + wd ** 2)) * i21)
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
        u0 = np.dot(structure.reduced_k00_inv, reduced_p0) + np.dot(structure.ku0, ut)
        unrestrianed_ut = structure.undo_disp_boundary_condition(ut, structure.mass_bounds)
        unrestrianed_u0 = structure.undo_disp_boundary_condition(u0, structure.zero_mass_bounds)

        disp = structure.undo_disp_condensation(unrestrianed_ut, unrestrianed_u0)
        nodal_disp[0, 0] = disp
        return a2s, b2s, modal_loads, nodal_disp

    def get_dynamic_sensitivity(self, modes, time_step):
        structure = self.structure
        modes_count = modes.shape[1]
        # fv: equivalent global force vector for a yield component's udef
        members = structure.members
        pv = np.matrix(np.zeros((
            structure.yield_specs.intact_components_count, structure.yield_specs.intact_components_count
        )))
        pv_column = 0

        members_nodal_forces_sensitivity = np.matrix(np.zeros((
            structure.members_count, structure.yield_specs.intact_components_count), dtype=object
        ))
        members_disps_sensitivity = np.matrix(np.zeros((
            structure.members_count, structure.yield_specs.intact_components_count), dtype=object
        ))
        nodal_disp_sensitivity = np.matrix(np.zeros((
            1, structure.yield_specs.intact_components_count), dtype=object
        ))
        modal_load_sensitivity = np.matrix(np.zeros((
            1, structure.yield_specs.intact_components_count), dtype=object
        ))
        a2_sensitivity = np.matrix(np.zeros((
            1, structure.yield_specs.intact_components_count), dtype=object
        ))
        b2_sensitivity = np.matrix(np.zeros((
            1, structure.yield_specs.intact_components_count), dtype=object
        ))

        for member_num, member in enumerate(members):
            for load in member.udefs.T:
                fv = np.matrix(np.zeros((structure.dofs_count, 1)))
                global_load = member.t.T * load.T
                local_node_base_dof = 0
                for node in member.nodes:
                    global_node_base_dof = structure.node_dofs_count * node.num
                    for i in range(structure.node_dofs_count):
                        fv[global_node_base_dof + i] = global_load[local_node_base_dof + i]
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
                nodal_disp_sensitivity[0, pv_column] = affected_struc_disp[0, 0]
                modal_load_sensitivity[0, pv_column] = affected_modal_load[0, 0]
                a2_sensitivity[0, pv_column] = affected_a2s[0, 0]
                b2_sensitivity[0, pv_column] = affected_b2s[0, 0]
                affected_member_disps = self.get_members_disps(affected_struc_disp[0, 0])
                current_affected_member_ycns = 0

                for affected_member_num, affected_member_disp in enumerate(affected_member_disps):
                    fixed_external = -load.T if member_num == affected_member_num else None
                    affected_member_response = structure.members[affected_member_num].get_response(affected_member_disp[0, 0], fixed_external)
                    affected_member_nodal_force = affected_member_response.nodal_force
                    affected_member_yield_components_force = affected_member_response.yield_components_force
                    members_nodal_forces_sensitivity[affected_member_num, pv_column] = affected_member_nodal_force
                    members_disps_sensitivity[affected_member_num, pv_column] = affected_member_disp[0, 0]
                    pv[current_affected_member_ycns:(current_affected_member_ycns + structure.members[affected_member_num].yield_specs.components_count), pv_column] = affected_member_yield_components_force
                    current_affected_member_ycns = current_affected_member_ycns + structure.members[affected_member_num].yield_specs.components_count
                pv_column += 1

        sensitivity = DynamicSensitivity(
            pv=pv,
            nodal_disp=nodal_disp_sensitivity,
            members_nodal_forces=members_nodal_forces_sensitivity,
            members_disps=members_disps_sensitivity,
            modal_loads=modal_load_sensitivity,
            a2s=a2_sensitivity,
            b2s=b2_sensitivity,
        )
        return sensitivity
