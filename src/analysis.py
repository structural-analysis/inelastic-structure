import numpy as np
import enum
from dataclasses import dataclass
from scipy.linalg import cho_solve

from src.models.loads import Loads
from src.program.prepare import RawData, VarsCount
from src.program.main import MahiniMethod
from src.models.structure import Structure
from src.functions import get_elastoplastic_response


@dataclass
class InternalResponses:
    p0: np.matrix
    members_nodal_forces: np.matrix
    members_nodal_strains: np.matrix
    members_nodal_stresses: np.matrix
    members_internal_moments: np.matrix
    members_top_internal_strains: np.matrix
    members_bottom_internal_strains: np.matrix
    members_top_internal_stresses: np.matrix
    members_bottom_internal_stresses: np.matrix


@dataclass
class StaticSensitivity:
    pv: np.matrix
    nodal_disp: np.matrix
    members_nodal_forces: np.matrix
    members_disps: np.matrix
    members_nodal_strains: np.matrix
    members_nodal_stresses: np.matrix


@dataclass
class DynamicSensitivity:
    pv: np.matrix
    nodal_disp: np.matrix
    members_nodal_forces: np.matrix
    members_disps: np.matrix
    modal_loads: np.matrix
    a2s: np.matrix
    b2s: np.matrix


class DynamicAnalysisMethod(enum.Enum):
    NEWMARK = "newmark"
    DUHAMEL = "duhamel"


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
            self.elastic_members_nodal_strains = internal_responses.members_nodal_strains
            self.elastic_members_nodal_stresses = internal_responses.members_nodal_stresses
            self.elastic_members_internal_moments = internal_responses.members_internal_moments
            self.elastic_members_top_internal_strains = internal_responses.members_top_internal_strains
            self.elastic_members_bottom_internal_strains = internal_responses.members_bottom_internal_strains
            self.elastic_members_top_internal_stresses = internal_responses.members_top_internal_stresses
            self.elastic_members_bottom_internal_stresses = internal_responses.members_bottom_internal_stresses

            if self.structure.is_inelastic:
                self.p0 = internal_responses.p0
                self.d0 = self.get_nodal_disp_limits(self.elastic_nodal_disp[0, 0])
                sensitivity = self.get_sensitivity()
                self.pv = sensitivity.pv
                self.nodal_disp_sensitivity = sensitivity.nodal_disp
                self.members_disps_sensitivity = sensitivity.members_disps
                self.members_nodal_forces_sensitivity = sensitivity.members_nodal_forces
                self.members_nodal_strains_sensitivity = sensitivity.members_nodal_strains
                self.members_nodal_stresses_sensitivity = sensitivity.members_nodal_stresses
                self.dv = self.get_nodal_disp_limits_sensitivity_rows()
                raw_data = RawData(self)
                mahini_method = MahiniMethod(raw_data)
                self.plastic_vars = mahini_method.solve()

        elif self.type == "dynamic":
            dynamic_analysis_method = DynamicAnalysisMethod.DUHAMEL
            self.damping = self.general_info["dynamic_analysis"]["damping"]
            structure = self.structure
            loads = self.loads
            time_steps = loads.dynamic[0].magnitude.shape[0]
            self.time_steps = time_steps
            self.time = loads.dynamic[0].time
            dt = self.time[1][0, 0] - self.time[0][0, 0]
            if dynamic_analysis_method == DynamicAnalysisMethod.DUHAMEL:
                modes = np.matrix(structure.modes)
                modes_count = modes.shape[1]
                self.m_modal = structure.get_modal_property(structure.condensed_m, modes)
                self.k_modal = structure.get_modal_property(structure.condensed_k, modes)

                self.modal_loads = np.matrix(np.zeros((time_steps, 1), dtype=object))
                modal_load = np.matrix(np.zeros((modes_count, 1)))
                modal_loads = np.matrix(np.zeros((1, 1)), dtype=object)
                modal_loads[0, 0] = modal_load
                self.modal_loads[0, 0] = modal_loads
                a_duhamel = np.matrix(np.zeros((modes_count, 1)))
                a_duhamels = np.matrix(np.zeros((1, 1)), dtype=object)
                self.a_duhamel = np.matrix(np.zeros((time_steps, 1), dtype=object))
                a_duhamels[0, 0] = a_duhamel
                self.a_duhamel[0, 0] = a_duhamels

                b_duhamel = np.matrix(np.zeros((modes_count, 1)))
                b_duhamels = np.matrix(np.zeros((1, 1)), dtype=object)
                self.b_duhamel = np.matrix(np.zeros((time_steps, 1), dtype=object))
                b_duhamels[0, 0] = b_duhamel
                self.b_duhamel[0, 0] = b_duhamels
                self.modal_disp_history = np.matrix(np.zeros((time_steps, 1), dtype=object))

            elif dynamic_analysis_method == DynamicAnalysisMethod.NEWMARK:
                
                dense_disp = np.matrix(np.zeros((structure.condensed_m.shape[0], 1)))

                self.dense_disp_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
                self.dense_disp_history[0, 0] = dense_disp

                dense_velcoc = np.matrix(np.zeros((structure.condensed_m.shape[0], 1)))
                self.dense_veloc_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
                self.dense_veloc_history[0, 0] = dense_velcoc

                self.dense_accel_history = np.matrix(np.zeros((time_steps, 1), dtype=object))

                condense_load = np.matrix(np.zeros((structure.condensed_m.shape[0], 1)))

                gamma = 0.5
                beta = 1 / 6
                self.dense_accel_history[0, 0] = self.get_initial_accel(
                    condense_load,
                    self.dense_disp_history[0, 0],
                    self.dense_veloc_history[0, 0],
                )
                a1, a2, a3, inv_k_hat = self.get_newmark_props(gamma, beta, dt)

            self.total_load = np.zeros((structure.dofs_count, 1))
            self.elastic_nodal_disp_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.elastic_members_disps_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.elastic_members_nodal_forces_history = np.matrix(np.zeros((time_steps, 1), dtype=object))

            if self.structure.is_inelastic:
                self.nodal_disp_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
                self.members_nodal_forces_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
                self.members_disps_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
                self.modal_loads_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
                self.plastic_vars_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
                self.a2_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
                self.b2_sensitivity_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
                self.p0_history = np.zeros((time_steps, 1), dtype=object)
                self.p0_history[0, 0] = np.matrix(np.zeros((structure.yield_specs.components_count, 1)))
                self.d0_history = np.zeros((time_steps, 1), dtype=object)
                self.pv_history = np.zeros((time_steps, 1), dtype=object)
                initial_pv = np.matrix(np.zeros((
                    structure.yield_specs.components_count, structure.yield_specs.components_count
                )))
                self.pv_history[0, 0] = initial_pv
                vars_count = VarsCount(self)
                self.load_level = 0
                # self.constraints_force = np.matrix(np.zeros((plastic_vars_count, 1)))
                # self.constraints_force_prev = np.matrix(np.zeros((plastic_vars_count, 1)))
                self.plastic_multipliers_prev = np.matrix(np.zeros((vars_count.plastic_vars_count, 1)))
                self.plastic_multipliers_history = np.matrix(np.zeros((time_steps, 1), dtype=object))

            for time_step in range(1, time_steps):
                print(f"{time_step=}")
                print(f"{self.time[time_step][0, 0]=}")
                self.total_load = loads.get_total_load(structure, loads, time_step)
                if dynamic_analysis_method == DynamicAnalysisMethod.NEWMARK:
                    elastic_nodal_disp, dense_disp, dense_veloc, dense_accel = self.get_dynamic_nodal_disp_newmark(
                        gamma=gamma,
                        beta=beta,
                        dt=dt,
                        a1=a1,
                        a2=a2,
                        a3=a3,
                        inv_k_hat=inv_k_hat,
                        total_load=self.total_load,
                        u_prev=self.dense_disp_history[time_step - 1, 0],
                        v_prev=self.dense_veloc_history[time_step - 1, 0],
                        a_prev=self.dense_accel_history[time_step - 1, 0],
                    )
                    self.dense_disp_history[time_step, 0] = dense_disp
                    self.dense_veloc_history[time_step, 0] = dense_veloc
                    self.dense_accel_history[time_step, 0] = dense_accel
                elif dynamic_analysis_method == DynamicAnalysisMethod.DUHAMEL:
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
                    raw_data = RawData(self)
                    mahini_method = MahiniMethod(raw_data)
                    self.plastic_vars = mahini_method.solve_dynamic()
                    self.plastic_vars_history[time_step, 0] = self.plastic_vars
                    self.delta_plastic_multipliers = self.plastic_vars["pms_history"][-1]
                    self.load_level = self.plastic_vars["load_level_history"][-1]
                    self.plastic_multipliers = self.delta_plastic_multipliers + self.plastic_multipliers_prev
                    self.plastic_multipliers_history[time_step, 0] = self.plastic_multipliers

                    self.plastic_multipliers_prev = self.plastic_multipliers
                    # self.constraints_force_prev = self.constraints_force.copy()
                    # self.constraints_force = self.get_prev_constraints_force()

                    phi_x = structure.phi * self.plastic_multipliers
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
        members_nodal_strains = np.matrix(np.zeros((structure.members.num, 1), dtype=object))
        members_nodal_stresses = np.matrix(np.zeros((structure.members.num, 1), dtype=object))
        members_internal_moments = np.matrix(np.zeros((structure.members.num, 1), dtype=object))
        members_top_internal_strains = np.matrix(np.zeros((structure.members.num, 1), dtype=object))
        members_bottom_internal_strains = np.matrix(np.zeros((structure.members.num, 1), dtype=object))
        members_top_internal_stresses = np.matrix(np.zeros((structure.members.num, 1), dtype=object))
        members_bottom_internal_stresses = np.matrix(np.zeros((structure.members.num, 1), dtype=object))
        p0 = np.matrix(np.zeros((structure.yield_specs.components_count, 1)))
        base_p0_row = 0

        for i, member in enumerate(structure.members.list):
            member_response = member.get_response(members_disps[i, 0])
            yield_components_force = member_response.yield_components_force
            p0[base_p0_row:(base_p0_row + member.yield_specs.components_count)] = yield_components_force
            base_p0_row = base_p0_row + member.yield_specs.components_count

            # TODO: we can clean and simplify appending member response to members_{responses}
            # each member should contain only its responses, not zero response of other elements.
            # but in members_{responses}, instead of appending with i (member num in structure.members.list),
            # we should attach with member.num, and if one member has not some response it will not appended.
            # we can use a dataclass like: 
            # MemberResponse:
            # member: object
            # response: Response
            members_nodal_forces[i, 0] = member_response.nodal_force
            members_nodal_strains[i, 0] = member_response.nodal_strains
            members_nodal_stresses[i, 0] = member_response.nodal_stresses
            members_internal_moments[i, 0] = member_response.internal_moments
            members_top_internal_strains[i, 0] = member_response.top_internal_strains
            members_bottom_internal_strains[i, 0] = member_response.bottom_internal_strains
            members_top_internal_stresses[i, 0] = member_response.top_internal_stresses
            members_bottom_internal_stresses[i, 0] = member_response.bottom_internal_stresses

        return InternalResponses(
            p0=p0,
            members_nodal_forces=members_nodal_forces,
            members_nodal_strains=members_nodal_strains,
            members_nodal_stresses=members_nodal_stresses,
            members_internal_moments=members_internal_moments,
            members_top_internal_strains=members_top_internal_strains,
            members_bottom_internal_strains=members_bottom_internal_strains,
            members_top_internal_stresses=members_top_internal_stresses,
            members_bottom_internal_stresses=members_bottom_internal_stresses,
        )

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
        members_nodal_forces_sensitivity = np.matrix(np.zeros((structure.members.num, structure.yield_specs.components_count), dtype=object))
        nodal_disp_sensitivity = np.matrix(np.zeros((1, structure.yield_specs.components_count), dtype=object))
        members_disps_sensitivity = np.matrix(np.zeros((structure.members.num, structure.yield_specs.components_count), dtype=object))
        members_nodal_strains_sensitivity = np.matrix(np.zeros((structure.members.num, structure.yield_specs.components_count), dtype=object))
        members_nodal_stresses_sensitivity = np.matrix(np.zeros((structure.members.num, structure.yield_specs.components_count), dtype=object))
        pv_column = 0

        for member_num, member in enumerate(members):
            # FIXME: GENERALIZE PLEASE
            for comp_num, force in enumerate(member.udefs.T):
                fv = np.matrix(np.zeros((structure.dofs_count, 1)))
                global_force = member.t.T * force.T
                local_node_base_dof = 0
                for node in member.nodes:
                    global_node_base_dof = structure.node_dofs_count * node.num
                    for i in range(structure.node_dofs_count):
                        fv[global_node_base_dof + i] = global_force[local_node_base_dof + i]
                    local_node_base_dof += structure.node_dofs_count

                affected_structure_disp = self.get_nodal_disp(fv)
                nodal_disp_sensitivity[0, pv_column] = affected_structure_disp[0, 0]
                # TODO: it is good if any member has a disp vector (and disps matrix from sensitivity) property which is filled after analysis.
                # then we can iterate only on members instead of members count.
                # each member also has a num property and no need to get their position in the list.
                # there is another shortcut to do a clean way. create a AffectedMember class with num, disp, disps properties
                # which it's objects are created after analysis.
                affected_member_disps = self.get_members_disps(affected_structure_disp[0, 0])
                current_affected_member_ycns = 0
                for affected_member_num, affected_member_disp in enumerate(affected_member_disps):
                    fixed_force_shape = (structure.members.list[affected_member_num].dofs_count, 1)
                    fixed_force = -force.T if member_num == affected_member_num else np.matrix(np.zeros((fixed_force_shape)))
                    if structure.members.list[affected_member_num].__class__.__name__ in ["WallMember", "PlateMember"]:
                        # NOTE: yield_specs.components_count has different meanings in different members.
                        fixed_stress_shape = (structure.members.list[affected_member_num].yield_specs.components_count, 1)
                        if member_num == affected_member_num:
                            fixed_stress = -structure.members.list[affected_member_num].usefs.T[comp_num].T
                        else:
                            fixed_stress = np.matrix(np.zeros((fixed_stress_shape)))
                    else:
                        fixed_stress = None
                    affected_member_response = structure.members.list[affected_member_num].get_response(affected_member_disp[0, 0], fixed_force, fixed_stress)
                    affected_member_nodal_force = affected_member_response.nodal_force
                    affected_member_yield_components_force = affected_member_response.yield_components_force
                    if member.__class__.__name__ in ["WallMember", "PlateMember"]:
                        # FIXME: GENERALIZE PLEASE
                        if member_num == affected_member_num:
                            usef = structure.members.list[affected_member_num].usefs.T[comp_num]
                            affected_member_yield_components_force -= usef.T
                        affected_member_nodal_strains = affected_member_response.nodal_strains
                        affected_member_nodal_stresses = affected_member_response.nodal_stresses
                        members_nodal_strains_sensitivity[affected_member_num, pv_column] = affected_member_nodal_strains
                        members_nodal_stresses_sensitivity[affected_member_num, pv_column] = affected_member_nodal_stresses
                    members_nodal_forces_sensitivity[affected_member_num, pv_column] = affected_member_nodal_force
                    members_disps_sensitivity[affected_member_num, pv_column] = affected_member_disp[0, 0]
                    pv[current_affected_member_ycns:(current_affected_member_ycns + structure.members.list[affected_member_num].yield_specs.components_count), pv_column] = affected_member_yield_components_force
                    current_affected_member_ycns = current_affected_member_ycns + structure.members.list[affected_member_num].yield_specs.components_count
                pv_column += 1

        sensitivity = StaticSensitivity(
            pv=pv,
            nodal_disp=nodal_disp_sensitivity,
            members_nodal_forces=members_nodal_forces_sensitivity,
            members_disps=members_disps_sensitivity,
            members_nodal_strains=members_nodal_strains_sensitivity,
            members_nodal_stresses=members_nodal_stresses_sensitivity,
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
                dv[i, j] = self.nodal_disp_sensitivity[0, j][dof, 0]
        return dv

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

    def get_initial_accel(self, p0, u0, v0):
        a0 = np.dot(np.linalg.inv(self.structure.condensed_m), (p0 - np.dot(self.structure.c, v0) - np.dot(self.structure.condensed_k, u0)))
        return a0

    def get_newmark_props(self, gamma, beta, dt):
        # gamma = 0.5
        # beta = 1 / 6
        structure = self.structure
        a1 = structure.condensed_m / (beta * dt **2) + gamma * structure.c / (beta * dt)
        a2 = structure.condensed_m / (beta * dt) + structure.c * (gamma / beta - 1)
        a3 = structure.condensed_m * (1 / (2 * beta) - 1) + structure.c * dt * (gamma / (2 * beta) - 1)
        k_hat = structure.condensed_k + a1
        inv_k_hat = np.linalg.inv(k_hat)
        return a1, a2, a3, inv_k_hat

    def get_dynamic_nodal_disp_newmark(self, gamma, beta, dt, a1, a2, a3, inv_k_hat, total_load, u_prev, v_prev, a_prev):
        nodal_disp = np.matrix(np.zeros((1, 1), dtype=object))
        condense_load, reduced_p0 = self.loads.apply_static_condensation(self.structure, total_load)
        phat = condense_load + np.dot(a1, u_prev) + np.dot(a2, v_prev) + np.dot(a3, a_prev)
        dense_disp = np.dot(inv_k_hat, phat)
        dense_veloc = gamma / (beta * dt) * (dense_disp - u_prev) + (1 - gamma / beta) * v_prev + dt * (1 - gamma / (2 * beta)) * a_prev
        dense_accel = 1 / (beta * dt ** 2) * (dense_disp - u_prev) - 1 / (beta * dt) * v_prev - (1 / (2 * beta) - 1) * a_prev
        u0 = np.dot(self.structure.reduced_k00_inv, reduced_p0) - np.dot(self.structure.reduced_k00_inv, np.dot(self.structure.reduced_k0t, dense_disp))
        unrestrianed_ut = self.structure.undo_disp_boundary_condition(dense_disp, self.structure.mass_bounds)
        unrestrianed_u0 = self.structure.undo_disp_boundary_condition(u0, self.structure.zero_mass_bounds)
        disp = self.structure.undo_disp_condensation(unrestrianed_ut, unrestrianed_u0)
        nodal_disp[0, 0] = disp
        return nodal_disp, dense_disp, dense_veloc, dense_accel

    def get_dynamic_sensitivity(self, modes, time_step):
        structure = self.structure
        modes_count = modes.shape[1]
        # fv: equivalent global force vector for a yield component's udef
        members = structure.members.list
        pv = np.matrix(np.zeros((
            structure.yield_specs.components_count, structure.yield_specs.components_count
        )))
        pv_column = 0

        members_nodal_forces_sensitivity = np.matrix(np.zeros((
            structure.members.num, structure.yield_specs.components_count), dtype=object
        ))
        members_disps_sensitivity = np.matrix(np.zeros((
            structure.members.num, structure.yield_specs.components_count), dtype=object
        ))
        nodal_disp_sensitivity = np.matrix(np.zeros((
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
                    fixed_force = -load.T if member_num == affected_member_num else None
                    affected_member_response = structure.members.list[affected_member_num].get_response(affected_member_disp[0, 0], fixed_force)
                    affected_member_nodal_force = affected_member_response.nodal_force
                    affected_member_yield_components_force = affected_member_response.yield_components_force
                    members_nodal_forces_sensitivity[affected_member_num, pv_column] = affected_member_nodal_force
                    members_disps_sensitivity[affected_member_num, pv_column] = affected_member_disp[0, 0]
                    pv[current_affected_member_ycns:(current_affected_member_ycns + structure.members.list[affected_member_num].yield_specs.components_count), pv_column] = affected_member_yield_components_force
                    current_affected_member_ycns = current_affected_member_ycns + structure.members.list[affected_member_num].yield_specs.components_count
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

    # def get_prev_constraints_force(self):
    #     phi = self.structure.phi
    #     constraints_force = (
    #         phi.T * self.pv * phi * self.delta_plastic_multipliers +
    #         phi.T * self.p0 * self.load_level -
    #         phi.T * self.p0_prev * self.load_level_prev +
    #         self.constraints_force_prev
    #     )
    #     return constraints_force
