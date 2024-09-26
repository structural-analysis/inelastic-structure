import numpy as np
from scipy.linalg import cho_solve
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class InternalResponses:
    p0: np.matrix
    members_nodal_forces: np.matrix
    members_nodal_strains: np.matrix
    members_nodal_stresses: np.matrix
    members_nodal_moments: np.matrix


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


@dataclass
class IDuhamels:
    i1: np.matrix
    i2: np.matrix
    i3: np.matrix
    i4: np.matrix


def get_nodal_disp(structure, loads, total_load):
    reduced_total_load = loads.apply_boundary_conditions(structure.boundaries_dof_mask, total_load)
    reduced_disp = cho_solve(structure.kc, reduced_total_load)
    nodal_disp = np.matrix(np.zeros((1, 1), dtype=object))
    disp = structure.undo_disp_boundaries(reduced_disp)
    nodal_disp[0, 0] = disp
    return nodal_disp


def get_members_disps(structure, disp):
    members_disps = np.matrix(np.zeros((structure.members_count, 1), dtype=object))
    for i_member, member in enumerate(structure.members):
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


def get_internal_responses(structure, members_disps):
    # calculate p0
    members_nodal_forces = np.matrix(np.zeros((structure.members_count, 1), dtype=object))
    members_nodal_strains = np.matrix(np.zeros((structure.members_count, 1), dtype=object))
    members_nodal_stresses = np.matrix(np.zeros((structure.members_count, 1), dtype=object))
    members_nodal_moments = np.matrix(np.zeros((structure.members_count, 1), dtype=object))
    p0 = np.matrix(np.zeros((structure.yield_specs.intact_components_count, 1)))
    base_p0_row = 0

    for i, member in enumerate(structure.members):
        member_response = member.get_response(members_disps[i, 0])
        yield_components_force = member_response.yield_components_force
        p0[base_p0_row:(base_p0_row + member.yield_specs.components_count)] = yield_components_force
        base_p0_row = base_p0_row + member.yield_specs.components_count

        # TODO: we can clean and simplify appending member response to members_{responses}
        # each member should contain only its responses, not zero response of other elements.
        # but in members_{responses}, instead of appending with i (member num in structure.members),
        # we should attach with member.num, and if one member has not some response it will not appended.
        # we can use a dataclass like:
        # MemberResponse:
        # member: object
        # response: Response
        members_nodal_forces[i, 0] = member_response.nodal_force
        members_nodal_strains[i, 0] = member_response.nodal_strains
        members_nodal_stresses[i, 0] = member_response.nodal_stresses
        members_nodal_moments[i, 0] = member_response.nodal_moments

    return InternalResponses(
        p0=p0,
        members_nodal_forces=members_nodal_forces,
        members_nodal_strains=members_nodal_strains,
        members_nodal_stresses=members_nodal_stresses,
        members_nodal_moments=members_nodal_moments,
    )


def get_nodal_disp_limits(structure, elastic_nodal_disp):
    disp_limits = structure.limits["disp_limits"]
    disp_limits_count = disp_limits.shape[0]
    d0 = np.matrix(np.zeros((disp_limits_count, 1)))
    for i, disp_limit in enumerate(disp_limits):
        node = disp_limit[0]
        node_dof = disp_limit[1]
        dof = structure.get_global_dof(node, node_dof)
        d0[i, 0] = elastic_nodal_disp[dof, 0]
    return d0


def get_sensitivity(structure, loads):
    # fv: equivalent global force vector for a yield component's udef
    members = structure.members
    pv = np.matrix(np.zeros((structure.yield_specs.intact_components_count, structure.yield_specs.intact_components_count)))
    members_nodal_forces_sensitivity = np.matrix(np.zeros((structure.members_count, structure.yield_specs.intact_components_count), dtype=object))
    nodal_disp_sensitivity = np.matrix(np.zeros((1, structure.yield_specs.intact_components_count), dtype=object))
    members_disps_sensitivity = np.matrix(np.zeros((structure.members_count, structure.yield_specs.intact_components_count), dtype=object))
    members_nodal_strains_sensitivity = np.matrix(np.zeros((structure.members_count, structure.yield_specs.intact_components_count), dtype=object))
    members_nodal_stresses_sensitivity = np.matrix(np.zeros((structure.members_count, structure.yield_specs.intact_components_count), dtype=object))
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

            affected_structure_disp = get_nodal_disp(structure=structure, loads=loads, total_load=fv)
            nodal_disp_sensitivity[0, pv_column] = affected_structure_disp[0, 0]
            # TODO: it is good if any member has a disp vector (and disps matrix from sensitivity) property which is filled after analysis.
            # then we can iterate only on members instead of members count.
            # each member also has a num property and no need to get their position in the list.
            # there is another shortcut to do a clean way. create a AffectedMember class with num, disp, disps properties
            # which it's objects are created after analysis.
            affected_member_disps = get_members_disps(structure, affected_structure_disp[0, 0])
            current_affected_member_ycns = 0
            for affected_member_num, affected_member_disp in enumerate(affected_member_disps):
                fixed_external_shape = (structure.members[affected_member_num].dofs_count, 1)
                fixed_external = -force.T if member_num == affected_member_num else np.matrix(np.zeros((fixed_external_shape)))
                if structure.members[affected_member_num].__class__.__name__ in ["WallMember", "PlateMember"]:
                    # NOTE: yield_specs.components_count has different meanings in different members.
                    fixed_internal_shape = (structure.members[affected_member_num].yield_specs.components_count, 1)
                    if member_num == affected_member_num:
                        fixed_internal = -structure.members[affected_member_num].udets.T[comp_num].T
                    else:
                        fixed_internal = np.matrix(np.zeros((fixed_internal_shape)))
                else:
                    fixed_internal = None
                affected_member_response = structure.members[affected_member_num].get_response(affected_member_disp[0, 0], fixed_external, fixed_internal)
                affected_member_nodal_force = affected_member_response.nodal_force
                affected_member_yield_components_force = affected_member_response.yield_components_force
                if member.__class__.__name__ in ["WallMember", "PlateMember"]:
                    # FIXME: GENERALIZE PLEASE
                    if member_num == affected_member_num:
                        udet = structure.members[affected_member_num].udets.T[comp_num]
                        affected_member_yield_components_force -= udet.T
                    affected_member_nodal_strains = affected_member_response.nodal_strains
                    affected_member_nodal_stresses = affected_member_response.nodal_stresses
                    members_nodal_strains_sensitivity[affected_member_num, pv_column] = affected_member_nodal_strains
                    members_nodal_stresses_sensitivity[affected_member_num, pv_column] = affected_member_nodal_stresses
                members_nodal_forces_sensitivity[affected_member_num, pv_column] = affected_member_nodal_force
                members_disps_sensitivity[affected_member_num, pv_column] = affected_member_disp[0, 0]
                pv[current_affected_member_ycns:(current_affected_member_ycns + structure.members[affected_member_num].yield_specs.components_count), pv_column] = affected_member_yield_components_force
                current_affected_member_ycns = current_affected_member_ycns + structure.members[affected_member_num].yield_specs.components_count
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


def get_nodal_disp_limits_sensitivity_rows(structure, nodal_disp_sensitivity):
    disp_limits = structure.limits["disp_limits"]
    disp_limits_count = disp_limits.shape[0]
    dv = np.matrix(np.zeros((disp_limits_count, structure.yield_specs.intact_components_count)))
    for i, disp_limit in enumerate(disp_limits):
        node = disp_limit[0]
        node_dof = disp_limit[1]
        dof = structure.get_global_dof(node, node_dof)
        for j in range(structure.yield_specs.intact_components_count):
            dv[i, j] = nodal_disp_sensitivity[0, j][dof, 0]
    return dv


def get_dynamic_nodal_disp(structure, loads, time, time_step, modes, previous_modal_loads, total_load, a1s, b1s, i_duhamels):

    nodal_disp = np.matrix(np.zeros((1, 1), dtype=object))
    modal_loads = np.matrix(np.zeros((1, 1), dtype=object))
    condense_load, reduced_p0 = loads.apply_static_condensation(structure, total_load)
    modal_loads[0, 0] = loads.get_modal_load(condense_load, modes)
    modal_disps, a2s, b2s = get_modal_disp(
        structure=structure,
        time=time,
        time_step=time_step,
        modes=modes,
        previous_modal_loads=previous_modal_loads,
        modal_loads=modal_loads,
        a1s=a1s,
        b1s=b1s,
        i_duhamels=i_duhamels,
    )

    ut = np.dot(modes, modal_disps[0, 0])
    u0 = np.dot(structure.reduced_k00_inv, reduced_p0) + np.dot(structure.ku0, ut)

    disp = structure.undo_disp_condensation(ut, u0)

    nodal_disp[0, 0] = disp
    return a2s, b2s, modal_loads, nodal_disp


def get_modal_disp(structure, time, time_step, modes, previous_modal_loads, modal_loads, a1s, b1s, i_duhamels: IDuhamels):
    modes_count = modes.shape[1]

    modal_disp = np.matrix(np.zeros((modes_count, 1)))
    modal_disps = np.matrix(np.zeros((1, 1), dtype=object))
    a2 = np.matrix(np.zeros((modes_count, 1)))
    a2s = np.matrix(np.zeros((1, 1), dtype=object))
    b2 = np.matrix(np.zeros((modes_count, 1)))
    b2s = np.matrix(np.zeros((1, 1), dtype=object))

    t1 = time[time_step - 1, 0]
    t2 = time[time_step, 0]
    for mode_num in range(modes_count):
        wn = structure.wns[mode_num]
        wd = structure.wds[mode_num]

        i1 = i_duhamels.i1[mode_num, 0]
        i2 = i_duhamels.i2[mode_num, 0]
        i3 = i_duhamels.i3[mode_num, 0]
        i4 = i_duhamels.i4[mode_num, 0]

        mn = structure.m_modal[mode_num, mode_num]
        p1 = previous_modal_loads[0, 0][mode_num, 0]
        p2 = modal_loads[0, 0][mode_num, 0]
        a1 = a1s[0, 0][mode_num, 0]
        b1 = b1s[0, 0][mode_num, 0]
        a2_mode, b2_mode, modal_disp_mode = get_abx_duhamel(
            structure.damping, t1, t2, i1, i2, i3, i4, wn, wd, mn, a1, b1, p1, p2
        )
        a2[mode_num, 0] = a2_mode
        b2[mode_num, 0] = b2_mode
        modal_disp[mode_num, 0] = modal_disp_mode
    a2s[0, 0] = a2
    b2s[0, 0] = b2
    modal_disps[0, 0] = modal_disp
    return modal_disps, a2s, b2s


@lru_cache()
def get_i_duhamel(damping, t1, t2, wn, wd):
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


def get_abx_duhamel(damping, t1, t2, i1, i2, i3, i4, wn, wd, mn, a1, b1, p1, p2):
    deltat = t2 - t1
    deltap = p2 - p1
    a2 = a1 + (p1 - t1 * deltap / deltat) * i1 + (deltap / deltat) * i4
    b2 = b1 + (p1 - t1 * deltap / deltat) * i2 + (deltap / deltat) * i3
    un = (np.exp(-1 * damping * wn * t2) / (mn * wd)) * (a2 * np.sin(wd * t2) - b2 * np.cos(wd * t2))
    return a2, b2, un


def get_dynamic_sensitivity(structure, loads, time, time_step, modes, i_duhamels):
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

            affected_a2s, affected_b2s, affected_modal_load, affected_struc_disp = get_dynamic_nodal_disp(
                structure=structure,
                loads=loads,
                time=time,
                time_step=time_step,
                modes=modes,
                previous_modal_loads=initial_modal_loads,
                total_load=fv,
                a1s=a1s,
                b1s=b1s,
                i_duhamels=i_duhamels,
            )
            nodal_disp_sensitivity[0, pv_column] = affected_struc_disp[0, 0]
            modal_load_sensitivity[0, pv_column] = affected_modal_load[0, 0]
            a2_sensitivity[0, pv_column] = affected_a2s[0, 0]
            b2_sensitivity[0, pv_column] = affected_b2s[0, 0]
            affected_member_disps = get_members_disps(structure, affected_struc_disp[0, 0])
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
    )
    return sensitivity


def get_a_and_b_sensitivity(structure, modes, time, time_step, i_duhamels: IDuhamels, modal_unit_loads):
    modes_count = modes.shape[1]

    a2 = np.matrix(np.zeros((modes_count, 1)))
    a2s = np.matrix(np.zeros((1, 1), dtype=object))
    b2 = np.matrix(np.zeros((modes_count, 1)))
    b2s = np.matrix(np.zeros((1, 1), dtype=object))
    a2_sensitivity = np.matrix(np.zeros((
        1, structure.yield_specs.intact_components_count), dtype=object
    ))
    b2_sensitivity = np.matrix(np.zeros((
        1, structure.yield_specs.intact_components_count), dtype=object
    ))

    t1 = time[time_step - 1, 0]
    t2 = time[time_step, 0]
    dt = t2 - t1

    for component_num in range(structure.yield_specs.intact_components_count):
        # np.savetxt(f"temp/modal_load-c{component_num}-step-{time_step}", modal_unit_loads[0, component_num], delimiter="\n")
        for mode_num in range(modes_count):
            p = modal_unit_loads[0, component_num][mode_num, 0]
            a2[mode_num, 0] = p / dt * (i_duhamels.i4[mode_num, 0] - t1 * i_duhamels.i1[mode_num, 0])
            b2[mode_num, 0] = p / dt * (i_duhamels.i3[mode_num, 0] - t1 * i_duhamels.i2[mode_num, 0])  
        a2s[0, 0] = a2
        b2s[0, 0] = b2
        # np.savetxt(f"temp/ad-c{component_num}-step-{time_step}", a2, delimiter="\n")
        # np.savetxt(f"temp/bd-c{component_num}-step-{time_step}", b2, delimiter="\n")
        a2_sensitivity[0, component_num] = a2s[0, 0]
        b2_sensitivity[0, component_num] = b2s[0, 0]
    return a2_sensitivity, b2_sensitivity


def get_modes_i1_to_i4(time, time_step, damping, modes, wns, wds):
    modes_count = modes.shape[1]

    i1s = np.matrix(np.zeros((modes_count, 1)))
    i2s = np.matrix(np.zeros((modes_count, 1)))
    i3s = np.matrix(np.zeros((modes_count, 1)))
    i4s = np.matrix(np.zeros((modes_count, 1)))

    t1 = time[time_step - 1, 0]
    t2 = time[time_step, 0]
    for mode_num in range(modes_count):
        wn = wns[mode_num]
        wd = wds[mode_num]
        i1, i2, i3, i4 = get_i_duhamel(damping, t1, t2, wn, wd)
        i1s[mode_num, 0] = i1
        i2s[mode_num, 0] = i2
        i3s[mode_num, 0] = i3
        i4s[mode_num, 0] = i4
    return IDuhamels(i1=i1s, i2=i2s, i3=i3s, i4=i4s)


def get_modal_unit_loads(structure, loads, modes):
    # fv: equivalent global force vector for a yield component's udef
    modal_unit_loads = np.matrix(np.zeros((1, structure.yield_specs.intact_components_count), dtype=object))
    component_counter = 0
    for member in structure.members:
        for load in member.udefs.T:
            fv = np.matrix(np.zeros((structure.dofs_count, 1)))
            global_load = member.t.T * load.T
            local_node_base_dof = 0
            for node in member.nodes:
                global_node_base_dof = structure.node_dofs_count * node.num
                for i in range(structure.node_dofs_count):
                    fv[global_node_base_dof + i] = global_load[local_node_base_dof + i]
                local_node_base_dof += structure.node_dofs_count
            condense_load, _ = loads.apply_static_condensation(structure, fv)
            modal_loads = loads.get_modal_load(condense_load, modes)
            modal_unit_loads[0, component_counter] = modal_loads
            component_counter += 1
    return modal_unit_loads
