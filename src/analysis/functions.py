import numpy as np
from scipy.linalg import cho_solve
from dataclasses import dataclass
from line_profiler import profile
from functools import lru_cache


@dataclass
class InternalResponses:
    p0: np.array
    members_nodal_forces: np.array
    members_nodal_strains: np.array
    members_nodal_stresses: np.array
    members_nodal_moments: np.array


@dataclass
class StaticSensitivity:
    pv: np.array
    nodal_disp: np.array
    members_nodal_forces: np.array
    members_disps: np.array
    members_nodal_strains: np.array
    members_nodal_stresses: np.array
    members_nodal_moments: np.array

@dataclass
class DynamicSensitivity:
    pv: np.array
    nodal_disp: np.array
    members_nodal_forces: np.array
    members_disps: np.array
    modal_loads: np.array
    a2s: np.array
    b2s: np.array


@dataclass
class A2B2Sensitivity:
    a2s: np.array
    b2s: np.array


def get_nodal_disp(structure, loads, total_load):
    reduced_total_load = loads.apply_boundary_conditions(structure.boundaries_dof_mask, total_load)
    reduced_disp = cho_solve(structure.kc, reduced_total_load)
    nodal_disp = structure.undo_disp_boundaries(reduced_disp)
    return nodal_disp


def get_members_disps(structure, disp):
    member_dofs_counts = np.array([member.dofs_count for member in structure.members], dtype=np.int32)
    member_nodes_counts = np.array([len(member.nodes) for member in structure.members], dtype=np.int32)
    member_node_dofs_counts = member_dofs_counts // member_nodes_counts

    all_node_nums = np.concatenate([np.array([node.num for node in member.nodes]) for member in structure.members])
    t_matrices = np.array([member.t for member in structure.members])
    node_offsets = np.repeat(all_node_nums, np.repeat(member_node_dofs_counts, member_nodes_counts))
    dof_offsets = np.tile(np.arange(member_node_dofs_counts[0]), sum(member_nodes_counts))

    v = disp[member_node_dofs_counts[0] * node_offsets + dof_offsets].reshape(-1, 1)

    # v_split = np.split(v, np.cumsum(member_dofs_counts)[:-1])
    cumsum_indices = np.cumsum(member_dofs_counts)[:-1]
    # v_split = np.split(v, cumsum_indices)
    v_split = np.array_split(v, cumsum_indices)
    # t_matrices = np.array([t for t in t_matrices])  # shape (n, m, p)
    # v_split = np.array([v.flatten() for v in v_split])  # shape (n, p)

    members_disps = np.zeros((structure.members_count, structure.max_member_dofs_count))
    computed_members_disps = np.squeeze(np.matmul(t_matrices, v_split), axis=2)
    members_disps[:computed_members_disps.shape[0], :computed_members_disps.shape[1]] = computed_members_disps

    # computed_members_disps = np.array([t @ v for t, v in zip(t_matrices, v_split)])
    # computed_members_disps = np.squeeze(computed_members_disps)
    # flat_members_disps = members_disps.flatten()
    # flat_members_disps_computed = np.concatenate(computed_members_disps)

    # Now use np.put to fill the target array
    # np.put(flat_members_disps, np.arange(len(flat_members_disps_computed)), flat_members_disps_computed)
    return members_disps


# def get_internal_responses(structure, members_disps):
#     # calculate p0
#     members_nodal_forces = np.zeros((structure.members_count, 1))
#     members_nodal_strains = np.zeros((structure.members_count, 1))
#     members_nodal_stresses = np.zeros((structure.members_count, 1))
#     members_nodal_moments = np.zeros((structure.members_count, 1))
#     p0 = np.zeros((structure.yield_specs.intact_components_count, 1))

#     base_p0_row = 0
#     members_responses = [member.get_response(members_disps[i, :]) for i, member in enumerate(structure.members)]

#     for i, member in enumerate(structure.members):
#         member_response = member.get_response(members_disps[i, :].reshape(-1, 1))
#         yield_components_force = member_response.yield_components_force
#         p0[base_p0_row:(base_p0_row + member.yield_specs.components_count)] = yield_components_force
#         base_p0_row = base_p0_row + member.yield_specs.components_count

#         # TODO: we can clean and simplify appending member response to members_{responses}
#         # each member should contain only its responses, not zero response of other elements.
#         # but in members_{responses}, instead of appending with i (member num in structure.members),
#         # we should attach with member.num, and if one member has not some response it will not appended.
#         # we can use a dataclass like:
#         # MemberResponse:
#         # member: object
#         # response: Response
#         members_nodal_forces[i, 0] = member_response.nodal_force
#         members_nodal_strains[i, 0] = member_response.nodal_strains
#         members_nodal_stresses[i, 0] = member_response.nodal_stresses
#         members_nodal_moments[i, 0] = member_response.nodal_moments

#     return InternalResponses(
#         p0=p0,
#         members_nodal_forces=members_nodal_forces,
#         members_nodal_strains=members_nodal_strains,
#         members_nodal_stresses=members_nodal_stresses,
#         members_nodal_moments=members_nodal_moments,
#     )


def get_internal_responses(structure, members_disps):
    # Compute member responses
    members_responses = [
        member.get_response(members_disps[i, :])
        for i, member in enumerate(structure.members)
    ]

    # Extract attributes from member responses
    members_nodal_forces = np.array(
        [mr.nodal_force for mr in members_responses]
    )
    members_nodal_strains = np.array(
        [mr.nodal_strains for mr in members_responses]
    )
    members_nodal_stresses = np.array(
        [mr.nodal_stresses for mr in members_responses]
    )
    members_nodal_moments = np.array(
        [mr.nodal_moments for mr in members_responses]
    )

    # Concatenate yield component forces
    yield_components_forces = [
        mr.yield_components_force for mr in members_responses
    ]
    p0 = np.concatenate(yield_components_forces, axis=0)

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
    d0 = np.zeros(disp_limits_count)
    for i, disp_limit in enumerate(disp_limits):
        node = disp_limit[0]
        node_dof = disp_limit[1]
        dof = structure.get_global_dof(node, node_dof)
        d0[i] = elastic_nodal_disp[dof]
    return d0


def get_sensitivity(structure, loads):
    # fv: equivalent global force vector for a yield component's udef
    members = structure.members
    pv = np.zeros((structure.yield_specs.intact_components_count, structure.yield_specs.intact_components_count))
    nodal_disp_sensitivity = np.zeros((structure.dofs_count, structure.yield_specs.intact_components_count))
    members_nodal_forces_sensitivity = np.zeros((structure.members_count, structure.max_member_dofs_count, structure.yield_specs.intact_components_count))
    members_disps_sensitivity = np.zeros((structure.members_count, structure.max_member_dofs_count, structure.yield_specs.intact_components_count))
    members_nodal_strains_sensitivity = np.zeros((structure.members_count, structure.max_member_nodal_components_count, structure.yield_specs.intact_components_count))
    members_nodal_stresses_sensitivity = np.zeros((structure.members_count, structure.max_member_nodal_components_count, structure.yield_specs.intact_components_count))
    members_nodal_moments_sensitivity = np.zeros((structure.members_count, structure.max_member_nodal_components_count, structure.yield_specs.intact_components_count))
    pv_column = 0

    for member_num, member in enumerate(members):
        # FIXME: GENERALIZE PLEASE
        for comp_num, force in enumerate(member.udefs.T):
            fv = np.zeros((structure.dofs_count, 1))
            global_force = np.dot(member.t.T, force.T)
            local_node_base_dof = 0
            for node in member.nodes:
                global_node_base_dof = structure.node_dofs_count * node.num
                for i in range(structure.node_dofs_count):
                    fv[global_node_base_dof + i] = global_force[local_node_base_dof + i]
                local_node_base_dof += structure.node_dofs_count

            affected_structure_disp = get_nodal_disp(structure=structure, loads=loads, total_load=fv)
            nodal_disp_sensitivity[:, pv_column] = affected_structure_disp
            # TODO: it is good if any member has a disp vector (and disps matrix from sensitivity) property which is filled after analysis.
            # then we can iterate only on members instead of members count.
            # each member also has a num property and no need to get their position in the list.
            # there is another shortcut to do a clean way. create a AffectedMember class with num, disp, disps properties
            # which it's objects are created after analysis.
            affected_member_disps = get_members_disps(structure, affected_structure_disp)
            current_affected_member_ycns = 0
            for affected_member_num, affected_member_disp in enumerate(affected_member_disps):
                fixed_external_shape = structure.members[affected_member_num].dofs_count
                fixed_external = -force.T if member_num == affected_member_num else np.zeros(fixed_external_shape)
                if structure.members[affected_member_num].__class__.__name__ in ["WallMember", "PlateMember"]:
                    # NOTE: yield_specs.components_count has different meanings in different members.
                    fixed_internal_shape = (structure.members[affected_member_num].yield_specs.components_count, 1)
                    if member_num == affected_member_num:
                        fixed_internal = -structure.members[affected_member_num].udets.T[comp_num].T
                    else:
                        fixed_internal = np.matrix(np.zeros((fixed_internal_shape)))
                else:
                    fixed_internal = None

                affected_member_response = structure.members[affected_member_num].get_response(affected_member_disp, fixed_external, fixed_internal)
                affected_member_nodal_force = affected_member_response.nodal_force
                affected_member_yield_components_force = affected_member_response.yield_components_force

                # FIXME: GENERALIZE PLEASE
                # for wall and plate members:
                if member.__class__.__name__ in ["WallMember", "PlateMember"]:
                    if member_num == affected_member_num:
                        udet = structure.members[affected_member_num].udets.T[comp_num]
                        affected_member_yield_components_force -= udet.T
                    affected_member_nodal_strains = affected_member_response.nodal_strains
                    affected_member_nodal_stresses = affected_member_response.nodal_stresses
                    affected_member_nodal_moments = affected_member_response.nodal_moments
                    members_nodal_strains_sensitivity[affected_member_num, :, pv_column] = np.pad(affected_member_nodal_strains, (0, structure.max_member_nodal_components_count - affected_member_nodal_strains.size))
                    members_nodal_stresses_sensitivity[affected_member_num, :, pv_column] = np.pad(affected_member_nodal_stresses, (0, structure.max_member_nodal_components_count - affected_member_nodal_stresses.size))
                    members_nodal_moments_sensitivity[affected_member_num, :, pv_column] = np.pad(affected_member_nodal_moments, (0, structure.max_member_nodal_components_count - affected_member_nodal_moments.size))

                members_nodal_forces_sensitivity[affected_member_num, :, pv_column] = affected_member_nodal_force
                members_disps_sensitivity[affected_member_num, :, pv_column] = affected_member_disp

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
        members_nodal_moments=members_nodal_moments_sensitivity,
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
            dv[i, j] = nodal_disp_sensitivity[dof, j]
    return dv


def get_dynamic_nodal_disp(structure, loads, t1, t2, modes, total_load, previous_modal_loads, previous_a2s, previous_b2s):
    condense_load, reduced_p0 = loads.apply_static_condensation(structure, total_load)
    modal_loads = loads.get_modal_load(condense_load, structure.selected_modes)
    modal_disps, a2s, b2s, a_factor, b_factor = get_modal_disp(
        structure=structure,
        t1=t1,
        t2=t2,
        modal_loads=modal_loads,
        previous_modal_loads=previous_modal_loads,
        previous_a2s=previous_a2s,
        previous_b2s=previous_b2s,
    )
    ut = np.dot(modes, modal_disps)
    u0 = np.dot(structure.reduced_k00_inv, reduced_p0) + np.dot(structure.ku0, ut)
    nodal_disp = structure.undo_disp_condensation(ut, u0)

    return a2s, b2s, a_factor, b_factor, modal_loads, nodal_disp


def get_modal_disp(structure, t1, t2, modal_loads, previous_modal_loads, previous_a2s, previous_b2s):
    deltat = t2 - t1

    wns = np.array(structure.wns[:structure.selected_modes_count])
    wds = np.array(structure.wds[:structure.selected_modes_count])
    mns = np.diag(structure.m_modal)
    p1s = np.array(previous_modal_loads).flatten()
    p2s = np.array(modal_loads).flatten()
    previous_a2s = np.array(previous_a2s).flatten()
    previous_b2s = np.array(previous_b2s).flatten()

    i1s, i2s, i3s, i4s, a_factor, b_factor = get_is_duhamel(structure.damping, t1, t2, wns, wds)

    deltaps = p2s - p1s
    a2s = get_a_duhamel(t1, deltat, i1s, i4s, previous_a2s, p1s, deltaps)
    b2s = get_b_duhamel(t1, deltat, i2s, i3s, previous_b2s, p1s, deltaps)
    modal_disps = get_disps_duhamel(structure.damping, t2, wns, wds, mns, a2s, b2s)

    return modal_disps, a2s, b2s, a_factor, b_factor

# # Before Optimization
# def get_is_duhamel(damping, t1, t2, wns, wds):
#     i12 = (np.exp(damping * wns * t2) / ((damping * wns) ** 2 + wds ** 2)) * (damping * wns * np.cos(wds * t2) + wds * np.sin(wds * t2))
#     i11 = (np.exp(damping * wns * t1) / ((damping * wns) ** 2 + wds ** 2)) * (damping * wns * np.cos(wds * t1) + wds * np.sin(wds * t1))
#     i1 = i12 - i11
#     i22 = (np.exp(damping * wns * t2) / ((damping * wns) ** 2 + wds ** 2)) * (damping * wns * np.sin(wds * t2) - wds * np.cos(wds * t2))
#     i21 = (np.exp(damping * wns * t1) / ((damping * wns) ** 2 + wds ** 2)) * (damping * wns * np.sin(wds * t1) - wds * np.cos(wds * t1))
#     i2 = i22 - i21
#     i3 = (t2 - (damping * wns / ((damping * wns) ** 2 + wds ** 2))) * i22 + (wds / ((damping * wns) ** 2 + wds ** 2)) * i12 - ((t1 - (damping * wns / ((damping * wns) ** 2 + wds ** 2))) * i21 + (wds / ((damping * wns) ** 2 + wds ** 2)) * i11)
#     i4 = (t2 - (damping * wns / ((damping * wns) ** 2 + wds ** 2))) * i12 - (wds / ((damping * wns) ** 2 + wds ** 2)) * i22 - ((t1 - (damping * wns / ((damping * wns) ** 2 + wds ** 2))) * i11 - (wds / ((damping * wns) ** 2 + wds ** 2)) * i21)
#     return i1, i2, i3, i4


@profile
def get_is_duhamel(damping, t1, t2, wns, wds):
    D = damping * wns
    D2 = D * D
    wds2 = wds * wds
    denom = D2 + wds2
    inv_denom = 1.0 / denom

    alpha = D * inv_denom
    beta = wds * inv_denom

    # Precompute exponentials
    D_t1 = D * t1
    D_t2 = D * t2
    E1 = np.exp(D_t1)
    E2 = np.exp(D_t2)

    # Precompute trigonometric functions
    wds_t1 = wds * t1
    wds_t2 = wds * t2
    C1 = np.cos(wds_t1)
    S1 = np.sin(wds_t1)
    C2 = np.cos(wds_t2)
    S2 = np.sin(wds_t2)

    # Precompute products
    D_C1 = D * C1
    D_C2 = D * C2
    D_S1 = D * S1
    D_S2 = D * S2
    wds_C1 = wds * C1
    wds_C2 = wds * C2
    wds_S1 = wds * S1
    wds_S2 = wds * S2

    # Compute i1 and i2 terms
    common_factor_E1 = E1 * inv_denom
    common_factor_E2 = E2 * inv_denom

    i11 = common_factor_E1 * (D_C1 + wds_S1)
    i12 = common_factor_E2 * (D_C2 + wds_S2)
    i1 = i12 - i11

    i21 = common_factor_E1 * (D_S1 - wds_C1)
    i22 = common_factor_E2 * (D_S2 - wds_C2)
    i2 = i22 - i21

    # Compute i3 and i4 terms
    t1_minus_alpha = t1 - alpha
    t2_minus_alpha = t2 - alpha

    i3 = (t2_minus_alpha * i22 + beta * i12) - (t1_minus_alpha * i21 + beta * i11)
    i4 = (t2_minus_alpha * i12 - beta * i22) - (t1_minus_alpha * i11 - beta * i21)

    a_factor = - i1 * alpha - i2 * beta + (t2 - t1) * i12
    b_factor = i1 * beta - alpha * i2 + (t2 - t1) * i22
    return i1, i2, i3, i4, a_factor, b_factor


def get_a_duhamel(t1, deltat, i1s, i4s, a1s, p1s, deltaps):
    return a1s + (p1s - t1 * deltaps / deltat) * i1s + (deltaps / deltat) * i4s


def get_b_duhamel(t1, deltat, i2s, i3s, b1s, p1s, deltaps):
    return b1s + (p1s - t1 * deltaps / deltat) * i2s + (deltaps / deltat) * i3s


def get_disps_duhamel(damping, t2, wns, wds, mns, a2s, b2s):
    return (np.exp(-1 * damping * wns * t2) / (mns * wds)) * (a2s * np.sin(wds * t2) - b2s * np.cos(wds * t2))


def get_dynamic_sensitivity(structure, loads, deltat):
    # modes_count = modes.shape[1]
    selected_modes_count = structure.selected_modes_count
    # fv: equivalent global force vector for a yield component's udef
    members = structure.members
    pv = np.zeros((
        structure.yield_specs.intact_components_count, structure.yield_specs.intact_components_count
    ))
    pv_column = 0

    members_nodal_forces_sensitivity = np.zeros((
        structure.members_count, structure.max_member_dofs_count, structure.yield_specs.intact_components_count
    ))
    members_disps_sensitivity = np.zeros((
        structure.members_count, structure.max_member_dofs_count, structure.yield_specs.intact_components_count
    ))
    nodal_disp_sensitivity = np.zeros((
        structure.dofs_count, structure.yield_specs.intact_components_count
    ))
    modal_load_sensitivity = np.zeros((
        selected_modes_count, structure.yield_specs.intact_components_count
    ))
    a2_sensitivity = np.zeros((
        selected_modes_count, structure.yield_specs.intact_components_count
    ))
    b2_sensitivity = np.zeros((
        selected_modes_count, structure.yield_specs.intact_components_count
    ))

    a1s_empty = np.zeros(selected_modes_count)
    b1s_empty = np.zeros(selected_modes_count)
    initial_modal_load_empty = np.zeros(selected_modes_count)

    for member_num, member in enumerate(members):
        for load in member.udefs.T:
            fv = np.zeros((structure.dofs_count, 1))
            global_load = np.dot(member.t.T, load.T)
            local_node_base_dof = 0
            for node in member.nodes:
                global_node_base_dof = structure.node_dofs_count * node.num
                for i in range(structure.node_dofs_count):
                    fv[global_node_base_dof + i] = global_load[local_node_base_dof + i]
                local_node_base_dof += structure.node_dofs_count

                # affected_struc_disp, p2_modes, a2_modes, b2_mdoes = self.get_dynamic_unit_nodal_disp(fv, modes, time_step)
                a1s = a1s_empty
                b1s = b1s_empty
                initial_modal_loads = initial_modal_load_empty

            affected_a2s, affected_b2s, _, _, affected_modal_load, affected_struc_disp = get_dynamic_nodal_disp(
                structure=structure,
                loads=loads,
                t1=0,
                t2=deltat,
                modes=structure.selected_modes,
                previous_modal_loads=initial_modal_loads,
                total_load=fv,
                previous_a2s=a1s,
                previous_b2s=b1s,
            )
            nodal_disp_sensitivity[:, pv_column] = affected_struc_disp
            modal_load_sensitivity[:, pv_column] = affected_modal_load
            a2_sensitivity[:, pv_column] = affected_a2s
            b2_sensitivity[:, pv_column] = affected_b2s
            affected_member_disps = get_members_disps(structure, affected_struc_disp)
            current_affected_member_ycns = 0

            # for affected_member_num, affected_member_disp in enumerate(affected_member_disps):
            for affected_member_num in range(affected_member_disps.shape[0]):
                fixed_external = -load.T if member_num == affected_member_num else None
                affected_member_response = structure.members[affected_member_num].get_response(affected_member_disps[affected_member_num, :], fixed_external)
                affected_member_nodal_force = affected_member_response.nodal_force
                affected_member_yield_components_force = affected_member_response.yield_components_force
                members_nodal_forces_sensitivity[affected_member_num, :, pv_column] = affected_member_nodal_force
                members_disps_sensitivity[affected_member_num, :, pv_column] = affected_member_disps[affected_member_num, :]
                pv[current_affected_member_ycns:(current_affected_member_ycns + structure.members[affected_member_num].yield_specs.components_count), pv_column] = affected_member_yield_components_force
                current_affected_member_ycns = current_affected_member_ycns + structure.members[affected_member_num].yield_specs.components_count
            pv_column += 1

    sensitivity = DynamicSensitivity(
        modal_loads=modal_load_sensitivity,
        a2s=a2_sensitivity,
        b2s=b2_sensitivity,
        pv=pv,
        nodal_disp=nodal_disp_sensitivity,
        members_nodal_forces=members_nodal_forces_sensitivity,
        members_disps=members_disps_sensitivity,
    )
    return sensitivity


def get_a2s_b2s_sensitivity(a_factor, b_factor, a2s_b2s_sensitivity_constant):
    a2s_sensitivity = np.multiply(a_factor[:, np.newaxis], a2s_b2s_sensitivity_constant)
    b2s_sensitivity = np.multiply(b_factor[:, np.newaxis], a2s_b2s_sensitivity_constant)
    sensitivity = A2B2Sensitivity(
        a2s=a2s_sensitivity,
        b2s=b2s_sensitivity,
    )
    return sensitivity


def get_a2s_b2s_sensitivity_constant(structure, loads, deltat, modal_loads_sensitivity):
    """ checking some examples shows that normalized_a2_sensitivity and normalized_b2_sensitivity are equal,
    so we only compute one of them and use for a2 and b2.
    """

    selected_modes_count = structure.selected_modes_count
    # fv: equivalent global force vector for a yield component's udef
    members = structure.members

    pv_column = 0

    a2s_sensitivity = np.zeros((
        selected_modes_count, structure.yield_specs.intact_components_count
    ))

    a1s_empty = np.zeros(selected_modes_count)
    b1s_empty = np.zeros(selected_modes_count)
    initial_modal_load_empty = np.zeros(selected_modes_count)

    for member in members:
        for load in member.udefs.T:
            fv = np.zeros((structure.dofs_count, 1))
            global_load = np.dot(member.t.T, load.T)
            local_node_base_dof = 0
            for node in member.nodes:
                global_node_base_dof = structure.node_dofs_count * node.num
                for i in range(structure.node_dofs_count):
                    fv[global_node_base_dof + i] = global_load[local_node_base_dof + i]
                local_node_base_dof += structure.node_dofs_count

                a1s = a1s_empty
                b1s = b1s_empty
                initial_modal_loads = initial_modal_load_empty

            condense_load, _ = loads.apply_static_condensation(structure, fv)
            modal_loads = loads.get_modal_load(condense_load, structure.selected_modes)

            wns = np.array(structure.wns[:structure.selected_modes_count])
            wds = np.array(structure.wds[:structure.selected_modes_count])
            p1s = np.array(initial_modal_loads).flatten()
            p2s = np.array(modal_loads).flatten()
            a1s = np.array(a1s).flatten()
            b1s = np.array(b1s).flatten()

            i1s, _, _, i4s, _, _ = get_is_duhamel(
                damping=structure.damping,
                t1=0,
                t2=deltat,
                wns=wns,
                wds=wds,
            )

            deltaps = p2s - p1s
            affected_a2s = get_a_duhamel(0, deltat, i1s, i4s, a1s, p1s, deltaps)
            a2s_sensitivity[:, pv_column] = affected_a2s

            pv_column += 1

    first_elements = a2s_sensitivity[:, 0]
    mask = first_elements != 0
    a2s_sensitivity[mask] /= first_elements[mask, np.newaxis]

    # normalized_a2s_b2s_sensitivity = a2s_sensitivity / a2s_sensitivity[:, 0][:, np.newaxis]
    a2s_b2s_sensitivity_constant = 1 / deltat * np.multiply(a2s_sensitivity, modal_loads_sensitivity[:, 0][:, np.newaxis])
    return a2s_b2s_sensitivity_constant
