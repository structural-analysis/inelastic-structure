import numpy as np
from scipy.linalg import cho_solve
from dataclasses import dataclass


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


def get_nodal_disp(structure, loads, total_load):
    j = 0
    o = 0
    reduced_total_load = loads.apply_boundary_conditions(structure.boundaries_dof, total_load)
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
