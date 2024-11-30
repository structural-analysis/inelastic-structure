import os
import numpy as np
from enum import Enum

from .functions import get_elastoplastic_response, load_chunk, delete_chunk, get_activated_plastic_points
from .settings import settings
from .analysis.initial_analysis import AnalysisType

outputs_dir = "output/examples/"


class DesiredResponse(list, Enum):
    TRUSS2D = [
        "load_levels",
        "nodal_disp",
        "members_disps",
        "members_nodal_forces",
    ]
    FRAME2D = [
        "load_levels",
        "nodal_disp",
        "members_disps",
        "members_nodal_forces",
    ]
    FRAME3D = [
        "load_levels",
        "nodal_disp",
        "members_disps",
        "members_nodal_forces",
    ]
    WALL2D = [
        "load_levels",
        "nodal_disp",
        "nodal_strains",
        "nodal_stresses",
        "members_disps",
        "members_nodal_forces",
        "members_nodal_strains",
        "members_nodal_stresses",
    ]
    PLATE2D = [
        "load_levels",
        "members_disps",
        "members_nodal_forces",
        "nodal_disp",
        "nodal_moments",
        "members_nodal_moments",
    ]


def calculate_responses(initial_analysis, inelastic_analysis=None):
    if initial_analysis.analysis_type is AnalysisType.STATIC:
        responses = calculate_static_responses(initial_analysis, inelastic_analysis)
    if initial_analysis.analysis_type is AnalysisType.DYNAMIC:
        responses = calculate_dynamic_responses(initial_analysis, inelastic_analysis)
    return responses


def calculate_static_responses(initial_analysis, inelastic_analysis=None):
    structure = initial_analysis.structure
    if structure.is_inelastic:
        pms_history = inelastic_analysis.plastic_vars["pms_history"]
        phi_x_history = inelastic_analysis.plastic_vars["phi_pms_history"]
        load_level_history = inelastic_analysis.plastic_vars["load_level_history"]
        increments_count = len(load_level_history)

        load_levels = np.zeros(increments_count)

        plastic_points = np.zeros(increments_count, dtype=object)

        nodal_disp_sensitivity = initial_analysis.nodal_disp_sensitivity
        nodal_disp = np.zeros((increments_count, structure.dofs_count))

        members_nodal_forces_sensitivity = initial_analysis.members_nodal_forces_sensitivity
        members_nodal_forces = np.zeros((increments_count, structure.members_count, structure.max_member_dofs_count))

        members_nodal_moments_sensitivity = initial_analysis.members_nodal_moments_sensitivity
        members_nodal_moments = np.zeros((increments_count, structure.members_count, structure.max_member_nodal_components_count))
        nodal_moments = np.zeros((increments_count, structure.nodal_components_count))

        members_disps_sensitivity = initial_analysis.members_disps_sensitivity
        members_disps = np.zeros((increments_count, structure.members_count, structure.max_member_dofs_count))

        members_nodal_strains_sensitivity = initial_analysis.members_nodal_strains_sensitivity
        members_nodal_strains = np.zeros((increments_count, structure.members_count, structure.max_member_nodal_components_count))
        nodal_strains = np.zeros((increments_count, structure.nodal_components_count))

        members_nodal_stresses_sensitivity = initial_analysis.members_nodal_stresses_sensitivity
        members_nodal_stresses = np.zeros((increments_count, structure.members_count, structure.max_member_nodal_components_count))
        nodal_stresses = np.zeros((increments_count, structure.nodal_components_count))

        for i in range(increments_count):
            phi_x = phi_x_history[i]
            load_level = load_level_history[i]
            load_levels[i] = load_level

            elastoplastic_nodal_disp = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=initial_analysis.elastic_nodal_disp,
                sensitivity=nodal_disp_sensitivity,
            )
            nodal_disp[i, :] = elastoplastic_nodal_disp

            elastoplastic_members_nodal_forces = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=initial_analysis.elastic_members_nodal_forces,
                sensitivity=members_nodal_forces_sensitivity,
            )

            elastoplastic_members_disps = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=initial_analysis.elastic_members_disps,
                sensitivity=members_disps_sensitivity,
            )

            members_nodal_forces[i, :, :] = elastoplastic_members_nodal_forces
            members_disps[i, :] = elastoplastic_members_disps

            if has_any_response(initial_analysis.elastic_members_nodal_strains):
                elastoplastic_members_nodal_strains = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_x,
                    elastic_response=initial_analysis.elastic_members_nodal_strains,
                    sensitivity=members_nodal_strains_sensitivity,
                )

                elastoplastic_members_nodal_stresses = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_x,
                    elastic_response=initial_analysis.elastic_members_nodal_stresses,
                    sensitivity=members_nodal_stresses_sensitivity,
                )

                members_nodal_strains[i, :, :] = elastoplastic_members_nodal_strains
                members_nodal_stresses[i, :, :] = elastoplastic_members_nodal_stresses

                nodal_strains[i, :] = average_nodal_responses(structure=structure, members_responses=elastoplastic_members_nodal_strains)
                nodal_stresses[i, :] = average_nodal_responses(structure=structure, members_responses=elastoplastic_members_nodal_stresses)

            if has_any_response(initial_analysis.elastic_members_nodal_moments):
                elastoplastic_members_nodal_moments = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_x,
                    elastic_response=initial_analysis.elastic_members_nodal_moments,
                    sensitivity=members_nodal_moments_sensitivity,
                )

                members_nodal_moments[i, :, :] = elastoplastic_members_nodal_moments
                nodal_moments[i, :] = average_nodal_responses(structure=structure, members_responses=elastoplastic_members_nodal_moments)

        responses = {
            "plastic_points": plastic_points,
            "load_levels": load_levels,
            "nodal_disp": nodal_disp,
            "members_disps": members_disps,
            "members_nodal_forces": members_nodal_forces,
        }

        if has_any_response(members_nodal_strains):
            responses.update(
                {
                    "nodal_strains": nodal_strains,
                    "nodal_stresses": nodal_stresses,
                    "members_nodal_strains": members_nodal_strains,
                    "members_nodal_stresses": members_nodal_stresses,
                }
            )

        if has_any_response(members_nodal_moments):
            responses.update(
                {
                    "nodal_moments": nodal_moments,
                    "members_nodal_moments": members_nodal_moments,
                }
            )

    elif not structure.is_inelastic:  # if structure is elastic
        # for elastic analysis, there is only one increment so for responses size we use 1.
        increments_count = 1
        nodal_disp = np.zeros((increments_count, structure.dofs_count))
        members_disps = np.zeros((increments_count, structure.members_count, structure.max_member_dofs_count))
        members_nodal_forces = np.zeros((increments_count, structure.members_count, structure.max_member_dofs_count))
        members_nodal_strains = np.zeros((increments_count, structure.members_count, structure.max_member_nodal_components_count))
        members_nodal_stresses = np.zeros((increments_count, structure.members_count, structure.max_member_nodal_components_count))
        nodal_strains = np.zeros((increments_count, structure.nodal_components_count))
        nodal_stresses = np.zeros((increments_count, structure.nodal_components_count))
        members_nodal_moments = np.zeros((increments_count, structure.members_count, structure.max_member_nodal_components_count))
        nodal_moments = np.zeros((increments_count, structure.nodal_components_count))

        load_levels = np.array([structure.limits["load_limit"][0]])
        nodal_disp[0, :] = structure.limits["load_limit"][0] * initial_analysis.elastic_nodal_disp
        members_disps[0, :, :] = structure.limits["load_limit"][0] * initial_analysis.elastic_members_disps
        members_nodal_forces[0, :, :] = structure.limits["load_limit"][0] * initial_analysis.elastic_members_nodal_forces

        responses = {
            "load_levels": load_levels,
            "nodal_disp": nodal_disp,
            "members_disps": members_disps,
            "members_nodal_forces": members_nodal_forces,

        }

        if has_any_response(initial_analysis.elastic_members_nodal_strains):
            members_nodal_strains[0, :, :] = structure.limits["load_limit"][0] * initial_analysis.elastic_members_nodal_strains
            members_nodal_stresses[0, :, :] = structure.limits["load_limit"][0] * initial_analysis.elastic_members_nodal_stresses
            nodal_strains[0, :] = average_nodal_responses(structure=structure, members_responses=members_nodal_strains[0, :, :])
            nodal_stresses[0, :] = average_nodal_responses(structure=structure, members_responses=members_nodal_stresses[0, :, :])
            responses.update(
                {
                    "members_nodal_strains": members_nodal_strains,
                    "members_nodal_stresses": members_nodal_stresses,
                    "nodal_strains": nodal_strains,
                    "nodal_stresses": nodal_stresses,
                }
            )

        if has_any_response(initial_analysis.elastic_members_nodal_moments):
            members_nodal_moments[0, :, :] = structure.limits["load_limit"][0] * initial_analysis.elastic_members_nodal_moments
            nodal_moments[0, :] = average_nodal_responses(structure=structure, members_responses=members_nodal_moments[0, :, :])
            responses.update(
                {
                    "members_nodal_moments": members_nodal_moments,
                    "nodal_moments": nodal_moments,
                }
            )

    return responses


def average_nodal_responses(structure, members_responses):
    comp_count = 3  # response_components_count
    nodes_map = structure.nodes_map
    nodal_responses = np.zeros(structure.nodes_count * comp_count)
    for node in structure.nodes:
        node_sum_response = np.zeros(comp_count)
        for attached_member in nodes_map[node.num].attached_members:
            start = comp_count * attached_member.member_node_num
            end = comp_count * (attached_member.member_node_num + 1)
            member_node_response = members_responses[attached_member.member.num, start:end]
            node_sum_response += member_node_response
        node_average_response = node_sum_response / len(nodes_map[node.num].attached_members)
        nodal_responses[comp_count * node.num:comp_count * (node.num + 1)] = node_average_response
    return nodal_responses


def calculate_dynamic_responses(initial_analysis, inelastic_analysis):
    structure = initial_analysis.structure
    if structure.is_inelastic:
        plastic_vars_history = inelastic_analysis.plastic_vars_history
        final_inc_phi_pms_history = inelastic_analysis.final_inc_phi_pms_history

        elastic_members_nodal_forces_history = initial_analysis.elastic_members_nodal_forces_history
        elastic_members_disps_history = initial_analysis.elastic_members_disps_history
        elastic_nodal_disp_history = initial_analysis.elastic_nodal_disp_history

        responses = np.zeros(initial_analysis.time_steps, dtype=object)
        for time_step in range(1, initial_analysis.time_steps):
            plastic_vars = plastic_vars_history[time_step, 0]
            pms_history = plastic_vars["pms_history"]
            phi_pms_history = plastic_vars["phi_pms_history"]
            load_level_history = plastic_vars["load_level_history"]
            final_inc_phi_pms_prev = final_inc_phi_pms_history[time_step - 1, :]

            increments_count = len(load_level_history)
            load_levels = np.zeros(increments_count)

            nodal_disp_sensitivity = load_chunk(time_step=time_step, response="nodal_disp")
            members_nodal_forces_sensitivity = load_chunk(time_step=time_step, response="members_nodal_forces")
            members_disps_sensitivity = load_chunk(time_step=time_step, response="members_disps")

            plastic_points = np.zeros(increments_count, dtype=object)
            nodal_disp = np.zeros((increments_count, structure.dofs_count))
            members_nodal_forces = np.zeros((increments_count, structure.members_count, structure.max_member_dofs_count))
            members_disps = np.zeros((increments_count, structure.members_count, structure.max_member_dofs_count))

            delete_chunk(time_step=time_step, response="nodal_disp")
            delete_chunk(time_step=time_step, response="members_nodal_forces")
            delete_chunk(time_step=time_step, response="members_disps")

            for i in range(increments_count):
                phi_pms = phi_pms_history[i] + final_inc_phi_pms_prev
                load_level = load_level_history[i]

                load_levels[i] = load_level
                elastoplastic_nodal_disp = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_pms,
                    elastic_response=elastic_nodal_disp_history[time_step, :],
                    sensitivity=nodal_disp_sensitivity,
                )
                nodal_disp[i, :] = elastoplastic_nodal_disp

                elastoplastic_members_nodal_forces = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_pms,
                    elastic_response=elastic_members_nodal_forces_history[time_step, :, :],
                    sensitivity=members_nodal_forces_sensitivity,
                )
                members_nodal_forces[i, :, :] = elastoplastic_members_nodal_forces

                elastoplastic_members_disps = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_pms,
                    elastic_response=elastic_members_disps_history[time_step, :, :],
                    sensitivity=members_disps_sensitivity,
                )
                members_disps[i, :, :] = elastoplastic_members_disps

                plastic_points[i] = get_activated_plastic_points(pms=pms_history[i], intact_pieces=initial_analysis.initial_data.intact_pieces)

            responses[time_step] = {
                "plastic_points": plastic_points,
                "members_nodal_forces": members_nodal_forces,
                "members_disps": members_disps,

    elif not structure.is_inelastic:  # if structure is elastic
        load_limit = structure.limits["load_limit"][0]
        elastic_nodal_disp_history = initial_analysis.elastic_nodal_disp_history
        elastic_members_nodal_forces_history = initial_analysis.elastic_members_nodal_forces_history
        elastic_members_disps_history = initial_analysis.elastic_members_disps_history
        elastic_members_nodal_strains_history = initial_analysis.elastic_members_nodal_strains_history
        elastic_members_nodal_stresses_history = initial_analysis.elastic_members_nodal_stresses_history
        responses = np.zeros((initial_analysis.time_steps), dtype=object)

        # for elastic analysis, there is only one increment so for responses size we use 1.
        increments_count = 1

        for time_step in range(1, initial_analysis.time_steps):
            nodal_disp = np.zeros((increments_count, structure.dofs_count))
            members_disps = np.zeros((increments_count, structure.members_count, structure.max_member_dofs_count))
            members_nodal_forces = np.zeros((increments_count, structure.members_count, structure.max_member_dofs_count))
            members_nodal_strains = np.zeros([increments_count, structure.members_count], dtype=object)
            members_nodal_stresses = np.zeros([increments_count, structure.members_count], dtype=object)
            nodal_strains = np.zeros([increments_count, 1], dtype=object)
            nodal_stresses = np.zeros([increments_count, 1], dtype=object)
            load_levels = np.zeros(increments_count)

            load_levels[0] = load_limit
            # elastoplastic_nodal_disp = get_elastoplastic_response(
            #     load_level=load_level,
            #     phi_x=phi_x,
            #     elastic_response=elastic_nodal_disp_history[time_step, 0],
            #     sensitivity=nodal_disp_sensitivity,
            # )
            # nodal_disp[i, 0] = elastoplastic_nodal_disp[0, 0]
            for i in range(increments_count):
                nodal_disp[i, :] = elastic_nodal_disp_history[time_step, :] * load_limit
                # elastoplastic_members_nodal_forces = get_elastoplastic_response(
                #     load_level=load_level,
                #     phi_x=phi_x,
                #     elastic_response=elastic_members_nodal_forces_history[time_step, 0],
                #     sensitivity=members_nodal_forces_sensitivity,
                # )
                elastic_members_nodal_forces = elastic_members_nodal_forces_history[time_step, :, :] * load_limit
                elastic_members_disps = elastic_members_disps_history[time_step, :, :] * load_limit
                elastic_members_nodal_strains = elastic_members_nodal_strains_history[time_step, :] * load_limit
                elastic_members_nodal_stresses = elastic_members_nodal_stresses_history[time_step, :] * load_limit

                members_nodal_forces[i, :, :] = elastic_members_nodal_forces
                members_disps[i, :, :] = elastic_members_disps

                if has_any_response(members_nodal_strains):
                    nodal_strains[0, 0] = average_nodal_responses(structure=structure, members_responses=members_nodal_strains)
                    nodal_stresses[0, 0] = average_nodal_responses(structure=structure, members_responses=members_nodal_stresses)

                # elastoplastic_members_disps = get_elastoplastic_response(
                #     load_level=load_level,
                #     phi_x=phi_x,
                #     elastic_response=elastic_members_disps_history[time_step, 0],
                #     sensitivity=members_disps_sensitivity,
                # )

                responses[time_step] = {
                    "load_levels": load_levels,
                    "nodal_disp": nodal_disp,
                    "members_disps": members_disps,
                    "members_nodal_forces": members_nodal_forces,
                    "members_nodal_strains": members_nodal_strains,
                    "members_nodal_stresses": members_nodal_stresses,
                }
                if has_any_response(members_nodal_strains):
                    responses[time_step].update(
                        {
                            "nodal_strains": nodal_strains,
                            "nodal_stresses": nodal_stresses,
                        }
                    )

    return responses


def has_any_response(array):
    answer = False
    if array.any():
        return True
    return answer


def write_static_responses_to_file(example_name, responses, desired_responses):
    for response in responses:
        if response in desired_responses:
            write_response_to_file(
                example_name=example_name,
                response=responses[response],
                response_name=response,
            )


def write_dynamic_responses_to_file(example_name, structure_type, responses, desired_responses, time_steps):
    for time_step in range(1, time_steps):
        for response in responses[time_step]:
            if response in desired_responses:
                example_name_with_time_step = os.path.join(
                    example_name,
                    structure_type,
                    "increments",
                    str(time_step),
                )
                write_response_to_file(
                    example_name=example_name_with_time_step,
                    response=responses[time_step][response],
                    response_name=response,
                )


def write_response_to_file(example_name, response, response_name):
    for increment in range(response.shape[0]):
        response_dir = os.path.join(outputs_dir, example_name, str(increment), response_name)
        os.makedirs(response_dir, exist_ok=True)
        response_elements_count = len(np.shape(response[increment]))
<<<<<<< HEAD
=======

>>>>>>> 528f6fd853ba9b1ba390cfd01c50b4b35576ff72
        if response_elements_count == 0:
            dir = os.path.join(response_dir, "0.csv")
            np.savetxt(fname=dir, X=np.array([response[increment]]), delimiter=",", fmt=f'%.{settings.output_digits}e')
        elif response_elements_count == 1:
            if response_name == "plastic_points":
                dir = os.path.join(response_dir, "0.csv")
                np.savetxt(fname=dir, X=np.array(response[increment]), delimiter=",", fmt=f'%.{settings.output_digits}e')
            else:
                dir = os.path.join(response_dir, "0.csv")
                np.savetxt(fname=dir, X=np.array(response[increment, :]), delimiter=",", fmt=f'%.{settings.output_digits}e')
        else:
            for i in range(np.shape(response[increment])[0]):
                dir = os.path.join(response_dir, f"{str(i)}.csv")
                np.savetxt(fname=dir, X=np.array(response[increment, i, :]), delimiter=",", fmt=f'%.{settings.output_digits}e')
