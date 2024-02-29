import os
import numpy as np

from .functions import get_elastoplastic_response
from .settings import settings
from .analysis.initial_analysis import AnalysisType

outputs_dir = "output/examples/"


def calculate_responses(initial_analysis, inelastic_analysis=None):
    if initial_analysis.analysis_type is AnalysisType.STATIC:
        responses = calculate_static_responses(initial_analysis, inelastic_analysis)
    if initial_analysis.analysis_type is AnalysisType.DYNAMIC:
        responses = calculate_dynamic_responses(initial_analysis, inelastic_analysis)
    return responses


def calculate_static_responses(initial_analysis, inelastic_analysis=None):
    structure = initial_analysis.structure
    if structure.is_inelastic:
        phi_x_history = inelastic_analysis.plastic_vars["phi_pms_history"]
        load_level_history = inelastic_analysis.plastic_vars["load_level_history"]
        increments_count = len(load_level_history)

        load_levels = np.zeros([increments_count, 1], dtype=object)

        nodal_disp_sensitivity = initial_analysis.nodal_disp_sensitivity
        nodal_disp = np.zeros([increments_count, 1], dtype=object)

        members_nodal_forces_sensitivity = initial_analysis.members_nodal_forces_sensitivity
        members_nodal_forces = np.zeros([increments_count, structure.members_count], dtype=object)

        members_nodal_strains_sensitivity = initial_analysis.members_nodal_strains_sensitivity
        members_nodal_strains = np.zeros([increments_count, structure.members_count], dtype=object)
        nodal_strains = np.zeros([increments_count, 1], dtype=object)

        members_nodal_stresses_sensitivity = initial_analysis.members_nodal_stresses_sensitivity
        members_nodal_stresses = np.zeros([increments_count, structure.members_count], dtype=object)
        nodal_stresses = np.zeros([increments_count, 1], dtype=object)

        members_disps_sensitivity = initial_analysis.members_disps_sensitivity
        members_disps = np.zeros([increments_count, structure.members_count], dtype=object)

        for i in range(increments_count):
            phi_x = phi_x_history[i]
            load_level = load_level_history[i]
            load_levels[i, 0] = np.matrix([[load_level]])

            elastoplastic_nodal_disp = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=initial_analysis.elastic_nodal_disp,
                sensitivity=nodal_disp_sensitivity,
            )
            nodal_disp[i, 0] = elastoplastic_nodal_disp[0, 0]

            elastoplastic_members_nodal_forces = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=initial_analysis.elastic_members_nodal_forces,
                sensitivity=members_nodal_forces_sensitivity,
            )
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
            for j in range(structure.members_count):
                members_nodal_forces[i, j] = elastoplastic_members_nodal_forces[j, 0]
                members_nodal_strains[i, j] = elastoplastic_members_nodal_strains[j, 0]
                members_nodal_stresses[i, j] = elastoplastic_members_nodal_stresses[j, 0]

            elastoplastic_members_disps = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=initial_analysis.elastic_members_disps,
                sensitivity=members_disps_sensitivity,
            )
            if has_any_response(members_nodal_strains):
                nodal_strains[i, 0] = average_nodal_responses(structure=structure, members_responses=elastoplastic_members_nodal_strains.T)
                nodal_stresses[i, 0] = average_nodal_responses(structure=structure, members_responses=elastoplastic_members_nodal_stresses.T)

            for j in range(structure.members_count):
                members_disps[i, j] = elastoplastic_members_disps[j, 0]

        responses = {
            "load_levels": load_levels,
            "nodal_disp": nodal_disp,
            "members_disps": members_disps,
            "members_nodal_forces": members_nodal_forces,
            "members_nodal_strains": members_nodal_strains,
            "members_nodal_stresses": members_nodal_stresses,
        }
        if has_any_response(members_nodal_strains):
            responses.update(
                {
                    "nodal_strains": nodal_strains,
                    "nodal_stresses": nodal_stresses,
                }
            )

    elif not structure.is_inelastic:  # if structure is elastic
        nodal_disp = np.zeros([1, 1], dtype=object)
        members_disps = np.zeros([1, structure.members_count], dtype=object)
        members_nodal_forces = np.zeros([1, structure.members_count], dtype=object)
        members_nodal_strains = np.zeros([1, structure.members_count], dtype=object)
        members_nodal_stresses = np.zeros([1, structure.members_count], dtype=object)
        nodal_strains = np.zeros([1, 1], dtype=object)
        nodal_stresses = np.zeros([1, 1], dtype=object)

        nodal_disp[0, 0] = structure.limits["load_limit"][0] * initial_analysis.elastic_nodal_disp[0, 0]
        for i in range(structure.members_count):
            members_disps[0, i] = structure.limits["load_limit"][0] * initial_analysis.elastic_members_disps[i, 0]
            members_nodal_forces[0, i] = structure.limits["load_limit"][0] * initial_analysis.elastic_members_nodal_forces[i, 0]
            members_nodal_strains[0, i] = structure.limits["load_limit"][0] * initial_analysis.elastic_members_nodal_strains[i, 0]
            members_nodal_stresses[0, i] = structure.limits["load_limit"][0] * initial_analysis.elastic_members_nodal_stresses[i, 0]

        if has_any_response(members_nodal_strains):
            nodal_strains[0, 0] = average_nodal_responses(structure=structure, members_responses=members_nodal_strains)
            nodal_stresses[0, 0] = average_nodal_responses(structure=structure, members_responses=members_nodal_stresses)

        responses = {
            "nodal_disp": nodal_disp,
            "members_disps": members_disps,
            "members_nodal_forces": members_nodal_forces,
            "members_nodal_strains": members_nodal_strains,
            "members_nodal_stresses": members_nodal_stresses,
        }
        if has_any_response(members_nodal_strains):
            responses.update(
                {
                    "nodal_strains": nodal_strains,
                    "nodal_stresses": nodal_stresses,
                }
            )
    return responses


def average_nodal_responses(structure, members_responses):
    comp_count = 3  # response_components_count
    nodes_map = structure.nodes_map
    nodal_responses = np.matrix(np.zeros((structure.nodes_count * comp_count, 1)))
    for node in structure.nodes:
        node_sum_response = np.matrix(np.zeros((comp_count, 1)))
        for attached_member in nodes_map[node.num].attached_members:
            start = comp_count * attached_member.member_node_num
            end = comp_count * (attached_member.member_node_num + 1)
            member_node_response = members_responses[0, attached_member.member.num][start:end]
            node_sum_response += member_node_response
        node_average_response = node_sum_response / len(nodes_map[node.num].attached_members)
        nodal_responses[comp_count * node.num:comp_count * (node.num + 1), 0] = node_average_response
    return nodal_responses


def calculate_dynamic_responses(initial_analysis, inelastic_analysis):
    structure = initial_analysis.structure
    if structure.is_inelastic:
        plastic_vars_history = inelastic_analysis.plastic_vars_history
        final_inc_phi_pms_history = inelastic_analysis.final_inc_phi_pms_history

        nodal_disp_sensitivity_history = initial_analysis.nodal_disp_sensitivity_history
        members_nodal_forces_sensitivity_history = initial_analysis.members_nodal_forces_sensitivity_history
        members_disps_sensitivity_history = initial_analysis.members_disps_sensitivity_history

        elastic_members_nodal_forces_history = initial_analysis.elastic_members_nodal_forces_history
        elastic_members_disps_history = initial_analysis.elastic_members_disps_history
        elastic_nodal_disp_history = initial_analysis.elastic_nodal_disp_history

        responses = np.matrix(np.zeros((initial_analysis.time_steps, 1), dtype=object))
        for time_step in range(1, initial_analysis.time_steps):
            plastic_vars = plastic_vars_history[time_step, 0]
            phi_pms_history = plastic_vars["phi_pms_history"]
            load_level_history = plastic_vars["load_level_history"]
            final_inc_phi_pms_prev = final_inc_phi_pms_history[time_step - 1, 0]

            increments_count = len(load_level_history)
            load_levels = np.zeros([increments_count, 1], dtype=object)

            nodal_disp_sensitivity = nodal_disp_sensitivity_history[time_step, 0]
            nodal_disp = np.zeros([increments_count, 1], dtype=object)

            members_nodal_forces_sensitivity = members_nodal_forces_sensitivity_history[time_step, 0]
            members_nodal_forces = np.zeros([increments_count, structure.members_count], dtype=object)

            members_disps_sensitivity = members_disps_sensitivity_history[time_step, 0]
            members_disps = np.zeros([increments_count, structure.members_count], dtype=object)
            for i in range(increments_count):
                phi_pms = phi_pms_history[i] + final_inc_phi_pms_prev
                load_level = load_level_history[i]

                load_levels[i, 0] = np.matrix([[load_level]])

                elastoplastic_nodal_disp = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_pms,
                    elastic_response=elastic_nodal_disp_history[time_step, 0],
                    sensitivity=nodal_disp_sensitivity,
                )
                nodal_disp[i, 0] = elastoplastic_nodal_disp[0, 0]

                elastoplastic_members_nodal_forces = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_pms,
                    elastic_response=elastic_members_nodal_forces_history[time_step, 0],
                    sensitivity=members_nodal_forces_sensitivity,
                )
                for j in range(structure.members_count):
                    members_nodal_forces[i, j] = elastoplastic_members_nodal_forces[j, 0]

                elastoplastic_members_disps = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_pms,
                    elastic_response=elastic_members_disps_history[time_step, 0],
                    sensitivity=members_disps_sensitivity,
                )
                for j in range(structure.members_count):
                    members_disps[i, j] = elastoplastic_members_disps[j, 0]

            responses[time_step, 0] = {
                "nodal_disp": nodal_disp,
                "members_nodal_forces": members_nodal_forces,
                "members_disps": members_disps,
                "load_levels": load_levels,
            }
    else:
        load_limit = structure.limits["load_limit"][0]
        elastic_members_nodal_forces_history = initial_analysis.elastic_members_nodal_forces_history
        elastic_members_disps_history = initial_analysis.elastic_members_disps_history
        elastic_nodal_disp_history = initial_analysis.elastic_nodal_disp_history
        responses = np.matrix(np.zeros((initial_analysis.time_steps, 1), dtype=object))

        # for elastic analysis, there is only one increment so for responses size we use 1.
        increments_count = 1

        for time_step in range(1, initial_analysis.time_steps):
            nodal_disp = np.zeros([increments_count, 1], dtype=object)
            members_nodal_forces = np.zeros([increments_count, structure.members_count], dtype=object)
            members_disps = np.zeros([increments_count, structure.members_count], dtype=object)
            # elastoplastic_nodal_disp = get_elastoplastic_response(
            #     load_level=load_level,
            #     phi_x=phi_x,
            #     elastic_response=elastic_nodal_disp_history[time_step, 0],
            #     sensitivity=nodal_disp_sensitivity,
            # )
            # nodal_disp[i, 0] = elastoplastic_nodal_disp[0, 0]
            for i in range(increments_count):
                nodal_disp[i, 0] = elastic_nodal_disp_history[time_step, 0][0, 0] * load_limit
                # elastoplastic_members_nodal_forces = get_elastoplastic_response(
                #     load_level=load_level,
                #     phi_x=phi_x,
                #     elastic_response=elastic_members_nodal_forces_history[time_step, 0],
                #     sensitivity=members_nodal_forces_sensitivity,
                # )
                elastic_members_nodal_forces = elastic_members_nodal_forces_history[time_step, 0] * load_limit
                for j in range(structure.members_count):
                    members_nodal_forces[i, j] = elastic_members_nodal_forces[j, 0]

                # elastoplastic_members_disps = get_elastoplastic_response(
                #     load_level=load_level,
                #     phi_x=phi_x,
                #     elastic_response=elastic_members_disps_history[time_step, 0],
                #     sensitivity=members_disps_sensitivity,
                # )
                elastic_members_disps = elastic_members_disps_history[time_step, 0] * load_limit
                for j in range(structure.members_count):
                    members_disps[i, j] = elastic_members_disps[j, 0]

                responses[time_step, 0] = {
                    "nodal_disp": nodal_disp,
                    "members_nodal_forces": members_nodal_forces,
                    "members_disps": members_disps,
                }
    return responses


def has_any_response(array):
    answer = False
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if isinstance(array[i, j], np.matrix):
                if array[i, j].any():
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
        for response in responses[time_step, 0]:
            if response in desired_responses:
                example_name_with_time_step = os.path.join(
                    example_name,
                    structure_type,
                    "increments",
                    str(time_step),
                )
                write_response_to_file(
                    example_name=example_name_with_time_step,
                    response=responses[time_step, 0][response],
                    response_name=response,
                )


def write_response_to_file(example_name, response, response_name):
    for increment in range(response.shape[0]):
        response_dir = os.path.join(outputs_dir, example_name, str(increment), response_name)
        os.makedirs(response_dir, exist_ok=True)
        for i in range(response.shape[1]):
            dir = os.path.join(response_dir, f"{str(i)}.csv")
            np.savetxt(fname=dir, X=np.array(response[increment, i]), delimiter=",", fmt=f'%.{settings.output_digits}e')
