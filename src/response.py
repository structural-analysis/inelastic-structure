import os
import numpy as np
from src.functions import get_elastoplastic_response
# from src.analysis import Analysis

outputs_dir = "output/examples/"


def calculate_responses(analysis):
    if analysis.type == "static":
        responses = calculate_static_responses(analysis)
    if analysis.type == "dynamic":
        responses = calculate_dynamic_responses(analysis)
    return responses


def calculate_static_responses(analysis):
    structure = analysis.structure
    if structure.is_inelastic:
        pms_history = analysis.plastic_vars["pms_history"]
        load_level_history = analysis.plastic_vars["load_level_history"]
        increments_count = len(load_level_history)
        phi = structure.phi

        load_levels = np.zeros([increments_count, 1], dtype=object)

        nodal_disp_sensitivity = analysis.nodal_disp_sensitivity
        nodal_disp = np.zeros([increments_count, 1], dtype=object)

        members_nodal_forces_sensitivity = analysis.members_nodal_forces_sensitivity
        members_nodal_forces = np.zeros([increments_count, structure.members.num], dtype=object)

        members_disps_sensitivity = analysis.members_disps_sensitivity
        members_disps = np.zeros([increments_count, structure.members.num], dtype=object)

        for i in range(increments_count):
            pms = pms_history[i]
            load_level = load_level_history[i]
            phi_x = phi * pms

            load_levels[i, 0] = np.matrix([[load_level]])

            elastoplastic_nodal_disp = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=analysis.elastic_nodal_disp,
                sensitivity=nodal_disp_sensitivity,
            )
            nodal_disp[i, 0] = elastoplastic_nodal_disp[0, 0]

            elastoplastic_members_nodal_forces = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=analysis.elastic_members_nodal_forces,
                sensitivity=members_nodal_forces_sensitivity,
            )
            for j in range(structure.members.num):
                members_nodal_forces[i, j] = elastoplastic_members_nodal_forces[j, 0]

            elastoplastic_members_disps = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=analysis.elastic_members_disps,
                sensitivity=members_disps_sensitivity,
            )
            for j in range(structure.members.num):
                members_disps[i, j] = elastoplastic_members_disps[j, 0]
        responses = {
            "nodal_disp": nodal_disp,
            "members_nodal_forces": members_nodal_forces,
            "members_disps": members_disps,
            "load_levels": load_levels,
        }
    else:
        nodal_disp = np.zeros([1, 1], dtype=object)
        members_disps = np.zeros([1, structure.members.num], dtype=object)
        members_nodal_forces = np.zeros([1, structure.members.num], dtype=object)
        internal_moments = np.zeros([1, structure.members.num], dtype=object)
        top_internal_strains = np.zeros([1, structure.members.num], dtype=object)
        bottom_internal_strains = np.zeros([1, structure.members.num], dtype=object)
        top_internal_stresses = np.zeros([1, structure.members.num], dtype=object)
        bottom_internal_stresses = np.zeros([1, structure.members.num], dtype=object)

        nodal_disp[0, 0] = structure.limits["load_limit"][0] * analysis.elastic_nodal_disp[0, 0]
        for i in range(structure.members.num):
            members_disps[0, i] = structure.limits["load_limit"][0] * analysis.elastic_members_disps[i, 0]
            members_nodal_forces[0, i] = structure.limits["load_limit"][0] * analysis.elastic_members_nodal_forces[i, 0]
            internal_moments[0, i] = structure.limits["load_limit"][0] * analysis.elastic_members_internal_moments[i, 0]
            top_internal_strains[0, i] = structure.limits["load_limit"][0] * analysis.elastic_members_top_internal_strains[i, 0]
            bottom_internal_strains[0, i] = structure.limits["load_limit"][0] * analysis.elastic_members_bottom_internal_strains[i, 0]
            top_internal_stresses[0, i] = structure.limits["load_limit"][0] * analysis.elastic_members_top_internal_stresses[i, 0]
            bottom_internal_stresses[0, i] = structure.limits["load_limit"][0] * analysis.elastic_members_bottom_internal_stresses[i, 0]

        responses = {
            "nodal_disp": nodal_disp,
            "members_disps": members_disps,
            "members_nodal_forces": members_nodal_forces,
            "internal_moments": internal_moments,
            "top_internal_strains": top_internal_strains,
            "bottom_internal_strains": bottom_internal_strains,
            "top_internal_stresses": top_internal_stresses,
            "bottom_internal_stresses": bottom_internal_stresses,
        }

    # if analysis.type == "dynamic":
    #     members_nodal_forces = np.zeros([increments_num, structure.members.num], dtype=object)

    return responses


def calculate_dynamic_responses(analysis):
    structure = analysis.structure
    if structure.is_inelastic:
        plastic_vars_history = analysis.plastic_vars_history
        nodal_disp_sensitivity_history = analysis.nodal_disp_sensitivity_history
        members_nodal_forces_sensitivity_history = analysis.members_nodal_forces_sensitivity_history
        members_disps_sensitivity_history = analysis.members_disps_sensitivity_history

        elastic_members_nodal_forces_history = analysis.elastic_members_nodal_forces_history
        elastic_members_disps_history = analysis.elastic_members_disps_history
        elastic_nodal_disp_history = analysis.elastic_nodal_disp_history

        plastic_multipliers_history = analysis.plastic_multipliers_history

        responses = np.matrix(np.zeros((analysis.time_steps, 1), dtype=object))
        for time_step in range(1, analysis.time_steps):
            plastic_vars = plastic_vars_history[time_step, 0]
            pms_history = plastic_vars["pms_history"]
            load_level_history = plastic_vars["load_level_history"]
            increments_count = len(load_level_history)
            phi = structure.phi

            load_levels = np.zeros([increments_count, 1], dtype=object)

            nodal_disp_sensitivity = nodal_disp_sensitivity_history[time_step, 0]
            nodal_disp = np.zeros([increments_count, 1], dtype=object)

            members_nodal_forces_sensitivity = members_nodal_forces_sensitivity_history[time_step, 0]
            members_nodal_forces = np.zeros([increments_count, structure.members.num], dtype=object)

            members_disps_sensitivity = members_disps_sensitivity_history[time_step, 0]
            members_disps = np.zeros([increments_count, structure.members.num], dtype=object)
            for i in range(increments_count):
                pms = pms_history[i]
                load_level = load_level_history[i]
                plastic_multipliers = pms + plastic_multipliers_history[time_step, 0]
                phi_x = phi * plastic_multipliers

                load_levels[i, 0] = np.matrix([[load_level]])

                elastoplastic_nodal_disp = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_x,
                    elastic_response=elastic_nodal_disp_history[time_step, 0],
                    sensitivity=nodal_disp_sensitivity,
                )
                nodal_disp[i, 0] = elastoplastic_nodal_disp[0, 0]
                elastoplastic_members_nodal_forces = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_x,
                    elastic_response=elastic_members_nodal_forces_history[time_step, 0],
                    sensitivity=members_nodal_forces_sensitivity,
                )
                for j in range(structure.members.num):
                    members_nodal_forces[i, j] = elastoplastic_members_nodal_forces[j, 0]

                elastoplastic_members_disps = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_x,
                    elastic_response=elastic_members_disps_history[time_step, 0],
                    sensitivity=members_disps_sensitivity,
                )
                for j in range(structure.members.num):
                    members_disps[i, j] = elastoplastic_members_disps[j, 0]

            responses[time_step, 0] = {
                "nodal_disp": nodal_disp,
                "members_nodal_forces": members_nodal_forces,
                "members_disps": members_disps,
                "load_levels": load_levels,
            }
    else:
        load_limit = structure.limits["load_limit"][0]
        elastic_members_nodal_forces_history = analysis.elastic_members_nodal_forces_history
        elastic_members_disps_history = analysis.elastic_members_disps_history
        elastic_nodal_disp_history = analysis.elastic_nodal_disp_history
        responses = np.matrix(np.zeros((analysis.time_steps, 1), dtype=object))

        # for elastic analysis, there is only one increment so for responses size we use 1.
        increments_count = 1

        for time_step in range(1, analysis.time_steps):
            nodal_disp = np.zeros([increments_count, 1], dtype=object)
            members_nodal_forces = np.zeros([increments_count, structure.members.num], dtype=object)
            members_disps = np.zeros([increments_count, structure.members.num], dtype=object)
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
                for j in range(structure.members.num):
                    members_nodal_forces[i, j] = elastic_members_nodal_forces[j, 0]

                # elastoplastic_members_disps = get_elastoplastic_response(
                #     load_level=load_level,
                #     phi_x=phi_x,
                #     elastic_response=elastic_members_disps_history[time_step, 0],
                #     sensitivity=members_disps_sensitivity,
                # )
                elastic_members_disps = elastic_members_disps_history[time_step, 0] * load_limit
                for j in range(structure.members.num):
                    members_disps[i, j] = elastic_members_disps[j, 0]

                responses[time_step, 0] = {
                    "nodal_disp": nodal_disp,
                    "members_nodal_forces": members_nodal_forces,
                    "members_disps": members_disps,
                }
    return responses


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
            print(f"{response[increment, i]=}")
            np.savetxt(fname=dir, X=np.array(response[increment, i]), delimiter=",")


    # members_yield_points_count = 0
    # for member in structure.members:
    #     members_yield_points_count += len(member.yield_points)

    # yield_points_data = np.matrix(np.zeros((members_yield_points_count, 6)))
    # yp_counter = 0
    # for i, member in enumerate(structure.members):
    #     member_yield_points_count = len(member.yield_points)
    #     for j in range(member_yield_points_count):
    #         if member.has_axial_yield:
    #             components_count = 2
    #             components_dofs = [0, 2]
    #             yield_capacity = [member.section.ap, member.section.mp]
    #             node_dof = 3
    #             yield_points_data[yp_counter, 0:6] = np.array([[
    #                 components_count,
    #                 components_dofs[0],
    #                 components_dofs[1],
    #                 yield_capacity[0],
    #                 yield_capacity[1],
    #                 node_dof,
    #             ]])
    #         else:
    #             components_count = 1
    #             components_dofs = [2]
    #             yield_capacity = [member.section.mp]
    #             node_dof = 3
    #             yield_points_data[yp_counter, 0:4] = np.array([[
    #                 components_count,
    #                 components_dofs[0],
    #                 yield_capacity[0],
    #                 node_dof,
    #             ]])
    #         yp_counter += 1
    # np.savetxt(fname=yield_points_data_path, X=yield_points_data, delimiter=",")
