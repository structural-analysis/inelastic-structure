import os
import numpy as np
from src.analysis import Analysis

outputs_dir = "output/examples/"


def calculate_responses(analysis: Analysis):
    structure = analysis.structure
    if structure.is_inelastic:
        pms_history = analysis.plastic_vars["pms_history"]
        load_level_history = analysis.plastic_vars["load_level_history"]
        increments_count = len(load_level_history)
        phi = structure.phi

        load_levels = np.zeros([increments_count, 1], dtype=object)

        nodal_disps_sensitivity = analysis.nodal_disps_sensitivity
        nodal_disps = np.zeros([increments_count, 1], dtype=object)

        members_forces_sensitivity = analysis.members_forces_sensitivity
        members_forces = np.zeros([increments_count, structure.members.num], dtype=object)

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
                sensitivity=nodal_disps_sensitivity,
            )
            nodal_disps[i, 0] = elastoplastic_nodal_disp[0, 0]

            elastoplastic_members_forces = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=analysis.elastic_members_nodal_forces,
                sensitivity=members_forces_sensitivity,
            )
            for j in range(structure.members.num):
                members_forces[i, j] = elastoplastic_members_forces[j, 0]

            elastoplastic_members_disps = get_elastoplastic_response(
                load_level=load_level,
                phi_x=phi_x,
                elastic_response=analysis.elastic_members_disps,
                sensitivity=members_disps_sensitivity,
            )
            for j in range(structure.members.num):
                members_disps[i, j] = elastoplastic_members_disps[j, 0]
        responses = {
            "nodal_disps": nodal_disps,
            "members_forces": members_forces,
            "members_disps": members_disps,
            "load_levels": load_levels,
        }
    else:
        nodal_disps = structure.limits["load_limit"][0] * analysis.elastic_nodal_disp
        members_disps = structure.limits["load_limit"][0] * analysis.elastic_members_disps
        members_forces = structure.limits["load_limit"][0] * analysis.elastic_members_nodal_forces
        internal_moments = structure.limits["load_limit"][0] * analysis.elastic_members_internal_moments
        top_internal_strains = structure.limits["load_limit"][0] * analysis.elastic_members_top_internal_strains
        bottom_internal_strains = structure.limits["load_limit"][0] * analysis.elastic_members_bottom_internal_strains
        top_internal_stresses = structure.limits["load_limit"][0] * analysis.elastic_members_top_internal_stresses
        bottom_internal_stresses = structure.limits["load_limit"][0] * analysis.elastic_members_bottom_internal_stresses

        responses = {
            "nodal_disps": nodal_disps,
            "members_disps": members_disps,
            "members_forces": members_forces,
            "internal_moments": internal_moments,
            "top_internal_strains": top_internal_strains,
            "bottom_internal_strains": bottom_internal_strains,
            "top_internal_stresses": top_internal_stresses,
            "bottom_internal_stresses": bottom_internal_stresses,
        }

    # if analysis.type == "dynamic":
    #     members_forces = np.zeros([increments_num, structure.members.num], dtype=object)

    return responses


def write_responses_to_file(example_name, responses, desired_responses):
    for response in responses:
        if response in desired_responses:
            write_response_to_file(
                example_name=example_name,
                response=responses[response],
                response_name=response,
            )


def write_response_to_file(example_name, response, response_name):
    for increment in range(response.shape[0]):
        response_dir = os.path.join(outputs_dir, example_name, str(increment), response_name)
        os.makedirs(response_dir, exist_ok=True)
        for i in range(response.shape[1]):
            dir = os.path.join(response_dir, f"{str(i)}.csv")
            np.savetxt(fname=dir, X=np.array(response[increment, i]), delimiter=",")


def get_elastoplastic_response(load_level, phi_x, elastic_response, sensitivity):
    scaled_elastic_response = np.matrix(np.dot(load_level, elastic_response))
    plastic_response = sensitivity * phi_x
    elastoplastic_response = scaled_elastic_response + plastic_response
    return elastoplastic_response

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
