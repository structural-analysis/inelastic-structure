import os
import numpy as np
from src.settings import settings

outputs_dir = "output/examples/"

desired_responses = [
    "nodal_disp",
    "members_nodal_forces",
    "members_disps",
]


def aggregate_dynamic_responses(structure_type_path):
    increments_path = os.path.join(structure_type_path, "increments")
    time_steps = len(get_inner_folders(increments_path)) + 1
    time_step_list = [str(time_step) for time_step in range(1, time_steps)]
    responses = initialize_responses(increments_path)

    for time_step in time_step_list:
        time_step_path = os.path.join(increments_path, time_step)
        increments = get_inner_folders(time_step_path)
        final_increment = max([int(increment) for increment in increments])

        for response in desired_responses:
            response_path = os.path.join(time_step_path, str(final_increment), response)
            elements = get_inner_folders(response_path)
            for element in elements:
                element_path = os.path.join(response_path, element)
                element_name = element.replace(".csv", "")
                result = np.loadtxt(fname=element_path, usecols=range(1), delimiter=",", ndmin=2, skiprows=0, dtype=float)
                dofs_count = result.shape[0]
                for dof in range(dofs_count):
                    if time_step == "1":
                        responses[response][element_name][str(dof)] = np.zeros((time_steps, 1))
                    responses[response][element_name][str(dof)][int(time_step), 0] = result[dof]
    return responses


def get_inner_folders(path):
    dirs_list = os.listdir(path)
    return dirs_list


def initialize_responses(increments_path):
    responses = {}
    for response in desired_responses:
        responses[response] = {}
        time_step_path = os.path.join(increments_path, "1")
        increments = get_inner_folders(time_step_path)
        final_increment = max([int(increment) for increment in increments])
        response_path = os.path.join(time_step_path, str(final_increment), response)
        elements = get_inner_folders(response_path)
        for element in elements:
            element_name = element.replace(".csv", "")
            responses[response][element_name] = {}
    return responses


def write_responses(responses, structure_type_path):
    aggregate_path = os.path.join(structure_type_path, "aggregatation")
    for response in responses:
        response_path = os.path.join(aggregate_path, response)
        for element in responses[response]:
            element_path = os.path.join(response_path, element)
            for dof in responses[response][element]:
                dof_path = os.path.join(element_path, f"{dof}.csv")
                dof_response = responses[response][element][dof]
                os.makedirs(element_path, exist_ok=True)
                np.savetxt(fname=dof_path, X=dof_response, delimiter=",", fmt=f'%.{settings.output_digits}e')


def get_structure_types(example_name):
    structure_types = []
    if os.path.isdir(os.path.join(outputs_dir, example_name, "elastic")):
        structure_types.append("elastic")
    if os.path.isdir(os.path.join(outputs_dir, example_name, "inelastic")):
        structure_types.append("inelastic")
    return structure_types


def aggregate_responses(example_name):
    structure_types = get_structure_types(example_name)
    for structure_type in structure_types:
        structure_type_path = os.path.join(outputs_dir, example_name, structure_type)
        responses = aggregate_dynamic_responses(structure_type_path)
        write_responses(responses, structure_type_path)
