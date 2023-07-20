import os
import yaml
import numpy as np

from src.settings import settings

example_name = settings.example_name
examples_dir = "input/examples/"
output_dir = "output/examples/"

yield_surface_path = os.path.join(examples_dir, example_name, "visualization/yield_surface.csv")
yield_points_path = os.path.join(examples_dir, example_name, "visualization/yield_points.csv")
increments_path = os.path.join(examples_dir, example_name, "visualization/increments.csv")
frames_path = os.path.join(examples_dir, example_name, "members/frames2d.csv")

output_increments_path = os.path.join(output_dir, example_name)

yield_data_path = os.path.join(output_dir, example_name, "yield_data.csv")

increments_array = np.loadtxt(fname=increments_path, skiprows=1, delimiter=",", dtype=str, ndmin=1)
yield_points_array = np.loadtxt(fname=yield_points_path, skiprows=1, delimiter=",", dtype=str, ndmin=1)
yield_surface_array = np.loadtxt(fname=yield_surface_path, skiprows=1, delimiter=",", dtype=str)
frames_array = np.loadtxt(fname=frames_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
# yield_data_array = np.loadtxt(fname=yield_data_path, usecols=range(6), delimiter=",", ndmin=2, dtype=float)

SELECTED_YIELD_POINT = 4
FROM_TIME_STEP = 36

def find_subdirs(path):
    subdirs = os.listdir(path)
    subdirs_int = sorted([int(subdir) for subdir in subdirs if subdir.isdigit()])
    return subdirs_int


def get_analysis_type():
    general_file_path = os.path.join(examples_dir, example_name, "general.yaml")
    with open(general_file_path, "r") as file_path:
        general_properties = yaml.safe_load(file_path)
    if dynamic_analysis := general_properties.get("dynamic_analysis"):
        if dynamic_analysis.get("enabled"):
            analysis_type = "dynamic"
        else:
            analysis_type = "static"
    else:
        analysis_type = "static"
    return analysis_type


analysis_type = get_analysis_type()

members_count = frames_array.shape[0]

member_yield_points_count = 2
node_dofs_count = 3

if analysis_type == "static":
    if increments_array[0] == "all":
        selected_increments = find_subdirs(f"{output_increments_path}")
    else:
        selected_increments = [int(inc) for inc in increments_array]

    if yield_points_array[0] == "all":
        selected_yield_points = [i for i in range(members_count * member_yield_points_count)]
    else:
        selected_yield_points = [int(inc) for inc in yield_points_array]
    selected_figs = selected_increments

elif analysis_type == "dynamic":
    time_steps_path = os.path.join(output_dir, example_name, "inelastic", "increments")
    selected_time_steps = find_subdirs(time_steps_path)
    # selected_time_steps = [step for step in range(FROM_TIME_STEP, 50)]
    selected_yield_point = SELECTED_YIELD_POINT
    selected_figs = selected_time_steps

def get_yield_components_data():
    yield_components_count = int(yield_surface_array[0])
    if yield_components_count == 1:
        yield_components = {
            "x": yield_surface_array[1],
        }
        yield_components_dof = [
            int(yield_surface_array[2])
        ]

    elif yield_components_count == 2:
        yield_components = {
            "x": yield_surface_array[1],
            "y": yield_surface_array[2],
        }
        yield_components_dof = [
            int(yield_surface_array[3]),
            int(yield_surface_array[4]),
        ]

    elif yield_components_count == 3:
        yield_components = {
            "x": yield_surface_array[1],
            "y": yield_surface_array[2],
            "z": yield_surface_array[3],
        }
        yield_components_dof = [
            int(yield_surface_array[4]),
            int(yield_surface_array[5]),
            int(yield_surface_array[6]),
        ]
    yield_components_data = {
        "yield_components_count": yield_components_count,
        "yield_components": yield_components,
        "yield_components_dof": yield_components_dof,
    }
    return yield_components_data


def get_static_yield_points():
    capacities = get_capacities()
    yield_components_data = get_yield_components_data()
    yield_components_count = yield_components_data.get("yield_components_count")
    yield_components_dof = yield_components_data.get("yield_components_dof")

    increments_yield_points = []

    for increment in selected_increments:
        members_nodal_forces_path = os.path.join(output_dir, example_name, f"{increment}/members_nodal_forces")
        files = os.listdir(members_nodal_forces_path)
        members = sorted([int(file.replace(".csv", "")) for file in files])

        members_nodal_forces = []
        for member in members:
            member_nodal_forces = np.loadtxt(fname=f"{members_nodal_forces_path}/{member}.csv", delimiter=",", ndmin=1, dtype=float)
            members_nodal_forces.append(member_nodal_forces)

        yield_points_count = len(selected_yield_points)
        increment_yield_points = np.zeros((yield_components_count, yield_points_count))

        for yield_point_i, yield_point in enumerate(selected_yield_points):
            member = yield_point // member_yield_points_count
            member_yield_point = yield_point % member_yield_points_count
            for i, yield_component in enumerate(yield_components_dof):
                member_dof = member_yield_point * node_dofs_count + yield_component                
                increment_yield_points[i, yield_point_i] = members_nodal_forces[member][member_dof] / capacities[member][yield_component]["value"]

        if yield_components_count == 1:
            yield_points = {
                "x": increment_yield_points[0, :],
                "label": selected_yield_points,
            }
        elif yield_components_count == 2:
            yield_points = {
                "x": increment_yield_points[0, :],
                "y": increment_yield_points[1, :],
                "label": selected_yield_points,
            }
        elif yield_components_count == 3:
            yield_points = {
                "x": increment_yield_points[0, :],
                "y": increment_yield_points[1, :],
                "z": increment_yield_points[2, :],
                "label": selected_yield_points,
            }

        increments_yield_points.append(yield_points)
    return increments_yield_points


def get_dynamic_yield_points():
    capacities = get_capacities()
    yield_components_data = get_yield_components_data()
    yield_components_count = yield_components_data.get("yield_components_count")
    yield_components_dof = yield_components_data.get("yield_components_dof")

    time_steps_incs = []
    for time_step in selected_time_steps:
        time_step_path = os.path.join(output_dir, example_name, "inelastic", "increments", str(time_step))
        incs = find_subdirs(time_step_path)
        time_step_incs_state = np.zeros((yield_components_count, len(incs)))
        for inc in incs:
            members_nodal_forces_path = os.path.join(time_step_path, f"{inc}/members_nodal_forces")
            member = selected_yield_point // member_yield_points_count
            member_nodal_forces = np.loadtxt(fname=f"{members_nodal_forces_path}/{member}.csv", delimiter=",", ndmin=1, dtype=float)
            member_yield_point = selected_yield_point % member_yield_points_count
            for i, yield_component in enumerate(yield_components_dof):
                member_dof = member_yield_point * node_dofs_count + yield_component                
                time_step_incs_state[i, inc] = member_nodal_forces[member_dof] / capacities[member][yield_component]["value"]

            if yield_components_count == 1:
                incs_state = {
                    "x": time_step_incs_state[0, :],
                    "label": incs,
                }
            elif yield_components_count == 2:
                incs_state = {
                    "x": time_step_incs_state[0, :],
                    "y": time_step_incs_state[1, :],
                    "label": incs,
                }
            elif yield_components_count == 3:
                incs_state = {
                    "x": time_step_incs_state[0, :],
                    "y": time_step_incs_state[1, :],
                    "z": time_step_incs_state[2, :],
                    "label": incs,
                }

            time_steps_incs.append(incs_state)
    return time_steps_incs


def get_yield_surface():
    np_surface = [-1, -0.15, 0.15, 1, 0.15, -0.15, -1]
    mp_surface = [0, 1, 1, 0, -1, -1, 0]
    yield_surface = {
        "x": np_surface,
        "y": mp_surface,
    }
    return yield_surface


def get_capacities():
    examples_dir = "input/examples/"
    frame_members_file = "members/frames2d.csv"
    frame_members_path = os.path.join(examples_dir, example_name, frame_members_file)
    frames_array = np.loadtxt(fname=frame_members_path, usecols=range(2), delimiter=",", ndmin=1, skiprows=1, dtype=str)
    
    nonlinear_capacity_dir = f"{output_dir}/{settings.example_name}/nonlinear_capacity"
    sections = [section.replace(".csv", "") for section in os.listdir(nonlinear_capacity_dir)]
    sections_capacity = {}
    for section in sections:
        capacities = np.loadtxt(fname=os.path.join(nonlinear_capacity_dir, f"{section}.csv"), usecols=range(3), delimiter=",", ndmin=2, dtype=str)
        for i in range(capacities.shape[0]):
            if not section in sections_capacity and not sections_capacity.get(section):
                sections_capacity[section] = {}
            if not sections_capacity[section].get(int(capacities[i, 0])):
                sections_capacity[section][int(capacities[i, 0])] = {}
            sections_capacity[section][int(capacities[i, 0])]["name"] = capacities[i, 1]
            sections_capacity[section][int(capacities[i, 0])]["value"] = float(capacities[i, 2])
    frames_capacity = [sections_capacity[frames_array[f_num, 1]] for f_num in range(frames_array.shape[0])]
    return frames_capacity
