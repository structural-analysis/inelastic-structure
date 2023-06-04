import enum
import numpy as np
import os

from src.settings import settings
example_name = settings.example_name


def find_subdirs(path):
    subdirs = os.listdir(path)
    subdirs_int = sorted([int(subdir) for subdir in subdirs])
    return subdirs_int


examples_dir = "input/examples/"
output_dir = "output/examples/"

yield_surface_path = os.path.join(examples_dir, example_name, "visualization/yield_surface.csv")
yield_points_path = os.path.join(examples_dir, example_name, "visualization/yield_points.csv")
increments_path = os.path.join(examples_dir, example_name, "visualization/increments.csv")
frames_path = os.path.join(examples_dir, example_name, "members/frames.csv")

output_increments_path = os.path.join(output_dir, example_name)

yield_data_path = os.path.join(output_dir, example_name, "yield_data.csv")

increments_array = np.loadtxt(fname=increments_path, skiprows=1, delimiter=",", dtype=str, ndmin=1)
yield_points_array = np.loadtxt(fname=yield_points_path, skiprows=1, delimiter=",", dtype=str, ndmin=1)
yield_surface_array = np.loadtxt(fname=yield_surface_path, skiprows=1, delimiter=",", dtype=str)
frames_array = np.loadtxt(fname=frames_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
# yield_data_array = np.loadtxt(fname=yield_data_path, usecols=range(6), delimiter=",", ndmin=2, dtype=float)

members_count = frames_array.shape[0]

member_yield_points_count = 2
node_dofs_count = 3

if increments_array[0] == "all":
    selected_increments = find_subdirs(output_increments_path)
else:
    selected_increments = [int(inc) for inc in increments_array]

if yield_points_array[0] == "all":
    selected_yield_points = [i for i in range(members_count * member_yield_points_count)]
else:
    selected_yield_points = [int(inc) for inc in yield_points_array]


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


def get_yield_points():
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


def get_yield_surface():
    np_surface = [-1, -0.77, 0.77, 1, 0.77, -0.77, -1]
    mp_surface = [0, 1, 1, 0, -1, -1, 0]
    yield_surface = {
        "x": np_surface,
        "y": mp_surface,
    }
    return yield_surface


def get_capacities():
    examples_dir = "input/examples/"
    frame_members_file = "members/frames.csv"
    frame_members_path = os.path.join(examples_dir, example_name, frame_members_file)
    frames_array = np.loadtxt(fname=frame_members_path, usecols=range(1), delimiter=",", ndmin=1, skiprows=1, dtype=str)
    
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
    frames_capacity = [sections_capacity[frames_array[f_num]] for f_num in range(frames_array.shape[0])]
    return frames_capacity
