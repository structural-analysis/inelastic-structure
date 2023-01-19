import numpy as np
import os

from src.settings import settings
example_name = settings.example_name


def find_subdir_count(path):
    count1 = 0
    for _, dirs, _ in os.walk(path):
        count1 += len(dirs)

    return count1


examples_dir = "input/examples/"
output_dir = "output/examples/"

yield_surface_path = os.path.join(examples_dir, example_name, "visualization/yield_surface.csv")
yield_points_path = os.path.join(examples_dir, example_name, "visualization/yield_points.csv")
increments_path = os.path.join(examples_dir, example_name, "visualization/increments.csv")
frames_path = os.path.join(examples_dir, example_name, "members/frames.csv")

output_increments_path = os.path.join(output_dir, example_name)
total_increments_count = find_subdir_count(output_increments_path)
yield_data_path = os.path.join(output_dir, example_name, "yield_data.csv")

increments_array = np.loadtxt(fname=increments_path, skiprows=1, delimiter=",", dtype=str)
yield_points_array = np.loadtxt(fname=yield_points_path, skiprows=1, delimiter=",", dtype=str)
yield_surface_array = np.loadtxt(fname=yield_surface_path, skiprows=1, delimiter=",", dtype=str)
frames_array = np.loadtxt(fname=frames_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)
yield_data_array = np.loadtxt(fname=yield_data_path, usecols=range(6), delimiter=",", ndmin=2, dtype=float)

members_count = frames_array.shape[0]

member_yield_points_count = 2
node_dofs_count = 3

if increments_array[0] == "all":
    selected_increments_count = total_increments_count

if yield_points_array[0] == "all":
    selected_yield_points_count = members_count * member_yield_points_count


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


def get_yield_points(selected_increments_count, selected_yield_points_count):
    yield_components_data = get_yield_components_data()
    yield_components_count = yield_components_data.get("yield_components_count")
    yield_components_dof = yield_components_data.get("yield_components_dof")

    increments_yield_points = []

    for increment in range(selected_increments_count):
        members_nodal_forces_path = os.path.join(output_dir, example_name, f"{increment}/members_nodal_forces.csv")
        members_nodal_forces = np.loadtxt(fname=members_nodal_forces_path, delimiter=",", ndmin=2, dtype=float)

        yield_points_count = members_nodal_forces.shape[1]
        increment_yield_points = np.zeros((yield_components_count, yield_points_count))

        for member_num in range(yield_points_count):
            for member_yield_point in range(member_yield_points_count):
                for yield_component in range(len(yield_components_dof)):
                    dof = member_yield_point * node_dofs_count + yield_component
                    increment_yield_points[yield_component, member_num] = members_nodal_forces[dof, member_num]

    if yield_components_count == 1:
        yield_points = {
            "x": increment_yield_points[0, :],
            "label": [i for i in range(members_count)],
        }
    elif yield_components_count == 2:
        yield_points = {
            "x": increment_yield_points[0, :],
            "y": increment_yield_points[1, :],
            "label": [i for i in range(members_count)],
        }
    elif yield_components_count == 3:
        yield_points = {
            "x": increment_yield_points[0, :],
            "y": increment_yield_points[1, :],
            "z": increment_yield_points[2, :],
            "label": [i for i in range(members_count)],
        }

    increments_yield_points.append(yield_points)
    # x_points = [0.15, 0.3, 0.45, 0.6, 1]
    # y_points = [0.2, 0.3, 0.7, 1.0, 0.5]
    # points_label = [0, 1, 2, 3, 4]
    # yield_points = {
    #     "x": x_points,
    #     "y": y_points,
    #     "label": points_label,
    # }
    return increments_yield_points


def get_yield_surface():
    np_surface = [-1, -0.77, 0.77, 1, 0.77, -0.77, -1]
    mp_surface = [0, 1, 1, 0, -1, -1, 0]
    yield_surface = {
        "x": np_surface,
        "y": mp_surface,
    }
    return yield_surface
