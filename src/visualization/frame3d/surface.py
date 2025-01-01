import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


from src.settings import settings



# element = 30
# element_point_num = 0
# section_name="IPB180"

# element = 106
# element_point_num = 0
# section_name="IPE270"

element = 3
element_point_num = 0
section_name="sec1"


selected_time_steps = []


@dataclass
class Point3d:
    n: float
    my: float
    mz: float


@dataclass
class Point2d:
    x: float
    y: float


def get_point3d_marker(point: Point3d):
    # marker options: "o", "x", "+", "s", "^", "v"
    # Check if the point lies on the surface
    if np.isclose(abs(point.n) + (8/9) * abs(point.my) + (8/9) * point.mz, 1, rtol=1e-1) or np.isclose((1/2) * abs(point.n) + abs(point.my) + point.mz, 1, rtol=1e-1):
        marker = "x" # Cross Marker
        color = "red"
        size = 10
    else:
        marker = "o" # Circle Marker
        color = "black"
        size = 5
    return marker, color, size


def get_point2d_marker(on_yield_surface_condition):
    # marker options: "o", "x", "+", "s", "^", "v"
    # Check if the point lies on the surface
    if on_yield_surface_condition:
        marker = "o" # Cross Marker
        color = "red"
        size = 25
    else:
        marker = "o" # Circle Marker
        color = "black"
        size = 5
    return marker, color, size


def get_responses_path():
    example_name = settings.example_name
    outputs_dir = "output/examples/"
    inelastic_response_path = os.path.join(outputs_dir, example_name, "inelastic")

    # example_path = "H:\\Doctora\\Thesis\\temp\\3d-4story-3span-dynamic-inelastic"
    # inelastic_response_path = os.path.join(example_path, "inelastic")

    aggregation_path = os.path.join(inelastic_response_path, "aggregatation")
    return aggregation_path


def get_section_capacity(section_name):
    example_name = settings.example_name
    outputs_dir = "output/examples/"
    nonlinear_capacity_path = os.path.join(outputs_dir, example_name, "nonlinear_capacity")

    # example_path = "H:\\Doctora\\Thesis\\temp\\3d-4story-3span-dynamic-inelastic"
    # nonlinear_capacity_path = os.path.join(example_path, "nonlinear_capacity")

    section_capacity_path = os.path.join(nonlinear_capacity_path, f"{section_name}.csv")
    capacities = np.loadtxt(fname=section_capacity_path, usecols=range(3), delimiter=",", ndmin=2, dtype=str)
    return capacities


def get_yield_point_responses_history(element, element_point_num):
    responses_path = get_responses_path()
    element_members_nodal_forces_path = os.path.join(responses_path, "members_nodal_forces", str(element))
    base_dofs = [0, 5, 4]
    components_dof = [base_dof + element_point_num * 6 for base_dof in base_dofs]
    n = components_dof[0]
    my = components_dof[1]
    mz = components_dof[2]
    n_history = np.loadtxt(fname=os.path.join(element_members_nodal_forces_path, f"{n}.csv"), usecols=range(1), delimiter=",", ndmin=1, dtype=float)
    my_history = np.loadtxt(fname=os.path.join(element_members_nodal_forces_path, f"{my}.csv"), usecols=range(1), delimiter=",", ndmin=1, dtype=float)
    mz_history = np.loadtxt(fname=os.path.join(element_members_nodal_forces_path, f"{mz}.csv"), usecols=range(1), delimiter=",", ndmin=1, dtype=float)
    return n_history, my_history, mz_history


def plot_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Define the ranges for n and my
    n = np.linspace(-1, 1, 100)
    my = np.linspace(-1, 1, 100)
    n, my = np.meshgrid(n, my)

    # Calculate mz based on the conditions
    mz1 = (1 - np.abs(n) - (8/9) * np.abs(my)) * (9/8)
    mz2 = 1 - (1/2) * np.abs(n) - np.abs(my)

    # Apply the conditions
    mask1 = np.abs(n) >= 0.2
    mask2 = np.abs(n) < 0.2

    mz1 = np.where(mask1, mz1, np.nan)
    mz2 = np.where(mask2, mz2, np.nan)

    # Ensure mz is within valid range
    mz1[mz1 < 0] = np.nan
    mz1[mz1 + (8/9) * np.abs(my) > 1] = np.nan
    mz2[mz2 < 0] = np.nan
    mz2[mz2 + np.abs(my) > 1] = np.nan

    # Plot the surfaces
    stride_size = 5
    ax.plot_surface(my, mz1, n, color="b", alpha=0.3, edgecolor="b", linewidth=0.2, rstride=stride_size, cstride=stride_size)
    ax.plot_surface(my, -mz1, n, color="b", alpha=0.3, edgecolor="b", linewidth=0.2, rstride=stride_size, cstride=stride_size)
    ax.plot_surface(my, mz2, n, color="b", alpha=0.3, edgecolor="b", linewidth=0.2, rstride=stride_size, cstride=stride_size)
    ax.plot_surface(my, -mz2, n, color="b", alpha=0.3, edgecolor="b", linewidth=0.2, rstride=stride_size, cstride=stride_size)

    # Labels and title
    ax.set_xlabel("My")
    ax.set_ylabel("Mz")
    ax.set_zlabel("N")
    ax.set_title("Force State In 3D N-My-Mz")

    section_capacity = get_section_capacity(section_name=section_name)
    n_capacity = float(section_capacity[0, 2])
    my_capacity = float(section_capacity[1, 2])
    mz_capacity = float(section_capacity[2, 2])
    n_history, my_history, mz_history = get_yield_point_responses_history(element=element, element_point_num=element_point_num)
    if selected_time_steps:
        selected_n_history = n_history[selected_time_steps[0]:selected_time_steps[1] + 1] / n_capacity
        selected_my_history = my_history[selected_time_steps[0]:selected_time_steps[1] + 1] / my_capacity
        selected_mz_history = mz_history[selected_time_steps[0]:selected_time_steps[1] + 1] / mz_capacity
    else:
        selected_n_history = n_history / n_capacity
        selected_my_history = my_history / my_capacity
        selected_mz_history = mz_history / mz_capacity

    # Plot the point
    points = []
    for i in range(selected_n_history.shape[0]):
        points.append(Point3d(n=selected_n_history[i], my=selected_my_history[i], mz=selected_mz_history[i]))

    for point in points:
        marker, color, size = get_point3d_marker(point)
        ax.scatter(point.n, point.my, point.mz, color=color, s=size, label="Point", marker=marker)
    ax.view_init(azim=145, elev=20)
    plt.show()


def plot_2d_n_my():
    n = np.linspace(-1, 1, 500)
    my_pos = (1 - np.abs(n)) * (9/8)
    my_neg = -(1 - np.abs(n)) * (9/8)
    my2_pos = 1 - (1/2) * np.abs(n)
    my2_neg = -(1 - (1/2) * np.abs(n))

    my_pos[np.abs(n) < 0.2] = np.nan  # Mask invalid values for |n| < 0.2
    my_neg[np.abs(n) < 0.2] = np.nan  # Mask invalid values for |n| < 0.2
    my2_pos[np.abs(n) >= 0.2] = np.nan  # Mask invalid values for |n| ≥ 0.2
    my2_neg[np.abs(n) >= 0.2] = np.nan  # Mask invalid values for |n| ≥ 0.2

    plt.plot(n, my_pos, color="black")
    plt.plot(n, my_neg, color="black")
    plt.plot(n, my2_pos, color="black")
    plt.plot(n, my2_neg, color="black")

    plt.xlabel("N")
    plt.ylabel("My")
    plt.title("Force State In N-My Plane")

    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)


    section_capacity = get_section_capacity(section_name=section_name)
    n_capacity = float(section_capacity[0, 2])
    my_capacity = float(section_capacity[1, 2])
    n_history, my_history, _ = get_yield_point_responses_history(element=element, element_point_num=element_point_num)
    if selected_time_steps:
        selected_n_history = n_history[selected_time_steps[0]:selected_time_steps[1] + 1] / n_capacity
        selected_my_history = my_history[selected_time_steps[0]:selected_time_steps[1] + 1] / my_capacity
    else:
        selected_n_history = n_history / n_capacity
        selected_my_history = my_history / my_capacity

    # Plot the point
    points = []
    for i in range(selected_n_history.shape[0]):
        points.append(Point2d(x=selected_n_history[i], y=selected_my_history[i]))

    for point in points:
        on_yield_surface_condition = np.isclose(0.5 * abs(point.x) + abs(point.y), 1) or np.isclose(abs(point.x) + (8/9) * abs(point.y), 1)
        marker, color, size = get_point2d_marker(on_yield_surface_condition)
        plt.scatter(point.x, point.y, color=color, s=size, label="Point", marker=marker)


    plt.show()


def plot_2d_n_mz():
    n = np.linspace(-1, 1, 500)
    mz_pos = (1 - np.abs(n)) * (9/8)
    mz_neg = -(1 - np.abs(n)) * (9/8)
    mz2_pos = 1 - (1/2) * np.abs(n)
    mz2_neg = -(1 - (1/2) * np.abs(n))

    mz_pos[np.abs(n) < 0.2] = np.nan  # Mask invalid values for |n| < 0.2
    mz_neg[np.abs(n) < 0.2] = np.nan  # Mask invalid values for |n| < 0.2
    mz2_pos[np.abs(n) >= 0.2] = np.nan  # Mask invalid values for |n| ≥ 0.2
    mz2_neg[np.abs(n) >= 0.2] = np.nan  # Mask invalid values for |n| ≥ 0.2

    plt.plot(n, mz_pos, color="black")
    plt.plot(n, mz_neg, color="black")
    plt.plot(n, mz2_pos, color="black")
    plt.plot(n, mz2_neg, color="black")

    plt.xlabel("N")
    plt.ylabel("Mz")
    plt.title("Force State In N-Mz Plane")

    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)


    section_capacity = get_section_capacity(section_name=section_name)
    n_capacity = float(section_capacity[0, 2])
    mz_capacity = float(section_capacity[2, 2])
    n_history, _, mz_history = get_yield_point_responses_history(element=element, element_point_num=element_point_num)
    if selected_time_steps:
        selected_n_history = n_history[selected_time_steps[0]:selected_time_steps[1] + 1] / n_capacity
        selected_mz_history = mz_history[selected_time_steps[0]:selected_time_steps[1] + 1] / mz_capacity
    else:
        selected_n_history = n_history / n_capacity
        selected_mz_history = mz_history / mz_capacity

    # Plot the point
    points = []
    for i in range(selected_n_history.shape[0]):
        points.append(Point2d(x=selected_n_history[i], y=selected_mz_history[i]))

    for point in points:
        on_yield_surface_condition = np.isclose(0.5 * abs(point.x) + abs(point.y), 1) or np.isclose(abs(point.x) + (8/9) * abs(point.y), 1)
        marker, color, size = get_point2d_marker(on_yield_surface_condition)
        plt.scatter(point.x, point.y, color=color, s=size, label="Point", marker=marker)

    plt.show()


def plot_2d_my_mz():
    my = np.linspace(-1, 1, 500)

    # Calculate mz based on the conditions
    mz1 = 1 - np.abs(my)

    # Plot the lines based on the equations
    plt.plot(my, mz1, color="black")
    plt.plot(my, -mz1, color="black")

    # Add labels and title
    plt.xlabel("My")
    plt.ylabel("Mz")
    plt.title("Force State In My-Mz Plane")
    
    # Add grid and axes
    plt.grid(True)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)

    section_capacity = get_section_capacity(section_name=section_name)
    my_capacity = float(section_capacity[1, 2])
    mz_capacity = float(section_capacity[2, 2])
    _, my_history, mz_history = get_yield_point_responses_history(element=element, element_point_num=element_point_num)
    if selected_time_steps:
        selected_my_history = my_history[selected_time_steps[0]:selected_time_steps[1] + 1] / my_capacity
        selected_mz_history = mz_history[selected_time_steps[0]:selected_time_steps[1] + 1] / mz_capacity
    else:
        selected_my_history = my_history / my_capacity
        selected_mz_history = mz_history / mz_capacity

    # Plot the point
    points = []
    for i in range(selected_my_history.shape[0]):
        points.append(Point2d(x=selected_my_history[i], y=selected_mz_history[i]))

    for point in points:
        on_yield_surface_condition = np.isclose(abs(point.x) + abs(point.y), 1, rtol=2e-2)
        marker, color, size = get_point2d_marker(on_yield_surface_condition)
        plt.scatter(point.x, point.y, color=color, s=size, label="Point", marker=marker)


    plt.show()


plot_2d_n_my()


# plot_3d()
