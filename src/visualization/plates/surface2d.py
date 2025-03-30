import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.settings import settings
from .shrunk import build_discretizing_points, compute_convex_hull_2d


mp = 6000
yield_point_num = 6 # 5, 6
target_incs = [100, 200, 400]
every_n_incs = 20
xi_vals = np.linspace(-1.95, 1.95, 12)
theta_vals = np.linspace(0, 2 * np.pi, 16, endpoint=False)
alpha = 0.7
edgecolor = "#2b3a52"
facecolor = "#bfd8ff"


@dataclass
class Point3d:
    mx: float
    my: float
    mxy: float

def get_example_path():
    example_name = settings.example_name
    outputs_dir = "output/examples/"
    example_path = os.path.join(outputs_dir, example_name)
    return example_path

def find_subdirs(path):
    subdirs = os.listdir(path)
    subdirs_int = sorted([int(subdir) for subdir in subdirs if subdir.isdigit()])
    return subdirs_int

def interpolate(k0, a0, k2, a2, k1):
    return a0 + (a2 - a0) * (k1 - k0) / (k2 - k0)

def get_surface_size_for_yield_point_in_inc(example_path, yield_point_num, target_inc):
    curvatures_file_path = os.path.join(example_path, str(target_inc), "yield_points_mises_curvatures/0.csv")
    target_curvature = np.loadtxt(fname=curvatures_file_path, usecols=range(1), delimiter=",", ndmin=2, dtype=str)[yield_point_num]
    surface_size = interpolate(0, 1, 5, 0.2, float(target_curvature[0]))
    return surface_size

def get_state_points(mp, yield_point_num, target_inc):
    example_path = get_example_path()  # Ensure this function is defined
    increments = find_subdirs(example_path)  # Ensure this function is defined
    surface_size = get_surface_size_for_yield_point_in_inc(example_path, yield_point_num, target_inc)
    
    # Convert increments to integers and filter up to target_inc
    increments = sorted([int(inc) for inc in increments if int(inc) <= target_inc])

    # Select one increment per n increments
    filtered_increments = [inc for i, inc in enumerate(increments) if i % every_n_incs == 0]

    # Ensure target_inc is included
    if target_inc not in filtered_increments:
        filtered_increments.append(target_inc)
        filtered_increments.sort()  # Keep order sorted

    all_mx, all_my, all_mxy = [], [], []

    for inc in filtered_increments:
        inc_nodal_moments_array_path = os.path.join(example_path, str(inc), "yield_points_forces", "0.csv")
        
        if not os.path.exists(inc_nodal_moments_array_path):
            print(f"Warning: Missing file for increment {inc}, skipping...")
            continue
        
        moments = pd.read_csv(inc_nodal_moments_array_path, header=None)
        
        # Extract and normalize moments
        mx = moments.iloc[::3, 0].values / mp
        my = moments.iloc[1::3, 0].values / mp
        mxy = moments.iloc[2::3, 0].values / mp

        if yield_point_num < len(mx):  # Ensure index is within bounds
            all_mx.append(mx[yield_point_num])
            all_my.append(my[yield_point_num])
            all_mxy.append(mxy[yield_point_num])
        else:
            print(f"Warning: yield_point_num {yield_point_num} out of bounds for increment {inc}")

    # Convert lists into Point3d objects
    points = [Point3d(mx=mx, my=my, mxy=mxy) for mx, my, mxy in zip(all_mx, all_my, all_mxy)]

    return points, surface_size


def visualize_shape_projection(coords, mp, yield_point_num, target_incs):
    """
    Plot 3 shapes for different increments in a single figure:
      - Rows: Different target increments
      - Columns: Different views ("Mx-My", "Mx-Mxy", "My-Mxy")
    """
    views = ["mx-my", "mx-mxy", "my-mxy"]
    num_rows = len(target_incs)
    num_cols = len(views)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 6 * num_rows))  # Adjust figure size

    all_points = coords.reshape(-1, 3)  # Flatten coordinates

    for row, target_inc in enumerate(target_incs):
        points, surface_size = get_state_points(mp, yield_point_num, target_inc)
        for col, view in enumerate(views):
            ax = axs[row, col] if num_rows > 1 else axs[col]  # Handle single-row case

            # Select view
            if view == "mx-my":
                selected_view = all_points[:, :2]
                xlabel, ylabel = r"$M_x$", r"$M_y$"
            elif view == "mx-mxy":
                selected_view = all_points[:, [0, 2]]
                xlabel, ylabel = r"$M_x$", r"$M_{xy}$"
            elif view == "my-mxy":
                selected_view = all_points[:, 1:]
                xlabel, ylabel = r"$M_y$", r"$M_{xy}$"

            # Compute Convex Hull
            hull_points = compute_convex_hull_2d(selected_view)
            hull_points_closed = np.vstack([hull_points, hull_points[0]])

            # Define styles
            shapes = [
                (1.0, "--k", "gray", "original"),
                (surface_size, "-k", "red", "softened"),
            ]

            # Plot each shape
            for alpha_val, line_style, fill_color, label_str in shapes:
                scaled = alpha_val * hull_points_closed
                ax.plot(scaled[:, 0], scaled[:, 1], line_style, label=label_str)

            # Plot yield points
            for point in points:
                if view == "mx-my":
                    ax.scatter(point.mx, point.my, c='b', marker='o', alpha=0.5)
                elif view == "mx-mxy":
                    ax.scatter(point.mx, point.mxy, c='b', marker='o', alpha=0.5)
                elif view == "my-mxy":
                    ax.scatter(point.my, point.mxy, c='b', marker='o', alpha=0.5)

            # Set labels and aesthetics
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_aspect("equal", "box")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc="best", fontsize=10)
            ax.set_title(f"Inc: {target_inc}, View: {view}", fontsize=14)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    coords = build_discretizing_points(xi_vals, theta_vals)
    visualize_shape_projection(coords, mp, yield_point_num, target_incs)
