import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.settings import settings
from .shrunk import build_discretizing_points, compute_convex_hull_2d


mp = 6000
yield_point_num = 21
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


def get_state_points(mp, yield_point_num):
    example_path = get_example_path()  # Make sure this function is defined
    increments = find_subdirs(example_path)  # Make sure this function is defined

    # Prepare lists to store all points
    all_mx = []
    all_my = []
    all_mxy = []

    for inc in increments:
        inc_nodal_moments_array_path = os.path.join(example_path, str(inc), "yield_points_forces", "0.csv")
        moments = pd.read_csv(inc_nodal_moments_array_path, header=None)
        
        # Extract and normalize moments
        mx = moments.iloc[::3, 0].values / mp
        my = moments.iloc[1::3, 0].values / mp
        mxy = moments.iloc[2::3, 0].values / (mp)

        # Add to our collection
        all_mx.append(mx[yield_point_num])
        all_my.append(my[yield_point_num])
        all_mxy.append(mxy[yield_point_num])

    points = []
    for i in range(len(all_mx)):
        points.append(
            Point3d(
                mx=all_mx[i],
                my=all_my[i],
                mxy=all_mxy[i],
            )
        )

    return points


def visualize_shape_projection(coords, mp, yield_point_num, view):
    """
    Plot 3 shapes in the (Mx, My) plane:
      1) Original shape (alpha=1.0)
      2) Shrunk shape (alpha=0.8)
      3) Minimum shrunk shape (alpha=0.5)
    """
    points = get_state_points(mp, yield_point_num)
    # --- Flatten coords (N,3) -> keep only (Mx, My) ---
    all_points = coords.reshape(-1, 3)
    if view == "mx-my":
        selected_view = all_points[:, :2]
    elif view =="mx-mxy":
        selected_view = all_points[:, [0, 2]]
    elif view =="my-mxy":
        selected_view = all_points[:, 1:]

    # --- Get the 2D convex hull of all points, then "close" it ---
    hull_points = compute_convex_hull_2d(selected_view)
    hull_points_closed = np.vstack([hull_points, hull_points[0]])

    # --- Prepare a figure ---
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define each alpha + style
    shapes = [
        (1.0,  "--k",  "gray",  "original"),
        (0.8,  "-k",  "red",  "softened"),
    ]

    # --- Plot each shape ---
    for alpha_val, line_style, fill_color, label_str in shapes:
        # Scale points by alpha about origin
        scaled = alpha_val * hull_points_closed
        ax.plot(scaled[:, 0], scaled[:, 1], line_style, label=label_str)
        # # Optional fill
        # ax.fill(scaled[:, 0], scaled[:, 1], alpha=0.1, color=fill_color)

    if view == "mx-my":
        ax.set_xlabel(r"$M_x$", fontsize=14)
        ax.set_ylabel(r"$M_y$", fontsize=14)
        for point in points:
            ax.scatter(point.mx, point.my, c='b', marker='o', alpha=0.5)
    elif view =="mx-mxy":
        ax.set_xlabel(r"$M_x$", fontsize=14)
        ax.set_ylabel(r"$M_xy$", fontsize=14)
        for point in points:
            ax.scatter(point.mx, point.mxy, c='b', marker='o', alpha=0.5)
    elif view =="my-mxy":
        ax.set_xlabel(r"$M_y$", fontsize=14)
        ax.set_ylabel(r"$M_xy$", fontsize=14)
        for point in points:
            ax.scatter(point.my, point.mxy, c='b', marker='o', alpha=0.5)

    # --- Cosmetics ---
    ax.set_aspect("equal", "box")
    # ax.set_title("2D Projection With Isotropic Softening (Three Shapes)", fontsize=13)
    # Remove top and right spines (box effect)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc="best", fontsize=14)
    plt.show()


if __name__ == "__main__":
    coords = build_discretizing_points(xi_vals, theta_vals)
    visualize_shape_projection(coords, mp, yield_point_num, "mx-my")
    visualize_shape_projection(coords, mp, yield_point_num, "mx-mxy")
    visualize_shape_projection(coords, mp, yield_point_num, "my-mxy")
