import os
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import numpy as np

from src.settings import settings
from .j2 import build_discretizing_points_and_gradients

mp = 150000
yield_point_num = 31
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


def visualize_shape_with_single_piece_caps(coords, mp, yield_point_num):
    points = get_state_points(mp, yield_point_num)
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection="3d")

    n_xi, n_th, _ = coords.shape

    # A) Plot the "barrel" facets
    for i in range(n_xi - 1):
        for j in range(n_th):
            jn = (j + 1) % n_th
            face = [
                coords[i, j],
                coords[i + 1, j],
                coords[i + 1, jn],
                coords[i, jn],
            ]
            poly = Poly3DCollection(
                [face], alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)  # **White/Gray Edges**
            ax.add_collection3d(poly)

    # B) Build a SINGLE polygon for each end
    ring_low = coords[0]  # shape (M,3)
    ring_high = coords[-1]  # shape (M,3)

    # **Caps now match the main shape color**
    cap_poly_low = Poly3DCollection([ring_low], alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)
    cap_poly_high = Poly3DCollection([ring_high], alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)

    ax.add_collection3d(cap_poly_low)
    ax.add_collection3d(cap_poly_high)
    for point in points:
        ax.scatter(point.mx, point.my, point.mxy, c='b', marker='o', alpha=0.5)
    # ax.scatter(all_mx, all_my, all_mxy, c='b', marker='o', alpha=0.5)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-0.7, 0.7)

    # D) Axis labels with **larger fonts + padding**
    ax.set_xlabel(r"$M_x$", fontsize=16, labelpad=15)
    ax.set_ylabel(r"$M_y$", fontsize=16, labelpad=15, rotation=90)
    ax.set_zlabel(r"$M_{xy}$", fontsize=16, labelpad=15, rotation=90)

    ax.set_title(
        "Piecewised Von Mises Yield Surface", fontsize=14
    )

    # E) Set coordinate spacing to **every 0.5**
    ax.set_xticks(np.array([-1. , -0.5,  0. ,  0.5,  1.]))
    ax.set_yticks(np.array([-1. , -0.5,  0. ,  0.5,  1.]))
    ax.set_zticks(np.array([ -0.5,  0. ,  0.5]))

    # plt.tight_layout()
    plt.show()

coords, grads = build_discretizing_points_and_gradients(xi_vals, theta_vals)
visualize_shape_with_single_piece_caps(coords, mp, yield_point_num)
