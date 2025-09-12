import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

alpha = 0.7
edgecolor = "#2b3a52"
facecolor = "#bfd8ff"

###############################################################################
# 1) Build the main discretizing points & gradients
###############################################################################
def build_discretizing_points_and_gradients(xi_values, theta_values):
    """
    Returns:
      coords[i,j] = (sx, sy, txy) from Eq. (17).
      grads[i,j]  = gradient [df/dsx, df/dsy, df/dtxy] from Eq. (18).
    """
    n_xi = len(xi_values)
    n_th = len(theta_values)

    coords = np.zeros((n_xi, n_th, 3))
    grads = np.zeros((n_xi, n_th, 3))

    for i, xi in enumerate(xi_values):
        for j, theta in enumerate(theta_values):
            # Eq. (17): Intersection with plane-stress von Mises surface
            denom = 3.0 * (1.0 + np.sin(theta) ** 2)
            radicand = (4.0 - xi**2) / denom
            if radicand < 0.0:
                radicand = 0.0

            root_val = np.sqrt(radicand)

            sx = 0.5 * xi + 0.5 * root_val * np.cos(theta)
            sy = 0.5 * xi - 0.5 * root_val * np.cos(theta)
            txy = 0.5 * np.sqrt(2.0) * np.sin(theta) * root_val

            coords[i, j] = (sx, sy, txy)

            # Eq. (18): gradient of f = sx^2 + sy^2 - sx*sy + 3*(txy^2) - 1
            # => grad(f) = [2sx - sy, 2sy - sx, 6txy]
            dfd_sx = 2.0 * sx - sy
            dfd_sy = 2.0 * sy - sx
            dfd_txy = 6.0 * txy

            grads[i, j] = (dfd_sx, dfd_sy, dfd_txy)

    return coords, grads


###############################################################################
# 2) Visualize the shape with journal-level aesthetics
###############################################################################
def visualize_shape_with_single_piece_caps(coords):
    """
    Show the piecewise-linear shape from xi in [-1.95, +1.95],
    and add a SINGLE polygon cap at each boundary ring.
    """

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

    # C) Adjust axes and journal-style settings
    all_points = coords.reshape(-1, 3)
    # x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
    # y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
    # z_min, z_max = np.min(all_points[:, 2]), np.max(all_points[:, 2])

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

    # print(f"{np.floor(x_min)=}")
    # print(f"{np.floor(y_min)=}")
    # print(f"{np.floor(z_min)=}")

    # print(f"{np.ceil(x_max)=}")
    # print(f"{np.ceil(y_max)=}")
    # print(f"{np.ceil(z_max)=}")

    # print(f"{np.arange(np.floor(x_min), np.ceil(x_max) + 0.5, 0.5)=}")
    # print(f"{np.arange(np.floor(y_min), np.ceil(y_max) + 0.5, 0.5)=}")
    # print(f"{np.arange(np.floor(z_min), np.ceil(z_max) + 0.5, 0.5)=}")

    # E) Set coordinate spacing to **every 0.5**
    ax.set_xticks(np.array([-1. , -0.5,  0. ,  0.5,  1.]))
    ax.set_yticks(np.array([-1. , -0.5,  0. ,  0.5,  1.]))
    ax.set_zticks(np.array([ -0.5,  0. ,  0.5]))

    plt.show()


###############################################################################
# 3) Example usage
###############################################################################
if __name__ == "__main__":
    # Use xi in [-1.95, +1.95], no extension beyond that
    xi_vals = np.linspace(-1.95, 1.95, 12)
    theta_vals = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    # Build main discretizing points
    coords, grads = build_discretizing_points_and_gradients(xi_vals, theta_vals)

    # Visualize: "barrel" plus single-piece polygon caps
    visualize_shape_with_single_piece_caps(coords)
