import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from src.models.sections.functions import build_discretizing_points_and_gradients_hill

alpha = 0.7
edgecolor = "#2b3a52"
facecolor = "#bfd8ff"

###############################################################################
# Visualize the shape with journal-level aesthetics (Hill surface)
###############################################################################
def visualize_shape_with_single_piece_caps(coords):
    """
    Plot the piecewise-linear Hill yield surface given the computed coordinates.
    Includes the barrel facets and single polygon caps at each end.
    """
    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111, projection="3d")

    n_xi, n_th, _ = coords.shape

    # A) Plot the barrel facets connecting each adjacent ring
    for i in range(n_xi - 1):
        for j in range(n_th):
            jp = (j + 1) % n_th  # next theta index (wrap around)
            # Define a quadrilateral face between ring i and ring i+1
            face = [
                coords[i, j],
                coords[i + 1, j],
                coords[i + 1, jp],
                coords[i, jp],
            ]
            poly = Poly3DCollection([face], alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)
            ax.add_collection3d(poly)

    # B) Single polygon cap for the top (last ring) and bottom (first ring)
    ring_bottom = coords[0]    # xi minimum ring (bottom cap)
    ring_top    = coords[-1]   # xi maximum ring (top cap)
    cap_poly_bottom = Poly3DCollection([ring_bottom], alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)
    cap_poly_top    = Poly3DCollection([ring_top], alpha=alpha, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_collection3d(cap_poly_bottom)
    ax.add_collection3d(cap_poly_top)

    # C) Set axes limits for clarity
    # all_points = coords.reshape(-1, 3)
    # ax.set_xlim(np.min(all_points[:, 0]), np.max(all_points[:, 0]))
    # ax.set_ylim(np.min(all_points[:, 1]), np.max(all_points[:, 1]))
    # ax.set_zlim(np.min(all_points[:, 2]), np.max(all_points[:, 2]))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # D) Axis labels (with larger fonts and padding)
    ax.set_xlabel(r"$M_x$", fontsize=16, labelpad=15)
    ax.set_ylabel(r"$M_y$", fontsize=16, labelpad=15, rotation=90)
    ax.set_zlabel(r"$M_{xy}$", fontsize=16, labelpad=15, rotation=90)
    ax.set_title("Piecewise-Linear Hill Yield Surface", fontsize=14)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Define Hill yield parameters (example values)
    M0x, M0y, M0xy = 1, 0.5, 0.5    # yield capacities for Mx, My, and Mxy
    rho = 1.0   # anisotropy cross-coupling parameter
    gamma = 2 # exponent (2 for quadratic Hill yield criterion)

    # Generate discretized surface points
    xi_vals = np.linspace(-0.95, 0.95, 11) * np.sqrt(4.0 / (2.0 - rho))  # xi from -xi_max*0.95 to +xi_max*0.95
    theta_vals = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    coords, _ = build_discretizing_points_and_gradients_hill(xi_vals, theta_vals, M0x, M0y, M0xy, rho, gamma)

    # Visualize the piecewise-linear Hill yield surface
    visualize_shape_with_single_piece_caps(coords)
