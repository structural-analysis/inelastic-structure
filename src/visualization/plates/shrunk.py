import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

alpha = 0.7
edgecolor = "#2b3a52"
facecolor = "#bfd8ff"

###############################################################################
# 1) Build the main discretizing points & gradients
###############################################################################
def build_discretizing_points(xi_values, theta_values):
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

    return coords


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


def compute_convex_hull_2d(points_2d):
    """
    A simple 'monotone chain' algorithm to compute the convex hull of a set 
    of 2D points. Returns the hull points in counterclockwise order.
    """

    # Sort points lex order (by x, then by y)
    points_sorted = sorted(points_2d, key=lambda x: (x[0], x[1]))

    # Build lower hull
    lower = []
    for p in points_sorted:
        while len(lower) >= 2:
            cross_val = ((lower[-1][0] - lower[-2][0]) * (p[1] - lower[-2][1]) -
                         (lower[-1][1] - lower[-2][1]) * (p[0] - lower[-2][0]))
            if cross_val <= 0:
                lower.pop()
            else:
                break
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points_sorted):
        while len(upper) >= 2:
            cross_val = ((upper[-1][0] - upper[-2][0]) * (p[1] - upper[-2][1]) -
                         (upper[-1][1] - upper[-2][1]) * (p[0] - upper[-2][0]))
            if cross_val <= 0:
                upper.pop()
            else:
                break
        upper.append(p)

    # Remove duplicate end-points on each list
    # The last point of each list is the same as the first point of the other list
    lower.pop()
    upper.pop()
    hull = lower + upper  # Concatenate

    return np.array(hull)

def visualize_shape_projection_mx_my(coords):
    """
    Plots only the 'external edges' in the Mx-My plane (ignoring Mxy).
    The edges are found by computing the convex hull of (Mx, My) points,
    so you see only the outer boundary in 2D. The hull is explicitly
    closed by appending the first point again at the end.
    """
    # Flatten coords (N, 3) -> keep only (Mx, My)
    all_points = coords.reshape(-1, 3)
    xy_2d = all_points[:, :2]

    # Compute convex hull in 2D
    hull_points = compute_convex_hull_2d(xy_2d)

    # Append the first hull point to the end to close the polygon
    hull_points_closed = np.concatenate([hull_points, hull_points[:1]], axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1], "-k", linewidth=1.5)
    ax.fill(hull_points_closed[:, 0], hull_points_closed[:, 1], alpha=0.1, color="gray")

    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$M_x$", fontsize=14)
    ax.set_ylabel(r"$M_y$", fontsize=14, rotation=90)
    ax.set_title("2D Projection (External Edges) in Mx–My Plane", fontsize=13)
    plt.show()

def visualize_shape_projection_mx_my_3shapes(coords):
    """
    Plot 3 shapes in the (Mx, My) plane:
      1) Original shape (alpha=1.0)
      2) Shrunk shape (alpha=0.8)
      3) Minimum shrunk shape (alpha=0.5)
    """

    # --- Flatten coords (N,3) -> keep only (Mx, My) ---
    all_points = coords.reshape(-1, 3)
    xy_2d = all_points[:, :2]

    # --- Get the 2D convex hull of all points, then "close" it ---
    hull_points = compute_convex_hull_2d(xy_2d)
    hull_points_closed = np.vstack([hull_points, hull_points[0]])

    # --- Prepare a figure ---
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define each alpha + style
    shapes = [
        (1.0,  "-k",  "gray",  "α=1.0"),
        (0.8,  "--k",  "red",  r"α=$α_s$"),
        (0.5,  ":k",  "green", r"α=$α_{\min}$"),
    ]

    # --- Plot each shape ---
    for alpha_val, line_style, fill_color, label_str in shapes:
        # Scale points by alpha about origin
        scaled = alpha_val * hull_points_closed
        ax.plot(scaled[:, 0], scaled[:, 1], line_style, label=label_str)
        # # Optional fill
        # ax.fill(scaled[:, 0], scaled[:, 1], alpha=0.1, color=fill_color)

    # --- Cosmetics ---
    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$M_x$", fontsize=14)
    ax.set_ylabel(r"$M_y$", fontsize=14)
    # ax.set_title("2D Projection With Isotropic Softening (Three Shapes)", fontsize=13)
    # Remove top and right spines (box effect)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc="best", fontsize=14)
    plt.show()

###############################################################################
# 3) Example usage
###############################################################################
if __name__ == "__main__":
    # Use xi in [-1.95, +1.95], no extension beyond that
    xi_vals = np.linspace(-1.95, 1.95, 12)
    theta_vals = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    # Build main discretizing points
    coords = build_discretizing_points(xi_vals, theta_vals)
    visualize_shape_projection_mx_my_3shapes(coords)
