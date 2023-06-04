from typing import Tuple
import matplotlib.pyplot as plt

from src.settings import settings
from src.visualization.get_data import (
    get_yield_points,
    get_yield_surface,
    get_yield_components_data,
    selected_increments,
)

examples_dir = "input/examples/"
example_name = settings.example_name


def draw_yield_surface(ax, yield_surface):
    x_list = yield_surface.get("x")
    y_list = yield_surface.get("y")
    z_list = yield_surface.get("z")
    if not z_list:
        ax.plot(x_list, y_list, c="black")
    else:
        ax.plot_wireframe(x_list, y_list, z_list, c="black")


def draw_lines(ax, yield_surface):
    x_list = yield_surface.get("x")
    y_list = yield_surface.get("y")
    ax.plot(x_list, y_list, c="black")


def draw_yield_points(ax, yield_points):
    x_points = yield_points.get("x")
    y_points = yield_points.get("y")
    z_points = yield_points.get("z")
    points_labels = yield_points.get("label")
    if not z_points:
        for i, point_label in enumerate(points_labels):
            ax.scatter(x_points[i], y_points[i], color='red', s=10)
            ax.text(x_points[i], y_points[i], s=f"{point_label}", size=10, color='k')
    else:
        for i, point_label in range(len(points_labels)):
            ax.scatter(x_points[i], y_points[i], z_points[i], color='red', s=10)
            ax.text(x_points[i], y_points[i], z_points[i], s=f"{point_label}", size=10, color='k')


def draw_yield_condition(ax, yield_points, yield_surface):
    draw_yield_points(
        ax,
        yield_points,
    )
    draw_yield_surface(
        ax,
        yield_surface,
    )


def edit_plot_labes(yield_components):
    plt.xlabel(yield_components.get("x"))
    plt.ylabel(yield_components.get("y"))
    plt.title('Yield Surface')


def calculate_figure_grid_size(total_diagrams_count, maximum_diagram_in_row):
    grid_size_y = min(maximum_diagram_in_row, total_diagrams_count)
    grid_size_x = total_diagrams_count // maximum_diagram_in_row + 1
    return (grid_size_x, grid_size_y)


def add_diagram_to_figure(
    main_figure,
    figure_grid_size: Tuple[int, int],
    position: Tuple[int, int],
    dimension="2d"
):
    x_grid_num, y_grid_num = figure_grid_size[0], figure_grid_size[1]
    position_num = y_grid_num * (position[0]) + position[1] + 1
    # add_subplot(rows, columns, position from left up to right down)
    # !!! position starts from 1 not 0.
    if dimension == "2d":
        ax = main_figure.add_subplot(x_grid_num, y_grid_num, position_num)
    elif dimension == "3d":
        ax = main_figure.add_subplot(x_grid_num, y_grid_num, position_num, projection='3d')
    return ax


def generate_increments_yield_surfaces(selected_increments, yield_components_count, yield_components, increments_yield_points, yield_surface):
    increments_count = len(selected_increments)
    fig = plt.figure(figsize=plt.figaspect(0.5))
    maximum_diagram_in_row = 4
    figure_grid_size = calculate_figure_grid_size(increments_count, maximum_diagram_in_row)

    dimension = f"{yield_components_count}d"
    # dimension = "3d"
    for i, increment in enumerate(selected_increments):
        x_position = i // maximum_diagram_in_row
        y_position = i % figure_grid_size[1]
        position = (x_position, y_position)
        ax = add_diagram_to_figure(fig, figure_grid_size, position, dimension)
        draw_yield_condition(ax, increments_yield_points[i], yield_surface)
        edit_plot_labes(yield_components)
        ax.set_title(f"Increment {increment}", fontsize=10)
    plt.tight_layout(h_pad=1)
    plt.show()


increments_yield_points = get_yield_points()
yield_components_data = get_yield_components_data()
yield_components_count = yield_components_data.get("yield_components_count")
yield_components = yield_components_data.get("yield_components")
yield_surface = get_yield_surface()

generate_increments_yield_surfaces(selected_increments, yield_components_count, yield_components, increments_yield_points, yield_surface)


plt.show()


def surface_3d_example():
    import numpy as np
    import matplotlib.pyplot as plt

    point = np.array([1, 2, 3])
    normal = np.array([1, 1, 2])

    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)

    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))

    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    z2 = 2 * (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    # plot the surface
    fig = plt.figure()
    plt3d = fig.add_subplot(1, 1, 1, projection='3d')

    plt3d.plot_surface(xx, yy, z, rstride=1, cstride=1, alpha=0.5, color="gray")
    plt3d.plot_surface(xx, yy, z2, rstride=1, cstride=1, alpha=0.5, color="gray")
    plt.show()
