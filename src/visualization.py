from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import os

from src.settings import settings

example_name = settings.example_name

yield_components = {
    "x": "Np",
    "y": "Mp",
    "z": "M2p"
}


def get_yield_points():
    x_points = [0.15, 0.3, 0.45, 0.6, 1]
    y_points = [0.2, 0.3, 0.7, 1.0, 0.5]
    z_points = [1, 1, 1, 1, 1]
    points_label = [0, 1, 2, 3, 4]
    yield_points = {
        "x": x_points,
        "y": y_points,
        "z": z_points,
        "label": points_label,
    }
    return yield_points


def get_yield_surface():
    np_surface = [-1, -0.77, 0.77, 1, 0.77, -0.77, -1]
    # y values
    mp_surface = [0, 1, 1, 0, -1, -1, 0]
    z_surface = [0, 1, 1, 0, -1, -1, 0]
    yield_surface = {
        "x": np_surface,
        "y": mp_surface,
        "z": z_surface,
    }
    return yield_surface


def generate_lines(ax, yield_surface):
    x_list = yield_surface.get("x")
    y_list = yield_surface.get("y")
    z_list = yield_surface.get("z")
    if not z_list:
        ax.plot(x_list, y_list, c="black")
    else:
        ax.plot_wireframe(x_list, y_list, z_list, c="black")


def draw_yield_points(ax, yield_points):
    x_points = yield_points.get("x")
    y_points = yield_points.get("y")
    z_points = yield_points.get("z")
    points_labels = yield_points.get("label")
    if not z_points:
        for i in range(len(points_labels)):
            ax.scatter(x_points[i], y_points[i], color='red', s=10)
            ax.text(x_points[i], y_points[i], s=f"{i}", size=10, color='k')
    else:
        for i in range(len(points_labels)):
            ax.scatter(x_points[i], y_points[i], z_points[i], color='red', s=10)
            ax.text(x_points[i], y_points[i], z_points[i], s=f"{i}", size=10, color='k')
    # ax.scatter(x_points, y_points, s=10, c="red")
    # for i, txt in enumerate(points_labels):
    #     ax.annotate(txt, (x_points[i], y_points[i]))


def draw_yield_condition(ax, yield_points, yield_surface):
    # points = yield_points.get("x"), yield_points.get("y"), yield_points.get("label")
    draw_yield_points(
        ax,
        yield_points,
    )
    generate_lines(
        ax,
        yield_surface,
    )


def edit_plot_labes(yield_components):
    plt.xlabel(yield_components.get("x"))
    plt.ylabel(yield_components.get("y"))
    plt.title('Yield Surface')


def calculate_figure_grid_size(total_diagrams_num, maximum_diagram_in_row):
    grid_size_y = min(maximum_diagram_in_row, total_diagrams_num)
    grid_size_x = total_diagrams_num // maximum_diagram_in_row + 1
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


def generate_increments_yield_surfaces(increments_num, yield_components, yield_points, yield_surface):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    maximum_diagram_in_row = 4
    figure_grid_size = calculate_figure_grid_size(increments_num, maximum_diagram_in_row)
    dimension = "2d" if len(yield_components) == 2 else "3d"
    # dimension = "3d"
    for increment in range(increments_num):
        x_position = increment // maximum_diagram_in_row
        y_position = increment % figure_grid_size[1]
        position = (x_position, y_position)
        ax = add_diagram_to_figure(fig, figure_grid_size, position, dimension)
        draw_yield_condition(ax, yield_points, yield_surface)
        edit_plot_labes(yield_components)
        ax.set_title(f"Increment {increment}", fontsize=10)
    plt.tight_layout(h_pad=1)
    plt.show()


yield_points = get_yield_points()
yield_surface = get_yield_surface()
generate_increments_yield_surfaces(10, yield_components, yield_points, yield_surface)


def generate_elements(frames_array, nodes_array):
    for frame_label, frame_element in enumerate(frames_array):
        beginning_node = int(frame_element[1])
        end_node = int(frame_element[2])
        beginning_x = int(nodes_array[beginning_node][0])
        beginning_y = int(nodes_array[beginning_node][1])
        end_x = int(nodes_array[end_node][0])
        end_y = int(nodes_array[end_node][1])
        x_list = [beginning_x, end_x]
        y_list = [beginning_y, end_y]
        generate_lines(x_list, y_list)


def visualize_2d_frame(example_name):
    examples_dir = "input/examples/"
    frames_path = os.path.join(examples_dir, example_name, "members/frames.csv")
    global_cords_path = os.path.join(examples_dir, example_name, "global_cords.csv")
    frames_array = np.loadtxt(fname=frames_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)

    nodes = []
    nodes_array = np.loadtxt(fname=global_cords_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1)
    for i in range(nodes_array.shape[0]):
        x = nodes_array[i][0]
        y = nodes_array[i][1]
        nodes.append([x, y])
    generate_elements(frames_array, nodes_array)

# visualize_2d_frame(example_name)
# draw_yield_condition(yield_points, yield_surface)
# edit_plot_labes()

# plt.show()


def wireframe_3d_example():
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

    plt3d.plot_wireframe(xx, yy, z, rstride=1, cstride=1,)
    plt3d.plot_wireframe(xx, yy, z2, rstride=1, cstride=1,)
    plt.show()
