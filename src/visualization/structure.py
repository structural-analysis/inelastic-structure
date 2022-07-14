import os
import numpy as np
import matplotlib.pyplot as plt

from src.settings import settings
from src.visualization.drawing import add_diagram_to_figure, draw_lines


examples_dir = "input/examples/"
example_name = settings.example_name


def generate_frame_elements(frames_array, nodes_array):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = add_diagram_to_figure(fig, (1, 1), (0, 0))
    for frame_label, frame_element in enumerate(frames_array):
        beginning_node = int(frame_element[1])
        end_node = int(frame_element[2])
        beginning_x = int(nodes_array[beginning_node][0])
        beginning_y = int(nodes_array[beginning_node][1])
        end_x = int(nodes_array[end_node][0])
        end_y = int(nodes_array[end_node][1])
        x_list = [beginning_x, end_x]
        y_list = [beginning_y, end_y]
        element_coords = {
            "x": x_list,
            "y": y_list,
        }
        draw_lines(ax, element_coords)


def visualize_2d_frame(example_name):
    frames_path = os.path.join(examples_dir, example_name, "members/frames.csv")
    global_cords_path = os.path.join(examples_dir, example_name, "global_cords.csv")
    frames_array = np.loadtxt(fname=frames_path, usecols=range(4), delimiter=",", ndmin=2, skiprows=1, dtype=str)

    nodes = []
    nodes_array = np.loadtxt(fname=global_cords_path, usecols=range(2), delimiter=",", ndmin=2, skiprows=1)
    for i in range(nodes_array.shape[0]):
        x = nodes_array[i][0]
        y = nodes_array[i][1]
        nodes.append([x, y])
    generate_frame_elements(frames_array, nodes_array)
    plt.show()


visualize_2d_frame(example_name)
