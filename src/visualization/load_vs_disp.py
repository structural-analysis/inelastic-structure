import os
import numpy as np
import matplotlib.pyplot as plt

from ..settings import settings

outputs_dir = "output/examples/"


def draw_load_disp_history(example, node_num, dof, each_node_dof_count):
    list_to_draw = get_load_disp_history(example, node_num, dof, each_node_dof_count)
    save_load_disp_history_to_output(example, list_to_draw)
    loads = [load for load, _ in list_to_draw]
    disps = [disp for _, disp in list_to_draw]

    # Plotting the data
    plt.plot(disps, loads, marker='o', linestyle='-')
    plt.xlabel('Controlled Disp (m)')
    plt.ylabel('Load Factor (N)')
    plt.title('Load Factor vs. Disp')
    plt.grid(True)
    plt.show()


def get_load_disp_history(example, node_num, dof, each_node_dof_count):
    example_path = os.path.join(outputs_dir, example)
    increments = find_subdirs(example_path)
    list_to_draw = []
    list_to_draw.append((0, 0))
    for inc in increments:
        inc_nodal_disp_array_path = os.path.join(example_path, str(inc), "nodal_disp", "0.csv")
        inc_load_level_array_path = os.path.join(example_path, str(inc), "load_levels", "0.csv")
        inc_nodal_disp_array = np.loadtxt(fname=inc_nodal_disp_array_path, skiprows=0, delimiter=",", dtype=float, ndmin=1)
        inc_load_level_array = np.loadtxt(fname=inc_load_level_array_path, skiprows=0, delimiter=",", dtype=float, ndmin=1)
        list_to_draw.append((inc_load_level_array[0], np.abs(inc_nodal_disp_array[each_node_dof_count * (node_num + 1) - (each_node_dof_count - dof)])))
    return list_to_draw


def save_load_disp_history_to_output(example, list_to_draw):
    example_path = os.path.join(outputs_dir, example)
    visualization_dir_path = os.path.join(example_path, "visualization")
    path_to_save = os.path.join(visualization_dir_path, "load_disp_history.csv")
    # new_permissions = 0o777
    # print(path_to_save)
    # os.chmod(path_to_save, new_permissions)
    os.makedirs(visualization_dir_path, exist_ok=True)
    np.savetxt(fname=path_to_save, X=np.array(list_to_draw), delimiter=", ", fmt=f'%.{settings.output_digits}e')


def find_subdirs(path):
    subdirs = os.listdir(path)
    subdirs_int = sorted([int(subdir) for subdir in subdirs if subdir.isdigit()])
    return subdirs_int
