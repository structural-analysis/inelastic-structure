import os
import numpy as np
import matplotlib.pyplot as plt

from ..settings import settings

outputs_dir = "output/examples/"


def draw_mises_vs_disp_history(example, disp_node_num, mises_ip_num, disp_dof):
    list_to_draw = get_mises_disp_history(example, disp_node_num, mises_ip_num, disp_dof)
    save_mises_disp_history_to_output(example, list_to_draw)
    miseses = [mises for mises, _ in list_to_draw]
    disps = [disp for _, disp in list_to_draw]

    # Plotting the data
    plt.plot(disps, miseses, marker='o', linestyle='-')
    plt.xlabel('Controlled Disp (m)')
    plt.ylabel('Mises Moment (N.m)')
    plt.title('Mises Moment vs. Disp')
    plt.grid(True)
    plt.show()


def get_mises_disp_history(example, disp_node_num, mises_ip_num, disp_dof):
    example_path = os.path.join(outputs_dir, example)
    increments = find_subdirs(example_path)
    list_to_draw = []
    list_to_draw.append((0, 0))
    for inc in increments:
        inc_nodal_disp_array_path = os.path.join(example_path, str(inc), "nodal_disp", "0.csv")
        inc_yp_moments_array_path = os.path.join(example_path, str(inc), "yield_points_mises_moments", "0.csv")
        inc_nodal_disp_array = np.loadtxt(fname=inc_nodal_disp_array_path, skiprows=0, delimiter=",", dtype=float, ndmin=1)
        inc_yp_moments_array = np.loadtxt(fname=inc_yp_moments_array_path, skiprows=0, delimiter=",", dtype=float, ndmin=1)
        list_to_draw.append((np.abs(inc_yp_moments_array[mises_ip_num]), np.abs(inc_nodal_disp_array[3 * (disp_node_num + 1) - (3 - disp_dof)])))
    return list_to_draw


def save_mises_disp_history_to_output(example, list_to_draw):
    example_path = os.path.join(outputs_dir, example)
    visualization_dir_path = os.path.join(example_path, "visualization")
    path_to_save = os.path.join(visualization_dir_path, "ip_mises_disp_history.csv")
    os.makedirs(visualization_dir_path, exist_ok=True)
    np.savetxt(fname=path_to_save, X=np.array(list_to_draw), delimiter=", ", fmt=f'%.{settings.output_digits}e')


def find_subdirs(path):
    subdirs = os.listdir(path)
    subdirs_int = sorted([int(subdir) for subdir in subdirs if subdir.isdigit()])
    return subdirs_int
