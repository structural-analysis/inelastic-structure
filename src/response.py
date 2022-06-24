import os
import numpy as np
from src.models.functions import get_elements_max_dof_num


outputs_dir = "output/examples/"


def calculate_responses(structure, result, example_name):
    pms_history = result["pms_history"]
    load_level_history = result["load_level_history"]
    increments_num = len(load_level_history)
    phi = structure.phi
    total_dofs_num = structure.total_dofs_num
    max_element_dof_num = get_elements_max_dof_num(structure.elements.list)
    load_levels = np.zeros([increments_num, 1])

    nodal_disps_sensitivity_matrix = structure.nodal_disps_sensitivity_matrix
    nodal_disps = np.zeros([increments_num, total_dofs_num])

    elements_forces_sensitivity_matrix = structure.elements_forces_sensitivity_matrix
    elements_forces = np.zeros([increments_num, structure.elements.num], dtype=object)

    # elements displacements
    elements_disps_sensitivity_matrix = structure.elements_disps_sensitivity_matrix
    elements_disps = np.zeros([increments_num, structure.elements.num], dtype=object)

    for i in range(increments_num):
        pms = pms_history[i]
        load_level = load_level_history[i]
        phi_x = phi * pms

        load_levels[i, 0] = load_level

        # structure nodal displacements
        scaled_elastic_nodal_disp = np.matrix(np.dot(load_level, structure.elastic_nodal_disp))
        plastic_nodal_disp = nodal_disps_sensitivity_matrix * phi_x
        elastoplastic_nodal_disp = scaled_elastic_nodal_disp + plastic_nodal_disp[0, 0]
        nodal_disps[i, :] = np.asarray(elastoplastic_nodal_disp).reshape(-1)

        # elements forces
        scaled_elastic_elements_forces = np.matrix(np.dot(load_level, structure.elastic_elements_forces))
        plastic_elements_forces = elements_forces_sensitivity_matrix * phi_x
        elastoplastic_elements_forces = scaled_elastic_elements_forces + plastic_elements_forces
        for j in range(structure.elements.num):
            elements_forces[i, j] = elastoplastic_elements_forces[j, 0]

        # elements disps
        scaled_elastic_elements_disps = np.matrix(np.dot(load_level, structure.elastic_elements_disps))
        plastic_elements_disps = elements_disps_sensitivity_matrix * phi_x
        elastoplastic_elements_disps = scaled_elastic_elements_disps + plastic_elements_disps
        for j in range(structure.elements.num):
            elements_disps[i, j] = elastoplastic_elements_disps[j, 0]

    for i in range(increments_num):
        increment_dir = os.path.join(outputs_dir, example_name, str(i))
        os.makedirs(increment_dir, exist_ok=True)

        load_levels_path = os.path.join(increment_dir, "load_levels.csv")
        nodal_disps_path = os.path.join(increment_dir, "nodal_disps.csv")
        elements_forces_path = os.path.join(increment_dir, "elements_forces.csv")
        elements_disps_path = os.path.join(increment_dir, "elements_disps.csv")

        current_increment_elements_forces = elements_forces[i, :]
        empty_current_increment_elements_forces_compact = np.zeros([max_element_dof_num, structure.elements.num])
        current_increment_elements_forces_compact = np.matrix(empty_current_increment_elements_forces_compact)
        for j in range(structure.elements.num):
            current_increment_elements_forces_compact[:, j] = current_increment_elements_forces[j]

        current_increment_elements_disps = elements_disps[i, :]
        empty_current_increment_elements_disps_compact = np.zeros([max_element_dof_num, structure.elements.num])
        current_increment_elements_disps_compact = np.matrix(empty_current_increment_elements_disps_compact)
        for j in range(structure.elements.num):
            current_increment_elements_disps_compact[:, j] = current_increment_elements_disps[j]

        np.savetxt(fname=load_levels_path, X=np.array([load_levels[i, 0]]), delimiter=",")
        np.savetxt(fname=nodal_disps_path, X=nodal_disps[i, :], delimiter=",")
        np.savetxt(fname=elements_forces_path, X=current_increment_elements_forces_compact, delimiter=",")
        np.savetxt(fname=elements_disps_path, X=current_increment_elements_disps_compact, delimiter=",")
