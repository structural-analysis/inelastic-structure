import numpy as np
from src.workshop import create_structure
from src.functions import prepare_raw_data
from src.programming import solve_by_mahini_approach

structure = create_structure()
load_limit = structure.load_limit
mp_data = prepare_raw_data(structure=structure, load_limit=load_limit)
x_history = solve_by_mahini_approach(mp_data)
plastic_multipliers = x_history[-1][0:-mp_data["extra_numbers_num"]]

phi = structure.phi
phi_x = phi * plastic_multipliers

# elements forces
scaled_elastic_elements_forces = np.matrix(np.dot(load_limit, structure.elastic_elements_forces))
elements_forces_sensitivity_matrix = structure.elements_forces_sensitivity_matrix
plastic_elements_forces = elements_forces_sensitivity_matrix * phi_x
elastoplastic_elements_forces = scaled_elastic_elements_forces + plastic_elements_forces

# elements displacements
scaled_elastic_elements_disps = np.matrix(np.dot(load_limit, structure.elastic_elements_disps))
elements_disps_sensitivity_matrix = structure.elements_disps_sensitivity_matrix
plastic_elements_disps = elements_disps_sensitivity_matrix * phi_x
elastoplastic_elements_disps = scaled_elastic_elements_disps + plastic_elements_disps

# structure nodal displacements
scaled_elastic_nodal_disp = np.matrix(np.dot(load_limit, structure.elastic_nodal_disp))
nodal_disps_sensitivity_matrix = structure.nodal_disps_sensitivity_matrix
plastic_nodal_disp = nodal_disps_sensitivity_matrix * phi_x
elastoplastic_nodal_disp = scaled_elastic_nodal_disp + plastic_nodal_disp[0, 0]

if __name__ == "__main__":
    print(f"{plastic_multipliers=}")
    print(f"{elastoplastic_elements_forces=}")
    print(f"{elastoplastic_nodal_disp=}")
