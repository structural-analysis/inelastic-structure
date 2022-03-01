import numpy as np
from src.workshop import create_structure
from src.programming import prepare_raw_data, complementarity_programming

structure = create_structure()
load_limit = structure.load_limit
mp_data = prepare_raw_data(structure=structure, load_limit=load_limit)
plastic_multipliers = complementarity_programming(mp_data)

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
