import numpy as np
from src.settings import settings
from src.workshop import create_structure
from src.programming import prepare_raw_data, complementarity_programming

load_limit = settings.load_limit
structure = create_structure()
mp_data = prepare_raw_data(structure=structure, load_limit=load_limit)
plastic_multipliers = complementarity_programming(mp_data)
phi = structure.phi
phi_x = phi * plastic_multipliers
elastic_elements_forces = np.matrix(np.dot(load_limit, structure.elastic_elements_forces))
elements_forces_sensitivity_matrix = structure.elements_forces_sensitivity_matrix
plastic_elements_forces = elements_forces_sensitivity_matrix * phi_x

elastoplastic_elements_forces = elastic_elements_forces + plastic_elements_forces
