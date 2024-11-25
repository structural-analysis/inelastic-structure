import numpy as np


def get_elastoplastic_response(load_level, phi_x, elastic_response, sensitivity):
    scaled_elastic_response = np.dot(load_level, elastic_response)
    plastic_response = np.dot(sensitivity, phi_x)
    elastoplastic_response = scaled_elastic_response + plastic_response
    return elastoplastic_response
