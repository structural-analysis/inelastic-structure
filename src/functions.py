import os
import numpy as np


def get_elastoplastic_response(load_level, phi_x, elastic_response, sensitivity):
    scaled_elastic_response = load_level * elastic_response
    plastic_response = sensitivity @ phi_x
    elastoplastic_response = scaled_elastic_response + plastic_response
    return elastoplastic_response


def create_chunk(sensitivity, response):
    np.save(f"temp/{response}.npy", sensitivity)


def load_chunk(response):
    chunk = np.load(f"temp/{response}.npy")
    return chunk


def delete_chunk(response):
    os.remove(f"temp/{response}.npy")


def get_activated_plastic_points(pms, intact_pieces):
    activated_pieces_num = np.nonzero(pms)[0]
    activated_plastic_points = np.array([intact_pieces[activated_piece_num].ref_yield_point_num for activated_piece_num in activated_pieces_num])
    return activated_plastic_points
