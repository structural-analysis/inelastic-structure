import os
import numpy as np


def get_elastoplastic_response(load_level, phi_x, elastic_response, sensitivity):
    scaled_elastic_response = np.dot(load_level, elastic_response)
    plastic_response = np.dot(sensitivity, phi_x)
    elastoplastic_response = scaled_elastic_response + plastic_response
    return elastoplastic_response


def create_chunk(time_step, sensitivity, response):
    np.save(f"temp/{response}-{time_step}.npy", sensitivity)


def load_chunk(time_step, response):
    chunk = np.load(f"temp/{response}-{time_step}.npy")
    return chunk


def delete_chunk(time_step, response):
    os.remove(f"temp/{response}-{time_step}.npy")


def get_activated_plastic_points(phi_pms, intact_pieces):
    print(f"{phi_pms=}")
    print(f"{intact_pieces=}")
    print(f"{len(intact_pieces)=}")
    activated_pieces_num = np.nonzero(phi_pms)[0]
    activated_plastic_points = np.array([intact_pieces[activated_piece_num].ref_yield_point_num for activated_piece_num in activated_pieces_num])
    return activated_plastic_points
