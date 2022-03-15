import numpy as np

b = np.ones((variables_num))
b[-extra_numbers_num] = load_limit

table = np.matrix()
variables = []
basic_variables = []
entering_candidates = [
    {
        "variable_num": 1,
        "variable_cost": -100,
    },
    {
        "variable_num": 9,
        "variable_cost": -50,
    }
]


def get_min_cost_variable_num(entering_candidates):
    candidates_costs = [candidate["variable_cost"] for candidate in entering_candidates]
    min_cost_candidate_num = min(range(len(candidates_costs)), key=candidates_costs.__getitem__)
    min_cost_variable_num = entering_candidates[min_cost_candidate_num]["variable_num"]
    return min_cost_variable_num


def is_candidate_fpm(min_cost_variable_num, variables_num):
    # fpm: free plastic multiplier
    return is_variable_plastic_multiplier(min_cost_variable_num, variables_num)


def is_variable_plastic_multiplier(variable_num, variables_num):
    return False if variable_num >= variables_num -1 else True


def is_will_out_opm(will_out, variables_num):
    # opm: obstacle plastic multiplier
    return is_variable_plastic_multiplier(will_out, variables_num)


def get_yield_point_num_from_piece(piece, yield_points_pieces):
    for yield_point_num, yield_point_pieces in enumerate(yield_points_pieces):
        if piece in yield_point_pieces:
            return yield_point_num


def get_active_yield_points(basic_variables, yield_points_pieces):
    active_yield_points = []
    for variable in basic_variables:
        active_yield_point = get_yield_point_num_from_piece(variable, yield_points_pieces)
        if active_yield_point != None :
            active_yield_points.append(active_yield_point)
    return active_yield_points


def is_fpm_for_an_active_yield_point(fpm, active_yield_points):
    return True if fpm in active_yield_points else False


def update_entering_candidates(entering_candidates, will_out, cbar):
    return entering_candidates


def create_original_table():
    original_table = np.matrix()
    return original_table


def update_costs(c, a):


def update_will_in_column():



def create_initial_revised_table():
    initial_revised_table = np.matrix()
    return initial_revised_table



def get_will_out_variable(basic_variables, revised_table):
    will_out_row_num = None
    will_out = None
    # calculate min b/a and give will_out
    return will_out_row_num, will_out


def pivot(will_in, will_out_row_num, revised_table, basic_variables):
    # for example
    revised_table = None
    basic_variables = None
    return revised_table, basic_variables


def update_basic_variables(basic_variables, pivot_row_num, will_in):
    return basic_variables


def reset(basic_variables, b):
    # for example
    basic_variables += b
    b = 0
    return basic_variables, b
