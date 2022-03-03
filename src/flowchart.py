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


def update_entering_candidates(entering_candidates, will_out, cbar):
    return entering_candidates


def create_original_table():
    original_table = np.matrix()
    return original_table


def update_costs(c, , a):


def update_will_in_column():



def create_initial_revised_table():
    initial_revised_table = np.matrix()
    return initial_revised_table


def get_min_cost_variable_num(entering_candidates):
    candidates_costs = [candidate["variable_cost"] for candidate in entering_candidates]
    min_cost_candidate_num = min(range(len(candidates_costs)), key=candidates_costs.__getitem__)
    min_cost_variable_num = entering_candidates[min_cost_candidate_num]["variable_num"]
    return min_cost_variable_num


def is_candidate_fpm(min_cost_variable_num, variables_num):
    return is_variable_plastic_multiplier(min_cost_variable_num, variables_num)


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


def is_variable_plastic_multiplier(variable_num, variables_num):
    return False if variable_num >= variables_num else True


def reset(basic_variables, b):
    # for example
    basic_variables += b
    b = 0
    return basic_variables, b
