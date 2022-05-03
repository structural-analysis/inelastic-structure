import numpy as np
from enum import Enum
# from openpyxl import load_workbook
# import os

# app_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# workbook_path = app_dir + '/data/skew.xlsx'
# workbook = load_workbook(filename=workbook_path)


def get_slack_var_num(primary_var_num):
    return primary_var_num + variables_num


def get_primary_var_num(slack_var_num):
    return slack_var_num - variables_num


class FPM(Enum):
    var_num: int
    cost: float


class SlackCandidate():
    def __init__(self, var_num, cost):
        self.var_num = var_num
        self.cost = cost


def solve_by_mahini_approach(mp_data):

    global variables_num
    global extra_numbers_num
    global landa_var_num
    global a_matrix
    global b
    global c
    global yield_points_pieces
    global full_a_matrix

    variables_num = mp_data["variables_num"]
    extra_numbers_num = mp_data["extra_numbers_num"]
    landa_var_num = variables_num - extra_numbers_num
    a_matrix = np.array(mp_data["raw_a"])
    b = mp_data["b"]
    c = -1 * mp_data["c"]
    cbar = c
    yield_points_pieces = mp_data["yield_points_pieces"]

    full_a_matrix = get_full_a_matrix()
    basic_variables = get_initial_basic_variables()
    active_yield_points = []
    b_matrix_inv = np.eye(variables_num)
    cb = np.zeros(variables_num)
    b_history = np.zeros((variables_num))
    fpm = FPM
    fpm.var_num = landa_var_num
    fpm.cost = 0
    landa_bar_var_num = 2 * variables_num - extra_numbers_num
    fpm, b_matrix_inv, basic_variables, cb, will_out_row_num, will_out_var_num = enter_landa(fpm, b_matrix_inv, basic_variables, cb, cbar)

    while will_out_var_num != landa_bar_var_num:
        # b, b_history = reset(basic_variables, b_history)
        sorted_slack_candidates = get_sorted_slack_candidates(basic_variables, cbar)
        will_in_col_num = fpm.var_num
        abar = calculate_abar(will_in_col_num, b_matrix_inv)
        bbar = calculate_bbar(b_matrix_inv)
        will_out_row_num = get_will_out(abar, bbar)
        will_out_var_num = basic_variables[will_out_row_num]

        for slack_candidate in sorted_slack_candidates:
            if is_candidate_fpm(fpm, slack_candidate):
                break
            else:
                spm_var_num = get_primary_var_num(slack_candidate.var_num)
                r = calculate_r(
                    spm_var_num=spm_var_num,
                    basic_variables=basic_variables,
                    abar=abar,
                    b_matrix_inv=b_matrix_inv,
                )
                if r > 0:
                    continue
                else:
                    pass

        # if is_will_out_opm(will_out):
        #     # FIXME: first we must determine correct unloading
        #     pass
        # else:
        #     if is_fpm_for_an_active_yield_point(fpm, active_yield_points):
        #         pass

        b_matrix_inv = update_b_matrix_inverse(b_matrix_inv, abar, will_out_row_num)
        cb = update_cb(cb, will_in_col_num, will_out_row_num)
        pi_transpose = np.dot(cb, b_matrix_inv)
        cbar = calculate_cbar(pi_transpose)
        basic_variables = update_basic_variables(basic_variables, will_out_row_num, will_in_col_num)
        fpm = update_fpm(will_out_row_num, cbar)


    bbar = np.dot(b_matrix_inv, b)
    empty_xn = np.zeros((variables_num, 1))
    xn = np.matrix(empty_xn)
    for i in range(variables_num):
        if basic_variables[i] < variables_num:
            xn[basic_variables[i], 0] = bbar[i]
    plastic_multipliers = xn[0:-extra_numbers_num, 0]

    return plastic_multipliers


def enter_landa(fpm, b_matrix_inv, basic_variables, cb, cbar):
    will_in_col_num = fpm.var_num
    a = full_a_matrix[:, will_in_col_num]
    will_out_row_num = get_will_out(a, b)
    will_out_var_num = basic_variables[will_out_row_num]
    cb = update_cb(cb, will_in_col_num, will_out_row_num)
    basic_variables = update_basic_variables(basic_variables, will_out_row_num, will_in_col_num)
    b_matrix_inv = update_b_matrix_inverse(b_matrix_inv, a, will_out_row_num)
    fpm = update_fpm(will_out_row_num, cbar)
    return fpm, b_matrix_inv, basic_variables, cb, will_out_row_num, will_out_var_num


def get_will_in(fpm):
    will_in = fpm
    return will_in


def calculate_abar(col, b_matrix_inv):
    a = full_a_matrix[:, col]
    abar = np.dot(b_matrix_inv, a)
    return abar


def calculate_bbar(b_matrix_inv):
    bbar = np.dot(b_matrix_inv, b)
    return bbar


def calculate_cbar(pi_transpose):
    cbar = np.zeros(2 * variables_num)
    for i in range(2 * variables_num):
        cbar[i] = c[i] - np.dot(pi_transpose, full_a_matrix[:, i])
    return cbar


def update_cb(cb, will_in, will_out_row_num):
    cb[will_out_row_num] = c[will_in]
    return cb


def get_will_out(abar, bbar):
    # TODO: we check b/a to be positive, correct way is to check a to be positive
    # b is not always positive
    # TODO: exclude load variable, look mahini find_pivot function
    # TODO: do not divide zero values
    # TODO: use sign function like mahini find_pivot function
    # TODO: for hardening parameters extra care must be taken.
    # TODO: check mahini code line 123, make sure whether reset before this function is necessary or not.

    # ba = np.round(ba, 5)
    ba = bbar / abar
    minba = min(ba[ba > 0])
    will_out_row_num = np.where(ba == minba)[0][0]
    return will_out_row_num


def get_initial_basic_variables():
    basic_variables = np.zeros(variables_num, dtype=int)
    for i in range(variables_num):
        basic_variables[i] = variables_num + i
    return basic_variables


def get_full_a_matrix():
    full_a_matrix = np.zeros((variables_num, 2 * variables_num))
    full_a_matrix[0:variables_num, 0:variables_num] = a_matrix[0:variables_num, 0:variables_num]
    j = variables_num

    # Assigning diagonal arrays of y variables.
    for i in range(variables_num):
        full_a_matrix[i, j] = 1.0
        j += 1

    return full_a_matrix


def update_b_matrix_inverse(b_matrix_inv, abar, will_out_row_num):
    e = np.eye(variables_num)
    eta = np.zeros(variables_num)
    will_out_item = abar[will_out_row_num]

    for i, item in enumerate(abar):
        if i == will_out_row_num:
            eta[i] = 1 / will_out_item
        else:
            eta[i] = -item / will_out_item
    e[:, will_out_row_num] = eta
    updated_b_matrix_inv = np.dot(e, b_matrix_inv)
    return updated_b_matrix_inv


def is_variable_plastic_multiplier(variable_num):
    return False if variable_num >= landa_var_num else True


def is_candidate_fpm(fpm, slack_candidate):
    if fpm.cost <= slack_candidate.cost:
        return True
    else:
        return False


def is_will_out_opm(will_out):
    # opm: obstacle plastic multiplier
    return is_variable_plastic_multiplier(will_out)


def get_yield_point_num_from_piece(piece):
    for yield_point_num, yield_point_pieces in enumerate(yield_points_pieces):
        if piece in yield_point_pieces:
            return yield_point_num


def get_active_yield_points(basic_variables):
    active_yield_points = []
    for variable in basic_variables:
        active_yield_point = get_yield_point_num_from_piece(variable)
        if active_yield_point is not None:
            active_yield_points.append(active_yield_point)
    return active_yield_points


def is_fpm_for_an_active_yield_point(fpm, active_yield_points):
    return True if fpm in active_yield_points else False


def update_basic_variables(basic_variables, will_out_row_num, will_in_col_num):
    basic_variables[will_out_row_num] = will_in_col_num
    return basic_variables


def update_fpm(will_out_row_num, cbar):
    fpm = FPM
    fpm.var_num = will_out_row_num
    fpm.cost = cbar[will_out_row_num]
    return fpm


def get_sorted_slack_candidates(basic_variables, cbar):
    slack_candidates = []
    for var in basic_variables:
        if var < landa_var_num:
            slack_var_num = get_slack_var_num(var)
            slack_candidate = SlackCandidate(
                var_num=slack_var_num,
                cost=cbar[slack_var_num]
            )
            slack_candidates.append(slack_candidate)
    slack_candidates.sort(key=lambda y: y.cost)
    return slack_candidates


def reset(basic_variables, b_history):
    # INCOMPLETE:
    for i, basic_variable in enumerate(basic_variables):
        if basic_variable < variables_num:
            b_history[i] += b[i]
            b[i] = 0
    return b, b_history


def calculate_r(spm_var_num, basic_variables, abar, b_matrix_inv):
    spm_row_num = get_var_row_num(spm_var_num, basic_variables)
    r = abar[spm_row_num] / b_matrix_inv[spm_row_num, spm_var_num]
    return r


def get_var_row_num(var_num, basic_variables):
    row_num = np.where(basic_variables == var_num)[0][0]
    return row_num


def unload(pm_var_num, basic_variables, b_matrix_inv):
    # TODO: should handle if third pivot column is a y not x. possible bifurcation.
    # TODO: must handle landa-row separately like mahini unload (e.g. softening, ...)
    # TODO: loading whole b_inverse in input and output is costly, try like mahini method.

    exiting_row_num = get_var_row_num(pm_var_num, basic_variables)

    unloading_pivot_elements = [
        {
            "row": exiting_row_num,
            "column": get_slack_var_num(exiting_row_num),
        },
        {
            "row": pm_var_num,
            "column": get_slack_var_num(pm_var_num),
        },
        {
            "row": exiting_row_num,
            "column": basic_variables[pm_var_num],
        },
    ]

    for element in unloading_pivot_elements:
        abar = calculate_abar(element["column"], b_matrix_inv)
        b_matrix_inv = update_b_matrix_inverse(b_matrix_inv, abar, element["row"])
        basic_variables = update_basic_variables(basic_variables, element["row"], element["column"])

    return basic_variables, b_matrix_inv

    # active_yield_points = get_active_yield_points(basic_variables)
    # pi_transpose = np.dot(cb, b_matrix_inv)
    # cbar = calculate_cbar(pi_transpose)
    # cb = update_cb(cb, will_in, will_out_row_num)
    # old_fpm = fpm
    # entering_candidates = update_entering_candidates(entering_candidates, old_fpm, fpm, cbar)
