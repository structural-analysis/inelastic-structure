import numpy as np
# from openpyxl import load_workbook
# import os

# app_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# workbook_path = app_dir + '/data/skew.xlsx'
# workbook = load_workbook(filename=workbook_path)


def solve_by_mahini_approach(mp_data):

    global variables_num
    global a_matrix
    global b
    global c
    global extra_numbers_num
    global yield_points_pieces
    global full_a_matrix

    variables_num = mp_data["variables_num"]
    a_matrix = np.array(mp_data["raw_a"])
    b = mp_data["b"]
    c = -1 * mp_data["c"]
    extra_numbers_num = mp_data["extra_numbers_num"]
    yield_points_pieces = mp_data["yield_points_pieces"]

    full_a_matrix = get_full_a_matrix()
    basic_variables = get_initial_basic_variables()
    active_yield_points = []
    b_matrix_inv = np.eye(variables_num)
    cb = np.zeros(variables_num)
    b_history = np.zeros((variables_num))
    fpm = variables_num - extra_numbers_num

    entering_candidates = [
        {
            "variable_num": fpm,
            "variable_cost": 0,
        }
    ]

    landa_bar_var_num = 2 * variables_num - extra_numbers_num
    will_out = 0

    while will_out != landa_bar_var_num:
        min_candidate_index = 0
        sorted_candidates = get_sorted_candidates(entering_candidates)
        min_cost_variable_num = sorted_candidates[min_candidate_index]["variable_num"]
        abar_fpm = calculate_abar(fpm, b_matrix_inv)

        while True:

            if is_candidate_fpm(min_cost_variable_num):
                will_in = fpm
                abar = calculate_abar(will_in, b_matrix_inv)
                bbar = calculate_bbar(b_matrix_inv)
                will_out, will_out_row_num = get_will_out(abar, bbar, basic_variables)

                if is_will_out_opm(will_out):
                    # FIXME: first we must determine correct unloading
                    pass
                else:
                    if is_fpm_for_an_active_yield_point(fpm, active_yield_points):
                        pass

                old_fpm = fpm
                fpm = get_new_fpm(basic_variables, will_out_row_num)
                pi_transpose = np.dot(cb, b_matrix_inv)
                cbar = calculate_cbar(pi_transpose)
                cb = update_cb(cb, will_in, will_out_row_num)
                entering_candidates = update_entering_candidates(entering_candidates, old_fpm, fpm, cbar)
                basic_variables = update_basic_variables(basic_variables, will_out_row_num, will_in)
                active_yield_points = get_active_yield_points(basic_variables)
                b_matrix_inv = update_b_matrix_inverse(b_matrix_inv, will_in, will_out_row_num)
                break

            else:
                spm = min_cost_variable_num - variables_num
                r = calculate_r(
                    spm,
                    basic_variables,
                    abar_fpm,
                    b_matrix_inv,
                )
                if r < 0:
                    b, b_history = reset(
                        basic_variables,
                        b_history)
                    will_in = spm
                    abar = calculate_abar(will_in, b_matrix_inv)
                    bbar = calculate_bbar(b_matrix_inv)
                    will_out, will_out_row_num = get_will_out(abar, bbar, basic_variables)
                    # reset and unload here
                else:
                    min_candidate_index += 1
                    min_cost_variable_num = sorted_candidates[min_candidate_index]["variable_num"]

    bbar = np.dot(b_matrix_inv, b)
    empty_xn = np.zeros((variables_num, 1))
    xn = np.matrix(empty_xn)
    for i in range(variables_num):
        if basic_variables[i] < variables_num:
            xn[basic_variables[i], 0] = bbar[i]
    plastic_multipliers = xn[0:-extra_numbers_num, 0]

    return plastic_multipliers


def get_will_in(fpm):
    will_in = fpm
    return will_in


def calculate_abar(will_in, b_matrix_inv):
    a = full_a_matrix[:, will_in]
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


def get_will_out(abar, bbar, basic_variables):
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
    will_out = basic_variables[will_out_row_num]
    return will_out, will_out_row_num


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


def get_fpm(will_out, variables_num):
    # TODO: check wether is possible to a x be will_out or not
    fpm = will_out - variables_num
    return int(fpm)


def update_b_matrix_inverse(b_matrix_inv, will_in, will_out_row_num):
    e = np.eye(variables_num)
    eta = np.zeros(variables_num)
    abar = calculate_abar(will_in, b_matrix_inv)
    will_out_item = abar[will_out_row_num]

    for i, item in enumerate(abar):
        if i == will_out_row_num:
            eta[i] = 1 / will_out_item
        else:
            eta[i] = -item / will_out_item
    e[:, will_out_row_num] = eta
    updated_b_matrix_inv = np.dot(e, b_matrix_inv)
    return updated_b_matrix_inv


def get_sorted_candidates(entering_candidates):
    sorted_candidates = sorted(entering_candidates, key=lambda d: d['variable_cost']) 
    # candidates_costs = [candidate["variable_cost"] for candidate in entering_candidates]
    # min_cost_candidate_num = min(range(len(candidates_costs)), key=candidates_costs.__getitem__)
    # min_cost_variable_num = entering_candidates[min_cost_candidate_num]["variable_num"]
    return sorted_candidates


def is_variable_plastic_multiplier(variable_num):
    return False if variable_num >= variables_num - extra_numbers_num else True


def is_candidate_fpm(min_cost_variable_num):
    # fpm: free plastic multiplier
    if is_variable_plastic_multiplier(min_cost_variable_num) or min_cost_variable_num == variables_num - extra_numbers_num:
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


def update_basic_variables(basic_variables, will_out_row_num, will_in):
    basic_variables[will_out_row_num] = will_in
    return basic_variables


def get_new_fpm(basic_variables, will_out_row_num):
    new_fpm = basic_variables[will_out_row_num] - variables_num
    return new_fpm


def update_entering_candidates(
        entering_candidates,
        old_fpm,
        new_fpm,
        cbar):

    for candidate in entering_candidates:
        if candidate["variable_num"] == old_fpm:
            entering_candidates.remove(candidate)
            break

    new_fpm_candidate = {
        "variable_num": new_fpm,
        "variable_cost": cbar[new_fpm],
    }

    # if fpm != lambda:
    if old_fpm != variables_num - extra_numbers_num:
        old_fpm_slack_num = variables_num + old_fpm
        old_fpm_slack_candidate = {
            "variable_num": old_fpm_slack_num,
            "variable_cost": cbar[old_fpm_slack_num],
        }
        entering_candidates.append(old_fpm_slack_candidate)

    entering_candidates.append(new_fpm_candidate)
    return entering_candidates


def reset(basic_variables, b_history):
    # INCOMPLETE:
    for i, basic_variable in enumerate(basic_variables):
        if basic_variable < variables_num:
            b_history[i] += b[i]
            b[i] = 0
    return b, b_history


def calculate_r(spm, basic_variables, abar_fpm, b_matrix_inv):
    slack_column_num = spm
    spm_row_num = np.where(basic_variables == spm)[0][0]
    r = abar_fpm[spm_row_num] / b_matrix_inv[spm_row_num, slack_column_num]
    return r


def unload(will_out_row_num, basic_variables, b_matrix_inv, will_in):
    # TODO: should handle if third pivot column is a y not x. possible bifurcation.
    # TODO: must handle landa-row separately like mahini unload (e.g. softening, ...)
    prow = will_out_row_num
    unloading_pivot_elements = [
        {"row": prow, "column": prow + variables_num},
        {"row": basic_variables[prow], "column": basic_variables[prow] + variables_num},
        {"row": prow, "column": basic_variables[basic_variables[prow]]},
    ]
    for element in unloading_pivot_elements:

        # TODO: NEXT WEEK:
        # 2 - unload and enter functions are like each-other and update and return same things
        # 3 - we can write all lines like cbar, ... inside unload and enter functions

        b_matrix_inv = update_b_matrix_inverse(b_matrix_inv, will_in, element["row"])
        basic_variables = update_basic_variables(basic_variables, element["row"], element["column"])

    new_fpm = get_new_fpm(basic_variables, will_out_row_num)
    return new_fpm, basic_variables, b_matrix_inv

    # active_yield_points = get_active_yield_points(basic_variables)
    # pi_transpose = np.dot(cb, b_matrix_inv)
    # cbar = calculate_cbar(pi_transpose)
    # cb = update_cb(cb, will_in, will_out_row_num)
    # old_fpm = fpm
    # entering_candidates = update_entering_candidates(entering_candidates, old_fpm, fpm, cbar)
