import numpy as np
from src.settings import settings

computational_zero = settings.computational_zero


def get_slack_var_num(primary_var_num):
    return primary_var_num + variables_num


def get_primary_var_num(slack_var_num):
    return slack_var_num - variables_num


class FPM():
    var_num: int
    cost: float


class WillOut():
    row_num: int
    var_num: int


class SlackCandidate():
    def __init__(self, var_num, cost):
        self.var_num = var_num
        self.cost = cost


def solve_by_mahini_approach(analysis_data):

    global variables_num
    global landa_var_num
    global a_matrix
    global b
    global c
    global yield_points_pieces
    global full_a_matrix

    variables_num = analysis_data["variables_num"]
    landa_var_num = analysis_data["landa_var_num"]
    a_matrix = np.array(analysis_data["raw_a"])
    b = analysis_data["b"]
    c = -1 * analysis_data["c"]
    bbar = b
    yield_points_pieces = analysis_data["yield_points_pieces"]
    limits_slacks = analysis_data["limits_slacks"]

    full_a_matrix = get_full_a_matrix()
    basic_variables = get_initial_basic_variables()
    b_matrix_inv = np.eye(variables_num)
    cb = np.zeros(variables_num)
    empty_x_cumulative = np.zeros((variables_num, 1))
    x_cumulative = np.matrix(empty_x_cumulative)
    x_history = []
    fpm = FPM
    fpm.var_num = landa_var_num
    fpm.cost = 0
    fpm, b_matrix_inv, basic_variables, cb, will_out_row_num, will_out_var_num = enter_landa(fpm, b_matrix_inv, basic_variables, cb)
    landa_row_num = will_out_row_num

    while limits_slacks.issubset(set(basic_variables)):
        sorted_slack_candidates = get_sorted_slack_candidates(basic_variables, b_matrix_inv, cb)
        will_in_col_num = fpm.var_num
        abar = calculate_abar(will_in_col_num, b_matrix_inv)
        bbar = calculate_bbar(b_matrix_inv, bbar)
        will_out_row_num = get_will_out(abar, bbar, landa_row_num)
        will_out_var_num = basic_variables[will_out_row_num]
        x_cumulative, bbar = reset(basic_variables, x_cumulative, bbar)
        x_history.append(x_cumulative.copy())

        for slack_candidate in sorted_slack_candidates + [fpm]:
            if not is_candidate_fpm(fpm, slack_candidate):
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
                    print("unload r < 0")
                    basic_variables, b_matrix_inv, cb = unload(
                        pm_var_num=spm_var_num,
                        basic_variables=basic_variables,
                        b_matrix_inv=b_matrix_inv,
                        cb=cb,
                    )
                    break
            else:
                if is_will_out_var_opm(will_out_var_num):
                    print("unload opm")
                    opm_var_num = will_out_var_num
                    basic_variables, b_matrix_inv, cb = unload(
                        pm_var_num=opm_var_num,
                        basic_variables=basic_variables,
                        b_matrix_inv=b_matrix_inv,
                        cb=cb,
                    )
                    break
                else:
                    print("enter fpm")
                    basic_variables, b_matrix_inv, cb, fpm = enter_fpm(
                        basic_variables=basic_variables,
                        b_matrix_inv=b_matrix_inv,
                        cb=cb,
                        will_out_row_num=will_out_row_num,
                        will_in_col_num=will_in_col_num,
                        abar=abar,
                    )
                    break

    bbar = calculate_bbar(b_matrix_inv, bbar)
    x_cumulative, bbar = reset(basic_variables, x_cumulative, bbar)
    x_history.append(x_cumulative.copy())

    pms_history = []
    load_level_history = []
    for x in x_history:
        pms = x[0:landa_var_num]
        load_level = x[landa_var_num][0, 0]
        pms_history.append(pms)
        load_level_history.append(load_level)

    result = {
        "pms_history": pms_history,
        "load_level_history": load_level_history
    }
    return result


def enter_landa(fpm, b_matrix_inv, basic_variables, cb):
    will_in_col_num = fpm.var_num
    a = full_a_matrix[:, will_in_col_num]
    will_out_row_num = get_will_out(a, b)
    will_out_var_num = basic_variables[will_out_row_num]
    basic_variables = update_basic_variables(basic_variables, will_out_row_num, will_in_col_num)
    b_matrix_inv = update_b_matrix_inverse(b_matrix_inv, a, will_out_row_num)
    cb = update_cb(cb, will_in_col_num, will_out_row_num)
    cbar = calculate_cbar(cb, b_matrix_inv)
    fpm = update_fpm(will_out_row_num, cbar)
    return fpm, b_matrix_inv, basic_variables, cb, will_out_row_num, will_out_var_num


def enter_fpm(basic_variables, b_matrix_inv, cb, will_out_row_num, will_in_col_num, abar):
    b_matrix_inv = update_b_matrix_inverse(b_matrix_inv, abar, will_out_row_num)
    cb = update_cb(cb, will_in_col_num, will_out_row_num)
    cbar = calculate_cbar(cb, b_matrix_inv)
    basic_variables = update_basic_variables(basic_variables, will_out_row_num, will_in_col_num)
    fpm = update_fpm(will_out_row_num, cbar)
    return basic_variables, b_matrix_inv, cb, fpm


def unload(pm_var_num, basic_variables, b_matrix_inv, cb):
    # TODO: should handle if third pivot column is a y not x. possible bifurcation.
    # TODO: must handle landa-row separately like mahini unload (e.g. softening, ...)
    # TODO: loading whole b_matrix_inv in input and output is costly, try like mahini method.
    # TODO: check line 60 of unload and line 265 in mclp of mahini code
    # (probable usage: in case when unload is last step)

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
        cb = update_cb(
            cb=cb,
            will_in_col_num=element["column"],
            will_out_row_num=element["row"]
        )
        basic_variables = update_basic_variables(
            basic_variables=basic_variables,
            will_out_row_num=element["row"],
            will_in_col_num=element["column"]
        )

    return basic_variables, b_matrix_inv, cb


def calculate_abar(col, b_matrix_inv):
    a = full_a_matrix[:, col]
    abar = np.dot(b_matrix_inv, a)
    return abar


def calculate_bbar(b_matrix_inv, bbar):
    bbar = np.dot(b_matrix_inv, bbar)
    return bbar


def calculate_cbar(cb, b_matrix_inv):
    pi_transpose = np.dot(cb, b_matrix_inv)
    cbar = np.zeros(2 * variables_num)
    for i in range(2 * variables_num):
        cbar[i] = c[i] - np.dot(pi_transpose, full_a_matrix[:, i])
    return cbar


def update_cb(cb, will_in_col_num, will_out_row_num):
    cb[will_out_row_num] = c[will_in_col_num]
    return cb


def get_will_out(abar, bbar, landa_row_num=None):
    # TODO: see mahini find_pivot for handling hardening parameters
    # TODO: handle unbounded problem,
    # when there is no positive a remaining (structure failure), e.g. stop the process.
    # IMPORTANT TODO: sort twice: first time based on slackcosts like mahini find_pivot

    abar = zero_out_small_values(abar)
    positive_abar_indices = np.array(np.where(abar > 0)[0], dtype=int)
    positive_abar = abar[positive_abar_indices]
    ba = bbar[positive_abar_indices] / positive_abar
    zipped_ba = np.row_stack([positive_abar_indices, ba])
    mask = np.argsort(zipped_ba[1], kind="stable")
    sorted_zipped_ba = zipped_ba[:, mask]

    will_out_row_num = int(sorted_zipped_ba[0, 0])
    if landa_row_num:
        if landa_row_num == int(sorted_zipped_ba[0, 0]):
            will_out_row_num = int(sorted_zipped_ba[0, 1])

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


def is_will_out_var_opm(will_out_var_num):
    # opm: obstacle plastic multiplier
    return is_variable_plastic_multiplier(will_out_var_num)


# def get_yield_point_num_from_piece(piece):
#     for yield_point_num, yield_point_pieces in enumerate(yield_points_pieces):
#         if piece in yield_point_pieces:
#             return yield_point_num


# def get_active_yield_points(basic_variables):
#     active_yield_points = []
#     for variable in basic_variables:
#         active_yield_point = get_yield_point_num_from_piece(variable)
#         if active_yield_point is not None:
#             active_yield_points.append(active_yield_point)
#     return active_yield_points


# def is_fpm_for_an_active_yield_point(fpm, active_yield_points):
#     return True if fpm in active_yield_points else False


def update_basic_variables(basic_variables, will_out_row_num, will_in_col_num):
    basic_variables[will_out_row_num] = will_in_col_num
    return basic_variables


def update_fpm(will_out_row_num, cbar):
    fpm = FPM
    fpm.var_num = will_out_row_num
    fpm.cost = cbar[will_out_row_num]
    return fpm


def get_sorted_slack_candidates(basic_variables, b_matrix_inv, cb):
    cbar = calculate_cbar(cb, b_matrix_inv)
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


def reset(basic_variables, x_cumulative, bbar):
    for i, basic_variable in enumerate(basic_variables):
        if basic_variable < variables_num:
            x_cumulative[basic_variables[i], 0] += bbar[i]
            bbar[i] = 0
    return x_cumulative, bbar


def calculate_r(spm_var_num, basic_variables, abar, b_matrix_inv):
    spm_row_num = get_var_row_num(spm_var_num, basic_variables)
    r = abar[spm_row_num] / b_matrix_inv[spm_row_num, spm_var_num]
    return r


def get_var_row_num(var_num, basic_variables):
    row_num = np.where(basic_variables == var_num)[0][0]
    return row_num


def zero_out_small_values(array):
    low_values_flags = abs(array) < settings.computational_zero
    array[low_values_flags] = 0
    return array
