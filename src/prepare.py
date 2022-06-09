import numpy as np


def get_analysis_data(structure):

    load_limit = structure.limits["load_limit"]
    disp_limits = structure.limits["disp_limits"]
    phi = structure.phi
    p0 = structure.p0
    pv = structure.pv
    d0 = structure.d0
    dv = structure.dv

    yield_points_pieces = structure.yield_points_pieces

    phi_pv_phi = phi.T * pv * phi
    phi_p0 = phi.T * p0

    dv_phi = dv * phi

    disp_limits_num = disp_limits.shape[0]
    limits_num = 1 + disp_limits_num * 2  # + force_limits.shape[0], ...
    yield_pieces_num = phi.shape[1]
    variables_num = limits_num + yield_pieces_num
    landa_var_num = yield_pieces_num
    landa_bar_var_num = 2 * variables_num - limits_num
    empty_a = np.zeros((variables_num, variables_num))
    raw_a = np.matrix(empty_a)

    raw_a[0:yield_pieces_num, 0:yield_pieces_num] = phi_pv_phi
    raw_a[0:yield_pieces_num, yield_pieces_num] = phi_p0
    raw_a[yield_pieces_num, yield_pieces_num] = 1.0
    b = np.ones((variables_num))
    b[yield_pieces_num] = load_limit

    if disp_limits.any():
        raw_a[yield_pieces_num + 1:(yield_pieces_num + disp_limits_num + 1), 0:yield_pieces_num] = dv_phi
        raw_a[(yield_pieces_num + disp_limits_num + 1):(yield_pieces_num + 2 * disp_limits_num + 1), 0:yield_pieces_num] = - dv_phi

        raw_a[yield_pieces_num + 1:(yield_pieces_num + disp_limits_num + 1), yield_pieces_num] = d0
        raw_a[(yield_pieces_num + disp_limits_num + 1):(yield_pieces_num + 2 * disp_limits_num + 1), yield_pieces_num] = - d0

        b[yield_pieces_num + 1:(yield_pieces_num + disp_limits_num + 1)] = abs(disp_limits[:, 2])
        b[(yield_pieces_num + disp_limits_num + 1):(yield_pieces_num + 2 * disp_limits_num + 1)] = abs(disp_limits[:, 2])

    # # for dynamic
    # inequality_condition = np.full((variables_num), "lt")

    c = np.zeros(2 * variables_num)
    c[0:yield_pieces_num] = 1.0

    limits_slacks = set(range(landa_bar_var_num, 2 * variables_num))
    analysis_data = {
        "variables_num": variables_num,
        "landa_var_num": landa_var_num,
        "raw_a": raw_a,
        "b": b,
        "c": c,
        "yield_points_pieces": yield_points_pieces,
        "limits_slacks": limits_slacks,
    }
    return analysis_data
