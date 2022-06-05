import numpy as np


def get_analysis_data(structure):

    load_limit = structure.limits["load_limit"]
    disp_limits = structure.limits["disp_limits"]
    phi = structure.phi
    p0 = structure.p0
    pv = structure.pv
    yield_points_pieces = structure.yield_points_pieces

    phi_pv_phi = phi.T * pv * phi
    phi_p0 = phi.T * p0

    # if there's more constraints, extra numbers will vary.
    # TODO: handle extra limits in programming and responses.
    extra_limits_num = disp_limits.shape[0]  # + force_limits.shape[0], ...
    total_yield_pieces_num = phi.shape[1]
    variables_num = extra_limits_num + total_yield_pieces_num
    landa_var_num = variables_num - extra_limits_num
    landa_bar_var_num = 2 * variables_num - extra_limits_num
    empty_a = np.zeros((variables_num, variables_num))
    raw_a = np.matrix(empty_a)

    raw_a[0:total_yield_pieces_num, 0:total_yield_pieces_num] = phi_pv_phi[0:total_yield_pieces_num, 0:total_yield_pieces_num]
    raw_a[0:total_yield_pieces_num, total_yield_pieces_num] = phi_p0[0:total_yield_pieces_num, 0]
    raw_a[total_yield_pieces_num, total_yield_pieces_num] = 1.0
    b = np.ones((variables_num))

    b[-extra_limits_num] = load_limit
    # # for dynamic
    # inequality_condition = np.full((variables_num), "lt")

    c = np.zeros(2 * variables_num)
    c[0:total_yield_pieces_num] = 1.0

    analysis_data = {
        "variables_num": variables_num,
        "landa_var_num": landa_var_num,
        "landa_bar_var_num": landa_bar_var_num,
        "raw_a": raw_a,
        "b": b,
        "c": c,
        "yield_points_pieces": yield_points_pieces,
    }
    return analysis_data
