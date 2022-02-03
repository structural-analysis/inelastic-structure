import numpy as np


def prepare_raw_data(structure, load_limit, include_displacement_limit=False):

    # possible minmax are:
    # minimization: min, maximization: max

    minmax = "max"

    # analysis_type = "static"
    phi = structure.phi
    p0 = structure.p0
    pv = structure.pv

    phi_pv_phi = phi.T * pv * phi
    phi_p0 = phi.T * p0

    extra_numbers_num = 2 if include_displacement_limit else 1
    total_yield_pieces_num = phi.shape[1]
    variables_num = extra_numbers_num + total_yield_pieces_num

    empty_a = np.zeros((variables_num, variables_num))
    raw_a = np.matrix(empty_a)

    # np.savetxt("phiPvPhi.csv", phi_pv_phi, delimiter=",")
    # np.savetxt("phiP0.csv", phi_p0, delimiter=",")

    raw_a[0:total_yield_pieces_num, 0:total_yield_pieces_num] = phi_pv_phi[0:total_yield_pieces_num, 0:total_yield_pieces_num]
    raw_a[0:total_yield_pieces_num, total_yield_pieces_num] = phi_p0[0:total_yield_pieces_num, 0]
    raw_a[total_yield_pieces_num, total_yield_pieces_num] = 1.0
    b = np.ones((variables_num))

    # if analysis_type == "dynamic":
    #     pass
    #     b[0:-extra_numbers_num] = b[0:-extra_numbers_num] - np.dot(phi_pv_phi, xn_previous[0:-extra_numbers_num])
    # elif analysis_type == "static":
    #     b[0:-extra_numbers_num] = b[0:-extra_numbers_num]

    b[-extra_numbers_num] = load_limit

    # possible inequality_condition are:
    # lt: Less Than or Equal  gt: Larger Than or Eqaul  eq: Equal
    inequality_condition = np.full((variables_num), "lt")

    c = np.zeros(2 * variables_num)
    c[0:total_yield_pieces_num] = 1.0
    mp_data = {
        "variables_num": variables_num,
        "extra_numbers_num": extra_numbers_num,
        "raw_a": raw_a,
        "b": b,
        "c": c,
        "minmax": minmax,
        "inequality_condition": inequality_condition
    }
    return mp_data


def _zero_out_small_values(matrix, floor=1e-8):
    # where values are low
    low_values_flags = abs(matrix) < floor
    # all low values set to 0
    matrix[low_values_flags] = 0
    return matrix


def _find_pivot_element(basic_variables, tableau):
    variables_num = basic_variables.shape[0]
    for i in range(variables_num):
        if tableau[i, variables_num] > 0.0:
            min_ba = tableau[i, variables_num + 3] / tableau[i, variables_num]
    for i in range(variables_num):
        if tableau[i, variables_num] > 0.0:
            ba = tableau[i, variables_num + 3] / tableau[i, variables_num]
            if ba <= min_ba:
                min_ba = ba
                will_out_row_num = i
                will_out = basic_variables[will_out_row_num]
    return will_out_row_num, will_out


def _pivot_operation_on_pivot_element(will_out_row_num, will_in, basic_variables, tableau):
    variables_num = basic_variables.shape[0]
    basic_variables[will_out_row_num] = will_in
    pivot_element = tableau[will_out_row_num, variables_num]
    all_columns_num = variables_num + 4
    all_rows_num = variables_num + 1

    # Pivot Operation on pivot row
    for i in range(all_columns_num):
        tableau[will_out_row_num, i] = tableau[will_out_row_num, i] / pivot_element

    tableau = _zero_out_small_values(tableau)

    # Pivot Operation on other rows
    for i in range(all_rows_num):
        if i != will_out_row_num:
            pivot_columnt_i_member = tableau[i, variables_num]
            for j in range(all_columns_num):
                tableau[i, j] = tableau[i, j] - pivot_columnt_i_member * tableau[will_out_row_num, j]

    tableau = _zero_out_small_values(tableau)
    return basic_variables, tableau


def complementarity_programming(mp_data):
    # CA==1: hameye C haye manfi daraye a namosbat hastand
    # CA==2: hameye C haye manfi daraye a namosbat nistand
    # nC: No. of Constraint Equations
    # nV: No. of Variables
    # minmax: A parameter which declares that optimization is minimization of maximization
    # 1: Minimization   2: Maximization
    # equ: is a parameter to show the constraint is equality or unequality
    # 1: Less Than or Equal  2: Larger Than or Eqaul  3: Equal
    # A: Matrix of Coefficient of Variables
    # b: Vector of Constants
    # C: Vector of Cost Factors
    # f: Value of optimization
    # dn: Parameter to check that d is nonnegative or not. 1: Negative, 2: Nonnegative
    # ba: bi/ais
    # minba: minimume of bi/ais
    # bV: Basic Variable no.
    # !jj: A Parameter that in each iteration if xy==0 its value will increase by 1. At the first jj=0.
    # !xy: A parameter to distinguish wheather if x is in basis, y=xs or not? xy==0: x in basis and y=xs,
    # vice versa. xy==1: x in basis and y/=Xs, vice versa.
    variables_num = mp_data["variables_num"]
    raw_a = mp_data["raw_a"]
    b = mp_data["b"]
    c = mp_data["c"]
    minmax = mp_data["minmax"]
    inequality_condition = mp_data["inequality_condition"]
    extra_numbers_num = mp_data["extra_numbers_num"]

    cbar = np.zeros((2 * variables_num))
    # ------------------------------------Construction of Aj
    # Apply Inequality Conditions
    for i in range(variables_num):
        if inequality_condition[i] == "gt":
            b[i] = -b[i]
            raw_a[i, 0:variables_num] = -raw_a[i, 0:variables_num]

    # -------------Creating Minimization Objective Function
    if minmax == "max":  # Maximization
        c[0:variables_num] = -c[0:variables_num]
    # ----------------------------Creation of Aj Matrix
    # the first [1:nC,1:nV] arrays of Aj are arrays of A, and the others except the main diagonal arrays that are (1), are zero.
    a = np.zeros((variables_num, 2 * variables_num))
    a[0:variables_num, 0:variables_num] = raw_a[0:variables_num, 0:variables_num]
    j = variables_num

    # Assigning diagonal arrays of y variables.
    for i in range(variables_num):
        a[i, j] = 1.0
        j += 1

    # In dynamic analysis we may have negative b, so to achieve canonical form we must use 2phase mathematical programming.
    # So that we must check b values and if we had negative ones, we must use 2phase programming.
    # So at the first we check if we have negative b or not:
    # sorted_b = np.sort(b)
    # sorted_b_indices = np.argsort(b)

    # set the initial basic variables
    basic_variables = np.zeros(variables_num)
    for i in range(variables_num):
        basic_variables[i] = variables_num + i

    # if np.any(b < 0):
    negative_constraints_num = 0
    negative_constraints = []
    for i in range(variables_num):
        if b[i] < 0:
            b[i] = -b[i]
            a[i, 0:variables_num] = -a[i, 0:variables_num]
            basic_variables[i] = i + 2 * variables_num
            negative_constraints_num += 1
            negative_constraints.append(i)

    a2phase = np.zeros((variables_num, 2 * variables_num + negative_constraints_num))
    a2phase[0:variables_num, 0:2 * variables_num] = a[0:variables_num, 0:2 * variables_num]
    j = 0
    # Assigning diagonal arrays of z variables.
    for i in range(len(negative_constraints)):
        a2phase[negative_constraints[i], 2 * variables_num + j] = 1
        j += 1
    # np.savetxt("Aj2Phase.csv", aj2phase, delimiter=",")

    # # Computation of vector d
    # d = np.zeros((2 * variables_num + negative_constraints_num))
    # d[0:2 * variables_num] = -np.sum(a[0:variables_num, 0:2 * variables_num], axis=0)

    tableau = np.zeros((variables_num + 2, variables_num + 4))

    j = 0
    # Assigning diagonal arrays of y variables.
    for i in range(variables_num):
        tableau[i, j] = 1.0
        j += 1

    tableau[0:variables_num, variables_num + 3] = b
    tableau[variables_num, variables_num + 1] = 1.0
    tableau[variables_num + 1, variables_num + 2] = 1.0
    tableau[variables_num + 1, variables_num + 3] = -np.sum(b)
    tableau = _zero_out_small_values(tableau)

    # dbar = np.zeros(2 * variables_num + negative_constraints_num)
    # dbar = np.dot(tableau[variables_num + 1, 0:variables_num], a2phase[0:variables_num, 0:2 * variables_num + negative_constraints_num]) + d
    # dbar = _zero_out_small_values(dbar)

    # At the first the entering element is Loading multiplier (Lambda)
    will_in = variables_num - 1

    # Calculation of Abars
    tableau[0:variables_num, variables_num] = a[0:variables_num, will_in]

    # Finding the exiting variable (r)
    will_out_row_num, will_out = _find_pivot_element(basic_variables, tableau)

    # Pivot Operation
    basic_variables, tableau = _pivot_operation_on_pivot_element(will_out_row_num, will_in, basic_variables, tableau)
    cbar[0:2 * variables_num] = np.dot(tableau[variables_num, 0:variables_num], a[0:variables_num, 0:2 * variables_num]) + c[0:2 * variables_num]
    cbar = _zero_out_small_values(cbar)
    will_in = int(will_out - variables_num)
    index_ca = np.argsort(cbar)
    atemp = np.dot(tableau[0:variables_num, 0:variables_num], a[0:variables_num, index_ca[will_in]])
    atemp = _zero_out_small_values(atemp)

    #  -----------------------------Checking if all Abars are non-positive or not
    for i, ai in enumerate(atemp):
        if ai > 0 and basic_variables[i] != variables_num - 1 and basic_variables[i] != 2 * variables_num - 1:
            unbounded = False
            break
        elif ai <= 0 and basic_variables[i] != variables_num - 1 and basic_variables[i] != 2 * variables_num - 1:
            unbounded = True
    # while unbounded is True:
    #     if atemp[i] > 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
    #         unbounded = False
    #     elif atemp[i] <= 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
    #         unbounded = True
    #     i += 1
    while will_out != 2 * variables_num - 1 and unbounded is False:
        # Calculation of Abars
        # tableau[0:nC, nC] = Aj[0:nC, s]
        tableau[0:variables_num, variables_num] = np.dot(tableau[0:variables_num, 0:variables_num], a[0:variables_num, will_in])
        tableau[variables_num, variables_num] = cbar[will_in]
        tableau = _zero_out_small_values(tableau)
        will_out_row_num, will_out = _find_pivot_element(basic_variables, tableau)
        # Pivot Operation
        basic_variables, tableau = _pivot_operation_on_pivot_element(will_out_row_num, will_in, basic_variables, tableau)
        cbar[0:2 * variables_num] = c[0:2 * variables_num] + np.dot(tableau[variables_num, 0:variables_num], a[0:variables_num, 0:2 * variables_num])
        cbar = _zero_out_small_values(cbar)
        will_in = int(will_out - variables_num)
        index_ca = np.argsort(cbar)
        atemp = np.dot(tableau[0:variables_num, 0:variables_num], a[0:variables_num, index_ca[will_in]])
        atemp = _zero_out_small_values(atemp)
        # unbounded = True
        # i = 0
        #  -----------------------------Checking if all Abars are non-positive or not
        for i, ai in enumerate(atemp):
            if ai > 0 and basic_variables[i] != variables_num - 1 and basic_variables[i] != 2 * variables_num - 1:
                unbounded = False
                break
            elif ai <= 0 and basic_variables[i] != variables_num - 1 and basic_variables[i] != 2 * variables_num - 1:
                unbounded = True
        # while unbounded is True:
        #     print(i)
        #     print(atemp[i])
        #     if atemp[i] > 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
        #         unbounded = False
        #     elif atemp[i] <= 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
        #         unbounded = True
        #     i += 1
    empty_xn = np.zeros((variables_num, 1))
    xn = np.matrix(empty_xn)
    for i in range(variables_num):
        if int(basic_variables[i]) < variables_num:
            xn[int(basic_variables[i]), 0] = tableau[i, variables_num + 3]
    plastic_multipliers = xn[0:-extra_numbers_num, 0]
    return plastic_multipliers
