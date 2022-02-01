import numpy as np


def prepare_raw_data(structure, load_limit=95500, include_displacement_limit=False):
    """
    In this function we get elements internal forces due to
    Loading on structure and unit deformation loading and compute
    input data for Mathematical Programmig.

    :return: [description]
    :rtype: [type]
    """
    analysis_type = "static"
    phi = structure.phi
    p0 = structure.p0
    pv = structure.pv

    phi_pv_phi = phi.T * pv * phi
    phi_p0 = phi.T * p0

    extra_numbers_num = 2 if include_displacement_limit else 1
    yield_lines_num = phi.shape[1]
    variables_num = extra_numbers_num + yield_lines_num
    constraints_num = phi.shape[1]

    empty_a = np.zeros((variables_num, variables_num))
    a = np.matrix(empty_a)

    # np.savetxt("phiPvPhi.csv", phi_pv_phi, delimiter=",")
    # np.savetxt("phiP0.csv", phi_p0, delimiter=",")
    a[0:yield_lines_num, 0:yield_lines_num] = phi_pv_phi[0:yield_lines_num, 0:yield_lines_num]
    a[0:yield_lines_num, yield_lines_num] = phi_p0[0:yield_lines_num, 0]
    a[yield_lines_num, yield_lines_num] = 1.0
    b = np.ones((constraints_num))
    if analysis_type == "dynamic":
        pass
        # b[0:-extra_numbers_num] = b[0:-extra_numbers_num] - np.dot(phi_pv_phi, xn_previous[0:-extra_numbers_num])
    elif analysis_type == "static":
        b[0:-extra_numbers_num] = b[0:-extra_numbers_num]
    b[-extra_numbers_num] = load_limit
    # possible minmax are:
    # minimization: min, maximization: max
    minmax = "max"
    # possible inequality_condition are:
    # lt: Less Than or Equal  gt: Larger Than or Eqaul  eq: Equal
    inequality_condition = np.full((constraints_num), "lt")
    # 1: Less Than or Equal  2: Larger Than or Eqaul  3: Equal
    c = np.zeros(2 * variables_num)
    c[0:yield_lines_num] = 1.0
    mp_data = {
        "a": a,
        "b": b,
        "c": c,
        "minmax": minmax,
        "inequality_condition": inequality_condition
    }
    return mp_data


def _zero_out_small_values(matrix):
    """Zero out the small values

    :param matrix: matrix
    :type matrix: matrix of digits
    :return: matrix
    :rtype: matrix of digits
    """
    low_value_limit = 1e-8
    # Where values are low
    low_values_flags = abs(matrix) < low_value_limit
    # All low values set to 0
    matrix[low_values_flags] = 0
    return matrix


def find_pivot_element(basic_variables, tableau):
    nc = tableau.shape[0] - 2
    for i in range(nc):
        if tableau[i, nc] > 0.0:
            min_ba = tableau[i, nc + 3] / tableau[i, nc]

    for i in range(nc):
        if tableau[i, nc] > 0.0:
            ba = tableau[i, nc + 3] / tableau[i, nc]
            if ba <= min_ba:
                min_ba = ba
                row_number_of_exiting_value = i
                r = basic_variables[row_number_of_exiting_value]
    return row_number_of_exiting_value, r


def pivot_operation_on_pivot_element(row_number_of_exiting_value, s, bv, tableau):
    # Pivot Operation on ars
    nc = tableau.shape[0] - 2
    bv[row_number_of_exiting_value] = s
    ars = tableau[row_number_of_exiting_value, nc]

    for i in range(nc + 4):
        tableau[row_number_of_exiting_value, i] = tableau[row_number_of_exiting_value, i] / ars

    tableau = _zero_out_small_values(tableau)

    # !--------------------Pivot Operation on other rows
    for i in range(nc + 1):
        if i != row_number_of_exiting_value:
            ais = tableau[i, nc]
            for j in range(nc + 4):
                tableau[i, j] = tableau[i, j] - ais * tableau[row_number_of_exiting_value, j]

    tableau = _zero_out_small_values(tableau)
    return bv, tableau


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
    a = mp_data["a"]
    b = mp_data["b"]
    c = mp_data["c"]
    minmax = mp_data["minmax"]
    inequality_condition = mp_data["inequality_condition"]

    nc = a.shape[0]
    nv = a.shape[1]
    cbar = np.zeros((nv + nc))
    # ------------------------------------Construction of Aj
    # Apply Inequality Conditions
    for i in range(nc):
        if inequality_condition[i] == "gt":
            b[i] = -b[i]
            a[i, 0:nv] = -a[i, 0:nv]

    # -------------Creating Minimization Objective Function
    if minmax == "max":  # Maximization
        c[0:nv] = -c[0:nv]
    # ----------------------------Creation of Aj Matrix
    # the first [1:nC,1:nV] arrays of Aj are arrays of A, and the others except the main diagonal arrays that are (1), are zero.
    aj = np.zeros((nc, nc + nv))
    aj[0:nc, 0:nv] = a[0:nc, 0:nv]
    j = nv
    # Assigning diagonal arrays of y variables.
    for i in range(nc):
        aj[i, j] = 1.0
        j += 1
    # In dynamic analysis we may have negative b, so to achieve canonical form we must use 2phase mathematical programming.
    # So that we must check b values and if we had negative ones, we must use 2phase programming.
    # So at the first we check if we have negative b or not:
    # sorted_b = np.sort(b)
    # sorted_b_indices = np.argsort(b)

    # set the initial basic variables
    bv = np.zeros(nc)
    for i in range(nc):
        bv[i] = nc + i

    # if np.any(b < 0):
    num_of_negative_constraint = 0
    negative_constraints = []
    for i in range(nc):
        if b[i] < 0:
            b[i] = -b[i]
            aj[i, 0:nv] = -aj[i, 0:nv]
            bv[i] = i + nv + nc
            num_of_negative_constraint += 1
            negative_constraints.append(i)

    aj2phase = np.zeros((nc, nv + nc + num_of_negative_constraint))
    aj2phase[0:nc, 0:nv + nc] = aj[0:nc, 0:nv + nc]
    j = 0
    # Assigning diagonal arrays of z variables.
    for i in range(len(negative_constraints)):
        aj2phase[negative_constraints[i], nv + nc + j] = 1
        j += 1
    # np.savetxt("Aj2Phase.csv", aj2phase, delimiter=",")
    # Computation of vector d
    d = np.zeros((nv + nc + num_of_negative_constraint))
    # for i in range(nV+nC):
    d[0:nc + nv] = -np.sum(aj[0:nc, 0:nv + nc], axis=0)

    tableau = np.zeros((nc + 2, nc + 4))
    j = 0
    # Assigning diagonal arrays of y variables.
    for i in range(nc):
        tableau[i, j] = 1.0
        j += 1
    tableau[0:nc, nc + 3] = b
    tableau[nc, nc + 1] = 1.0
    tableau[nc + 1, nc + 2] = 1.0
    tableau[nc + 1, nc + 3] = -np.sum(b)
    tableau = _zero_out_small_values(tableau)

    dbar = np.zeros(nv + nc + num_of_negative_constraint)
    dbar = np.dot(tableau[nc + 1, 0:nc], aj2phase[0:nc, 0:nv + nc + num_of_negative_constraint]) + d
    dbar = _zero_out_small_values(dbar)

    # At the first the entering element is Loading multiplier (Lambda)
    s = nc - 1
    # Calculation of Abars
    tableau[0:nc, nc] = aj[0:nc, s]
    # Finding the exiting variable (r)
    row_number_of_exiting_value, r = find_pivot_element(s, bv, tableau)
    # Pivot Operation
    bv, tableau = pivot_operation_on_pivot_element(row_number_of_exiting_value, s, bv, tableau)
    cbar[0:nc + nv] = np.dot(tableau[nc, 0:nc], aj[0:nc, 0:nc + nv]) + c[0:nc + nv]
    cbar = _zero_out_small_values(cbar)
    s = int(r - nc)
    index_ca = np.argsort(cbar)
    atemp = np.dot(tableau[0:nv, 0:nc], aj[0:nc, index_ca[s]])
    atemp = _zero_out_small_values(atemp)

    #  -----------------------------Checking if all Abars are non-positive or not
    for i, ai in enumerate(atemp):
        if ai > 0 and bv[i] != nc - 1 and bv[i] != nc + nv - 1:
            unbounded = False
            break
        elif ai <= 0 and bv[i] != nc - 1 and bv[i] != nc + nv - 1:
            unbounded = True
    # while unbounded is True:
    #     if atemp[i] > 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
    #         unbounded = False
    #     elif atemp[i] <= 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
    #         unbounded = True
    #     i += 1
    while r != nc + nc - 1 and unbounded is False:
        # Calculation of Abars
        # tableau[0:nC, nC] = Aj[0:nC, s]
        tableau[0:nc, nc] = np.dot(tableau[0:nc, 0:nc], aj[0:nc, s])
        tableau[nc, nc] = cbar[s]
        tableau = _zero_out_small_values(tableau)
        row_number_of_exiting_value, r = find_pivot_element(s, bv, tableau)
        # Pivot Operation
        bv, tableau = pivot_operation_on_pivot_element(row_number_of_exiting_value, s, bv, tableau)
        cbar[0:nc + nv] = c[0:nc + nv] + np.dot(tableau[nc, 0:nc], aj[0:nc, 0:nc + nv])
        cbar = _zero_out_small_values(cbar)
        s = int(r - nc)
        index_ca = np.argsort(cbar)
        atemp = np.dot(tableau[0:nv, 0:nc], aj[0:nc, index_ca[s]])
        atemp = _zero_out_small_values(atemp)
        # unbounded = True
        # i = 0
        #  -----------------------------Checking if all Abars are non-positive or not
        for i, ai in enumerate(atemp):
            if ai > 0 and bv[i] != nc - 1 and bv[i] != nc + nv - 1:
                unbounded = False
                break
            elif ai <= 0 and bv[i] != nc - 1 and bv[i] != nc + nv - 1:
                unbounded = True
        # while unbounded is True:
        #     print(i)
        #     print(atemp[i])
        #     if atemp[i] > 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
        #         unbounded = False
        #     elif atemp[i] <= 0 and bV[i] != nC-1 and bV[i] != nC+nV-1:
        #         unbounded = True
        #     i += 1
    xn = np.zeros(nc)
    for i in range(nc):
        if int(bv[i]) < nc:
            xn[int(bv[i])] = tableau[i, nc + 3]

    return xn
