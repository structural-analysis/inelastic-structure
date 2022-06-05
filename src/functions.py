import math
import numpy as np


# math functions
def sin(t):
    return math.sin(t)


def cos(t):
    return math.cos(t)


def sqrt(t):
    return math.sqrt(t)


def apply_boundry_conditions(joints_restraints, matrix):
    # Applying Boundary Conditions to Stiffness Matrix or Force Vector
    reduced_matrix = matrix
    fixed_dof_counter = 0
    if matrix.shape[1] == 1:
        # if we have a vector, for example force vector.
        for bc_counter in range(len(joints_restraints)):
            # delete row 1
            reduced_matrix = np.delete(
                reduced_matrix,
                3 * joints_restraints[bc_counter, 0] + joints_restraints[bc_counter, 1] - fixed_dof_counter,
                0,
            )
            fixed_dof_counter += 1
    elif matrix.shape[1] != 1:
        # if we have a matrix like stiffness or mass.
        if matrix.shape[0] != matrix.shape[1]:
            # This condition is for K0t condensation of stiffness matrix
            moment_dof_counter = 0
            non_moment_index = 0
            for bc_counter in range(len(joints_restraints)):
                if joints_restraints[bc_counter, 1] == 2:
                    reduced_matrix = np.delete(reduced_matrix, joints_restraints[bc_counter, 0]-moment_dof_counter, 0)  # delete row 1
                    moment_dof_counter += 1
                else:
                    # delete column
                    reduced_matrix = np.delete(reduced_matrix, 3 * joints_restraints[bc_counter, 0]+joints_restraints[bc_counter, 1]-(3*joints_restraints[bc_counter, 0]+joints_restraints[bc_counter, 1])//3-non_moment_index, 1)
                    non_moment_index += 1
        else:
            for bc_counter in range(len(joints_restraints)):
                # delete column
                reduced_matrix = np.delete(reduced_matrix, 3 * joints_restraints[bc_counter, 0]+joints_restraints[bc_counter, 1]-fixed_dof_counter, 1)
                # delete row
                reduced_matrix = np.delete(reduced_matrix, 3 * joints_restraints[bc_counter, 0]+joints_restraints[bc_counter, 1]-fixed_dof_counter, 0)
                fixed_dof_counter += 1
    return reduced_matrix


def condense_boundary_condition(boundaries):
    # Computation of Natural Frequencies and Shape Modes
    jj = 0
    non_zero_boundaries = boundaries
    zero_mass_boundaries = boundaries
    jz = 0
    for bc_counter in range(len(non_zero_boundaries)):
        if (3 * non_zero_boundaries[bc_counter - jj, 0] + non_zero_boundaries[bc_counter - jj, 1]) % 3 == 2:
            non_zero_boundaries = np.delete(non_zero_boundaries, bc_counter - jj, 0)
            jj += 1
        elif (3 * non_zero_boundaries[bc_counter - jj, 0] + non_zero_boundaries[bc_counter - jj, 1]) % 3 != 2:
            non_zero_boundaries[bc_counter - jj, 1] = non_zero_boundaries[bc_counter - jj, 1] - (3 * non_zero_boundaries[bc_counter - jj, 0] + non_zero_boundaries[bc_counter - jj, 1]) // 3

        if (3 * zero_mass_boundaries[bc_counter - jz, 0] + zero_mass_boundaries[bc_counter - jz, 1]) % 3 == 2:
            zero_mass_boundaries[bc_counter - jz, 1] = zero_mass_boundaries[bc_counter - jz, 1] - 2 * ((3 * zero_mass_boundaries[bc_counter - jz, 0] + zero_mass_boundaries[bc_counter - jz, 1]) // 3 + 1)
        elif (3 * zero_mass_boundaries[bc_counter - jz, 0] + zero_mass_boundaries[bc_counter - jz, 1]) % 3 != 2:
            zero_mass_boundaries = np.delete(zero_mass_boundaries, bc_counter - jz, 0)
            jz += 1
    return non_zero_boundaries, zero_mass_boundaries


def static_condensation(structure_stiffness, structure_mass, boundaries):
    non_zero_boundaries, zero_mass_boundaries = condense_boundary_condition(boundaries)
    mtt = structure_mass
    ktt = structure_stiffness
    k00 = structure_stiffness
    k0t = structure_stiffness
    # for zero mass rows and columns
    ii = 0
    # for non-zero mass rows and columns
    jj = 0
    n = int(np.shape(structure_stiffness)[0] / 3)
    for i in range(n):
        mtt = np.delete(mtt, 3 * i + 2 - ii, 1)
        mtt = np.delete(mtt, 3 * i + 2 - ii, 0)
        ktt = np.delete(ktt, 3 * i + 2 - ii, 1)
        ktt = np.delete(ktt, 3 * i + 2 - ii, 0)
        k0t = np.delete(k0t, 3 * i + 2 - ii, 1)
        ii += 1
        k00 = np.delete(k00, 3 * i + 1 - jj, 1)
        k00 = np.delete(k00, 3 * i + 1 - jj, 0)
        k0t = np.delete(k0t, 3 * i + 1 - jj, 0)
        jj += 1
        k00 = np.delete(k00, 3 * i + 1 - jj, 1)
        k00 = np.delete(k00, 3 * i + 1 - jj, 0)
        k0t = np.delete(k0t, 3 * i + 1 - jj, 0)
        jj += 1
    kttr = apply_boundry_conditions(non_zero_boundaries, ktt)
    mttr = apply_boundry_conditions(non_zero_boundaries, mtt)
    k00r = apply_boundry_conditions(zero_mass_boundaries, k00)
    k0tr = apply_boundry_conditions(boundaries, k0t)
    k00_inverse = np.linalg.inv(k00r)
    ku0 = -(np.dot(k00_inverse, k0tr))
    khat = kttr - np.dot(np.dot(np.transpose(k0tr), k00_inverse), k0tr)
    return khat, mttr, ku0, k00_inverse, k00r


def load_condensation(force, ku0, boundaries):
    # Input: Loading considering all of degrees of freedoms. For a frame it is: 3*n
    # Output: The result is a load that is condensed and the boundary conditions are applied.
    pt = force
    p0 = force
    n = int(np.shape(force)[0] / 3)
    non_zero_boundaries, zero_mass_boundaries = condense_boundary_condition(boundaries)
    # for zero mass rows and columns
    ii = 0
    # for non-zero mass rows and columns
    jj = 0
    for i in range(n):
        pt = np.delete(pt, 3 * i + 2 - ii, 0)
        ii += 1
        p0 = np.delete(p0, 3 * i + 1 - jj, 0)
        jj += 1
        p0 = np.delete(p0, 3 * i + 1 - jj, 0)
        jj += 1
    ptr = apply_boundry_conditions(non_zero_boundaries, pt)

    p0r = apply_boundry_conditions(zero_mass_boundaries, p0)

    phat = ptr + np.dot(np.transpose(ku0), p0r)

    return phat, p0r


def prepare_raw_data(structure, include_displacement_limit=False):

    # possible minmax are:
    # minimization: min, maximization: max

    minmax = "max"

    # analysis_type = "static"
    load_limit = structure.load_limit
    phi = structure.phi
    p0 = structure.p0
    pv = structure.pv
    yield_points_pieces = structure.yield_points_pieces

    phi_pv_phi = phi.T * pv * phi
    phi_p0 = phi.T * p0

    extra_vars_num = 2 if include_displacement_limit else 1
    total_yield_pieces_num = phi.shape[1]
    variables_num = extra_vars_num + total_yield_pieces_num
    landa_var_num = variables_num - extra_vars_num
    landa_bar_var_num = 2 * variables_num - extra_vars_num
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
    #     b[0:-extra_vars_num] = b[0:-extra_vars_num] - np.dot(phi_pv_phi, xn_previous[0:-extra_vars_num])
    # elif analysis_type == "static":
    #     b[0:-extra_vars_num] = b[0:-extra_vars_num]

    b[-extra_vars_num] = load_limit
    # b[-extra_vars_num] = 1

    # possible inequality_condition are:
    # lt: Less Than or Equal  gt: Larger Than or Eqaul  eq: Equal
    inequality_condition = np.full((variables_num), "lt")

    c = np.zeros(2 * variables_num)
    c[0:total_yield_pieces_num] = 1.0

    mp_data = {
        "variables_num": variables_num,
        "landa_var_num": landa_var_num,
        "landa_bar_var_num": landa_bar_var_num,
        "extra_vars_num": extra_vars_num,
        "raw_a": raw_a,
        "b": b,
        "c": c,
        "minmax": minmax,
        "inequality_condition": inequality_condition,
        "yield_points_pieces": yield_points_pieces,
        "load_limit": load_limit,
    }
    return mp_data



def get_elements_max_dof_num(elements):
    max_dof_num = 0
    for element in elements:
        if element.total_dofs_num >= max_dof_num:
            max_dof_num = element.total_dofs_num
    return max_dof_num
