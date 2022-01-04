import numpy as np


def apply_boundry_conditions(joints_restraints, matrix):
    # Applying Boundary Conditions to Stiffness Matrix or Force Vector
    reduced_matrix = matrix
    fixed_dof_counter = 0
    if matrix.shape[1] == 1:
        # if we have a vector, for example force vector.
        for bc_index in range(len(joints_restraints)):
            reduced_matrix = np.delete(reduced_matrix, 3*joints_restraints[bc_index, 0]+joints_restraints[bc_index, 1]-fixed_dof_counter, 0)  # delete row 1
            fixed_dof_counter += 1
    elif matrix.shape[1] != 1:
        # if we have a matrix like stiffness or mass.
        if matrix.shape[0] != matrix.shape[1]:
            # This condition is for K0t condensation of stiffness matrix
            moment_dof_counter = 0
            non_moment_index = 0
            for bc_index in range(len(joints_restraints)):
                if joints_restraints[bc_index, 1] == 2:
                    # np.delete(nonZeroMassJTR,BC-jj,0)
                    reduced_matrix = np.delete(reduced_matrix, joints_restraints[bc_index, 0]-moment_dof_counter, 0)  # delete row 1
                    moment_dof_counter += 1
                else:
                    # nonZeroMassJTR[BC-jj,1]-(3*nonZeroMassJTR[BC-jj,0]+nonZeroMassJTR[BC-jj,1])//3
                    reduced_matrix = np.delete(reduced_matrix, 3*joints_restraints[bc_index, 0]+joints_restraints[bc_index, 1]-(3*joints_restraints[bc_index, 0]+joints_restraints[bc_index, 1])//3-non_moment_index, 1)  # delete column 1
                    non_moment_index += 1
        else:
            for bc_index in range(len(joints_restraints)):
                reduced_matrix = np.delete(reduced_matrix, 3*joints_restraints[bc_index, 0]+joints_restraints[bc_index, 1]-fixed_dof_counter, 1)  # delete column 1
                reduced_matrix = np.delete(reduced_matrix, 3*joints_restraints[bc_index, 0]+joints_restraints[bc_index, 1]-fixed_dof_counter, 0)  # delete row 1
                fixed_dof_counter += 1
    return reduced_matrix


# # Computation of Natural Frequencies and Shape Modes
def condense_BC(JTR):
    jj = 0
    nonZeroJTR = JTR
    zeroMassJTR = JTR
    jz = 0
    for BC in range(len(nonZeroJTR)):
        if (3*nonZeroJTR[BC-jj, 0]+nonZeroJTR[BC-jj, 1]) % 3 == 2:
            nonZeroJTR = np.delete(nonZeroJTR, BC-jj, 0)
            jj += 1
        elif (3*nonZeroJTR[BC-jj, 0]+nonZeroJTR[BC-jj, 1]) % 3 != 2:
            nonZeroJTR[BC-jj, 1] = nonZeroJTR[BC-jj, 1]-(3*nonZeroJTR[BC-jj, 0]+nonZeroJTR[BC-jj, 1])//3

        if (3*zeroMassJTR[BC-jz, 0]+zeroMassJTR[BC-jz, 1]) % 3 == 2:
            zeroMassJTR[BC-jz, 1] = zeroMassJTR[BC-jz, 1]-2*((3*zeroMassJTR[BC-jz, 0]+zeroMassJTR[BC-jz, 1])//3+1)
        elif (3*zeroMassJTR[BC-jz, 0]+zeroMassJTR[BC-jz, 1]) % 3 != 2:
            zeroMassJTR = np.delete(zeroMassJTR, BC-jz, 0)
            jz += 1
    return nonZeroJTR, zeroMassJTR


def static_condensation(Kt, M, JTR):
    nonZeroJTR, zeroMassJTR = condense_BC(JTR)
    Mtt = M
    Ktt = Kt
    K00 = Kt
    K0t = Kt
    ii = 0  # for zero mass rows and columns
    jj = 0  # for non-zero mass rows and columns
    n = int(np.shape(Kt)[0]/3)
    for i in range(n):
        Mtt = np.delete(Mtt, 3*i+2-ii, 1)  # delete column 1
        Mtt = np.delete(Mtt, 3*i+2-ii, 0)  # delete row 0
        Ktt = np.delete(Ktt, 3*i+2-ii, 1)  # delete column 1
        Ktt = np.delete(Ktt, 3*i+2-ii, 0)  # delete row 0
        K0t = np.delete(K0t, 3*i+2-ii, 1)
        ii += 1
        K00 = np.delete(K00, 3*i+1-jj, 1)  # delete column 1
        K00 = np.delete(K00, 3*i+1-jj, 0)  # delete row 0
        K0t = np.delete(K0t, 3*i+1-jj, 0)
        jj += 1
        K00 = np.delete(K00, 3*i+1-jj, 1)  # delete column 1
        K00 = np.delete(K00, 3*i+1-jj, 0)  # delete row 0
        K0t = np.delete(K0t, 3*i+1-jj, 0)
        jj += 1
    KttR = apply_boundry_conditions(nonZeroJTR, Ktt)
    MttR = apply_boundry_conditions(nonZeroJTR, Mtt)
    K00R = apply_boundry_conditions(zeroMassJTR, K00)
    K0tR = apply_boundry_conditions(JTR, K0t)
    K00Inv = np.linalg.inv(K00R)
    KU0 = -(np.dot(K00Inv, K0tR))
    Khat = KttR-np.dot(np.dot(np.transpose(K0tR), K00Inv), K0tR)
    return Khat, MttR, KU0, K00Inv, K00R


def load_condensation(P, KU0, JTR):
    # Input: Loading considering all of degrees of freedoms. For a frame it is: 3*n
    # Output: The result is a load that is condensed and the boundary conditions are applied.
    Pt = P
    P0 = P
    n = int(np.shape(P)[0]/3)
    nonZeroJTR, zeroMassJTR = condense_BC(JTR)
    ii = 0  # for zero mass rows and columns
    jj = 0  # for non-zero mass rows and columns
    for i in range(n):
        Pt = np.delete(Pt, 3*i+2-ii, 0)  # delete row 0
        ii += 1
        P0 = np.delete(P0, 3*i+1-jj, 0)  # delete row 0
        jj += 1
        P0 = np.delete(P0, 3*i+1-jj, 0)  # delete row 0
        jj += 1

    PtR = apply_boundry_conditions(nonZeroJTR, Pt)

    P0R = apply_boundry_conditions(zeroMassJTR, P0)

    Phat = PtR+np.dot(np.transpose(KU0), P0R)

    return Phat, P0R
