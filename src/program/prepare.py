import numpy as np


class RawData:
    def __init__(self, structure):
        self.load_limit = structure.limits["load_limit"]
        self.disp_limits = structure.limits["disp_limits"]
        self.phi = structure.phi
        self.p0 = structure.p0
        self.pv = structure.pv
        self.d0 = structure.d0
        self.dv = structure.dv

        self.q = structure.q
        self.h = structure.h
        self.w = structure.w
        self.cs = structure.cs

        self.disp_limits_num = self.disp_limits.shape[0]
        self.limits_num = 1 + self.disp_limits_num * 2
        self.yield_pieces_num = structure.yield_specs.pieces_num
        self.softening_vars_num = 2 * structure.yield_specs.points_num if structure.general.include_softening else 0
        self.yield_points_indices = structure.yield_points_indices

        self.vars_num = self.yield_pieces_num + self.softening_vars_num + 1
        self.slacks_num = self.yield_pieces_num + self.softening_vars_num + self.limits_num
        self.constraints_num = self.slacks_num

        self.table = self._create_table()
        self.landa_var_num = self.yield_pieces_num + self.softening_vars_num
        self.landa_bar_var_num = 2 * self.landa_var_num + 1

        self.limits_slacks = set(range(self.landa_bar_var_num, self.landa_bar_var_num + self.limits_num))
        self.b = self._get_b_column()
        self.c = self._get_costs_row()

    def _create_table(self):
        constraints_num = self.constraints_num
        yield_pieces_num = self.yield_pieces_num
        softening_vars_num = self.softening_vars_num
        disp_limits_num = self.disp_limits_num
        vars_num = self.vars_num

        phi_pv_phi = self.phi.T * self.pv * self.phi
        phi_p0 = self.phi.T * self.p0
        dv_phi = self.dv * self.phi
        empty_a = np.zeros((constraints_num, vars_num))
        raw_a = np.matrix(empty_a)
        raw_a[0:yield_pieces_num, 0:yield_pieces_num] = phi_pv_phi

        if softening_vars_num:
            raw_a[yield_pieces_num:(yield_pieces_num + softening_vars_num), 0:yield_pieces_num] = self.q
            raw_a[0:yield_pieces_num, yield_pieces_num:(yield_pieces_num + softening_vars_num)] = - self.h
            raw_a[yield_pieces_num:(yield_pieces_num + softening_vars_num), yield_pieces_num:(yield_pieces_num + softening_vars_num)] = self.w

        landa_base_num = yield_pieces_num + softening_vars_num
        raw_a[0:yield_pieces_num, landa_base_num] = phi_p0
        raw_a[landa_base_num, landa_base_num] = 1.0

        if self.disp_limits.any():
            disp_limit_base_num = yield_pieces_num + softening_vars_num + 1
            raw_a[disp_limit_base_num:(disp_limit_base_num + disp_limits_num), 0:yield_pieces_num] = dv_phi
            raw_a[(disp_limit_base_num + disp_limits_num):(disp_limit_base_num + 2 * disp_limits_num), 0:yield_pieces_num] = - dv_phi

            raw_a[disp_limit_base_num:(disp_limit_base_num + disp_limits_num), landa_base_num] = self.d0
            raw_a[(disp_limit_base_num + disp_limits_num):(disp_limit_base_num + 2 * disp_limits_num), landa_base_num] = - self.d0

        a_matrix = np.array(raw_a)
        columns_num = vars_num + self.slacks_num
        table = np.zeros((constraints_num, columns_num))
        table[0:constraints_num, 0:vars_num] = a_matrix

        # Assigning diagonal arrays of slack variables.
        j = vars_num
        for i in range(constraints_num):
            table[i, j] = 1.0
            j += 1

        return table

    def _get_b_column(self):
        yield_pieces_num = self.yield_pieces_num
        disp_limits_num = self.disp_limits_num

        b = np.ones((self.constraints_num))
        b[yield_pieces_num + self.softening_vars_num] = self.load_limit
        if self.softening_vars_num:
            b[yield_pieces_num:(yield_pieces_num + self.softening_vars_num)] = np.array(self.cs)[:, 0]

        if self.disp_limits.any():
            disp_limit_base_num = yield_pieces_num + self.softening_vars_num + 1
            b[disp_limit_base_num:(disp_limit_base_num + disp_limits_num)] = abs(self.disp_limits[:, 2])
            b[(disp_limit_base_num + disp_limits_num):(disp_limit_base_num + 2 * disp_limits_num)] = abs(self.disp_limits[:, 2])

        return b

    def _get_costs_row(self):
        c = np.zeros(self.vars_num + self.slacks_num)
        c[0:self.yield_pieces_num] = 1.0
        return -1 * c
