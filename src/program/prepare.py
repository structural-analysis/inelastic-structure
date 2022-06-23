import numpy as np


class RawData:
    def __init__(self, structure):
        self.load_limit = structure.limits["load_limit"]
        self.disp_limits = structure.limits["disp_limits"]
        self.disp_limits_num = self.disp_limits.shape[0]
        self.limits_num = 1 + self.disp_limits_num * 2
        self.phi = structure.phi
        self.p0 = structure.p0
        self.pv = structure.pv
        self.d0 = structure.d0
        self.dv = structure.dv
        self.yield_pieces_num = self.phi.shape[1]
        self.vars_num = self.limits_num + self.yield_pieces_num
        self.table = self._create_table()
        self.landa_var_num = self.yield_pieces_num
        self.landa_bar_var_num = 2 * self.vars_num - self.limits_num
        self.limits_slacks = set(range(self.landa_bar_var_num, 2 * self.vars_num))
        self.b = self._get_b_column()
        self.c = self._get_costs_row()
        # TODO: structure.h * -1 because alpha = 1-h*xbars

    def _create_table(self):
        vars_num = self.vars_num
        yield_pieces_num = self.yield_pieces_num
        disp_limits_num = self.disp_limits_num
        phi_pv_phi = self.phi.T * self.pv * self.phi
        phi_p0 = self.phi.T * self.p0
        dv_phi = self.dv * self.phi

        empty_a = np.zeros((vars_num, vars_num))
        raw_a = np.matrix(empty_a)
        raw_a[0:yield_pieces_num, 0:yield_pieces_num] = phi_pv_phi
        raw_a[0:yield_pieces_num, yield_pieces_num] = phi_p0
        raw_a[yield_pieces_num, yield_pieces_num] = 1.0

        if self.disp_limits.any():
            raw_a[yield_pieces_num + 1:(yield_pieces_num + disp_limits_num + 1), 0:yield_pieces_num] = dv_phi
            raw_a[(yield_pieces_num + disp_limits_num + 1):(yield_pieces_num + 2 * disp_limits_num + 1), 0:yield_pieces_num] = - dv_phi

            raw_a[yield_pieces_num + 1:(yield_pieces_num + disp_limits_num + 1), yield_pieces_num] = self.d0
            raw_a[(yield_pieces_num + disp_limits_num + 1):(yield_pieces_num + 2 * disp_limits_num + 1), yield_pieces_num] = - self.d0

        a_matrix = np.array(raw_a)
        table = np.zeros((vars_num, 2 * vars_num))
        table[0:vars_num, 0:vars_num] = a_matrix[0:vars_num, 0:vars_num]
        j = vars_num

        # Assigning diagonal arrays of slack variables.
        for i in range(vars_num):
            table[i, j] = 1.0
            j += 1

        return table

    def _get_b_column(self):
        yield_pieces_num = self.yield_pieces_num
        disp_limits_num = self.disp_limits_num
        b = np.ones((self.vars_num))
        b[yield_pieces_num] = self.load_limit

        if self.disp_limits.any():
            b[yield_pieces_num + 1:(yield_pieces_num + disp_limits_num + 1)] = abs(self.disp_limits[:, 2])
            b[(yield_pieces_num + disp_limits_num + 1):(yield_pieces_num + 2 * disp_limits_num + 1)] = abs(self.disp_limits[:, 2])

        return b

    def _get_costs_row(self):
        c = np.zeros(2 * self.vars_num)
        c[0:self.yield_pieces_num] = 1.0
        return -1 * c
