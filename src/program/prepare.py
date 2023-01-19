import numpy as np


class RawData:
    def __init__(self, analysis):
        structure = analysis.structure
        self.load_limit = structure.limits["load_limit"]
        self.disp_limits = structure.limits["disp_limits"]
        self.phi = structure.phi
        self.q = structure.q
        self.h = structure.h
        self.w = structure.w
        self.cs = structure.cs

        self.p0 = analysis.p0
        self.pv = analysis.pv
        self.d0 = analysis.d0
        self.dv = analysis.dv

        vars_count = VarsCount(analysis)
        self.disp_limits_count = vars_count.disp_limits_count
        self.limits_count = vars_count.limits_count
        self.plastic_vars_count = vars_count.plastic_vars_count
        self.softening_vars_count = vars_count.softening_vars_count
        self.yield_points_indices = vars_count.yield_points_indices

        self.primary_vars_count = vars_count.primary_vars_count
        self.constraints_count = vars_count.constraints_count
        self.slack_vars_count = vars_count.slack_vars_count
        self.total_vars_count = vars_count.total_vars_count

        self.b = self._get_b_column()
        self.table = self._create_table()

        if analysis.type == "dynamic":
            self.update_b_for_dynamic_analysis(analysis)

        self.landa_var = self.plastic_vars_count + self.softening_vars_count
        self.landa_bar_var = 2 * self.landa_var + 1

        self.limits_slacks = set(range(self.landa_bar_var, self.landa_bar_var + self.limits_count))

        self.c = self._get_costs_row()

    def _create_table(self):
        constraints_count = self.constraints_count
        yield_pieces_count = self.plastic_vars_count
        softening_vars_count = self.softening_vars_count
        disp_limits_count = self.disp_limits_count
        primary_vars_count = self.primary_vars_count
        phi_pv_phi = self.phi.T * self.pv * self.phi
        phi_p0 = self.phi.T * self.p0
        dv_phi = self.dv * self.phi
        raw_a = np.matrix(np.zeros((constraints_count, primary_vars_count)))
        raw_a[0:yield_pieces_count, 0:yield_pieces_count] = phi_pv_phi

        if softening_vars_count:
            raw_a[yield_pieces_count:(yield_pieces_count + softening_vars_count), 0:yield_pieces_count] = self.q
            raw_a[0:yield_pieces_count, yield_pieces_count:(yield_pieces_count + softening_vars_count)] = - self.h
            raw_a[yield_pieces_count:(yield_pieces_count + softening_vars_count), yield_pieces_count:(yield_pieces_count + softening_vars_count)] = self.w

        landa_base_num = yield_pieces_count + softening_vars_count
        raw_a[0:yield_pieces_count, landa_base_num] = phi_p0
        raw_a[landa_base_num, landa_base_num] = 1.0

        if self.disp_limits.any():
            disp_limit_base_num = yield_pieces_count + softening_vars_count + 1
            raw_a[disp_limit_base_num:(disp_limit_base_num + disp_limits_count), 0:yield_pieces_count] = dv_phi
            raw_a[(disp_limit_base_num + disp_limits_count):(disp_limit_base_num + 2 * disp_limits_count), 0:yield_pieces_count] = - dv_phi

            raw_a[disp_limit_base_num:(disp_limit_base_num + disp_limits_count), landa_base_num] = self.d0
            raw_a[(disp_limit_base_num + disp_limits_count):(disp_limit_base_num + 2 * disp_limits_count), landa_base_num] = - self.d0

        a_matrix = np.array(raw_a)
        columns_count = primary_vars_count + self.slack_vars_count
        table = np.zeros((constraints_count, columns_count))
        table[0:constraints_count, 0:primary_vars_count] = a_matrix

        # Assigning diagonal arrays of slack variables.
        # TODO: use np.eye instead
        j = primary_vars_count
        for i in range(constraints_count):
            table[i, j] = 1.0
            j += 1

        return table

    def _get_b_column(self):
        yield_pieces_count = self.plastic_vars_count
        disp_limits_count = self.disp_limits_count

        b = np.ones((self.constraints_count))
        b[yield_pieces_count + self.softening_vars_count] = self.load_limit
        if self.softening_vars_count:
            b[yield_pieces_count:(yield_pieces_count + self.softening_vars_count)] = np.array(self.cs)[:, 0]

        if self.disp_limits.any():
            disp_limit_base_num = yield_pieces_count + self.softening_vars_count + 1
            b[disp_limit_base_num:(disp_limit_base_num + disp_limits_count)] = abs(self.disp_limits[:, 2])
            b[(disp_limit_base_num + disp_limits_count):(disp_limit_base_num + 2 * disp_limits_count)] = abs(self.disp_limits[:, 2])

        return b

    def _get_costs_row(self):
        c = np.zeros(self.total_vars_count)
        c[0:self.plastic_vars_count] = 1.0
        return -1 * c

    def update_b_for_dynamic_analysis(self, analysis):
        self.b[0:self.plastic_vars_count] = (
            self.b[0:self.plastic_vars_count] -
            np.array(
                self.phi.T * analysis.pv_prev * self.phi * analysis.plastic_multipliers_prev
            ).flatten()
        )
        # print(f"{self.b=}")


class VarsCount:
    def __init__(self, analysis):
        structure = analysis.structure
        self.disp_limits = structure.limits["disp_limits"]

        self.disp_limits_count = self.disp_limits.shape[0]
        self.limits_count = 1 + self.disp_limits_count * 2
        self.plastic_vars_count = structure.yield_specs.pieces_count
        self.softening_vars_count = 2 * structure.yield_specs.points_count if structure.include_softening else 0
        self.yield_points_indices = structure.yield_points_indices

        self.primary_vars_count = self.plastic_vars_count + self.softening_vars_count + 1
        self.constraints_count = self.plastic_vars_count + self.softening_vars_count + self.limits_count
        self.slack_vars_count = self.constraints_count
        self.total_vars_count = self.primary_vars_count + self.slack_vars_count
