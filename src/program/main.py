import numpy as np
from .models import FPM, SlackCandidate
from .functions import zero_out_small_values


class MahiniMethod:
    def __init__(self, raw_data):
        self.primary_vars_num = raw_data.primary_vars_num
        self.slack_vars_num = raw_data.slack_vars_num
        self.total_vars_num = raw_data.total_vars_num
        self.plastic_vars_num = raw_data.plastic_vars_num
        self.softening_vars_num = raw_data.softening_vars_num
        self.constraints_num = raw_data.constraints_num
        self.yield_points_indices = raw_data.yield_points_indices

        self.landa_var = raw_data.landa_var
        self.limits_slacks = raw_data.limits_slacks
        self.table = raw_data.table
        self.b = raw_data.b
        self.c = raw_data.c

    def solve(self):
        bbar = self.b
        basic_variables = self.get_initial_basic_variables()
        b_matrix_inv = np.eye(self.slack_vars_num)
        cb = np.zeros(self.slack_vars_num)
        empty_x_cumulative = np.zeros((self.constraints_num, 1))
        x_cumulative = np.matrix(empty_x_cumulative)
        x_history = []
        fpm = FPM
        fpm.var = self.landa_var
        fpm.cost = 0
        fpm, b_matrix_inv, basic_variables, cb, will_out_row, will_out_var = self.enter_landa(fpm, b_matrix_inv, basic_variables, cb)
        landa_row = will_out_row

        while self.limits_slacks.issubset(set(basic_variables)):
            sorted_slack_candidates = self.get_sorted_slack_candidates(basic_variables, b_matrix_inv, cb)
            will_in_col = fpm.var
            abar = self.calculate_abar(will_in_col, b_matrix_inv)
            bbar = self.calculate_bbar(b_matrix_inv, bbar)
            will_out_row = self.get_will_out(abar, bbar, will_in_col, landa_row, basic_variables)
            will_out_var = basic_variables[will_out_row]
            x_cumulative, bbar = self.reset(basic_variables, x_cumulative, bbar)
            x_history.append(x_cumulative.copy())

            for slack_candidate in sorted_slack_candidates + [fpm]:
                if not self.is_candidate_fpm(fpm, slack_candidate):
                    spm_var = self.get_primary_var(slack_candidate.var)
                    r = self.calculate_r(
                        spm_var=spm_var,
                        basic_variables=basic_variables,
                        abar=abar,
                        b_matrix_inv=b_matrix_inv,
                    )
                    if r > 0:
                        continue
                    else:
                        print("unload r < 0")
                        # if self.softening_vars_num:
                        #     softening_var = int(spm_var + self.primary_vars_num)

                        basic_variables, b_matrix_inv, cb = self.unload(
                            pm_var=spm_var,
                            basic_variables=basic_variables,
                            b_matrix_inv=b_matrix_inv,
                            cb=cb,
                        )
                        break
                else:
                    if self.is_will_out_var_opm(will_out_var):
                        print("unload opm")
                        opm_var = will_out_var
                        basic_variables, b_matrix_inv, cb = self.unload(
                            pm_var=opm_var,
                            basic_variables=basic_variables,
                            b_matrix_inv=b_matrix_inv,
                            cb=cb,
                        )
                        break
                    else:
                        print("enter fpm")
                        basic_variables, b_matrix_inv, cb, fpm = self.enter_fpm(
                            basic_variables=basic_variables,
                            b_matrix_inv=b_matrix_inv,
                            cb=cb,
                            will_out_row=will_out_row,
                            will_in_col=will_in_col,
                            abar=abar,
                        )
                        break

        bbar = self.calculate_bbar(b_matrix_inv, bbar)
        x_cumulative, bbar = self.reset(basic_variables, x_cumulative, bbar)
        x_history.append(x_cumulative.copy())

        pms_history = []
        load_level_history = []
        for x in x_history:
            pms = x[0:self.landa_var]
            load_level = x[self.landa_var][0, 0]
            pms_history.append(pms)
            load_level_history.append(load_level)

        result = {
            "pms_history": pms_history,
            "load_level_history": load_level_history
        }
        return result

    def get_slack_var(self, primary_var):
        return primary_var + self.primary_vars_num

    def get_primary_var(self, slack_var):
        return slack_var - self.primary_vars_num

    def enter_landa(self, fpm, b_matrix_inv, basic_variables, cb):
        will_in_col = fpm.var
        a = self.table[:, will_in_col]
        will_out_row = self.get_will_out(a, self.b)
        will_out_var = basic_variables[will_out_row]
        basic_variables = self.update_basic_variables(basic_variables, will_out_row, will_in_col)
        b_matrix_inv = self.update_b_matrix_inverse(b_matrix_inv, a, will_out_row)
        cb = self.update_cb(cb, will_in_col, will_out_row)
        cbar = self.calculate_cbar(cb, b_matrix_inv)
        fpm = self.update_fpm(will_out_row, cbar)
        return fpm, b_matrix_inv, basic_variables, cb, will_out_row, will_out_var

    def enter_fpm(self, basic_variables, b_matrix_inv, cb, will_out_row, will_in_col, abar):
        b_matrix_inv = self.update_b_matrix_inverse(b_matrix_inv, abar, will_out_row)
        cb = self.update_cb(cb, will_in_col, will_out_row)
        cbar = self.calculate_cbar(cb, b_matrix_inv)
        basic_variables = self.update_basic_variables(basic_variables, will_out_row, will_in_col)
        fpm = self.update_fpm(will_out_row, cbar)
        return basic_variables, b_matrix_inv, cb, fpm

    def unload(self, pm_var, basic_variables, b_matrix_inv, cb):
        # TODO: should handle if third pivot column is a y not x. possible bifurcation.
        # TODO: must handle landa-row separately like mahini unload (e.g. softening, ...)
        # TODO: loading whole b_matrix_inv in input and output is costly, try like mahini method.
        # TODO: check line 60 of unload and line 265 in mclp of mahini code
        # (probable usage: in case when unload is last step)

        exiting_row = self.get_var_row(pm_var, basic_variables)

        unloading_pivot_elements = [
            {
                "row": exiting_row,
                "column": self.get_slack_var(exiting_row),
            },
            {
                "row": pm_var,
                "column": self.get_slack_var(pm_var),
            },
            {
                "row": exiting_row,
                "column": basic_variables[pm_var],
            },
        ]
        for element in unloading_pivot_elements:
            abar = self.calculate_abar(element["column"], b_matrix_inv)
            b_matrix_inv = self.update_b_matrix_inverse(b_matrix_inv, abar, element["row"])
            cb = self.update_cb(
                cb=cb,
                will_in_col=element["column"],
                will_out_row=element["row"]
            )
            basic_variables = self.update_basic_variables(
                basic_variables=basic_variables,
                will_out_row=element["row"],
                will_in_col=element["column"]
            )

        return basic_variables, b_matrix_inv, cb

    def calculate_abar(self, col, b_matrix_inv):
        a = self.table[:, col]
        abar = np.dot(b_matrix_inv, a)
        return abar

    def calculate_bbar(self, b_matrix_inv, bbar):
        bbar = np.dot(b_matrix_inv, bbar)
        return bbar

    def calculate_cbar(self, cb, b_matrix_inv):
        pi_transpose = np.dot(cb, b_matrix_inv)
        cbar = np.zeros(self.total_vars_num)
        for i in range(self.total_vars_num):
            cbar[i] = self.c[i] - np.dot(pi_transpose, self.table[:, i])
        return cbar

    def update_cb(self, cb, will_in_col, will_out_row):
        cb[will_out_row] = self.c[will_in_col]
        return cb

    def get_will_out(self, abar, bbar, will_in_col=None, landa_row=None, basic_variables=None):
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

        # if will in variable is landa
        will_out_row = int(sorted_zipped_ba[0, 0])

        # if will in variable is plastic or softening
        if landa_row and will_in_col:

            # if will in variable is plastic
            if will_in_col < self.plastic_vars_num or will_in_col == self.primary_vars_num:
                # skip landa variable from exiting
                if landa_row == sorted_zipped_ba[0, 0]:
                    will_out_row = int(sorted_zipped_ba[0, 1])

            # if will in variable is softening
            else:
                will_in_col_yield_point = self.get_softening_var_yield_point(will_in_col)

                for i, ba in enumerate(sorted_zipped_ba[0, :]):
                    ba = int(ba)
                    will_out_row = ba
                    will_out_var = basic_variables[will_out_row]

                    if will_out_var < self.plastic_vars_num:
                        # plastic primary
                        will_out_yield_point = self.get_plastic_var_yield_point(will_out_var)

                    elif self.plastic_vars_num <= will_out_var < self.primary_vars_num - 1:
                        # softening primary
                        will_out_yield_point = self.get_softening_var_yield_point(will_out_var)

                    elif self.primary_vars_num <= will_out_var < self.primary_vars_num + self.plastic_vars_num:
                        # plastic slack
                        primary_will_out_var = self.get_primary_var(will_out_var)
                        will_out_yield_point = self.get_plastic_var_yield_point(primary_will_out_var)

                    elif self.primary_vars_num + self.plastic_vars_num <= will_out_var < self.primary_vars_num + self.plastic_vars_num + self.softening_vars_num:
                        # softening slack
                        primary_will_out_var = self.get_primary_var(will_out_var)
                        will_out_yield_point = self.get_softening_var_yield_point(primary_will_out_var)

                    if landa_row != will_out_var and will_in_col_yield_point != will_out_yield_point:
                        will_out_row = int(sorted_zipped_ba[0, i + 1])
                        break
                    break

        return will_out_row

    def get_initial_basic_variables(self):
        basic_variables = np.zeros(self.constraints_num, dtype=int)
        for i in range(self.constraints_num):
            basic_variables[i] = self.primary_vars_num + i
        return basic_variables

    def update_b_matrix_inverse(self, b_matrix_inv, abar, will_out_row):
        e = np.eye(self.slack_vars_num)
        eta = np.zeros(self.slack_vars_num)
        will_out_item = abar[will_out_row]

        for i, item in enumerate(abar):
            if i == will_out_row:
                eta[i] = 1 / will_out_item
            else:
                eta[i] = -item / will_out_item
        e[:, will_out_row] = eta
        updated_b_matrix_inv = np.dot(e, b_matrix_inv)
        return updated_b_matrix_inv

    def is_candidate_fpm(self, fpm, slack_candidate):
        if fpm.cost <= slack_candidate.cost:
            return True
        else:
            return False

    def is_will_out_var_opm(self, will_out_var):
        # opm: obstacle plastic multiplier
        return True if will_out_var < (self.primary_vars_num - 1) else False

    def update_basic_variables(self, basic_variables, will_out_row, will_in_col):
        basic_variables[will_out_row] = will_in_col
        return basic_variables

    def update_fpm(self, will_out_row, cbar):
        fpm = FPM
        fpm.var = will_out_row
        fpm.cost = cbar[will_out_row]
        return fpm

    def get_sorted_slack_candidates(self, basic_variables, b_matrix_inv, cb):
        cbar = self.calculate_cbar(cb, b_matrix_inv)
        slack_candidates = []
        for var in basic_variables:
            if var < self.landa_var:
                slack_var = self.get_slack_var(var)
                slack_candidate = SlackCandidate(
                    var=slack_var,
                    cost=cbar[slack_var]
                )
                slack_candidates.append(slack_candidate)
        slack_candidates.sort(key=lambda y: y.cost)
        return slack_candidates

    def reset(self, basic_variables, x_cumulative, bbar):
        for i, basic_variable in enumerate(basic_variables):
            if basic_variable < self.primary_vars_num:
                x_cumulative[basic_variables[i], 0] += bbar[i]
                bbar[i] = 0
        return x_cumulative, bbar

    def calculate_r(self, spm_var, basic_variables, abar, b_matrix_inv):
        spm_row = self.get_var_row(spm_var, basic_variables)
        r = abar[spm_row] / b_matrix_inv[spm_row, spm_var]
        return r

    def get_var_row(self, var, basic_variables):
        row = np.where(basic_variables == var)[0][0]
        return row

    def get_plastic_var_yield_point(self, pm):
        for i, yield_point_indices in enumerate(self.yield_points_indices):
            if yield_point_indices["begin"] <= pm and pm <= yield_point_indices["end"]:
                yield_point = i
                break
        return yield_point

    def get_softening_var_yield_point(self, sm):
        yield_point = (sm - self.plastic_vars_num) // 2
        # TODO: complete for pm slacks and softening slacks
        return yield_point
