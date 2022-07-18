import numpy as np
from .models import FPM, SlackCandidate
from .functions import zero_out_small_values


class MahiniMethod:
    def __init__(self, raw_data):
        self.vars_num = raw_data.vars_num
        self.slacks_num = raw_data.slacks_num
        self.constraints_num = raw_data.constraints_num

        self.landa_var_num = raw_data.landa_var_num
        self.limits_slacks = raw_data.limits_slacks
        self.table = raw_data.table
        self.b = raw_data.b
        self.c = raw_data.c

    def solve(self):
        bbar = self.b
        basic_variables = self.get_initial_basic_variables()
        b_matrix_inv = np.eye(self.slacks_num)
        cb = np.zeros(self.slacks_num)
        empty_x_cumulative = np.zeros((self.constraints_num, 1))
        x_cumulative = np.matrix(empty_x_cumulative)
        x_history = []
        fpm = FPM
        fpm.var_num = self.landa_var_num
        fpm.cost = 0
        fpm, b_matrix_inv, basic_variables, cb, will_out_row_num, will_out_var_num = self.enter_landa(fpm, b_matrix_inv, basic_variables, cb)
        landa_row_num = will_out_row_num

        while self.limits_slacks.issubset(set(basic_variables)):
            sorted_slack_candidates = self.get_sorted_slack_candidates(basic_variables, b_matrix_inv, cb)
            will_in_col_num = fpm.var_num
            abar = self.calculate_abar(will_in_col_num, b_matrix_inv)
            bbar = self.calculate_bbar(b_matrix_inv, bbar)
            will_out_row_num = self.get_will_out(abar, bbar, landa_row_num)
            will_out_var_num = basic_variables[will_out_row_num]
            x_cumulative, bbar = self.reset(basic_variables, x_cumulative, bbar)
            x_history.append(x_cumulative.copy())

            for slack_candidate in sorted_slack_candidates + [fpm]:
                if not self.is_candidate_fpm(fpm, slack_candidate):
                    spm_var_num = self.get_var_num(slack_candidate.var_num)
                    r = self.calculate_r(
                        spm_var_num=spm_var_num,
                        basic_variables=basic_variables,
                        abar=abar,
                        b_matrix_inv=b_matrix_inv,
                    )
                    if r > 0:
                        continue
                    else:
                        print("unload r < 0")
                        basic_variables, b_matrix_inv, cb = self.unload(
                            pm_var_num=spm_var_num,
                            basic_variables=basic_variables,
                            b_matrix_inv=b_matrix_inv,
                            cb=cb,
                        )
                        break
                else:
                    if self.is_will_out_var_opm(will_out_var_num):
                        print("unload opm")
                        opm_var_num = will_out_var_num
                        basic_variables, b_matrix_inv, cb = self.unload(
                            pm_var_num=opm_var_num,
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
                            will_out_row_num=will_out_row_num,
                            will_in_col_num=will_in_col_num,
                            abar=abar,
                        )
                        break
        bbar = self.calculate_bbar(b_matrix_inv, bbar)
        x_cumulative, bbar = self.reset(basic_variables, x_cumulative, bbar)
        x_history.append(x_cumulative.copy())

        pms_history = []
        load_level_history = []
        for x in x_history:
            pms = x[0:self.landa_var_num]
            load_level = x[self.landa_var_num][0, 0]
            pms_history.append(pms)
            load_level_history.append(load_level)

        result = {
            "pms_history": pms_history,
            "load_level_history": load_level_history
        }
        return result

    def get_slack_num(self, var_num):
        return var_num + self.vars_num

    def get_var_num(self, slack_num):
        return slack_num - self.vars_num

    def enter_landa(self, fpm, b_matrix_inv, basic_variables, cb):
        will_in_col_num = fpm.var_num
        a = self.table[:, will_in_col_num]
        will_out_row_num = self.get_will_out(a, self.b)
        will_out_var_num = basic_variables[will_out_row_num]
        basic_variables = self.update_basic_variables(basic_variables, will_out_row_num, will_in_col_num)
        b_matrix_inv = self.update_b_matrix_inverse(b_matrix_inv, a, will_out_row_num)
        cb = self.update_cb(cb, will_in_col_num, will_out_row_num)
        cbar = self.calculate_cbar(cb, b_matrix_inv)
        fpm = self.update_fpm(will_out_row_num, cbar)
        return fpm, b_matrix_inv, basic_variables, cb, will_out_row_num, will_out_var_num

    def enter_fpm(self, basic_variables, b_matrix_inv, cb, will_out_row_num, will_in_col_num, abar):
        b_matrix_inv = self.update_b_matrix_inverse(b_matrix_inv, abar, will_out_row_num)
        cb = self.update_cb(cb, will_in_col_num, will_out_row_num)
        cbar = self.calculate_cbar(cb, b_matrix_inv)
        basic_variables = self.update_basic_variables(basic_variables, will_out_row_num, will_in_col_num)
        fpm = self.update_fpm(will_out_row_num, cbar)
        return basic_variables, b_matrix_inv, cb, fpm

    def unload(self, pm_var_num, basic_variables, b_matrix_inv, cb):
        # TODO: should handle if third pivot column is a y not x. possible bifurcation.
        # TODO: must handle landa-row separately like mahini unload (e.g. softening, ...)
        # TODO: loading whole b_matrix_inv in input and output is costly, try like mahini method.
        # TODO: check line 60 of unload and line 265 in mclp of mahini code
        # (probable usage: in case when unload is last step)

        exiting_row_num = self.get_var_row_num(pm_var_num, basic_variables)

        unloading_pivot_elements = [
            {
                "row": exiting_row_num,
                "column": self.get_slack_num(exiting_row_num),
            },
            {
                "row": pm_var_num,
                "column": self.get_slack_num(pm_var_num),
            },
            {
                "row": exiting_row_num,
                "column": basic_variables[pm_var_num],
            },
        ]
        for element in unloading_pivot_elements:
            abar = self.calculate_abar(element["column"], b_matrix_inv)
            b_matrix_inv = self.update_b_matrix_inverse(b_matrix_inv, abar, element["row"])
            cb = self.update_cb(
                cb=cb,
                will_in_col_num=element["column"],
                will_out_row_num=element["row"]
            )
            basic_variables = self.update_basic_variables(
                basic_variables=basic_variables,
                will_out_row_num=element["row"],
                will_in_col_num=element["column"]
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
        cbar = np.zeros(self.vars_num + self.slacks_num)
        for i in range(self.vars_num + self.slacks_num):
            cbar[i] = self.c[i] - np.dot(pi_transpose, self.table[:, i])
        return cbar

    def update_cb(self, cb, will_in_col_num, will_out_row_num):
        cb[will_out_row_num] = self.c[will_in_col_num]
        return cb

    def get_will_out(self, abar, bbar, landa_row_num=None):
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

        will_out_row_num = int(sorted_zipped_ba[0, 0])
        if landa_row_num:
            if landa_row_num == int(sorted_zipped_ba[0, 0]):
                will_out_row_num = int(sorted_zipped_ba[0, 1])

        return will_out_row_num

    def get_initial_basic_variables(self):
        basic_variables = np.zeros(self.constraints_num, dtype=int)
        for i in range(self.constraints_num):
            basic_variables[i] = self.vars_num + i
        return basic_variables

    def update_b_matrix_inverse(self, b_matrix_inv, abar, will_out_row_num):
        e = np.eye(self.slacks_num)
        eta = np.zeros(self.slacks_num)
        will_out_item = abar[will_out_row_num]

        for i, item in enumerate(abar):
            if i == will_out_row_num:
                eta[i] = 1 / will_out_item
            else:
                eta[i] = -item / will_out_item
        e[:, will_out_row_num] = eta
        updated_b_matrix_inv = np.dot(e, b_matrix_inv)
        return updated_b_matrix_inv

    def is_variable_plastic_multiplier(self, variable_num):
        return False if variable_num >= self.landa_var_num else True

    def is_candidate_fpm(self, fpm, slack_candidate):
        if fpm.cost <= slack_candidate.cost:
            return True
        else:
            return False

    def is_will_out_var_opm(self, will_out_var_num):
        # opm: obstacle plastic multiplier
        return self.is_variable_plastic_multiplier(will_out_var_num)

    def update_basic_variables(self, basic_variables, will_out_row_num, will_in_col_num):
        basic_variables[will_out_row_num] = will_in_col_num
        return basic_variables

    def update_fpm(self, will_out_row_num, cbar):
        fpm = FPM
        fpm.var_num = will_out_row_num
        fpm.cost = cbar[will_out_row_num]
        return fpm

    def get_sorted_slack_candidates(self, basic_variables, b_matrix_inv, cb):
        cbar = self.calculate_cbar(cb, b_matrix_inv)
        slack_candidates = []
        for var in basic_variables:
            if var < self.landa_var_num:
                slack_var_num = self.get_slack_num(var)
                slack_candidate = SlackCandidate(
                    var_num=slack_var_num,
                    cost=cbar[slack_var_num]
                )
                slack_candidates.append(slack_candidate)
        slack_candidates.sort(key=lambda y: y.cost)
        return slack_candidates

    def reset(self, basic_variables, x_cumulative, bbar):
        for i, basic_variable in enumerate(basic_variables):
            if basic_variable < self.vars_num:
                x_cumulative[basic_variables[i], 0] += bbar[i]
                bbar[i] = 0
        return x_cumulative, bbar

    def calculate_r(self, spm_var_num, basic_variables, abar, b_matrix_inv):
        spm_row_num = self.get_var_row_num(spm_var_num, basic_variables)
        r = abar[spm_row_num] / b_matrix_inv[spm_row_num, spm_var_num]
        return r

    def get_var_row_num(self, var_num, basic_variables):
        row_num = np.where(basic_variables == var_num)[0][0]
        return row_num
