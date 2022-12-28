import numpy as np
from src.program.models import FPM, SlackCandidate
from src.program.functions import zero_out_small_values


class MahiniMethod:
    def __init__(self, raw_data):
        self.primary_vars_count = raw_data.primary_vars_count
        self.slack_vars_count = raw_data.slack_vars_count
        self.total_vars_count = raw_data.total_vars_count
        self.plastic_vars_count = raw_data.plastic_vars_count
        self.softening_vars_count = raw_data.softening_vars_count
        self.constraints_count = raw_data.constraints_count
        self.yield_points_indices = raw_data.yield_points_indices

        self.landa_var = raw_data.landa_var
        self.limits_slacks = raw_data.limits_slacks
        self.table = raw_data.table
        self.b = raw_data.b
        self.c = raw_data.c

    def solve(self):
        from pprint import pprint
        bbar = self.b
        print(bbar)
        basic_variables = self.get_initial_basic_variables()
        b_matrix_inv = np.eye(self.slack_vars_count)
        cb = np.zeros(self.slack_vars_count)
        x_cumulative = np.matrix(np.zeros((self.constraints_count, 1)))
        x_history = []
        fpm = FPM
        fpm.var = self.landa_var
        fpm.cost = 0
        pprint(b_matrix_inv)
        print(f"{fpm.var=}")
        print("table")
        print(self.table)
        fpm, b_matrix_inv, basic_variables, cb, will_out_row, will_out_var = self.enter_landa(fpm, b_matrix_inv, basic_variables, cb)
        landa_row = will_out_row
        np.set_printoptions(precision=4, linewidth=200, suppress=True)
        print(f"{will_out_row=}")
        print(f"{will_out_var=}")
        print("//////////////////////////////////////////////////////////////////////////////////")
        pprint(b_matrix_inv)
        while self.limits_slacks.issubset(set(basic_variables)):
            sorted_slack_candidates = self.get_sorted_slack_candidates(basic_variables, b_matrix_inv, cb)
            will_in_col = fpm.var
            # print("***************")
            print(f"{will_in_col=}")
            abar = self.calculate_abar(will_in_col, b_matrix_inv)
            print("==================check=================")
            print(bbar)
            print(b_matrix_inv)
            bbar = self.calculate_bbar(b_matrix_inv, bbar)
            print(bbar)
            print("=================check==================")
            # print(f"{bbar=}")
            # print(f"{abar=}")
            will_out_row = self.get_will_out(abar, bbar, will_in_col, landa_row, basic_variables)
            will_out_var = basic_variables[will_out_row]
            print(f"{will_out_row=}")
            print(f"{will_out_var=}")
            x_cumulative, bbar = self.reset(basic_variables, x_cumulative, bbar)
            print("bbar reset ", bbar)
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
                        print("unload spm")
                        basic_variables, b_matrix_inv, cb, landa_row = self.unload(
                            pm_var=spm_var,
                            basic_variables=basic_variables,
                            b_matrix_inv=b_matrix_inv,
                            cb=cb,
                            landa_row=landa_row,
                        )
                        pprint(b_matrix_inv)
                        break
                else:
                    if self.is_will_out_var_opm(will_out_var):
                        print("unload opm")
                        opm_var = will_out_var
                        basic_variables, b_matrix_inv, cb, landa_row = self.unload(
                            pm_var=opm_var,
                            basic_variables=basic_variables,
                            b_matrix_inv=b_matrix_inv,
                            cb=cb,
                            landa_row=landa_row,
                        )
                        pprint(b_matrix_inv)
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
                        pprint(b_matrix_inv)
                        break
        print(f"before    {basic_variables=}")
        bbar = self.calculate_bbar(b_matrix_inv, bbar)
        print(f"{bbar=}")
        x_cumulative, bbar = self.reset(basic_variables, x_cumulative, bbar)
        print("bbar reset ", bbar)
        x_history.append(x_cumulative.copy())
        pms_history = []
        load_level_history = []
        for x in x_history:
            pms = x[0:self.plastic_vars_count]
            load_level = x[self.landa_var][0, 0]
            pms_history.append(pms)
            load_level_history.append(load_level)
        result = {
            "pms_history": pms_history,
            "load_level_history": load_level_history
        }
        return result

    def get_slack_var(self, primary_var):
        return primary_var + self.primary_vars_count

    def get_primary_var(self, slack_var):
        if slack_var < self.primary_vars_count + self.slack_vars_count:
            # x variables and y variables
            return slack_var - self.primary_vars_count
        else:
            # y variables from z variables
            return slack_var - self.constraints_count

    def enter_landa(self, fpm, b_matrix_inv, basic_variables, cb):
        will_in_col = fpm.var
        a = self.table[:, will_in_col]
        will_out_row = self.get_will_out(a, self.b)
        will_out_var = basic_variables[will_out_row]
        # print(f"+++++++++++++++++++++++++++++++++ {self.table[:, 0]=}")
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

    def unload(self, pm_var, basic_variables, b_matrix_inv, cb, landa_row):
        # TODO: should handle if third pivot column is a y not x. possible bifurcation.
        # TODO: loading whole b_matrix_inv in input and output is costly, try like mahini method.
        # TODO: check line 60 of unload and line 265 in mclp of mahini code
        # (probable usage: in case when unload is last step)
        pm_var_family = self.get_pm_var_family(pm_var)
        for primary_var in pm_var_family:
            if primary_var in basic_variables:
                exiting_row = self.get_var_row(primary_var, basic_variables)

                unloading_pivot_elements = [
                    {
                        "row": exiting_row,
                        "column": self.get_slack_var(exiting_row),
                    },
                    {
                        "row": primary_var,
                        "column": self.get_slack_var(primary_var),
                    },
                    {
                        "row": exiting_row,
                        "column": basic_variables[primary_var],
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

                    if element["column"] == self.landa_var:
                        landa_row = element["row"]

        return basic_variables, b_matrix_inv, cb, landa_row

    def get_pm_var_family(self, pm_var):
        if self.softening_vars_count:
            yield_point = self.get_plastic_var_yield_point(pm_var)
            pm_var_family = [
                pm_var,
                self.plastic_vars_count + yield_point * 2,
                self.plastic_vars_count + yield_point * 2 + 1,
            ]
        else:
            pm_var_family = [
                pm_var,
            ]
        return pm_var_family

    def calculate_abar(self, col, b_matrix_inv):
        # print(f"{col=}")
        a = self.table[:, col]
        abar = np.dot(b_matrix_inv, a)
        # print(f"asdasd as das d {self.table[:, 0]=}")
        return abar

    def calculate_bbar(self, b_matrix_inv, bbar):
        bbar = np.dot(b_matrix_inv, bbar)
        return bbar

    def calculate_cbar(self, cb, b_matrix_inv):
        pi_transpose = np.dot(cb, b_matrix_inv)
        cbar = np.zeros(self.total_vars_count)
        for i in range(self.total_vars_count):
            cbar[i] = self.c[i] - np.dot(pi_transpose, self.table[:, i])
        return cbar

    def update_cb(self, cb, will_in_col, will_out_row):
        cb[will_out_row] = self.c[will_in_col]
        return cb

    def get_will_out(self, abar, bbar, will_in_col=None, landa_row=None, basic_variables=None):
        # TODO: handle unbounded problem,
        # when there is no positive a remaining (structure failure), e.g. stop the process.

        abar = zero_out_small_values(abar)
        # print(f"{bbar=}")
        # print(f"{abar=}")
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
            if will_in_col < self.plastic_vars_count or will_in_col == self.primary_vars_count:
                # skip landa variable from exiting
                if landa_row == sorted_zipped_ba[0, 0]:
                    will_out_row = int(sorted_zipped_ba[0, 1])

            # if will in variable is softening
            else:
                will_out_row = int(sorted_zipped_ba[0, 0])
                # when we reach load or disp limit:

                will_in_col_yield_point = self.get_softening_var_yield_point(will_in_col)
                for i, ba_row in enumerate(sorted_zipped_ba[0, :]):
                    will_out_row = int(ba_row)
                    if will_out_row != landa_row:
                        # if exiting variable is load or disp limit
                        if will_out_row >= self.primary_vars_count - 1:
                            break
                        will_out_var = basic_variables[will_out_row]
                        will_out_yield_point = self.get_will_out_yield_point(will_out_var)
                        if will_in_col_yield_point != will_out_yield_point:
                            will_out_row = int(sorted_zipped_ba[0, i])
                            break
        return will_out_row

    def get_initial_basic_variables(self):
        basic_variables = np.zeros(self.constraints_count, dtype=int)
        for i in range(self.constraints_count):
            basic_variables[i] = self.primary_vars_count + i
        basic_variables = self.update_basic_variables_for_negative_b(basic_variables)
        return basic_variables

    def update_b_matrix_inverse(self, b_matrix_inv, abar, will_out_row):
        e = np.eye(self.slack_vars_count)
        eta = np.zeros(self.slack_vars_count)
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
        return True if will_out_var < (self.primary_vars_count - 1) else False

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
            if basic_variable < self.primary_vars_count:
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
        yield_point = (sm - self.plastic_vars_count) // 2
        # TODO: complete for pm slacks and softening slacks
        return yield_point

    def get_will_out_yield_point(self, will_out_var):
        if will_out_var < self.plastic_vars_count:
            # plastic primary
            will_out_yield_point = self.get_plastic_var_yield_point(will_out_var)

        elif self.plastic_vars_count <= will_out_var < self.primary_vars_count - 1:
            # softening primary
            will_out_yield_point = self.get_softening_var_yield_point(will_out_var)

        elif self.primary_vars_count <= will_out_var < self.primary_vars_count + self.plastic_vars_count:
            # plastic slack
            primary_will_out_var = self.get_primary_var(will_out_var)
            will_out_yield_point = self.get_plastic_var_yield_point(primary_will_out_var)

        elif self.primary_vars_count + self.plastic_vars_count <= will_out_var < self.primary_vars_count + self.plastic_vars_count + self.softening_vars_count:
            # softening slack
            primary_will_out_var = self.get_primary_var(will_out_var)
            will_out_yield_point = self.get_softening_var_yield_point(primary_will_out_var)

        return will_out_yield_point

    def update_basic_variables_for_negative_b(self, basic_variables):
        if any(self.b < 0):
            for i in range(self.constraints_count):
                if self.b[i] < 0:
                    basic_variables[i] = int(basic_variables[i] + self.constraints_count)
        return basic_variables
