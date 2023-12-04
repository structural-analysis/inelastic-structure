import numpy as np
from scipy.sparse import csr_matrix

from .models import FPM, SlackCandidate, Sifting
from .functions import zero_out_small_values
from ..analysis.initial_analysis import InitialData, AnalysisData
from ..settings import settings, SiftingType


class MahiniMethod:
    def __init__(self, initial_data: InitialData, analysis_data: AnalysisData):
        self.load_limit = initial_data.load_limit
        self.disp_limits = initial_data.disp_limits
        self.disp_limits_count = initial_data.disp_limits_count
        self.include_softening = initial_data.include_softening

        self.intact_points = initial_data.intact_points
        self.intact_pieces = initial_data.intact_pieces
        self.intact_points_count = initial_data.intact_points_count
        self.intact_components_count = initial_data.intact_components_count
        self.intact_pieces_count = initial_data.intact_pieces_count
        self.intact_plastic_vars_count = self.intact_pieces_count

        self.yield_points_indices = initial_data.yield_points_indices
        self.intact_phi = initial_data.intact_phi
        self.intact_q = initial_data.intact_q
        self.intact_h = initial_data.intact_h
        self.intact_w = initial_data.intact_w
        self.intact_cs = initial_data.intact_cs

        self.p0 = analysis_data.p0
        self.pv = analysis_data.pv
        self.d0 = analysis_data.d0
        self.dv = analysis_data.dv

        if settings.sifting_type is SiftingType.not_used:
            self.phi = self.intact_phi
            self.q = self.intact_q
            self.h = self.intact_h
            self.w = self.intact_w
            self.cs = self.intact_cs
            self.points = self.intact_points
            self.points_count = self.intact_points_count
            self.components_count = self.intact_components_count
            self.pieces_count = self.intact_pieces_count

        elif settings.sifting_type is SiftingType.mahini:
            initial_scores = self.load_limit * self.intact_phi.T * self.p0
            self.sifting = Sifting(intact_points=self.intact_points, scores=initial_scores)
            self.phi = self.sifting.sifted_phi
            # self.q =
            # self.h =
            # self.w =
            # self.cs =
            self.points = self.sifting.sifted_yield_points
            self.points_count = self.sifting.sifted_points_count
            self.components_count = self.sifting.sifted_components_count
            self.pieces_count = self.sifting.sifted_pieces_count

        self.limits_count = 1 + self.disp_limits_count * 2
        self.softening_vars_count = 2 * self.points_count if self.include_softening else 0
        self.plastic_vars_count = self.pieces_count
        self.primary_vars_count = self.plastic_vars_count + self.softening_vars_count + 1
        self.constraints_count = self.plastic_vars_count + self.softening_vars_count + self.limits_count
        self.slack_vars_count = self.constraints_count
        self.total_vars_count = self.primary_vars_count + self.slack_vars_count
        self.landa_var = self.plastic_vars_count + self.softening_vars_count
        self.landa_bar_var = 2 * self.landa_var + 1
        self.limits_slacks = set(range(self.landa_bar_var, self.landa_bar_var + self.limits_count))

        # IMPORTANT: must be placed after sifted variables
        self.b = self._get_b_column()
        self.c = self._get_costs_row()
        self.table = self._create_table()

        self.is_two_phase = True if any(self.b < 0) else False

    def _get_b_column(self):
        yield_pieces_count = self.plastic_vars_count
        disp_limits_count = self.disp_limits_count

        b = np.ones((self.constraints_count))
        b[yield_pieces_count + self.softening_vars_count] = self.load_limit
        if self.include_softening:
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

    def _create_table(self):
        phi_p0 = self.phi.T * self.p0
        phi_pv_phi = self.phi.T * self.pv * self.phi

        landa_base_num = self.plastic_vars_count + self.softening_vars_count
        dv_phi = self.dv * self.phi

        raw_a = np.matrix(np.zeros((self.constraints_count, self.primary_vars_count)))
        raw_a[0:self.plastic_vars_count, 0:self.plastic_vars_count] = phi_pv_phi
        raw_a[0:self.plastic_vars_count, landa_base_num] = phi_p0

        if self.include_softening:
            raw_a[self.plastic_vars_count:(self.plastic_vars_count + self.softening_vars_count), 0:self.plastic_vars_count] = self.q
            raw_a[0:self.plastic_vars_count, self.plastic_vars_count:(self.plastic_vars_count + self.softening_vars_count)] = - self.h
            raw_a[self.plastic_vars_count:(self.plastic_vars_count + self.softening_vars_count), self.plastic_vars_count:(self.plastic_vars_count + self.softening_vars_count)] = self.w
        raw_a[landa_base_num, landa_base_num] = 1.0

        if self.disp_limits.any():
            disp_limit_base_num = self.plastic_vars_count + self.softening_vars_count + 1
            raw_a[disp_limit_base_num:(disp_limit_base_num + self.disp_limits_count), 0:self.plastic_vars_count] = dv_phi
            raw_a[(disp_limit_base_num + self.disp_limits_count):(disp_limit_base_num + 2 * self.disp_limits_count), 0:self.plastic_vars_count] = - dv_phi

            raw_a[disp_limit_base_num:(disp_limit_base_num + self.disp_limits_count), landa_base_num] = self.d0
            raw_a[(disp_limit_base_num + self.disp_limits_count):(disp_limit_base_num + 2 * self.disp_limits_count), landa_base_num] = - self.d0

        a_matrix = np.array(raw_a)
        columns_count = self.primary_vars_count + self.slack_vars_count
        table = np.zeros((self.constraints_count, columns_count))
        table[0:self.constraints_count, 0:self.primary_vars_count] = a_matrix

        # Assigning diagonal arrays of slack variables.
        # TODO: use np.eye instead
        j = self.primary_vars_count
        for i in range(self.constraints_count):
            table[i, j] = 1.0
            j += 1
        return table

    def update_b_for_dynamic_analysis(self, pv_prev, plastic_multipliers_prev):
        self.b[0:self.plastic_vars_count] = (
            self.b[0:self.plastic_vars_count] -
            np.array(
                self.phi.T * pv_prev * self.phi * plastic_multipliers_prev
            ).flatten()
        )

    def solve_dynamic(self):
        basic_variables = self.get_initial_basic_variables()
        print(f"{self.is_two_phase=}")
        if self.is_two_phase:
            db = np.zeros(self.slack_vars_count)
            self.negative_b_count = np.count_nonzero(self.b < 0)
            new_table = self.create_new_table()
            negative_b_counter = 1
            for i in range(self.constraints_count):
                if self.b[i] < 0:
                    self.update_b_for_negative_b_row(i)
                    basic_variables = self.update_basic_variables_for_negative_b_row(
                        basic_variables=basic_variables,
                        negative_b_row=i,
                    )
                    new_table = self.update_table_for_negative_b_row(
                        new_table=new_table,
                        negative_b_row=i,
                        negative_b_num=negative_b_counter,
                    )
                    negative_b_counter += 1
            self.table = new_table.copy()
            self.d = self.calculate_initial_d(self.table)
        bbar = self.b
        b_matrix_inv = np.eye(self.slack_vars_count)
        cb = np.zeros(self.slack_vars_count)
        x_cumulative = np.matrix(np.zeros((self.constraints_count, 1)))
        x_history = []
        fpm = FPM
        fpm.var = self.landa_var
        fpm.cost = 0
        if self.is_two_phase:
            fpm, b_matrix_inv, basic_variables, cb, db, will_out_row, will_out_var = self.enter_landa_dynamic(
                fpm=fpm,
                b_matrix_inv=b_matrix_inv,
                basic_variables=basic_variables,
                cb=cb,
                db=db,
            )
        else:
            fpm, b_matrix_inv, basic_variables, cb, will_out_row, will_out_var = self.enter_landa(
                fpm=fpm,
                b_matrix_inv=b_matrix_inv,
                basic_variables=basic_variables,
                cb=cb,
            )

        landa_row = will_out_row
        while self.limits_slacks.issubset(set(basic_variables)):
            if self.is_two_phase:
                sorted_slack_candidates = self.get_sorted_slack_d_candidates(
                    basic_variables=basic_variables,
                    b_matrix_inv=b_matrix_inv,
                    db=db,
                )
            else:
                sorted_slack_candidates = self.get_sorted_slack_candidates(
                    basic_variables=basic_variables,
                    b_matrix_inv=b_matrix_inv,
                    cb=cb,
                )
            will_in_col = fpm.var
            abar = self.calculate_abar(will_in_col, b_matrix_inv)
            bbar = self.calculate_bbar(b_matrix_inv, bbar)
            # TODO: ZERO OUT ABAR HERE AND IF UNBOUNDED END THE COMPUTATION
            abar = zero_out_small_values(abar)
            # if not any(abar > 0):
            #     print("unbounded")
            #     input()
            #     break
            # print("not unbounded")
            # input()
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
                        print("unload spm")
                        if self.is_two_phase:
                            basic_variables, b_matrix_inv, cb, db, landa_row = self.unload_dynamic(
                                pm_var=spm_var,
                                basic_variables=basic_variables,
                                b_matrix_inv=b_matrix_inv,
                                cb=cb,
                                db=db,
                                landa_row=landa_row,
                            )
                        else:
                            basic_variables, b_matrix_inv, cb, landa_row = self.unload(
                                pm_var=spm_var,
                                basic_variables=basic_variables,
                                b_matrix_inv=b_matrix_inv,
                                cb=cb,
                                landa_row=landa_row,
                            )
                        break
                else:
                    if self.is_will_out_var_opm(will_out_var):
                        print("unload opm")
                        opm_var = will_out_var
                        if self.is_two_phase:
                            basic_variables, b_matrix_inv, cb, db, landa_row = self.unload_dynamic(
                                pm_var=opm_var,
                                basic_variables=basic_variables,
                                b_matrix_inv=b_matrix_inv,
                                cb=cb,
                                db=db,
                                landa_row=landa_row,
                            )
                        else:
                            basic_variables, b_matrix_inv, cb, landa_row = self.unload(
                                pm_var=spm_var,
                                basic_variables=basic_variables,
                                b_matrix_inv=b_matrix_inv,
                                cb=cb,
                                landa_row=landa_row,
                            )
                        break
                    else:
                        print("enter fpm")
                        if self.is_two_phase:
                            basic_variables, b_matrix_inv, cb, db, fpm = self.enter_fpm_dynamic(
                                basic_variables=basic_variables,
                                b_matrix_inv=b_matrix_inv,
                                cb=cb,
                                db=db,
                                will_out_row=will_out_row,
                                will_in_col=will_in_col,
                                abar=abar,
                            )
                        else:
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
            pms = x[0:self.plastic_vars_count]
            load_level = x[self.landa_var][0, 0]
            pms_history.append(pms)
            load_level_history.append(load_level)
        result = {
            "pms_history": pms_history,
            "load_level_history": load_level_history
        }
        return result

    def solve(self):
        basic_variables = self.get_initial_basic_variables()
        if any(self.b < 0):
            new_table = self.create_new_table()
            negative_b_counter = 1
            for i in range(self.constraints_count):
                if self.b[i] < 0:
                    self.update_b_for_negative_b_row(i)
                    basic_variables = self.update_basic_variables_for_negative_b_row(
                        basic_variables=basic_variables,
                        negative_b_row=i,
                    )
                    new_table = self.update_table_for_negative_b_row(
                        new_table=new_table,
                        negative_b_row=i,
                        negative_b_num=negative_b_counter,
                    )
                    negative_b_counter += 1
            self.table = new_table.copy()
        bbar = self.b
        b_matrix_inv = np.eye(self.slack_vars_count)
        cb = np.zeros(self.slack_vars_count)
        x_cumulative = np.matrix(np.zeros((self.constraints_count, 1)))
        x_history = []
        fpm = FPM
        fpm.var = self.landa_var
        fpm.cost = 0
        fpm, b_matrix_inv, basic_variables, cb, will_out_row, will_out_var = self.enter_landa(
            fpm=fpm,
            b_matrix_inv=b_matrix_inv,
            basic_variables=basic_variables,
            cb=cb,
        )
        landa_row = will_out_row

        while self.limits_slacks.issubset(set(basic_variables)):
            sorted_slack_candidates = self.get_sorted_slack_candidates(
                basic_variables=basic_variables,
                b_matrix_inv=b_matrix_inv,
                cb=cb,
            )
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
                        print("unload spm")
                        basic_variables, b_matrix_inv, cb, landa_row = self.unload(
                            pm_var=spm_var,
                            basic_variables=basic_variables,
                            b_matrix_inv=b_matrix_inv,
                            cb=cb,
                            landa_row=landa_row,
                        )
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
        phi_pms_history = []
        load_level_history = []
        if settings.sifting_type is SiftingType.not_used:
            for x in x_history:
                pms = x[0:self.plastic_vars_count]
                phi_pms = self.intact_phi * pms
                load_level = x[self.landa_var][0, 0]
                phi_pms_history.append(phi_pms)
                load_level_history.append(load_level)
        if settings.sifting_type is SiftingType.mahini:
            structure_sifted_yield_pieces = self.sifting.structure_sifted_yield_pieces
            for x in x_history:
                pms = x[0:self.plastic_vars_count]
                intact_pms = np.matrix(np.zeros((self.intact_plastic_vars_count, 1)))
                for piece in structure_sifted_yield_pieces:
                    intact_pms[piece.intact_num_in_structure, 0] = x[piece.sifted_num_in_structure, 0]
                intact_phi_pms = self.intact_phi * intact_pms
                load_level = x[self.landa_var][0, 0]
                phi_pms_history.append(intact_phi_pms)
                load_level_history.append(load_level)
        result = {
            "phi_pms_history": phi_pms_history,
            "load_level_history": load_level_history
        }
        return result

    def get_slack_var(self, primary_var):
        if primary_var < self.primary_vars_count:
            # y variables from x variables
            return primary_var + self.primary_vars_count
        else:
            # z variables from y variables
            return primary_var + self.constraints_count

    def get_primary_var(self, slack_var):
        if slack_var < self.primary_vars_count + self.slack_vars_count:
            # x variables from y variables
            return slack_var - self.primary_vars_count
        else:
            # y variables from z variables
            return slack_var - self.constraints_count

    def enter_landa_dynamic(self, fpm, b_matrix_inv, basic_variables, cb, db):
        will_in_col = fpm.var
        a = self.table[:, will_in_col]
        print("enter landa")
        will_out_row = self.get_will_out(a, self.b)
        will_out_var = basic_variables[will_out_row]
        basic_variables = self.update_basic_variables(basic_variables, will_out_row, will_in_col)
        b_matrix_inv = self.update_b_matrix_inverse(b_matrix_inv, a, will_out_row)
        cb = self.update_cb(cb, will_in_col, will_out_row)
        db = self.update_db(db, will_in_col, will_out_row)
        cbar = self.calculate_cbar(cb, b_matrix_inv)
        dbar = self.calculate_dbar(db, b_matrix_inv)
        if self.is_two_phase:
            fpm = self.update_fpm(will_out_row, dbar)
        else:
            fpm = self.update_fpm(will_out_row, cbar)
        return fpm, b_matrix_inv, basic_variables, cb, db, will_out_row, will_out_var

    def enter_landa(self, fpm, b_matrix_inv, basic_variables, cb):
        will_in_col = fpm.var
        a = self.table[:, will_in_col]
        print("enter landa")
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

    def enter_fpm_dynamic(self, basic_variables, b_matrix_inv, cb, db, will_out_row, will_in_col, abar):
        b_matrix_inv = self.update_b_matrix_inverse(b_matrix_inv, abar, will_out_row)
        cb = self.update_cb(cb, will_in_col, will_out_row)
        db = self.update_db(db, will_in_col, will_out_row)
        dbar = self.calculate_dbar(db, b_matrix_inv)
        basic_variables = self.update_basic_variables(basic_variables, will_out_row, will_in_col)
        fpm = self.update_fpm(will_out_row, dbar)
        return basic_variables, b_matrix_inv, cb, db, fpm

    def unload_dynamic(self, pm_var, basic_variables, b_matrix_inv, cb, db, landa_row):
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
                    db = self.update_db(
                        db=db,
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

        return basic_variables, b_matrix_inv, cb, db, landa_row

    # def unload_dynamic(self, pm_var, basic_variables, b_matrix_inv, db, landa_row):
    #     # TODO: should handle if third pivot column is a y not x. possible bifurcation.
    #     # TODO: loading whole b_matrix_inv in input and output is costly, try like mahini method.
    #     # TODO: check line 60 of unload and line 265 in mclp of mahini code
    #     # (probable usage: in case when unload is last step)
    #     pm_var_family = self.get_pm_var_family(pm_var)
    #     for primary_var in pm_var_family:
    #         if primary_var in basic_variables:
    #             exiting_row = self.get_var_row(primary_var, basic_variables)

    #             unloading_pivot_elements = [
    #                 {
    #                     "row": exiting_row,
    #                     "column": self.get_slack_var(exiting_row),
    #                 },
    #                 {
    #                     "row": primary_var,
    #                     "column": self.get_slack_var(primary_var),
    #                 },
    #                 {
    #                     "row": exiting_row,
    #                     "column": basic_variables[primary_var],
    #                 },
    #             ]
    #             for element in unloading_pivot_elements:
    #                 abar = self.calculate_abar(element["column"], b_matrix_inv)
    #                 b_matrix_inv = self.update_b_matrix_inverse(b_matrix_inv, abar, element["row"])
    #                 db = self.update_db(
    #                     cb=db,
    #                     will_in_col=element["column"],
    #                     will_out_row=element["row"]
    #                 )
    #                 basic_variables = self.update_basic_variables(
    #                     basic_variables=basic_variables,
    #                     will_out_row=element["row"],
    #                     will_in_col=element["column"]
    #                 )

    #                 if element["column"] == self.landa_var:
    #                     landa_row = element["row"]

    #     return basic_variables, b_matrix_inv, db, landa_row

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
        a = self.table[:, col]
        abar = np.dot(b_matrix_inv, a)
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

    def calculate_dbar(self, db, b_matrix_inv):
        sigma_transpose = np.dot(db, b_matrix_inv)
        dbar = np.zeros(self.total_vars_count + self.negative_b_count)
        for i in range(self.total_vars_count + self.negative_b_count):
            dbar[i] = self.d[i] - np.dot(sigma_transpose, self.table[:, i])
        return dbar

    def update_cb(self, cb, will_in_col, will_out_row):
        cb[will_out_row] = self.c[will_in_col]
        return cb

    def update_db(self, db, will_in_col, will_out_row):
        db[will_out_row] = self.d[will_in_col]
        return db

    def get_will_out(self, abar, bbar, will_in_col=None, landa_row=None, basic_variables=None):
        # TODO: handle unbounded problem,
        # when there is no positive a remaining (structure failure), e.g. stop the process.

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
        sparse_e = csr_matrix(e)
        sparse_b_inv = csr_matrix(b_matrix_inv)
        updated_b_matrix_inv = sparse_e.dot(sparse_b_inv)
        return updated_b_matrix_inv.toarray()

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

    def get_sorted_slack_d_candidates(self, basic_variables, b_matrix_inv, db):
        dbar = self.calculate_dbar(db, b_matrix_inv)
        slack_candidates = []
        for var in basic_variables:
            if var < self.landa_var:
                slack_var = self.get_slack_var(var)
                slack_candidate = SlackCandidate(
                    var=slack_var,
                    cost=dbar[slack_var]
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

    def update_table_for_negative_b_row(self, new_table, negative_b_row, negative_b_num):
        new_table[negative_b_row, :] = - new_table[negative_b_row, :]
        new_table[negative_b_row, self.primary_vars_count + negative_b_num] = 1
        return new_table

    def update_b_for_negative_b_row(self, negative_b_row):
        self.b[negative_b_row] = - self.b[negative_b_row]

    def update_basic_variables_for_negative_b_row(self, basic_variables, negative_b_row):
        basic_variables[negative_b_row] = int(basic_variables[negative_b_row] + self.constraints_count)
        return basic_variables

    def create_new_table(self):
        new_table = np.array(
            np.zeros((self.constraints_count, self.table.shape[1] + self.negative_b_count))
        )
        new_table[:, :-self.negative_b_count] = self.table
        return new_table

    def calculate_initial_d(self, table):
        d = np.zeros(self.total_vars_count + self.negative_b_count)
        d[:self.total_vars_count] = table.sum(axis=0)[:self.total_vars_count]
        return d
