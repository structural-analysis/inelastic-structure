import copy
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
# from line_profiler import profile

from .models import FPM, SlackCandidate, Sifting, SiftedResults
from .functions import zero_out_small_values, print_specific_properties
from ..analysis.initial_analysis import InitialData, AnalysisData
from ..settings import settings, SiftingType

# np.set_printoptions(threshold=np.inf, precision=4)


class MahiniMethod:
    def __init__(self, initial_data: InitialData, analysis_data: AnalysisData, final_inc_phi_pms_prev=None):
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
            initial_scores = self.load_limit * np.dot(self.intact_phi.T, self.p0)
            self.sifting = Sifting(
                intact_points=self.intact_points,
                intact_pieces=self.intact_pieces,
                intact_phi=self.intact_phi,
                include_softening=self.include_softening,
            )
            self.sifted_results_current: SiftedResults = self.sifting.create(scores=initial_scores)
            self.structure_sifted_yield_pieces_current = self.sifted_results_current.structure_sifted_yield_pieces.copy()
            self.phi = self.sifted_results_current.structure_sifted_output.phi
            self.q = self.sifted_results_current.structure_sifted_output.q
            self.h = self.sifted_results_current.structure_sifted_output.h
            self.w = self.sifted_results_current.structure_sifted_output.w
            self.cs = self.sifted_results_current.structure_sifted_output.cs

            self.points_count = len(self.sifted_results_current.sifted_yield_points)
            self.components_count = self.sifted_results_current.sifted_components_count
            self.pieces_count = self.sifted_results_current.sifted_pieces_count

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
        self.final_inc_phi_pms_prev = final_inc_phi_pms_prev

        print(f"{self.total_vars_count=}")
        print(f"{self.primary_vars_count=}")
        print(f"{self.slack_vars_count=}")
        print(f"{self.plastic_vars_count=}")
        print(f"{self.softening_vars_count=}")
        print(f"{self.limits_count=}")

        # IMPORTANT: must be placed after sifted variables
        self.b = self._get_b_column()

        c = self._get_costs_row()
        self.costs = c.copy()
        self.cb = np.zeros(self.slack_vars_count)
        self.activated_costs = self.cb.copy()

        self.table = self._create_table()
        self.basic_variables = self.get_initial_basic_variables()
        # if analysis is dynamic:
        if self.final_inc_phi_pms_prev is not None:
            self.update_b_for_dynamic_analysis()
            self.negative_b_count = np.count_nonzero(self.b < 0)

            is_two_phase = True if self.negative_b_count > 0 else False
            if is_two_phase:
                two_phase_table = self.create_two_phase_table()

                negative_b_counter = 1
                for i in range(self.constraints_count):
                    if self.b[i] < 0:
                        self.update_b_for_negative_b_row(i)
                        self.basic_variables = self.update_basic_variables_for_negative_b_row(
                            basic_variables=self.basic_variables,
                            negative_b_row=i,
                        )
                        two_phase_table = self.update_table_for_negative_b_row(
                            two_phase_table=two_phase_table,
                            negative_b_row=i,
                            negative_b_num=negative_b_counter,
                        )
                        negative_b_counter += 1
                self.table = two_phase_table.copy()
                d = self.calculate_initial_d(self.table)
                db = np.zeros(self.slack_vars_count)
                self.activated_costs = self.cb.copy()
                self.costs = c.copy()

    # NOTE: SIFTING+: take care in advanced sifting b/c self.cs will change or not?
    def _get_b_column(self):
        yield_pieces_count = self.plastic_vars_count
        disp_limits_count = self.disp_limits_count

        b = np.ones((self.constraints_count))
        b[yield_pieces_count + self.softening_vars_count] = self.load_limit
        if self.include_softening:
            b[yield_pieces_count:(yield_pieces_count + self.softening_vars_count)] = np.array(self.cs)[:]

        if self.disp_limits.any():
            disp_limit_base_num = yield_pieces_count + self.softening_vars_count + 1
            b[disp_limit_base_num:(disp_limit_base_num + disp_limits_count)] = abs(self.disp_limits[:, 2])
            b[(disp_limit_base_num + disp_limits_count):(disp_limit_base_num + 2 * disp_limits_count)] = abs(self.disp_limits[:, 2])

        return b

    def _get_costs_row(self):
        c = np.zeros(self.total_vars_count)
        c[0:self.plastic_vars_count] = 1.0
        return -1 * c

    # @profile
    def _create_table(self):
        phi_p0 = self.phi.T @ self.p0
        phi_pv = self.phi.T @ self.pv
        phi_pv_phi = phi_pv @ self.phi

        landa_base_num = self.plastic_vars_count + self.softening_vars_count
        dv_phi = self.dv @ self.phi

        raw_a = np.zeros((self.constraints_count, self.primary_vars_count))
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

        print(f"{self.q.shape=}")
        print(f"{self.h.shape=}")
        print(f"{self.w.shape=}")
        print(f"{dv_phi.shape=}")
        print(f"{self.d0.shape=}")

        columns_count = self.primary_vars_count + self.slack_vars_count
        table = np.zeros((self.constraints_count, columns_count))
        table[0:self.constraints_count, 0:self.primary_vars_count] = raw_a
        table[:self.constraints_count, self.primary_vars_count:self.total_vars_count] = np.eye(self.constraints_count)
        return table

    # def _create_table(self):
    #     # Efficient matrix multiplications using the @ operator
    #     phi_p0 = self.phi.T @ self.p0  # Shape: (n_p, 1)
    #     phi_pv_phi = self.phi.T @ self.pv @ self.phi  # Shape: (n_p, n_p)
    #     dv_phi = self.dv @ self.phi  # Shape: (n_d, n_p)

    #     # Variables for dimensions
    #     n_p = self.plastic_vars_count
    #     n_s = self.softening_vars_count
    #     n_d = self.disp_limits_count
    #     n_c = self.constraints_count
    #     n_v = self.primary_vars_count
    #     landa_base_num = n_p + n_s

    #     # List to collect rows for raw_a
    #     raw_a_rows = []

    #     # --- First n_p rows ---
    #     # Left block: phi_pv_phi (n_p x n_p)
    #     row0_left = phi_pv_phi

    #     # Middle block: -self.h (n_p x n_s) or empty if n_s == 0
    #     row0_middle = -self.h if self.include_softening else np.empty((n_p, 0))

    #     # Right block: zeros with phi_p0 in the landa column
    #     right_cols = n_v - n_p - n_s
    #     row0_right = np.zeros((n_p, right_cols))
    #     phi_p0 = phi_p0.reshape(-1, 1)  # Ensure phi_p0 is a column vector
    #     landa_col_index = landa_base_num - n_p - n_s
    #     row0_right[:, landa_col_index] = phi_p0.flatten()

    #     # Concatenate blocks horizontally
    #     row0 = np.hstack((row0_left, row0_middle, row0_right))
    #     raw_a_rows.append(row0)

    #     # --- Next n_s rows (if softening is included) ---
    #     if self.include_softening:
    #         # Left block: self.q (n_s x n_p)
    #         row1_left = self.q

    #         # Middle block: self.w (n_s x n_s)
    #         row1_middle = self.w

    #         # Right block: zeros (n_s x right_cols)
    #         row1_right = np.zeros((n_s, right_cols))

    #         # Concatenate blocks horizontally
    #         row1 = np.hstack((row1_left, row1_middle, row1_right))
    #         raw_a_rows.append(row1)

    #     # --- Row for landa_base_num ---
    #     row_landa = np.zeros((1, n_v))
    #     row_landa[0, landa_base_num] = 1.0
    #     raw_a_rows.append(row_landa)

    #     # --- Displacement limits (if any) ---
    #     if self.disp_limits.any():
    #         # Ensure dv_phi has correct shape
    #         dv_phi = dv_phi.reshape(n_d, -1)

    #         # Left block: dv_phi (n_d x n_p)
    #         pos_disp_left = dv_phi

    #         # Middle block: zeros (n_d x n_s)
    #         pos_disp_middle = np.zeros((n_d, n_s))

    #         # Right block: zeros with self.d0 in the landa column
    #         pos_disp_right = np.zeros((n_d, right_cols))
    #         pos_disp_right[:, landa_col_index] = self.d0.flatten()

    #         # Positive displacement limits row
    #         pos_disp_row = np.hstack((pos_disp_left, pos_disp_middle, pos_disp_right))
    #         raw_a_rows.append(pos_disp_row)

    #         # Negative displacement limits row
    #         neg_disp_row = -pos_disp_row
    #         raw_a_rows.append(neg_disp_row)

    #     # --- Combine all rows ---
    #     raw_a = np.vstack(raw_a_rows)

    #     # --- Create the full table ---
    #     columns_count = self.primary_vars_count + self.slack_vars_count
    #     table = np.zeros((self.constraints_count, columns_count))

    #     # Assign the constructed raw_a to the table
    #     table[:, :self.primary_vars_count] = raw_a

    #     # Assign slack variables using an identity matrix
    #     table[:, self.primary_vars_count:self.total_vars_count] = np.eye(self.constraints_count)

    #     return table

    def update_b_for_dynamic_analysis(self):
        # print(f"{self.phi.T=}")
        # print(f"{self.pv_prev=}")
        # print(f"{self.final_inc_phi_pms_prev=}")
        # print(f"{self.b[0:self.plastic_vars_count]=}")
        # input()
        self.b[0:self.plastic_vars_count] = self.b[0:self.plastic_vars_count] - self.phi.T @ self.pv @ self.final_inc_phi_pms_prev

    def solve(self):
        basic_variables = self.basic_variables
        activated_costs = self.activated_costs

        bbar = self.b
        b_matrix_inv = np.eye(self.slack_vars_count)
        phi_pms_cumulative = np.zeros(self.intact_components_count)
        load_level_cumulative = 0
        phi_pms_history = []

        load_level_cumulative = 0
        load_level_history = []
        pms_history = []

        if self.include_softening:
            h_sms_cumulative = np.zeros(self.intact_pieces_count)
            h_sms_history = []

        fpm = FPM(
            var=self.landa_var,
            cost=0,
        )

        increment = 0
        print("-------------------------------")
        print(f"{increment=}")
        fpm, b_matrix_inv, basic_variables, activated_costs, will_out_row, will_out_var = self.enter_landa(
            fpm=fpm,
            b_matrix_inv=b_matrix_inv,
            basic_variables=basic_variables,
            cb=activated_costs,
        )

        landa_row = will_out_row
        will_in_col = fpm.var
        abar = self.calculate_abar(will_in_col, b_matrix_inv)
        bbar = self.calculate_bbar(b_matrix_inv, bbar)
        will_out_row, sorted_zipped_ba = self.get_will_out(abar, bbar, will_in_col, landa_row, basic_variables)
        will_out_var = basic_variables[will_out_row]
        x, bbar = self.reset(basic_variables, bbar)

        if settings.sifting_type is SiftingType.not_used:
            pms = x[0:self.plastic_vars_count]
            phi_pms = self.intact_phi @ pms
            phi_pms_cumulative += phi_pms
            pms_history.append(pms)
            phi_pms_history.append(phi_pms_cumulative.copy())

            load_level = x[self.landa_var]
            load_level_cumulative += load_level
            load_level_history.append(load_level_cumulative)

            if self.include_softening:
                sms = x[self.plastic_vars_count:self.landa_var]
                h_sms = self.intact_h @ sms
                h_sms_cumulative += h_sms
                h_sms_history.append(h_sms_cumulative.copy())

        if settings.sifting_type is SiftingType.mahini:
            intact_pms = self.get_unsifted_pms(
                x=x,
                structure_sifted_yield_pieces=self.structure_sifted_yield_pieces_current,
            )
            intact_phi_pms = self.get_phi_pms(intact_pms=intact_pms)
            phi_pms_cumulative += intact_phi_pms
            pms_history.append(intact_pms)
            phi_pms_history.append(phi_pms_cumulative.copy())

            load_level = x[self.landa_var]
            load_level_cumulative += load_level
            load_level_history.append(load_level_cumulative)

            if self.include_softening:
                sms = x[self.plastic_vars_count:self.landa_var]
                h_sms = self.intact_h @ sms
                h_sms_cumulative += h_sms
                h_sms_history.append(h_sms_cumulative.copy())

        while self.limits_slacks.issubset(set(basic_variables)):

            increment = len(load_level_history)
            print("-------------------------------")
            print(f"{increment=}")
            print(f"{load_level_cumulative=}")
            print(f"will_in_col=x-{will_in_col}")
            print(f"{will_out_row=}")
            print(f"{will_out_var=}")

            # if settings.sifting_type == SiftingType.not_used:
            #     print("basic_variables:")
            #     for basic_variable in basic_variables:
            #         if basic_variable < self.landa_var:
            #             print(basic_variable)

            # if settings.sifting_type == SiftingType.mahini:
            #     print(f"global_will_in_col=x-{self.structure_sifted_yield_pieces_current[will_in_col].num_in_structure}")
            #     if will_out_row != 64:
            #         print(f"global_will_out_row={self.structure_sifted_yield_pieces_current[will_out_row].num_in_structure}")
            #     else:
            #         print("global_will_out_row=?")
            #     primary_will_out_var = self.get_primary_var(will_out_var)
            #     if primary_will_out_var < 0:
            #         primary_will_out_var = will_out_var
            #         print(f"will_out_var=x-{primary_will_out_var}")
            #         print(f"global_will_out_var=x-{self.structure_sifted_yield_pieces_current[primary_will_out_var].num_in_structure}")
            #     else:
            #         print(f"will_out_var=y-{primary_will_out_var}")
            #         # print(f"global_will_out_var=y-{self.structure_sifted_yield_pieces_current[primary_will_out_var].num_in_structure}")
            #     # print_specific_properties(self.structure_sifted_yield_pieces_current, ["ref_yield_point_num", "num_in_yield_point", "num_in_structure", "sifted_num_in_structure"])

            if settings.sifting_type == SiftingType.not_used:
                unsifted_primary_vars = []
                for basic_variable in basic_variables:
                    if basic_variable < self.landa_var:
                        unsifted_primary_vars.append(basic_variable)
                # print(f"{unsifted_primary_vars=}")

            # print(f"{bbar[will_out_row]=}")
            # print(f"{sorted_zipped_ba=}")
            # print(f"{abar[27]=}")
            # print(f"{bbar[27]=}")
            # print(f"{bbar[27]/abar[27]=}")

            # print(f"{abar[25]=}")
            # print(f"{bbar[25]=}")
            # print(f"{bbar[25]/abar[25]=}")

            # print(f"{basic_variables[2705]=}")
            # print(f"{basic_variables[2745]=}")

            # print(f"{abar[2705]=}")
            # print(f"{bbar[2705]=}")
            # print(f"{bbar[2705]/abar[2705]=}")

            # # var 3298 is in 2745 row
            # print(f"{abar[2745]=}")
            # print(f"{bbar[2745]=}")
            # print(f"{bbar[2745]/abar[2745]=}")

            sorted_slack_candidates, cbar = self.get_sorted_slack_candidates(
                basic_variables=basic_variables,
                b_matrix_inv=b_matrix_inv,
                cb=activated_costs,
            )

            if settings.sifting_type is SiftingType.mahini:
                b_matrix_inv_prev = b_matrix_inv.copy()
                basic_variables_prev = basic_variables.copy()
                cb_prev = activated_costs.copy()
                bbar_prev = bbar.copy()
                x_prev = x.copy()
                fpm_prev = FPM(var=fpm.var, cost=fpm.cost)

                # for basic in basic_variables:
                #     if basic < self.plastic_vars_count:
                #         print(f"{self.structure_sifted_yield_pieces_current[basic].num_in_structure}")
                #     elif basic < self.landa_var:
                #         print(f"{basic - self.sifted_results_current.sifted_pieces_count + self.intact_pieces_count}")

                plastic_vars_in_basic_variables_prev = self.get_plastic_vars_in_basic_variables(
                    basic_variables_prev,
                    self.landa_var,
                    self.structure_sifted_yield_pieces_current,
                )
            for slack_candidate in sorted_slack_candidates + [fpm]:
                if not self.is_candidate_fpm(fpm, slack_candidate):
                    spm_var = self.get_primary_var(slack_candidate.var)
                    # print(f"{slack_candidate.var=}")
                    # print(f"{spm_var=}")
                    r = self.calculate_r(
                        spm_var=spm_var,
                        basic_variables=basic_variables,
                        abar=abar,
                        b_matrix_inv=b_matrix_inv,
                    )
                    # print(f"{r=}")
                    # input()
                    if r > 0:
                        continue
                    else:
                        print("unload spm")
                        basic_variables, b_matrix_inv, activated_costs, landa_row = self.unload(
                            pm_var=spm_var,
                            basic_variables=basic_variables,
                            b_matrix_inv=b_matrix_inv,
                            cb=activated_costs,
                            landa_row=landa_row,
                        )
                        check_violation = False
                        break
                else:
                    if self.is_will_out_var_opm(will_out_var):
                        print("unload opm")
                        opm_var = will_out_var
                        basic_variables, b_matrix_inv, activated_costs, landa_row = self.unload(
                            pm_var=opm_var,
                            basic_variables=basic_variables,
                            b_matrix_inv=b_matrix_inv,
                            cb=activated_costs,
                            landa_row=landa_row,
                        )
                        check_violation = False
                        break
                    else:
                        print("enter fpm")
                        basic_variables, b_matrix_inv, activated_costs, fpm = self.enter_fpm(
                            basic_variables=basic_variables,
                            b_matrix_inv=b_matrix_inv,
                            cb=activated_costs,
                            will_out_row=will_out_row,
                            will_in_col=will_in_col,
                            abar=abar,
                        )
                        check_violation = True
                        break

            #     elif settings.sifting_type == SiftingType.mahini:
            #         basics_primary_vars = []
            #         primary_or_slack = []
            #         only_primary_basics = []
            #         for basic_variable in basic_variables:
            #             if basic_variable < self.landa_var:
            #                 primary_var = basic_variable
            #                 print(self.structure_sifted_yield_pieces_current[primary_var].num_in_structure)
            #                 basics_primary_vars.append(self.structure_sifted_yield_pieces_current[primary_var].num_in_structure)
            #                 primary_or_slack.append("primary")
            #                 only_primary_basics.append(self.structure_sifted_yield_pieces_current[primary_var].num_in_structure)
            #             else:
            #                 primary_var = self.get_primary_var(basic_variable)
            #                 if primary_var != self.landa_var:
            #                     print(self.structure_sifted_yield_pieces_current[primary_var].num_in_structure)
            #                     basics_primary_vars.append(self.structure_sifted_yield_pieces_current[primary_var].num_in_structure)
            #                     primary_or_slack.append("slack")
            #         # print(f"{basics_primary_vars=}")
            #         # print(f"{primary_or_slack=}")
            #         print(f"{only_primary_basics=}")
            #         print("--------------------------")
            #     input()

            #     # updated_indices=[105, 106, 65, 66, 718, 759, 719, 758, 1081, 1042, 1041, 1040, 1396, 1435, 1394, 1395, 2130, 2091, 2133, 2132, 2307, 2306, 2267, 2266, 2704, 2745, 2706, 2705, 3297, 3298, 3257, 3258, 3643, 3644, 3645, 3684, 4174, 4135, 4134, 4133, 4538, 4539, 4578, 4537, 4975, 4976, 4974, 4973, 5547, 5507, 5546, 5506, 5840, 5841, 5839, 5880, 6277, 6276, 6278, 6275, 6752, 6712, 6753, 6713]
            #     # inc 5
            #     # inc 14
            #     # updated_indices = [105, 106, 65, 66, 718, 678, 719, 679, 1081, 1080, 1041, 1040, 1434, 1435, 1433, 1474, 2051, 2091, 2090, 2092, 2307, 2306, 2267, 2266, 2746, 2745, 2747, 2786, 3297, 3298, 3337, 3338, 3643, 3644, 3683, 3642, 4174, 4173, 4134, 4133, 4538, 4539, 4578, 4537, 4975, 4976, 4974, 4973, 5547, 5507, 5546, 5506, 5840, 5841, 5839, 5880, 6277, 6276, 6278, 6275, 6752, 6712, 6753, 6713]
            #     # updated_indices = [105, 106, 65, 66, 718, 6-78, 719, 679, 1081, 1080, 1041, 1040, 1434, 1435, 1433, 1474, 2051, 2091, 2090, 2092, 2307, 2306, 2267, 2266, 2746, 2745, 2747, 2786, 3297, 3298, 3337, 3338, 3643, 3644, 3683, 3642, 4174, 4173, 4134, 4133, 4538, 4539, 4578, 4537, 4975, 4976, 4974, 4973, 5547, 5507, 5546, 5506, 5840, 5841, 5839, 5880, 6277, 6276, 6278, 6275, 6752, 6712, 6753, 6713]
            #     # np.savetxt(f"table_{increment}.csv", self.table[np.ix_(updated_indices, updated_indices)], delimiter=", ", fmt='%1.4e')
            #     # np.savetxt(f"phi_{increment}.csv", self.phi[:, updated_indices], delimiter=", ", fmt='%1.4e')
            #     # np.savetxt(f"b_matrix_inv_{increment}.csv", b_matrix_inv[np.ix_(updated_indices, updated_indices)], delimiter=", ", fmt='%1.4e')
            #     # input()
            #     #     print(f"{will_in_col=}")
            #     #     print(f"global_will_in_col=x-{self.structure_sifted_yield_pieces_current[will_in_col].num_in_structure}")
            #     # print(f"{updated_indices=}")
            #     #     # !!!!!!!!!!!! these values must check with increment 21 vaules,
            #     #     # beacause values are updated based on increment 21 values !!!!!!!!!!!!!!!!!!!!

            #     # np.savetxt("table.csv", self.table, delimiter=", ", fmt='%1.4e')
            #     # np.savetxt("phi.csv", self.phi, delimiter=", ", fmt='%1.4e')
            #     # np.savetxt("b_matrix_inv.csv", b_matrix_inv, delimiter=", ", fmt='%1.4e')
            #     # input()

            will_in_col = fpm.var
            abar = self.calculate_abar(will_in_col, b_matrix_inv)
            bbar = self.calculate_bbar(b_matrix_inv, bbar)
            will_out_row, sorted_zipped_ba = self.get_will_out(abar, bbar, will_in_col, landa_row, basic_variables)
            will_out_var = basic_variables[will_out_row]
            x, bbar = self.reset(basic_variables, bbar)

            # print("abar: ")
            # for num, el in enumerate(abar):
            #     print(num, ":", el)
            # print("current sifted yield pieces:")
            # print_specific_properties(
            #     obj_list=self.structure_sifted_yield_pieces_current,
            #     properties=[
            #         "sifted_num_in_structure",
            #         "num_in_structure",
            #         "ref_yield_point_num",
            #         "num_in_yield_point",
            #     ],
            # )
            # input()

            if settings.sifting_type is SiftingType.not_used:
                pms = x[0:self.plastic_vars_count]
                phi_pms = self.intact_phi @ pms
                phi_pms_cumulative += phi_pms
                pms_history.append(pms)
                phi_pms_history.append(phi_pms_cumulative.copy())

                load_level = x[self.landa_var]
                load_level_cumulative += load_level
                load_level_history.append(load_level_cumulative)

                if self.include_softening:
                    sms = x[self.plastic_vars_count:self.landa_var]
                    h_sms = self.intact_h @ sms
                    h_sms_cumulative += h_sms
                    h_sms_history.append(h_sms_cumulative.copy())

            if settings.sifting_type is SiftingType.mahini:
                intact_pms = self.get_unsifted_pms(
                    x=x,
                    structure_sifted_yield_pieces=self.structure_sifted_yield_pieces_current,
                )
                intact_phi_pms = self.get_phi_pms(intact_pms=intact_pms)

                phi_pms_cumulative += intact_phi_pms
                load_level = x[self.landa_var]
                load_level_cumulative += load_level

                if self.include_softening:
                    sms = x[self.plastic_vars_count:self.landa_var]
                    h_sms = self.intact_h @ sms
                    h_sms_cumulative += h_sms

                if check_violation:
                    if self.include_softening:
                        scores_current = self.calc_violation_scores(phi_pms_cumulative, load_level_cumulative, h_sms_cumulative)
                    else:
                        scores_current = self.calc_violation_scores(phi_pms_cumulative, load_level_cumulative)

                    sifted_results_old = self.sifted_results_current
                    structure_sifted_yield_pieces_old = self.structure_sifted_yield_pieces_current
                    violated_pieces = self.sifting.check_violation(
                        scores=scores_current,
                        structure_sifted_yield_pieces_old=structure_sifted_yield_pieces_old,
                    )
                    if violated_pieces:
                        # NOTE: in current increment some pieces are violated
                        # we want to roll back table to previous increment
                        # so we use previous increment plastic multipliers and load level to get sifted data and sorted pieces
                        # then we will modify previous increment to consider current increment's violated pieces in it's sifted data
                        if self.include_softening:
                            scores_prev = self.calc_violation_scores(phi_pms_history[-1], load_level_history[-1], h_sms_history[-1])
                        else:
                            scores_prev = self.calc_violation_scores(phi_pms_history[-1], load_level_history[-1])

                        print("++++ piece violation ++++")
                        print_specific_properties(violated_pieces, ["ref_yield_point_num", "num_in_yield_point", "num_in_structure"])
                        # print(f"top violated current score={scores_current[violated_pieces[0].num_in_structure]}")
                        # print(f"top violated prev score={scores_prev[violated_pieces[0].num_in_structure]}")

                        fpm = fpm_prev
                        will_in_col = fpm.var
                        if will_in_col < self.plastic_vars_count:
                            will_in_col_piece_num_in_structure = structure_sifted_yield_pieces_old[will_in_col].num_in_structure
                        else:
                            will_in_col_piece_num_in_structure = None

                        self.sifted_results_current: SiftedResults = self.sifting.update(
                            increment=increment,
                            plastic_vars_in_basic_variables_prev=plastic_vars_in_basic_variables_prev,
                            scores=scores_prev,
                            sifted_results_old=sifted_results_old,
                            violated_pieces=violated_pieces,
                            bbar_prev=bbar_prev,
                            b_matrix_inv_prev=b_matrix_inv_prev,
                            basic_variables_prev=basic_variables_prev,
                            landa_row=landa_row,
                            landa_var=self.landa_var,
                            pv=self.pv,
                            p0=self.p0,
                            will_in_col_piece_num_in_structure=will_in_col_piece_num_in_structure,
                            plastic_vars_count=self.plastic_vars_count,
                            softening_vars_count=self.softening_vars_count,
                            dv=self.dv,
                            constraints_count=self.constraints_count,
                            primary_vars_count=self.primary_vars_count,
                            disp_limits_count=self.disp_limits_count,
                            d0=self.d0,
                            disp_limits=self.disp_limits,
                        )

                        self.structure_sifted_yield_pieces_current = self.sifted_results_current.structure_sifted_yield_pieces
                        self.phi = self.sifted_results_current.structure_sifted_output.phi
                        self.q = self.sifted_results_current.structure_sifted_output.q
                        self.h = self.sifted_results_current.structure_sifted_output.h
                        self.w = self.sifted_results_current.structure_sifted_output.w
                        self.cs = self.sifted_results_current.structure_sifted_output.cs

                        b_matrix_inv = self.sifted_results_current.b_matrix_inv_updated
                        bbar = self.sifted_results_current.bbar_updated
                        # in_structure_indices = [num.num_in_structure for num in self.structure_sifted_yield_pieces_current]
                        # print(f"##################{in_structure_indices}")
                        basic_variables = basic_variables_prev
                        activated_costs = cb_prev
                        # plastic_vars_in_basic_variables = self.get_plastic_vars_in_basic_variables(
                        #     basic_variables,
                        #     self.landa_var,
                        #     self.structure_sifted_yield_pieces_current,
                        # )
                        # print(f"{plastic_vars_in_basic_variables=}")
                        # NOTE: not done for softening
                        # NOTE: SIFTING+: var nums should change
                        # TODO: check sifting with violation with disp limits
                        # disp limits are calculated in table creation.
                        # in mahini name is C_LastRows and updated after violation
                        # updated_indices = [piece.num_in_structure for piece in self.structure_sifted_yield_pieces_current]
                        self.table = self._create_table()

                        abar = self.calculate_abar(will_in_col, b_matrix_inv)
                        will_out_row, sorted_zipped_ba = self.get_will_out(abar, bbar, will_in_col, landa_row, basic_variables)
                        will_out_var = basic_variables[will_out_row]
                        # primary_will_out_var = self.get_primary_var(will_out_var)
                        # if primary_will_out_var < 0:
                        #     primary_will_out_var = will_out_var
                        x = x_prev
                        phi_pms_cumulative -= intact_phi_pms
                        load_level_cumulative -= load_level
                        if self.include_softening:
                            h_sms_cumulative -= h_sms
                    else:
                        pms_history.append(intact_pms)
                        phi_pms_history.append(phi_pms_cumulative.copy())
                        load_level_history.append(load_level_cumulative)
                        if self.include_softening:
                            h_sms_history.append(h_sms_cumulative.copy())
                else:
                    pms_history.append(intact_pms)
                    phi_pms_history.append(phi_pms_cumulative.copy())
                    load_level_history.append(load_level_cumulative)
                    if self.include_softening:
                        h_sms_history.append(h_sms_cumulative.copy())

        if self.final_inc_phi_pms_prev is not None:
            final_inc_phi_pms = self.final_inc_phi_pms_prev + phi_pms_history[-1]
            result = {
                "pms_history": pms_history,
                "phi_pms_history": phi_pms_history,
                "load_level_history": load_level_history,
                "final_inc_phi_pms": final_inc_phi_pms,
            }
        else:
            result = {
                "pms_history": pms_history,
                "phi_pms_history": phi_pms_history,
                "load_level_history": load_level_history,
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

    def enter_landa(self, fpm, b_matrix_inv, basic_variables, cb):
        will_in_col = fpm.var
        a = self.table[:, will_in_col]
        will_out_row, sorted_zipped_ba = self.get_will_out(a, self.b)
        will_out_var = basic_variables[will_out_row]
        print(f"{will_in_col=}")
        print(f"{will_out_row=}")
        print(f"{will_out_var=}")
        print("enter landa")
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
        # print(f"{pm_var_family=}")
        for primary_var in pm_var_family:
            # print("unload @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            # print(f"{primary_var=}")
            if primary_var in basic_variables:
                exiting_row = self.get_var_row(primary_var, basic_variables)
                # print("primary var in basic variables")
                # print(f"{primary_var=}")
                # print("first:")
                # print(f"in: {self.get_slack_var(exiting_row)=}")
                # print(f"out: {exiting_row=}")
                # print("---")
                # print("second:")
                # print(f"in: {self.get_slack_var(primary_var)=}")
                # print(f"out: {primary_var=}")
                # print("---")
                # print("third:")
                # print(f"in: {basic_variables[primary_var]=}")
                # print(f"out: {exiting_row=}")
                # print("---")

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
                # for var in basic_variables:
                #     if var < self.primary_vars_count:
                #         print(f"{var=}")
        return basic_variables, b_matrix_inv, cb, landa_row

    def get_pm_var_family(self, pm_var):
        if self.softening_vars_count:
            if pm_var < self.plastic_vars_count:
                yield_point = self.get_plastic_var_yield_point(pm_var)
                pm_var_family = [
                    self.plastic_vars_count + yield_point * 2,
                    self.plastic_vars_count + yield_point * 2 + 1,
                    pm_var,
                ]
            else:
                pm_var_family = [
                    pm_var,
                ]
        else:
            pm_var_family = [
                pm_var,
            ]
        return pm_var_family

    def calculate_abar(self, col, b_matrix_inv):
        # TODO: no need to calculate all table, just calculate a every time,
        # like cprow in mahini code.

        a = self.table[:, col]
        abar = np.dot(b_matrix_inv, a)
        return abar

    def calculate_bbar(self, b_matrix_inv, bbar):
        bbar = np.dot(b_matrix_inv, bbar)
        return bbar

    def calculate_cbar(self, cb, b_matrix_inv):
        # TODO: it seems we can calculate costs along with b_matrix_inv
        # and last member of selected a column, this will cause no need for full table calcs.
        # it seems fpm cost is Cprow(m + 4) in mahini code, check again.

        pi_transpose = np.dot(cb, b_matrix_inv)
        cbar = np.zeros(self.total_vars_count)
        for i in range(self.total_vars_count):
            cbar[i] = self.costs[i] - pi_transpose @ self.table[:, i]
        return cbar

    def calculate_dbar(self, db, b_matrix_inv):
        sigma_transpose = np.dot(db, b_matrix_inv)
        dbar = np.zeros(self.total_vars_count + self.negative_b_count)
        for i in range(self.total_vars_count + self.negative_b_count):
            dbar[i] = self.d[i] - np.dot(sigma_transpose, self.table[:, i])
        return dbar

    def update_cb(self, cb, will_in_col, will_out_row):
        cb[will_out_row] = self.costs[will_in_col]
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
        # print(f"{sorted_zipped_ba=}")
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
                        # print(f"{will_in_col_yield_point=}")
                        # print(f"{will_out_yield_point=}")
                        # input()

                        if will_in_col_yield_point != will_out_yield_point:
                            will_out_row = int(sorted_zipped_ba[0, i])
                            break
        return will_out_row, sorted_zipped_ba

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
        # print(f"{will_in_col=}")
        # for basic in basic_variables:
        #     if basic < self.landa_var:
        #         print(f"{basic=}")
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
        return slack_candidates, cbar

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

    def reset(self, basic_variables, bbar):
        x = np.zeros(self.constraints_count)
        for i, basic_variable in enumerate(basic_variables):
            if basic_variable < self.primary_vars_count:
                x[basic_variables[i]] = bbar[i]
                bbar[i] = 0
        return x, bbar

    def calculate_r(self, spm_var, basic_variables, abar, b_matrix_inv):
        spm_row = self.get_var_row(spm_var, basic_variables)
        r = abar[spm_row] / b_matrix_inv[spm_row, spm_var]
        return r

    def get_var_row(self, var, basic_variables):
        row = np.where(basic_variables == var)[0][0]
        return row

    def get_plastic_var_yield_point(self, pm):
        if settings.sifting_type is SiftingType.not_used:
            yield_point = self.intact_pieces[pm].ref_yield_point_num
        elif settings.sifting_type is SiftingType.mahini:
            yield_point = self.structure_sifted_yield_pieces_current[pm].ref_yield_point_num

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

    def update_table_for_negative_b_row(self, two_phase_table, negative_b_row, negative_b_num):
        two_phase_table[negative_b_row, :] = - two_phase_table[negative_b_row, :]
        two_phase_table[negative_b_row, self.primary_vars_count + negative_b_num] = 1
        return two_phase_table

    def update_b_for_negative_b_row(self, negative_b_row):
        self.b[negative_b_row] = - self.b[negative_b_row]

    def update_basic_variables_for_negative_b_row(self, basic_variables, negative_b_row):
        basic_variables[negative_b_row] = int(basic_variables[negative_b_row] + self.constraints_count)
        return basic_variables

    def create_two_phase_table(self):
        two_phase_table = np.array(
            np.zeros((self.constraints_count, self.table.shape[1] + self.negative_b_count))
        )
        two_phase_table[:, :-self.negative_b_count] = self.table
        return two_phase_table

    def calculate_initial_d(self, table):
        d = np.zeros(self.total_vars_count + self.negative_b_count)
        d[:self.total_vars_count] = table.sum(axis=0)[:self.total_vars_count]
        return d

    def calc_violation_scores(self, intact_phi_pms, load_level, intact_h_sms=None):
        if self.include_softening:
            scores = self.intact_phi.T @ self.pv @ intact_phi_pms + self.intact_phi.T @ self.p0 * load_level - intact_h_sms - np.ones(self.intact_pieces_count)
        else:
            scores = self.intact_phi.T @ self.pv @ intact_phi_pms + self.intact_phi.T @ self.p0 * load_level - np.ones(self.intact_pieces_count)
        return scores

    def get_unsifted_pms(self, x, structure_sifted_yield_pieces):
        intact_pms = np.zeros(self.intact_phi.shape[1])
        for piece in structure_sifted_yield_pieces:
            intact_pms[piece.num_in_structure] = x[piece.sifted_num_in_structure]
        return intact_pms

    def get_phi_pms(self, intact_pms):
        intact_phi_pms = np.dot(self.intact_phi, intact_pms)
        return intact_phi_pms

    def get_plastic_vars_in_basic_variables(self, basic_variables, landa_var, structure_sifted_yield_pieces):
        plastic_vars = []
        for basic_variable in basic_variables:
            if basic_variable < len(structure_sifted_yield_pieces):
                plastic_vars.append(structure_sifted_yield_pieces[basic_variable].num_in_structure)
        return plastic_vars
