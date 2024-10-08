import copy
import numpy as np
from scipy.sparse import csr_matrix

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
        self.pv_prev = analysis_data.pv_prev

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
            self.sifting = Sifting(
                intact_points=self.intact_points,
                intact_pieces=self.intact_pieces,
                intact_phi=self.intact_phi,
            )
            self.sifted_results_current: SiftedResults = self.sifting.create(scores=initial_scores)
            self.structure_sifted_yield_pieces_current = self.sifted_results_current.structure_sifted_yield_pieces.copy()
            self.phi = self.sifted_results_current.structure_sifted_phi
            # self.q
            # self.h
            # self.w
            # self.cs
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
            print(f"{self.b=}")
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

    def update_b_for_dynamic_analysis(self):
        self.b[0:self.plastic_vars_count] = (
            self.b[0:self.plastic_vars_count] -
            np.array(
                self.phi.T * self.pv_prev * self.final_inc_phi_pms_prev
            ).flatten()
        )

    def solve(self):
        basic_variables = self.basic_variables
        activated_costs = self.activated_costs

        bbar = self.b
        b_matrix_inv = np.eye(self.slack_vars_count)
        phi_pms_cumulative = np.matrix(np.zeros((self.intact_components_count, 1)))
        load_level_cumulative = 0
        phi_pms_history = []
        load_level_history = []
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
            phi_pms = self.intact_phi * pms
            load_level = x[self.landa_var][0, 0]
            phi_pms_cumulative += phi_pms
            load_level_cumulative += load_level
            phi_pms_history.append(phi_pms_cumulative.copy())
            load_level_history.append(load_level_cumulative)
        if settings.sifting_type is SiftingType.mahini:
            intact_phi_pms = self.get_unsifted_pms(
                x=x,
                structure_sifted_yield_pieces=self.structure_sifted_yield_pieces_current,
            )
            load_level = x[self.landa_var][0, 0]
            phi_pms_cumulative += intact_phi_pms
            load_level_cumulative += load_level
            phi_pms_history.append(phi_pms_cumulative.copy())
            load_level_history.append(load_level_cumulative)

        while self.limits_slacks.issubset(set(basic_variables)):
            increment = len(load_level_history)
            print("-------------------------------")
            print(f"{increment=}")
            print(f"{load_level_cumulative=}")
            print(f"will_in_col=x-{will_in_col}")
            print(f"{will_out_row=}")
            print(f"{will_out_var=}")

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
                plastic_vars_in_basic_variables_prev = self.get_plastic_vars_in_basic_variables(
                    basic_variables_prev,
                    self.landa_var,
                    self.structure_sifted_yield_pieces_current,
                )
                # print(f"input {plastic_vars_in_basic_variables_prev=}")

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
                phi_pms = self.intact_phi * pms
                load_level = x[self.landa_var][0, 0]
                phi_pms_cumulative += phi_pms
                load_level_cumulative += load_level
                phi_pms_history.append(phi_pms_cumulative.copy())
                load_level_history.append(load_level_cumulative)

            if settings.sifting_type is SiftingType.mahini:
                intact_phi_pms = self.get_unsifted_pms(
                    x=x,
                    structure_sifted_yield_pieces=self.structure_sifted_yield_pieces_current,
                )
                phi_pms_cumulative += intact_phi_pms
                load_level = x[self.landa_var][0, 0]
                load_level_cumulative += load_level

                if check_violation:
                    scores_current = self.calc_violation_scores(phi_pms_cumulative, load_level_cumulative)
                    # if increment == 23:
                    #     print(f"{scores_current[2131]=}")
                    #     print(f"{scores_current[2132]=}")
                    #     print(f"{scores_current.shape=}")
                    #     input()
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
                        scores_prev = self.calc_violation_scores(phi_pms_history[-1], load_level_history[-1])
                        print("++++ piece violation ++++")
                        print_specific_properties(violated_pieces, ["ref_yield_point_num", "num_in_yield_point", "num_in_structure"])
                        # print(f"{violated_pieces=}")
                        # print(f"top violated current score={scores_current[violated_pieces[0].num_in_structure]}")
                        # print(f"top violated prev score={scores_prev[violated_pieces[0].num_in_structure]}")

                        fpm = fpm_prev
                        will_in_col = fpm.var
                        will_in_col_piece_num_in_structure = structure_sifted_yield_pieces_old[will_in_col].num_in_structure
                        # print(f"{will_in_col=}")
                        # print(f"{will_in_col_piece_num_in_structure=}")
                        # print(f"before update {plastic_vars_in_basic_variables_prev=}")
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
                        )
                        self.structure_sifted_yield_pieces_current = self.sifted_results_current.structure_sifted_yield_pieces
                        self.phi = self.sifted_results_current.structure_sifted_phi
                        b_matrix_inv = self.sifted_results_current.b_matrix_inv_updated
                        bbar = self.sifted_results_current.bbar_updated
                        # self.q
                        # self.h
                        # self.w
                        # self.cs
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
                    else:
                        phi_pms_history.append(phi_pms_cumulative.copy())
                        load_level_history.append(load_level_cumulative)
                else:
                    phi_pms_history.append(phi_pms_cumulative.copy())
                    load_level_history.append(load_level_cumulative)

        if self.final_inc_phi_pms_prev is not None:
            final_inc_phi_pms = self.final_inc_phi_pms_prev + phi_pms_history[-1]
            result = {
                "phi_pms_history": phi_pms_history,
                "load_level_history": load_level_history,
                "final_inc_phi_pms": final_inc_phi_pms,
            }
        else:
            result = {
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
            cbar[i] = self.costs[i] - np.dot(pi_transpose, self.table[:, i])
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
        # print(f"{will_out_row=}")
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
        x = np.matrix(np.zeros((self.constraints_count, 1)))
        for i, basic_variable in enumerate(basic_variables):
            if basic_variable < self.primary_vars_count:
                x[basic_variables[i], 0] = bbar[i]
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

    def calc_violation_scores(self, intact_phi_pms, load_level):
        scores = self.intact_phi.T * self.pv * intact_phi_pms + self.intact_phi.T * self.p0 * load_level - np.matrix(np.ones((self.intact_pieces_count, 1)))
        return scores

    def get_unsifted_pms(self, x, structure_sifted_yield_pieces):
        intact_pms = np.matrix(np.zeros((self.intact_phi.shape[1], 1)))
        for piece in structure_sifted_yield_pieces:
            intact_pms[piece.num_in_structure, 0] = x[piece.sifted_num_in_structure, 0]
        intact_phi_pms = self.intact_phi * intact_pms
        return intact_phi_pms

    def get_plastic_vars_in_basic_variables(self, basic_variables, landa_var, structure_sifted_yield_pieces):
        plastic_vars = []
        for basic_variable in basic_variables:
            if basic_variable < landa_var:
                plastic_vars.append(structure_sifted_yield_pieces[basic_variable].num_in_structure)
        return plastic_vars
