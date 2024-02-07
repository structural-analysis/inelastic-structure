import copy
import numpy as np
from dataclasses import dataclass, field
from .functions import print_specific_properties

from ..models.yield_models import (
    SiftedYieldPiece,
    SiftedYieldPoint,
    ViolatedYieldPiece,
    ViolatedYieldPoint,
)
from ..settings import settings


class FPM():
    def __init__(self, var, cost):
        self.var: int = var
        self.cost: float = cost


class SlackCandidate():
    def __init__(self, var, cost):
        self.var = var
        self.cost = cost

    def __repr__(self):
        return f"SlackCandidate(var={self.var!r}, cost={self.cost!r})"


@dataclass
class SiftedResults:
    sifted_yield_points: list
    sifted_components_count: int
    sifted_pieces_count: int
    structure_sifted_yield_pieces: list
    structure_sifted_phi: np.matrix
    modified_structure_sifted_yield_pieces_indices: list = field(default_factory=list)
    bbar_updated: np.array = np.zeros((1, 1))
    b_matrix_inv_updated: np.array = np.zeros((1, 1))

    # def __copy__(self):
    #     # Create a new instance of the class
    #     new_obj = SiftedResults(
    #         copy.copy(self.sifted_yield_points),
    #         copy.copy(self.structure_sifted_yield_pieces),
    #         copy.copy(self.structure_sifted_phi),
    #         copy.copy(self.modified_structure_sifted_yield_pieces_indices),
    #         copy.copy(self.bbar_updated),
    #         copy.copy(self.b_matrix_inv_updated)
    #     )
    #     return new_obj

    # def __deepcopy__(self, memo):
    #     # Create a new instance of the class
    #     new_obj = SiftedResults(
    #         copy.deepcopy(self.sifted_yield_points, memo),
    #         copy.deepcopy(self.structure_sifted_yield_pieces, memo),
    #         copy.deepcopy(self.structure_sifted_phi, memo),
    #         copy.deepcopy(self.modified_structure_sifted_yield_pieces_indices, memo),
    #         copy.deepcopy(self.bbar_updated, memo),
    #         copy.deepcopy(self.b_matrix_inv_updated, memo)
    #     )
    #     return new_obj


class Sifting:
    # NOTE: SIFTING+: len(sifted_yield_points) != len(intact_yield_points)
    # so in advanced sifting we cannot loop through sifted yield points.
    # better to use unique id's for piece and points in sifted+
    def __init__(self, intact_points, intact_pieces, intact_phi):
        self.intact_points = intact_points
        self.intact_pieces = intact_pieces
        self.intact_phi = intact_phi

        # TODO: in create and update: calculate all q,h, ... in one loop of yield points.
        # self.sifted_q = self.get_sifted_q()
        # self.sifted_h = self.get_sifted_h()
        # self.sifted_w = self.get_sifted_w()
        # self.sifted_cs = self.get_sifted_cs()

    def create(self, scores):
        piece_num_in_structure = 0
        sifted_yield_points = []
        structure_sifted_yield_pieces = []
        sifted_components_count = 0
        sifted_pieces_count = 0
        for point in self.intact_points:
            point_selected_pieces = self.get_point_selected_pieces(point=point, scores=scores)
            point_sifted_yield_pieces = []
            point_sifted_yield_pieces_nums_in_intact_yield_point = []
            for piece in point_selected_pieces:
                sifted_yield_piece = SiftedYieldPiece(
                    ref_yield_point_num=point.num_in_structure,
                    num_in_yield_point=piece.num_in_yield_point,
                    sifted_num_in_structure=piece_num_in_structure,
                    num_in_structure=piece.num_in_structure,
                )
                point_sifted_yield_pieces.append(sifted_yield_piece)
                structure_sifted_yield_pieces.append(sifted_yield_piece)
                point_sifted_yield_pieces_nums_in_intact_yield_point.append(piece.num_in_yield_point)
                piece_num_in_structure += 1

            sifted_yield_points.append(
                SiftedYieldPoint(
                    ref_member_num=point.ref_member_num,
                    num_in_member=point.num_in_member,
                    num_in_structure=point.num_in_structure,
                    components_count=point.components_count,
                    pieces=point_sifted_yield_pieces,
                    pieces_count=len(point_sifted_yield_pieces),
                    sifted_yield_pieces_nums_in_intact_yield_point=point_sifted_yield_pieces_nums_in_intact_yield_point,
                    phi=point.phi[:, point_sifted_yield_pieces_nums_in_intact_yield_point],
                    q=point.q[:, point_sifted_yield_pieces_nums_in_intact_yield_point],
                    h=point.h[point_sifted_yield_pieces_nums_in_intact_yield_point, :],
                    w=point.w,
                    cs=point.cs,
                )
            )
            sifted_components_count += point.components_count
            sifted_pieces_count += len(point_sifted_yield_pieces)

        structure_sifted_phi = self.get_structure_sifted_phi(
            sifted_yield_points=sifted_yield_points,
            sifted_components_count=sifted_components_count,
            sifted_pieces_count=sifted_pieces_count
        )

        return SiftedResults(
            sifted_yield_points=sifted_yield_points,
            sifted_components_count=sifted_components_count,
            sifted_pieces_count=sifted_pieces_count,
            structure_sifted_yield_pieces=structure_sifted_yield_pieces,
            structure_sifted_phi=structure_sifted_phi,
        )

    def update(
            self,
            increment,
            plastic_vars_in_basic_variables_prev,
            scores,
            sifted_results_old,
            violated_pieces,
            bbar_prev,
            b_matrix_inv_prev,
            basic_variables_prev,
            landa_row,
            landa_var,
            pv,
            p0,
            will_in_col_piece_num_in_structure):

        violated_points = self.get_violated_points(violated_pieces)
        sifted_yield_points_updated = sifted_results_old.sifted_yield_points
        structure_sifted_yield_pieces_updated = sifted_results_old.structure_sifted_yield_pieces
        sifted_components_count = 0
        sifted_pieces_count = 0
        cumulative_point_pieces_count = 0
        modified_structure_sifted_yield_pieces_indices = []
        bbar_updated = bbar_prev
        # NOTE: SIFTING+:
        # we can not use intact_points anymore
        # insead of using point_num we should use num_in_structure somehow

        for point_num, point in enumerate(self.intact_points):
            point_violated_pieces = self.get_point_violated_pieces(
                point=point,
                violated_points=violated_points,
            )
            point_selected_pieces = self.get_point_selected_pieces(
                point=point,
                scores=scores,
                # will_in_col_piece_num_in_structure=will_in_col_piece_num_in_structure,
                # plastic_vars_in_basic_variables=plastic_vars_in_basic_variables_prev
            )
            point_pieces_current = self.get_point_final_pieces(
                selected_pieces=point_selected_pieces,
                violated_pieces=point_violated_pieces,
                will_in_col_piece_num_in_structure=will_in_col_piece_num_in_structure,
                plastic_vars_in_basic_variables=plastic_vars_in_basic_variables_prev
            )

            # if point_num == 1 or point_num == 5:
            #     print(f"point_violated_pieces for {point_num=}:")
            #     print_specific_properties(point_violated_pieces, ["num_in_structure"])
            #     print(f"point_selected_pieces for {point_num=}:")
            #     print_specific_properties(point_selected_pieces, ["num_in_structure", "score"])
            #     print(f"point_pieces_current for {point_num=}:")
            #     print_specific_properties(point_pieces_current, ["num_in_structure"])

            # final_num_in_structures = [piece.num_in_structure for piece in point_pieces_current]
            # print(f"{point_num=}")
            # print(f"{final_num_in_structures=}")
            point_updated = sifted_yield_points_updated[point_num]
            point_pieces_updated = point_updated.pieces
            point_sifted_yield_pieces_nums_in_intact_yield_point_updated = point_updated.sifted_yield_pieces_nums_in_intact_yield_point
            changed_indices_prev, changed_indices_current = self.get_point_changed_pieces(
                sifted_pieces_prev=point_pieces_updated,
                sifted_pieces_current=point_pieces_current,
            )
            # if point_num == 6:
            #     print(f"{point_pieces_updated[0].num_in_structure=}")
            #     print(f"{point_pieces_updated[1].num_in_structure=}")
            #     print(f"{point_pieces_updated[2].num_in_structure=}")
            #     print(f"{point_pieces_updated[3].num_in_structure=}")
            #     print("-----")
            #     print(f"{point_pieces_current[0].num_in_structure=}")
            #     print(f"{point_pieces_current[1].num_in_structure=}")
            #     print(f"{point_pieces_current[2].num_in_structure=}")
            #     print(f"{point_pieces_current[3].num_in_structure=}")
            #     print(f"{changed_indices_prev=}")
            #     print(f"{changed_indices_current=}")

            #     print("point selected pieces: ")
            #     print_specific_properties(
            #         obj_list=point_selected_pieces,
            #         properties=[
            #             "num_in_structure",
            #             "ref_yield_point_num",
            #             "num_in_yield_point",
            #         ],
            #     )
            #     print("point violated pieces: ")
            #     print_specific_properties(
            #         obj_list=point_violated_pieces,
            #         properties=[
            #             "num_in_structure",
            #             "ref_yield_point_num",
            #             "num_in_yield_point",
            #         ],
            #     )
            #     print("point final pieces: ")
            #     print_specific_properties(
            #         obj_list=point_pieces_current,
            #         properties=[
            #             "num_in_structure",
            #             "ref_yield_point_num",
            #             "num_in_yield_point",
            #         ],
            #     )
            # print(f"{point_pieces_current=}")
            # print("============")

            for i, prev_changed_index in enumerate(changed_indices_prev):
                current_piece_to_change = point_pieces_current[changed_indices_current[i]]
                sifted_num_in_structure = point_pieces_updated[prev_changed_index].sifted_num_in_structure
                sifted_yield_piece = SiftedYieldPiece(
                    ref_yield_point_num=point.num_in_structure,
                    num_in_yield_point=current_piece_to_change.num_in_yield_point,
                    sifted_num_in_structure=sifted_num_in_structure,
                    num_in_structure=current_piece_to_change.num_in_structure,
                )
                point_pieces_updated[prev_changed_index] = sifted_yield_piece
                point_sifted_yield_pieces_nums_in_intact_yield_point_updated[prev_changed_index] = current_piece_to_change.num_in_yield_point

                structure_sifted_yield_pieces_updated[sifted_num_in_structure] = sifted_yield_piece
                modified_structure_sifted_yield_pieces_indices.append(
                    prev_changed_index + cumulative_point_pieces_count
                )
                bbar_updated[prev_changed_index + cumulative_point_pieces_count] = -scores[
                    current_piece_to_change.num_in_structure
                ]
            # in_structure_indices = [num.num_in_structure for num in structure_sifted_yield_pieces_updated]
            # print(f"##################{in_structure_indices}")
            # input()
            cumulative_point_pieces_count += point.min_sifted_pieces_count
            point_phi_updated = point.phi[:, point_sifted_yield_pieces_nums_in_intact_yield_point_updated]
            point_q_updated = point.q[:, point_sifted_yield_pieces_nums_in_intact_yield_point_updated]
            point_h_updated = point.h[point_sifted_yield_pieces_nums_in_intact_yield_point_updated, :]
            point_w_updated = point.w
            point_cs_updated = point.cs

            sifted_yield_points_updated[point_num] = SiftedYieldPoint(
                ref_member_num=point.ref_member_num,
                num_in_member=point.num_in_member,
                num_in_structure=point.num_in_structure,
                components_count=point.components_count,
                pieces=point_pieces_updated,
                pieces_count=len(point_pieces_updated),
                sifted_yield_pieces_nums_in_intact_yield_point=point_sifted_yield_pieces_nums_in_intact_yield_point_updated,
                phi=point_phi_updated,
                q=point_q_updated,
                h=point_h_updated,
                w=point_w_updated,
                cs=point_cs_updated,
            )
            sifted_components_count += point.components_count
            sifted_pieces_count += len(point_pieces_updated)
        structure_sifted_phi = self.get_structure_sifted_phi(
            sifted_yield_points=sifted_yield_points_updated,
            sifted_components_count=sifted_components_count,
            sifted_pieces_count=sifted_pieces_count
        )
        if increment == 22:  # use when sifted
            pass
            # print(f"{will_in_col=}")
            # print(f"global_will_in_col=x-{self.structure_sifted_yield_pieces_current[will_in_col].num_in_structure}")
            # print(f"{updated_indices=}")
            # !!!!!!!!!!!! these values must check with increment 21 vaules,
            # beacause values are updated based on increment 21 values !!!!!!!!!!!!!!!!!!!!

            # np.savetxt("table.csv", self.table, delimiter=", ", fmt='%1.4e')
            # np.savetxt("phi.csv", self.phi, delimiter=", ", fmt='%1.4e')
            # np.savetxt("b_matrix_inv.csv", b_matrix_inv, delimiter=", ", fmt='%1.4e')
            # scores,
            # b_matrix_inv_prev,
            # basic_variables_prev,
            # landa_row,
            # landa_var,
            # pv,
            # p0,
            # will_in_col_piece_num_in_structure
            # input()

        return SiftedResults(
            sifted_yield_points=sifted_yield_points_updated,
            sifted_components_count=sifted_components_count,
            sifted_pieces_count=sifted_pieces_count,
            structure_sifted_yield_pieces=structure_sifted_yield_pieces_updated,
            structure_sifted_phi=structure_sifted_phi,
            modified_structure_sifted_yield_pieces_indices=modified_structure_sifted_yield_pieces_indices,
            bbar_updated=bbar_updated,
            b_matrix_inv_updated=self.get_b_matrix_inv_updated(
                b_matrix_inv_prev=b_matrix_inv_prev,
                modified_structure_sifted_yield_pieces_indices=modified_structure_sifted_yield_pieces_indices,
                basic_variables_prev=basic_variables_prev,
                landa_row=landa_row,
                landa_var=landa_var,
                phi=structure_sifted_phi,
                pv=pv,
                p0=p0,
            ),
        )

    def get_violated_points(self, violated_pieces):
        violated_points_dict = {}
        for piece in violated_pieces:
            if piece.ref_yield_point_num not in violated_points_dict:
                violated_points_dict[piece.ref_yield_point_num] = []
            violated_points_dict[piece.ref_yield_point_num].append(piece)
        violated_points = [
            ViolatedYieldPoint(num_in_structure=num, violated_pieces=pieces)
            for num, pieces in violated_points_dict.items()
        ]
        return violated_points

    def get_structure_sifted_phi(self, sifted_yield_points, sifted_components_count, sifted_pieces_count):
        structure_sifted_phi = np.matrix(np.zeros((sifted_components_count, sifted_pieces_count)))
        current_row_start = 0
        current_column_start = 0
        for yield_point in sifted_yield_points:
            current_row_end = current_row_start + yield_point.components_count
            current_column_end = current_column_start + yield_point.pieces_count
            structure_sifted_phi[current_row_start:current_row_end, current_column_start:current_column_end] = yield_point.phi
            current_row_start = current_row_end
            current_column_start = current_column_end
        return structure_sifted_phi

    # def get_sifted_q(self):
    #     sifted_q = np.matrix(np.zeros((2 * self.sifted_points_count, self.sifted_pieces_count)))
    #     pieces_counter = 0
    #     for i, yield_point in enumerate(self.sifted_yield_points):
    #         sifted_q[2 * i:2 * i + 2, pieces_counter:pieces_counter + yield_point.pieces_count] = yield_point.q
    #         pieces_counter += yield_point.pieces_count
    #     return sifted_q

    # def get_sifted_h(self):
    #     sifted_h = np.matrix(np.zeros((self.sifted_pieces_count, 2 * self.sifted_points_count)))
    #     pieces_counter = 0
    #     for i, yield_point in enumerate(self.sifted_yield_points):
    #         sifted_h[pieces_counter:pieces_counter + yield_point.pieces_count, 2 * i:2 * i + 2] = yield_point.h
    #         pieces_counter += yield_point.pieces_count
    #     return sifted_h

    # def get_sifted_w(self):
    #     sifted_w = np.matrix(np.zeros((2 * self.sifted_points_count, 2 * self.sifted_points_count)))
    #     for i, yield_point in enumerate(self.sifted_yield_points):
    #         sifted_w[2 * i:2 * i + 2, 2 * i:2 * i + 2] = yield_point.w
    #     return sifted_w

    # def get_sifted_cs(self):
    #     sifted_cs = np.matrix(np.zeros((2 * self.sifted_points_count, 1)))
    #     for i, yield_point in enumerate(self.sifted_yield_points):
    #         sifted_cs[2 * i:2 * i + 2, 0] = yield_point.cs
    #     return sifted_cs

    def check_violation(self, scores, structure_sifted_yield_pieces_old):
        for piece in structure_sifted_yield_pieces_old:
            scores[piece.num_in_structure] = 0

        violated_pieces = []
        violated_piece_nums = np.array(np.where(scores > settings.computational_zero)[0], dtype=int).flatten().tolist()

        for violated_piece_num in violated_piece_nums:
            violated_pieces.append(
                ViolatedYieldPiece(
                    ref_yield_point_num=self.intact_pieces[violated_piece_num].ref_yield_point_num,
                    num_in_structure=violated_piece_num,
                    num_in_yield_point=self.intact_pieces[violated_piece_num].num_in_yield_point,
                    score=scores[violated_piece_num],
                )
            )
        print("//// score of violated pieces: ")
        print_specific_properties(violated_pieces, ["num_in_structure", "score"])
        return violated_pieces

    def get_point_changed_pieces(self, sifted_pieces_prev, sifted_pieces_current):
        changed_indices_prev = [
            changed_index for changed_index, sifted_piece_prev in enumerate(sifted_pieces_prev)
            if sifted_piece_prev not in sifted_pieces_current
        ]
        changed_indices_current = [
            changed_index for changed_index, sifted_piece_current in enumerate(sifted_pieces_current)
            if sifted_piece_current not in sifted_pieces_prev
        ]
        return changed_indices_prev, changed_indices_current

    def get_point_violated_pieces(self, point, violated_points):
        point_violated_pieces = []
        for violated_point in violated_points:
            if violated_point.num_in_structure == point.num_in_structure:
                point_violated_pieces = violated_point.violated_pieces
        point_violated_pieces.sort(key=lambda x: x.score, reverse=True)
        # print(f"{point_violated_pieces=}")
        return point_violated_pieces

    def get_point_selected_pieces(self, point, scores):
        for piece in point.pieces:
            piece.score = scores[piece.num_in_structure]

        point.pieces.sort(key=lambda x: x.score, reverse=True)
        point_selected_pieces = point.pieces[0:point.min_sifted_pieces_count]
        # for piece in point_selected_pieces:
        #     print(f"{piece.num_in_structure=}")

        return point_selected_pieces

    def get_point_final_pieces(self, selected_pieces, violated_pieces, will_in_col_piece_num_in_structure, plastic_vars_in_basic_variables):
        unchanged_vars = [will_in_col_piece_num_in_structure] + plastic_vars_in_basic_variables
        final_pieces = selected_pieces[:]
        assign_indices = [index for index, piece in enumerate(selected_pieces) if piece.num_in_structure not in unchanged_vars]
        assign_indices.sort(reverse=True)
        if violated_pieces:
            if not violated_pieces[0] in final_pieces:
                final_pieces[assign_indices[0]] = violated_pieces[0]
        # for i, assign_index in enumerate(assign_indices):
        #     if i < len(violated_pieces):
        #         final_pieces[assign_index] = violated_pieces[i]
        #     else:
        #         break

        # print(f"{unchanged_vars=}")
        # for violated_piece in violated_pieces[:4]:
        #     print(f"{violated_piece=}")
        #     print(f"{counter=}")
        #     if violated_piece not in final_pieces and counter <= 4:
        #         print(f"{final_pieces[-counter]=}")
        #         if final_pieces[-counter] not in unchanged_vars:
        #             final_pieces[-counter] = violated_piece
        #             counter += 1
        #         elif final_pieces[-counter] in unchanged_vars and counter < 4:
        #             final_pieces[-counter - 1] = violated_piece
        #             counter += 2
        #     print(f"{final_pieces=}")
        #     print("---------")
        return final_pieces

    # def get_point_final_pieces(self, point, selected_pieces, violated_pieces, will_in_col_piece_num_in_structure, plastic_vars_in_basic_variables):
    #     print(f"/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*")
    #     print(f"{will_in_col_piece_num_in_structure=}")
    #     print(f"{selected_pieces=}")
    #     print(f"{violated_pieces=}")
    #     print(f"{plastic_vars_in_basic_variables=}")
    #     unchanged_vars = [will_in_col_piece_num_in_structure] + plastic_vars_in_basic_variables
    #     final_pieces = selected_pieces[:]
    #     counter = 1
    #     for violated_piece in violated_pieces[:point.min_sifted_pieces_count]:
    #         if violated_piece not in final_pieces and counter <= point.min_sifted_pieces_count:
    #             if final_pieces[-counter].num_in_structure not in unchanged_vars:
    #                 final_pieces[-counter] = violated_piece
    #                 counter += 1
    #             elif final_pieces[-counter].num_in_structure in unchanged_vars and counter < point.min_sifted_pieces_count:
    #                 final_pieces[-counter - 1] = violated_piece
    #                 counter += 2
    #     return final_pieces

    def get_b_matrix_inv_updated(
            self,
            b_matrix_inv_prev,
            modified_structure_sifted_yield_pieces_indices,
            basic_variables_prev,
            landa_var,
            landa_row,
            phi,
            pv,
            p0,):

        # NOTE:
        # j: indices of previous phi matrix columns which contents are updated
        # m: active pms for phi columns
        # v: original unsorted active pm rows used for b_inv columns including landa row
        # u: sorted active pm rows and landa row as last member

        active_pms, active_pms_rows = self.get_active_pms_stats(
            basic_variables_prev=basic_variables_prev,
            landa_var=landa_var,
        )
        j = modified_structure_sifted_yield_pieces_indices
        m = active_pms
        m.remove(landa_var)
        v = active_pms_rows
        u = active_pms_rows[:]
        u.remove(landa_row)
        u.sort()
        u.append(landa_row)

        a_sensitivity_part = phi.T[j, :] * pv * phi[:, m]
        a_elastic_part = phi.T[j, :] * p0
        a_updated = np.concatenate((a_sensitivity_part, a_elastic_part), axis=1)
        b_matrix_inv_prev[np.ix_(j, v)] = -a_updated * b_matrix_inv_prev[np.ix_(u, v)]
        return b_matrix_inv_prev

    def get_active_pms_stats(self, basic_variables_prev, landa_var):
        active_pms = []
        active_pms_rows = []
        for index, basic_variable in enumerate(basic_variables_prev):
            if basic_variable <= landa_var:
                active_pms.append(basic_variable)
                active_pms_rows.append(index)
        return active_pms, active_pms_rows
