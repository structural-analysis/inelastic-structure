import numpy as np
from dataclasses import dataclass, field

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

    def update(self, scores, sifted_results_prev, violated_pieces, bbar_prev, cb_prev):
        violated_points = self.get_violated_points(violated_pieces)
        sifted_yield_points_updated = sifted_results_prev.sifted_yield_points
        structure_sifted_yield_pieces_updated = sifted_results_prev.structure_sifted_yield_pieces
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
            point_selected_pieces = self.get_point_selected_pieces(point=point, scores=scores)
            point_pieces_current = self.get_point_final_pieces(
                point=point,
                selected_pieces=point_selected_pieces,
                violated_pieces=point_violated_pieces,
            )

            point_updated = sifted_yield_points_updated[point_num]
            point_pieces_updated = point_updated.pieces
            point_sifted_yield_pieces_nums_in_intact_yield_point_updated = point_updated.sifted_yield_pieces_nums_in_intact_yield_point
            changed_indices_prev, changed_indices_current = self.get_point_changed_pieces(
                sifted_pieces_prev=point_pieces_updated,
                sifted_pieces_current=point_pieces_current,
            )

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

        return SiftedResults(
            sifted_yield_points=sifted_yield_points_updated,
            sifted_components_count=sifted_components_count,
            sifted_pieces_count=sifted_pieces_count,
            structure_sifted_yield_pieces=structure_sifted_yield_pieces_updated,
            structure_sifted_phi=structure_sifted_phi,
            modified_structure_sifted_yield_pieces_indices=modified_structure_sifted_yield_pieces_indices,
            bbar_updated=bbar_updated,
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

    def check_violation(self, scores, structure_sifted_yield_pieces_prev):
        for piece in structure_sifted_yield_pieces_prev:
            scores[piece.num_in_structure] = 0

        violated_pieces = []
        violated_piece_nums = np.array(np.where(scores > settings.computational_zero)[0], dtype=int).flatten().tolist()
        for violated_piece_num in violated_piece_nums:
            violated_pieces.append(
                ViolatedYieldPiece(
                    ref_yield_point_num=self.intact_pieces[violated_piece_num].ref_yield_point_num,
                    num_in_structure=violated_piece_num,
                    num_in_yield_point=self.intact_pieces[violated_piece_num].num_in_yield_point
                )
            )
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
        return point_violated_pieces

    def get_point_selected_pieces(self, point, scores):
        for piece in point.pieces:
            piece.score = scores[piece.num_in_structure]

        point.pieces.sort(key=lambda x: x.score, reverse=True)
        point_selected_pieces = point.pieces[0:point.min_sifted_pieces_count]
        return point_selected_pieces

    def get_point_final_pieces(self, point, selected_pieces, violated_pieces):
        final_pieces = selected_pieces[:]
        counter = 1
        for violated_piece in violated_pieces[:point.min_sifted_pieces_count]:
            if violated_piece not in final_pieces:
                final_pieces[-counter] = violated_piece
                counter += 1
        return final_pieces
