import numpy as np
from dataclasses import dataclass

from ..models.yield_models import SiftedYieldPiece, SiftedYieldPoint


class FPM():
    var: int
    cost: float


class SlackCandidate():
    def __init__(self, var, cost):
        self.var = var
        self.cost = cost


@dataclass
class SiftedResults:
    sifted_yield_points: list
    structure_sifted_yield_pieces: list
    sifted_components_count: int
    sifted_pieces_count: int


class Sifting:
    def __init__(self, intact_points, scores):
        self.intact_points = intact_points
        self.scores = scores
        self.sifted_results = self.get_sifted_results()
        self.sifted_yield_points = self.sifted_results.sifted_yield_points
        self.structure_sifted_yield_pieces = self.sifted_results.structure_sifted_yield_pieces
        self.sifted_components_count = self.sifted_results.sifted_components_count
        self.sifted_pieces_count = self.sifted_results.sifted_pieces_count
        self.sifted_points_count = len(self.sifted_yield_points)
        self.sifted_phi = self.get_sifted_phi()

    def get_sifted_results(self):
        piece_num_in_structure = 0
        sifted_yield_points = []
        structure_sifted_yield_pieces = []
        sifted_components_count = 0
        sifted_pieces_count = 0
        for point in self.intact_points:
            for piece in point.pieces:
                piece.score = self.scores[piece.num_in_structure]
            point.pieces.sort(key=lambda x: x.score, reverse=True)
            point_sifted_yield_pieces = []
            sifted_yield_pieces_nums_in_intact_yield_point = []
            for piece in point.pieces[0:point.min_sifted_pieces_count]:
                sifted_yield_piece = SiftedYieldPiece(
                    ref_yield_point_num=point.num_in_structure,
                    sifted_num_in_yield_point=piece.num_in_yield_point,
                    sifted_num_in_structure=piece_num_in_structure,
                    intact_num_in_structure=piece.num_in_structure,
                )
                point_sifted_yield_pieces.append(sifted_yield_piece)
                structure_sifted_yield_pieces.append(sifted_yield_piece)
                sifted_yield_pieces_nums_in_intact_yield_point.append(piece.num_in_yield_point)
                piece_num_in_structure += 1
            sifted_yield_points.append(
                SiftedYieldPoint(
                    ref_member_num=point.ref_member_num,
                    num_in_member=point.num_in_member,
                    num_in_structure=point.num_in_structure,
                    components_count=point.components_count,
                    pieces=point_sifted_yield_pieces,
                    pieces_count=len(point_sifted_yield_pieces),
                    phi=point.phi[:, sifted_yield_pieces_nums_in_intact_yield_point],
                    q=point.q[:, sifted_yield_pieces_nums_in_intact_yield_point],
                    h=point.h[sifted_yield_pieces_nums_in_intact_yield_point, :],
                    w=point.w,
                    cs=point.cs,
                )
            )
            sifted_components_count += point.components_count
            sifted_pieces_count += len(point_sifted_yield_pieces)
        return SiftedResults(
            sifted_yield_points=sifted_yield_points,
            structure_sifted_yield_pieces=structure_sifted_yield_pieces,
            sifted_components_count=sifted_components_count,
            sifted_pieces_count=sifted_pieces_count,
        )

    def get_sifted_phi(self):
        sifted_phi = np.matrix(np.zeros((self.sifted_components_count, self.sifted_pieces_count)))
        current_row_start = 0
        current_column_start = 0
        for yield_point in self.sifted_yield_points:
            current_row_end = current_row_start + yield_point.components_count
            current_column_end = current_column_start + yield_point.pieces_count
            sifted_phi[current_row_start:current_row_end, current_column_start:current_column_end] = yield_point.phi
            current_row_start = current_row_end
            current_column_start = current_column_end
        return sifted_phi
