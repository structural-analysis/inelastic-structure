import numpy as np
from dataclasses import dataclass

from ..models.yield_models import SiftedYieldPiece, SiftedYieldPoint
from ..settings import settings


class FPM():
    var: int
    cost: float


class SlackCandidate():
    def __init__(self, var, cost):
        self.var = var
        self.cost = cost

    def __repr__(self):
        return f"SlackCandidate(var={self.var!r}, cost={self.cost!r})"


@dataclass
class SiftedResults:
    sifted_yield_points: list
    structure_sifted_yield_pieces: list
    sifted_components_count: int
    sifted_pieces_count: int


class Sifting:
    def __init__(self, intact_points, intact_phi, scores):
        self.intact_points = intact_points
        self.intact_phi = intact_phi
        self.scores = scores
        self.sifted_results = self.get_sifted_results()
        self.sifted_yield_points = self.sifted_results.sifted_yield_points
        self.structure_sifted_yield_pieces = self.sifted_results.structure_sifted_yield_pieces
        self.sifted_components_count = self.sifted_results.sifted_components_count
        self.sifted_pieces_count = self.sifted_results.sifted_pieces_count
        self.sifted_points_count = len(self.sifted_yield_points)
        # TODO: calculate all phi, q, ... in one loop of yield points.
        self.sifted_phi = self.get_sifted_phi()
        self.sifted_q = self.get_sifted_q()
        self.sifted_h = self.get_sifted_h()
        self.sifted_w = self.get_sifted_w()
        self.sifted_cs = self.get_sifted_cs()

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

    def get_sifted_q(self):
        sifted_q = np.matrix(np.zeros((2 * self.sifted_points_count, self.sifted_pieces_count)))
        pieces_counter = 0
        for i, yield_point in enumerate(self.sifted_yield_points):
            sifted_q[2 * i:2 * i + 2, pieces_counter:pieces_counter + yield_point.pieces_count] = yield_point.q
            pieces_counter += yield_point.pieces_count
        return sifted_q

    def get_sifted_h(self):
        sifted_h = np.matrix(np.zeros((self.sifted_pieces_count, 2 * self.sifted_points_count)))
        pieces_counter = 0
        for i, yield_point in enumerate(self.sifted_yield_points):
            sifted_h[pieces_counter:pieces_counter + yield_point.pieces_count, 2 * i:2 * i + 2] = yield_point.h
            pieces_counter += yield_point.pieces_count
        return sifted_h

    def get_sifted_w(self):
        sifted_w = np.matrix(np.zeros((2 * self.sifted_points_count, 2 * self.sifted_points_count)))
        for i, yield_point in enumerate(self.sifted_yield_points):
            sifted_w[2 * i:2 * i + 2, 2 * i:2 * i + 2] = yield_point.w
        return sifted_w

    def get_sifted_cs(self):
        sifted_cs = np.matrix(np.zeros((2 * self.sifted_points_count, 1)))
        for i, yield_point in enumerate(self.sifted_yield_points):
            sifted_cs[2 * i:2 * i + 2, 0] = yield_point.cs
        return sifted_cs

    def check_violation(self, scores):
        violated_pieces = np.array(np.where(scores > settings.computational_zero)[0], dtype=int).flatten().tolist()
        return violated_pieces

    def get_unsifted_pms(self, x):
        intact_pms = np.matrix(np.zeros((self.intact_phi.shape[1], 1)))
        for piece in self.structure_sifted_yield_pieces:
            intact_pms[piece.intact_num_in_structure, 0] = x[piece.sifted_num_in_structure, 0]
        intact_phi_pms = self.intact_phi * intact_pms
        return intact_phi_pms
