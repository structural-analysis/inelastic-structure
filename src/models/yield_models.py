from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional, Tuple


@dataclass
class SofteningVar:
    ref_yield_point_num: int
    num_in_yield_point: int
    num_in_structure: int


@dataclass
class YieldPiece:
    ref_yield_point_num: int
    num_in_yield_point: int
    num_in_structure: int
    score: float

    def __eq__(self, other):
        return self.ref_yield_point_num == other.ref_yield_point_num and self.num_in_structure == other.num_in_structure

    def __hash__(self):
        return hash((
            'ref_yield_point_num', self.ref_yield_point_num,
            'num_in_structure', self.num_in_structure
        ))


@dataclass
class YieldPoint:
    min_sifted_pieces_count: int
    ref_member_num: int
    num_in_member: int
    num_in_structure: int
    components_count: int
    pieces: List[YieldPiece]
    pieces_count: int
    phi: np.matrix
    q: np.matrix
    h: np.matrix
    w: np.matrix
    cs: np.matrix
    softening_vars: Optional[Tuple[SofteningVar, SofteningVar]] = None


@dataclass
class SiftedYieldPiece:
    ref_yield_point_num: int
    num_in_yield_point: int
    sifted_num_in_structure: int
    num_in_structure: int

    def __eq__(self, other):
        return self.ref_yield_point_num == other.ref_yield_point_num and self.num_in_structure == other.num_in_structure

    def __hash__(self):
        return hash((
            'ref_yield_point_num', self.ref_yield_point_num,
            'num_in_structure', self.num_in_structure
        ))


@dataclass
class SiftedYieldPoint:
    ref_member_num: int
    num_in_member: int
    num_in_structure: int
    components_count: int
    pieces: List[SiftedYieldPiece]
    pieces_count: int
    sifted_yield_pieces_nums_in_intact_yield_point: list
    phi: np.matrix
    q: np.matrix
    h: np.matrix
    w: np.matrix
    cs: np.matrix
    softening_vars: Optional[Tuple[SofteningVar, SofteningVar]] = None


@dataclass
class ViolatedYieldPiece:
    ref_yield_point_num: int
    num_in_yield_point: int
    num_in_structure: int
    score: float


@dataclass
class ViolatedYieldPoint:
    num_in_structure: int
    violated_pieces: List[ViolatedYieldPiece] = field(default_factory=list)
    softening_vars: Optional[Tuple[SofteningVar, SofteningVar]] = None


@dataclass
class IntactYieldPointsResults:
    intact_points: list
    intact_pieces: list
    intact_components_count: int


class MemberYieldSpecs:
    def __init__(self, section: object, points_count: int, include_softening: bool):
        self.section = section
        self.include_softening = include_softening
        self.points_count = points_count
        self.components_count = self.points_count * self.section.yield_specs.components_count
        self.yield_points = self.get_yield_points()

    def get_yield_points(self):
        yield_points = []
        if self.include_softening:
            softening_vars = (
                SofteningVar(
                    ref_yield_point_num=-1,
                    num_in_yield_point=0,
                    num_in_structure=-1,
                ),
                SofteningVar(
                    ref_yield_point_num=-1,
                    num_in_yield_point=1,
                    num_in_structure=-1,
                )
            )
        else:
            softening_vars = None

        for point_num in range(self.points_count):
            yield_pieces = []
            for piece_num in range(self.section.yield_specs.pieces_count):
                yield_pieces.append(
                    YieldPiece(
                        ref_yield_point_num=-1,
                        num_in_yield_point=piece_num,
                        num_in_structure=-1,
                        score=-1
                    )
                )

            yield_points.append(
                YieldPoint(
                    min_sifted_pieces_count=self.section.yield_specs.sifted_pieces_count,
                    ref_member_num=-1,
                    num_in_member=point_num,
                    num_in_structure=-1,
                    components_count=self.section.yield_specs.components_count,
                    pieces=yield_pieces,
                    pieces_count=self.section.yield_specs.pieces_count,
                    phi=self.section.yield_specs.phi,
                    q=self.section.softening.q,
                    h=self.section.softening.h,
                    w=self.section.softening.w,
                    cs=self.section.softening.cs,
                    softening_vars=softening_vars,
                )
            )
        return yield_points


class StructureYieldSpecs:
    def __init__(self, members, include_softening: bool):
        self.members = members
        self.include_softening = include_softening
        self.intact_yield_points_results = self.get_intact_yield_points_results()
        self.intact_points: list = self.intact_yield_points_results.intact_points
        self.intact_pieces: list = self.intact_yield_points_results.intact_pieces
        self.intact_components_count = self.intact_yield_points_results.intact_components_count
        self.intact_points_count = len(self.intact_points)
        self.intact_pieces_count = len(self.intact_pieces)
        self.intact_phi = self.create_intact_phi()
        self.intact_q = self.create_intact_q()
        self.intact_h = self.create_intact_h()
        self.intact_w = self.create_intact_w()
        self.intact_cs = self.create_intact_cs()

    def get_intact_yield_points_results(self):
        intact_points = []
        intact_pieces = []
        intact_components_count = 0
        point_num = 0
        piece_num = 0
        softening_var_num = 0
        process_softening = self.include_softening
        for member_num, member in enumerate(self.members):
            for point in member.yield_specs.yield_points:
                point.ref_member_num = member_num
                point.num_in_structure = point_num
                intact_points.append(point)
                for piece in point.pieces:
                    piece.ref_yield_point_num = point.num_in_structure
                    piece.num_in_structure = piece_num
                    intact_pieces.append(piece)
                    piece_num += 1
                if process_softening:
                    for softening_var in point.softening_vars:
                        softening_var.ref_yield_point_num = point_num
                        softening_var.num_in_structure = softening_var_num
                        softening_var_num += 1
                intact_components_count += point.components_count
                point_num += 1
        return IntactYieldPointsResults(
            intact_points=intact_points,
            intact_pieces=intact_pieces,
            intact_components_count=intact_components_count,
        )

    def create_intact_phi(self):
        intact_phi = np.zeros((self.intact_components_count, self.intact_pieces_count))
        current_row_start = 0
        current_column_start = 0
        for yield_point in self.intact_points:
            current_row_end = current_row_start + yield_point.components_count
            current_column_end = current_column_start + yield_point.pieces_count
            intact_phi[current_row_start:current_row_end, current_column_start:current_column_end] = yield_point.phi
            current_row_start = current_row_end
            current_column_start = current_column_end
        return intact_phi

    def create_intact_q(self):
        intact_q = np.zeros((2 * self.intact_points_count, self.intact_pieces_count))
        pieces_counter = 0
        for i, yield_point in enumerate(self.intact_points):
            intact_q[2 * i:2 * i + 2, pieces_counter:pieces_counter + yield_point.pieces_count] = yield_point.q
            pieces_counter += yield_point.pieces_count
        return intact_q

    def create_intact_h(self):
        intact_h = np.zeros((self.intact_pieces_count, 2 * self.intact_points_count))
        pieces_counter = 0
        for i, yield_point in enumerate(self.intact_points):
            intact_h[pieces_counter:pieces_counter + yield_point.pieces_count, 2 * i:2 * i + 2] = yield_point.h
            pieces_counter += yield_point.pieces_count
        return intact_h

    def create_intact_w(self):
        intact_w = np.zeros((2 * self.intact_points_count, 2 * self.intact_points_count))
        for i, yield_point in enumerate(self.intact_points):
            intact_w[2 * i:2 * i + 2, 2 * i:2 * i + 2] = yield_point.w
        return intact_w

    def create_intact_cs(self):
        intact_cs = np.zeros((2 * self.intact_points_count, 1))
        for i, yield_point in enumerate(self.intact_points):
            intact_cs[2 * i:2 * i + 2] = yield_point.cs
        return intact_cs
