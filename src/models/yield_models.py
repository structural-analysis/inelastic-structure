from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class YieldPiece:
    local_num: int
    selected: bool


@dataclass
class YieldPoint:
    selected: bool
    member_num: int
    components_count: int
    all_pieces: List[YieldPiece]
    all_pieces_count: int
    intact_phi: np.matrix
    sifted_pieces: List[YieldPiece]
    sifted_pieces_count: int
    sifted_phi: np.matrix
    softening_properties: object


class MemberYieldSpecs:
    def __init__(self, section: object, points_count: int):
        self.section = section
        self.points_count = points_count
        self.components_count = self.points_count * self.section.yield_specs.components_count
        self.yield_points = self.get_yield_points()

    def get_yield_points(self):
        yield_points = []
        for _ in range(self.points_count):
            yield_pieces = []
            yield_piece_num = 0
            for _ in range(self.section.yield_specs.pieces_count):
                yield_pieces.append(
                    YieldPiece(
                        local_num=yield_piece_num,
                        selected=True,
                    )
                )
                yield_piece_num += 1
            yield_points.append(
                YieldPoint(
                    selected=True,
                    member_num=-1,
                    components_count=self.section.yield_specs.components_count,
                    all_pieces=yield_pieces,
                    all_pieces_count=self.section.yield_specs.pieces_count,
                    intact_phi=self.section.yield_specs.phi,
                    sifted_pieces=[],
                    sifted_pieces_count=self.section.yield_specs.sifted_pieces_count,
                    sifted_phi=np.matrix(np.zeros((1, 1))),
                    softening_properties=self.section.softening,
                )
            )
        return yield_points


class StructureYieldSpecs:
    def __init__(self, members):
        self.members = members
        self.all_yield_points_stats: tuple = self.get_all_yield_points_stats()
        self.all_yield_points: list = self.all_yield_points_stats[0]
        self.all_components_count = self.all_yield_points_stats[1]
        self.all_pieces_count = self.all_yield_points_stats[2]
        self.all_points_count = len(self.all_yield_points)
        self.intact_phi = self.create_intact_phi()
        self.intact_q = self.create_intact_q()
        self.intact_h = self.create_intact_h()
        self.intact_w = self.create_intact_w()
        self.intact_cs = self.create_intact_cs()
        self.yield_points_indices = self.get_yield_points_indices()

    def get_all_yield_points_stats(self):
        all_yield_points = []
        all_components_count = 0
        all_pieces_count = 0
        for member_num, member in enumerate(self.members):
            for yield_point in member.yield_specs.yield_points:
                yield_point.member_num = member_num
                all_yield_points.append(yield_point)
                all_components_count += yield_point.components_count
                all_pieces_count += yield_point.all_pieces_count
        return all_yield_points, all_components_count, all_pieces_count

    def create_intact_phi(self):
        intact_phi = np.matrix(np.zeros((self.all_components_count, self.all_pieces_count)))
        current_row_start = 0
        current_column_start = 0
        for yield_point in self.all_yield_points:
            current_row_end = current_row_start + yield_point.components_count
            current_column_end = current_column_start + yield_point.all_pieces_count
            intact_phi[current_row_start:current_row_end, current_column_start:current_column_end] = yield_point.intact_phi
            current_row_start = current_row_end
            current_column_start = current_column_end
        return intact_phi

    def create_intact_q(self):
        intact_q = np.matrix(np.zeros((2 * self.all_points_count, self.all_pieces_count)))
        pieces_counter = 0
        for i, yield_point in enumerate(self.all_yield_points):
            intact_q[2 * i:2 * i + 2, pieces_counter:pieces_counter + yield_point.all_pieces_count] = yield_point.softening_properties.q
            pieces_counter += yield_point.all_pieces_count
        return intact_q

    def create_intact_h(self):
        intact_h = np.matrix(np.zeros((self.all_pieces_count, 2 * self.all_points_count)))
        pieces_counter = 0
        for i, yield_point in enumerate(self.all_yield_points):
            intact_h[pieces_counter:pieces_counter + yield_point.all_pieces_count, 2 * i:2 * i + 2] = yield_point.softening_properties.h
            pieces_counter += yield_point.all_pieces_count
        return intact_h

    def create_intact_w(self):
        intact_w = np.matrix(np.zeros((2 * self.all_points_count, 2 * self.all_points_count)))
        for i, yield_point in enumerate(self.all_yield_points):
            intact_w[2 * i:2 * i + 2, 2 * i:2 * i + 2] = yield_point.softening_properties.w
        return intact_w

    def create_intact_cs(self):
        intact_cs = np.matrix(np.zeros((2 * self.all_points_count, 1)))
        for i, yield_point in enumerate(self.all_yield_points):
            intact_cs[2 * i:2 * i + 2, 0] = yield_point.softening_properties.cs
        return intact_cs

    # TODO: can't we get yield point piece numbers from yield_points data?
    def get_yield_points_indices(self):
        yield_points_indices = []
        index_counter = 0
        for yield_point in self.all_yield_points:
            yield_points_indices.append(
                {
                    "begin": index_counter,
                    "end": index_counter + yield_point.all_pieces_count - 1,
                }
            )
            index_counter += yield_point.all_pieces_count
        return yield_points_indices
