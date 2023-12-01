from dataclasses import dataclass
import numpy as np


@dataclass
class YieldPiece:
    local_num: int
    selected: bool


@dataclass
class YieldPoint:
    selected: bool
    member_num: int
    components_count: int
    all_pieces: list()
    all_pieces_count: int
    intact_phi: np.matrix
    sifted_pieces: list
    sifted_pieces_count: int
    sifted_phi: np.matrix
    softening_properties: object
