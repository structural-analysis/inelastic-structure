from .sections import FrameSection


class Node:
    def __init__(self, num, x, y):
        self.num = num
        self.x = x
        self.y = y


class FrameYieldPoint:
    def __init__(self, section: FrameSection):
        self.pieces_num = section.yield_pieces_num
