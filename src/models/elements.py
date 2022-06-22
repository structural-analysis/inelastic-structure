import numpy as np
from math import sqrt

from .points import Node
from .sections import FrameSection, PlateSection


class YieldSpecs:
    points_num: int
    pieces_num: int
    components_num: int


class Elements:
    def __init__(self, elements_array):
        self.all = elements_array
        self.num = len(elements_array)
        self.yield_specs = self.get_yield_specs()

    def get_yield_specs(self):
        yield_points_num = 0
        yield_components_num = 0
        yield_pieces_num = 0

        for element in self.all:
            yield_points_num += element.yield_points_num
            yield_components_num += element.yield_components_num
            yield_pieces_num += element.yield_pieces_num

        yield_specs = YieldSpecs()
        yield_specs.points_num = yield_points_num
        yield_specs.components_num = yield_components_num
        yield_specs.pieces_num = yield_pieces_num

        return yield_specs


class FrameElement2D:
    # mp: bending capacity
    # udef: unit distorsions equivalent forces
    # ends_fixity: one of following: fix_fix, hinge_fix, fix_hinge, hinge_hinge
    def __init__(self, nodes: tuple[Node, Node], ends_fixity, section: FrameSection):
        self.nodes = nodes
        self.ends_fixity = ends_fixity
        self.section = section
        self.total_dofs_num = 6
        self.yield_points_num = 2
        self.yield_components_num = self.yield_points_num * self.section.yield_components_num
        self.yield_pieces_num = self.yield_points_num * self.section.yield_pieces_num
        self.has_axial_yield = self.section.has_axial_yield
        self.start = nodes[0]
        self.end = nodes[1]
        self.a = self.section.a
        self.i = self.section.ix
        self.e = self.section.e
        self.mp = self.section.mp
        self.l = self._length()
        self.k = self._stiffness()
        self.t = self._transform_matrix()
        self.udefs = self._udefs()

    def _length(self):
        a = self.start
        b = self.end
        l = sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        return l

    def _stiffness(self):
        l = self.l
        a = self.a
        i = self.i
        e = self.e
        ends_fixity = self.ends_fixity

        if (ends_fixity == "fix_fix"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 12.0 * e * i / (l ** 3.0), 6.0 * e * i / (l ** 2.0), 0.0, -12.0 * e * i / (l ** 3.0), 6.0 * e * i / (l ** 2.0)],
                [0.0, 6.0 * e * i / (l ** 2.0), 4.0 * e * i / (l), 0.0, -6.0 * e * i / (l ** 2.0), 2.0 * e * i / (l)],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -12.0 * e * i / (l ** 3.0), -6.0 * e * i / (l ** 2.0), 0.0, 12.0 * e * i / (l ** 3.0), -6.0 * e * i / (l ** 2.0)],
                [0.0, 6.0 * e * i / (l ** 2.0), 2.0 * e * i / (l), 0.0, -6.0 * e * i / (l ** 2.0), 4.0 * e * i / (l)]])

        elif (ends_fixity == "hinge_fix"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 3.0 * e * i / (l ** 3.0), 0.0, 0.0, -3.0 * e * i / (l ** 3.0), 3.0 * e * i / (l ** 2.0)],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -3.0 * e * i / (l ** 3.0), 0.0, 0.0, 3.0 * e * i / (l ** 3.0), -3.0 * e * i / (l ** 2.0)],
                [0.0, 3.0 * e * i / (l ** 2.0), 0.0, 0.0, -3.0 * e * i / (l ** 2.0), 3.0 * e * i / (l)]])

        elif (ends_fixity == "fix_hinge"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 3.0 * e * i / (l ** 3.0), 3.0 * e * i / (l ** 2.0), 0.0, -3.0 * e * i / (l ** 3.0), 0.0],
                [0.0, 3.0 * e * i / (l ** 2.0), 3.0 * e * i / (l), 0.0, -3.0 * e * i / (l ** 2.0), 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -3.0 * e * i / (l ** 3.0), -3.0 * e * i / (l ** 2.0), 0.0, 3.0 * e * i / (l ** 3.0), 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        elif (ends_fixity == "hinge_hinge"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        return k

    def _udefs(self):
        k = self.k
        k_size = k.shape[0]
        if self.has_axial_yield:
            udef_start_empty = np.zeros((k_size, 2))
            udef_end_empty = np.zeros((k_size, 2))
            udef_start = np.matrix(udef_start_empty)
            udef_end = np.matrix(udef_end_empty)
            udef_start[:, 0] = k[:, 0]
            udef_start[:, 1] = k[:, 2]
            udef_end[:, 0] = k[:, 3]
            udef_end[:, 1] = k[:, 5]
        else:
            udef_start_empty = np.zeros((k_size, 1))
            udef_end_empty = np.zeros((k_size, 1))
            udef_start = np.matrix(udef_start_empty)
            udef_end = np.matrix(udef_end_empty)
            udef_start[:, 0] = k[:, 2]
            udef_end[:, 0] = k[:, 5]

        udefs = (udef_start, udef_end)
        return udefs

    def _transform_matrix(self):
        a = self.start
        b = self.end
        l = self.l
        t = np.matrix([
            [(b.x - a.x) / l, -(b.y - a.y) / l, 0.0, 0.0, 0.0, 0.0],
            [(b.y - a.y) / l, (b.x - a.x) / l, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, (b.x - a.x) / l, -(b.y - a.y) / l, 0.0],
            [0.0, 0.0, 0.0, (b.y - a.y) / l, (b.x - a.x) / l, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        return t

    def get_nodal_force(self, displacements, fixed_forces):
        # displacements: numpy matrix
        # fixed_forces: numpy matrix
        k = self.k
        f = k * displacements + fixed_forces
        return f


class RectangularThinPlateElement:
    # k is calculated based on four integration points
    def __init__(self, nodes: tuple[Node, Node, Node, Node], section: PlateSection):
        self.t = section.t
        self.nodes = nodes
        self.lx = nodes[1].x - nodes[0].x
        self.ly = nodes[2].y - nodes[1].y
        self.k = self._stiffness(section.de)

    def _shape_functions(self, r, s):
        ax = (self.lx / 2)
        ay = (self.ly / 2)
        n = np.matrix([1 / 8 * (1 - r) * (1 - s) * (2 - r - s - r ** 2 - s ** 2),
                       1 / 8 * (1 - r) * (1 - s) * (+ay * (1 - s ** 2)),
                       1 / 8 * (1 - r) * (1 - s) * (-ax * (1 - r ** 2)),
                       1 / 8 * (1 + r) * (1 - s) * (2 + r - s - r ** 2 - s ** 2),
                       1 / 8 * (1 + r) * (1 - s) * (+ay * (1 - s ** 2)),
                       1 / 8 * (1 + r) * (1 - s) * (+ax * (1 - r ** 2)),
                       1 / 8 * (1 + r) * (1 + s) * (2 + r + s - r ** 2 - s ** 2),
                       1 / 8 * (1 + r) * (1 + s) * (-ay * (1 - s ** 2)),
                       1 / 8 * (1 + r) * (1 + s) * (+ax * (1 - r ** 2)),
                       1 / 8 * (1 - r) * (1 + s) * (2 - r + s - r ** 2 - s ** 2),
                       1 / 8 * (1 - r) * (1 + s) * (-ay * (1 - s ** 2)),
                       1 / 8 * (1 - r) * (1 + s) * (-ax * (1 - r ** 2))
                       ])
        return n

    def _shape_derivatives(self, r, s):
        ax = (self.lx / 2)
        ay = (self.ly / 2)
        b = np.matrix([[((0.125 - 0.125 * r) * (2 * s - 2) - 0.125 * (1 - s) * (-2 * r - 1) + (-2 * r - 1) * (0.125 * s - 0.125)) / ax ** 2,
                        0,
                        (-0.5 * ax * r * (1 - s) + 2 * ax * (0.125 - 0.125 * r) * (1 - s)) / ax ** 2,
                        ((0.125 - 0.125 * s) * (1 - 2 * r) + 0.125 * (1 - 2 * r) * (1 - s) + (0.125 * r + 0.125) * (2 * s - 2)) / ax ** 2,
                        0,
                        (-0.5 * ax * r * (1 - s) - 2 * ax * (1 - s) * (0.125 * r + 0.125)) / ax ** 2,
                        ((1 - 2 * r) * (0.125 * s + 0.125) + 0.125 * (1 - 2 * r) * (s + 1) + (0.125 * r + 0.125) * (-2 * s - 2)) / ax ** 2,
                        0,
                        (-0.5 * ax * r * (s + 1) - 2 * ax * (0.125 * r + 0.125) * (s + 1)) / ax ** 2,
                        ((0.125 - 0.125 * r) * (-2 * s - 2) + (-2 * r - 1) * (-0.125 * s - 0.125) - 0.125 * (-2 * r - 1) * (s + 1)) / ax ** 2,
                        0,
                        (-0.5 * ax * r * (s + 1) + 2 * ax * (0.125 - 0.125 * r) * (s + 1)) / ax ** 2],
                       [((0.125 - 0.125 * r) * (2 * s - 2) + (0.125 - 0.125 * r) * (2 * s + 1) + (0.125 * r - 0.125) * (-2 * s - 1)) / ay ** 2,
                        (4 * ay * s * (0.125 - 0.125 * r) - 2 * ay * (0.125 - 0.125 * r) * (1 - s)) / ay ** 2,
                        0,
                        ((-0.125 * r - 0.125) * (-2 * s - 1) + (0.125 * r + 0.125) * (2 * s - 2) + (0.125 * r + 0.125) * (2 * s + 1)) / ay ** 2,
                        (4 * ay * s * (0.125 * r + 0.125) - 2 * ay * (1 - s) * (0.125 * r + 0.125)) / ay ** 2,
                        0,
                        (2 * (1 - 2 * s) * (0.125 * r + 0.125) + (0.125 * r + 0.125) * (-2 * s - 2)) / ay ** 2,
                        (4 * ay * s * (0.125 * r + 0.125) + 2 * ay * (0.125 * r + 0.125) * (s + 1)) / ay ** 2,
                        0,
                        (2 * (0.125 - 0.125 * r) * (1 - 2 * s) + (0.125 - 0.125 * r) * (-2 * s - 2)) / ay ** 2,
                        (4 * ay * s * (0.125 - 0.125 * r) + 2 * ay * (0.125 - 0.125 * r) * (s + 1)) / ay ** 2,
                        0],
                       [2 * (-0.125 * r ** 2 - 0.125 * r - 0.125 * s ** 2 - 0.125 * s + (0.125 - 0.125 * r) * (2 * r + 1) + (-2 * s - 1) * (0.125 * s - 0.125) + 0.25) / (ax * ay),
                        2 * (0.25 * ay * s * (1 - s) + 0.125 * ay * (1 - s ** 2)) / (ax * ay),
                        2 * (-2 * ax * r * (0.125 - 0.125 * r) - 0.125 * ax * (1 - r ** 2)) / (ax * ay),
                        2 * (0.125 * r ** 2 - 0.125 * r + 0.125 * s ** 2 + 0.125 * s + (0.125 - 0.125 * s) * (-2 * s - 1) + (0.125 * r + 0.125) * (2 * r - 1) - 0.25) / (ax * ay),
                        2 * (-0.25 * ay * s * (1 - s) - 0.125 * ay * (1 - s ** 2)) / (ax * ay),
                        2 * (2 * ax * r * (0.125 * r + 0.125) - 0.125 * ax * (1 - r ** 2)) / (ax * ay),
                        2 * (-0.125 * r ** 2 + 0.125 * r - 0.125 * s ** 2 + 0.125 * s + (1 - 2 * r) * (0.125 * r + 0.125) + (1 - 2 * s) * (0.125 * s + 0.125) + 0.25) / (ax * ay),
                        2 * (0.25 * ay * s * (s + 1) - 0.125 * ay * (1 - s ** 2)) / (ax * ay),
                        2 * (-2 * ax * r * (0.125 * r + 0.125) + 0.125 * ax * (1 - r ** 2)) / (ax * ay),
                        2 * (0.125 * r ** 2 + 0.125 * r + 0.125 * s ** 2 - 0.125 * s + (0.125 - 0.125 * r) * (-2 * r - 1) + (1 - 2 * s) * (-0.125 * s - 0.125) - 0.25) / (ax * ay),
                        2 * (-0.25 * ay * s * (s + 1) + 0.125 * ay * (1 - s ** 2)) / (ax * ay),
                        2 * (2 * ax * r * (0.125 - 0.125 * r) + 0.125 * ax * (1 - r ** 2)) / (ax * ay)]])
        return b

    def _stiffness_integrand(self, r, s, de):
        b = self._shape_derivatives(r=r, s=s)
        ki = b.T * de * b
        return ki

    def _stiffness(self, de):
        ax = (self.lx / 2)
        ay = (self.ly / 2)
        kin = self._stiffness_integrand(r=-0.57735, s=-0.57735, de=de) + \
            self._stiffness_integrand(r=+0.57735, s=-0.57735, de=de) + \
            self._stiffness_integrand(r=+0.57735, s=+0.57735, de=de) + \
            self._stiffness_integrand(r=-0.57735, s=+0.57735, de=de)
        k = kin * ax * ay
        return k
