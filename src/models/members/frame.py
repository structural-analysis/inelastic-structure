import numpy as np

from ..points import Node
from ..sections.frame import FrameSection


class YieldSpecs:
    def __init__(self, section: FrameSection):
        self.points_num = 2
        self.components_num = self.points_num * section.yield_specs.components_num
        self.pieces_num = self.points_num * section.yield_specs.pieces_num


class Mass:
    def __init__(self, magnitude):
        self.magnitude = magnitude


class FrameMember2D:
    # mp: bending capacity
    # udef: unit distorsions equivalent forces
    # ends_fixity: one of following: fix_fix, hinge_fix, fix_hinge, hinge_hinge
    def __init__(self, nodes: tuple[Node, Node], ends_fixity, section: FrameSection, mass: Mass = None):
        self.nodes = nodes
        self.ends_fixity = ends_fixity
        self.section = section
        self.total_dofs_num = 6
        self.yield_specs = YieldSpecs(self.section)
        self.start = nodes[0]
        self.end = nodes[1]
        self.l = self._length()
        self.mass = mass if mass else None
        self.m = self._mass() if mass else None
        self.k = self._stiffness()
        self.t = self._transform_matrix()
        self.udefs = self._udefs()

    def _length(self):
        a = self.start
        b = self.end
        l = np.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        return l

    def _stiffness(self):
        l = self.l
        a = self.section.geometry.a
        i = self.section.geometry.ix
        e = self.section.material.e
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

    def _mass(self):
        l = self.l
        mass = self.mass.magnitude
        m = np.matrix(
            [
                [mass * l / 2, 0, 0, 0, 0, 0],
                [0, mass * l / 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, mass * l / 2, 0, 0],
                [0, 0, 0, 0, mass * l / 2, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        return m

    def _udefs(self):
        k = self.k
        k_size = k.shape[0]
        if self.section.nonlinear.has_axial_yield:
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
