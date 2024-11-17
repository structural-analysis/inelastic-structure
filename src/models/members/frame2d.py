import numpy as np
from dataclasses import dataclass

from ..points import Node
from ..sections.frame2d import Frame2DSection
from ..yield_models import MemberYieldSpecs


@dataclass
class Response:
    nodal_force: np.matrix
    yield_components_force: np.matrix
    nodal_strains: np.matrix = np.matrix(np.zeros([1, 1]))
    nodal_stresses: np.matrix = np.matrix(np.zeros([1, 1]))
    nodal_moments: np.matrix = np.matrix(np.zeros([1, 1]))


class Mass:
    def __init__(self, magnitude):
        self.magnitude = magnitude


class Frame2DMember:
    def __init__(self, num: int, nodes: tuple[Node, Node], ends_fixity, section: Frame2DSection, include_softening: bool, mass: Mass = None):
        self.num = num
        self.yield_points_count = 2
        self.nodes = nodes
        self.nodes_count = len(self.nodes)
        self.node_dofs_count = 3
        self.dofs_count = self.node_dofs_count * self.nodes_count
        # ends_fixity: one of following: fix_fix, hinge_fix, fix_hinge, hinge_hinge
        self.ends_fixity = ends_fixity
        self.section = section
        self.yield_specs = MemberYieldSpecs(
            section=self.section,
            points_count=self.yield_points_count,
            include_softening=include_softening,
        )
        self.l = self._length()
        self.mass = mass if mass else None
        self.m = self._mass() if mass else None
        self.k = self._stiffness()
        self.t = self._transform_matrix()
        # udef: unit distorsions equivalent forces (force, moment, ...) in nodes
        self.udefs = self.get_nodal_forces_from_unit_distortions()

    def _length(self):
        a = self.nodes[0]
        b = self.nodes[1]
        l = np.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        return l

    def _stiffness(self):
        # Reference for stiffness formula:
        # Bathe K. J., Finite Element Procedures, 1996, page 151.
        # Kassimali A., Matrix Analysis Of Structures, 2nd ed, 2011 page 260
        # Note: rotation in bathe is opposite of kassimali. it is affacted the stiffness elements signs
        l = self.l
        a = self.section.geometry.a
        i = self.section.geometry.i33
        e = self.section.material.e
        ends_fixity = self.ends_fixity

        if (ends_fixity == "fix_fix"):
            k = np.matrix([
                [e * a / l, 0, 0, -e * a / l, 0, 0],
                [0, 12 * e * i / (l ** 3), 6 * e * i / (l ** 2), 0, -12 * e * i / (l ** 3), 6 * e * i / (l ** 2)],
                [0, 6 * e * i / (l ** 2), 4 * e * i / (l), 0, -6 * e * i / (l ** 2), 2 * e * i / (l)],
                [-e * a / l, 0, 0, e * a / l, 0, 0],
                [0, -12 * e * i / (l ** 3), -6 * e * i / (l ** 2), 0, 12 * e * i / (l ** 3), -6 * e * i / (l ** 2)],
                [0, 6 * e * i / (l ** 2), 2 * e * i / (l), 0, -6 * e * i / (l ** 2), 4 * e * i / (l)]])

        # Kassimali A., Matrix Analysis Of Structures, 2nd ed, 2011 page 343
        elif (ends_fixity == "hinge_fix"):
            k = np.matrix([
                [e * a / l, 0, 0, -e * a / l, 0, 0],
                [0, 3 * e * i / (l ** 3), 0, 0, -3 * e * i / (l ** 3), 3 * e * i / (l ** 2)],
                [0, 0, 0, 0, 0, 0],
                [-e * a / l, 0, 0, e * a / l, 0, 0],
                [0, -3 * e * i / (l ** 3), 0, 0, 3 * e * i / (l ** 3), -3 * e * i / (l ** 2)],
                [0, 3 * e * i / (l ** 2), 0, 0, -3 * e * i / (l ** 2), 3 * e * i / (l)]])

        elif (ends_fixity == "fix_hinge"):
            k = np.matrix([
                [e * a / l, 0, 0, -e * a / l, 0, 0],
                [0, 3 * e * i / (l ** 3), 3 * e * i / (l ** 2), 0, -3 * e * i / (l ** 3), 0],
                [0, 3 * e * i / (l ** 2), 3 * e * i / (l), 0, -3 * e * i / (l ** 2), 0],
                [-e * a / l, 0, 0, e * a / l, 0, 0],
                [0, -3 * e * i / (l ** 3), -3 * e * i / (l ** 2), 0, 3 * e * i / (l ** 3), 0],
                [0, 0, 0, 0, 0, 0]])

        elif (ends_fixity == "hinge_hinge"):
            k = np.matrix([
                [e * a / l, 0, 0, -e * a / l, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-e * a / l, 0, 0, e * a / l, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]])

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

    def _transform_matrix(self):
        # Reference for transformation formula:
        # Papadrakakis M., Matrix Methods for Advanced Structural Analysis, 2017, page 28, page 92
        # Note: the transformation matrix in Kassimali A., Matrix Analysis Of Structures, 2nd ed, 2011 is not correct
        a = self.nodes[0]
        b = self.nodes[1]
        l = self.l
        t = np.matrix([
            [(b.x - a.x) / l, (b.y - a.y) / l, 0.0, 0.0, 0.0, 0.0],
            [-(b.y - a.y) / l, (b.x - a.x) / l, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, (b.x - a.x) / l, (b.y - a.y) / l, 0.0],
            [0.0, 0.0, 0.0, -(b.y - a.y) / l, (b.x - a.x) / l, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        return t

    def get_response(self, nodal_disp, fixed_external=None, fixed_internal=None):
        # fixed internal: fixed internal tractions like stress, force, moment, ... in gauss points of a member
        # fixed external: fixed external forces like force, moment, ... nodes of a member

        if fixed_external is None:
            fixed_external = np.matrix(np.zeros((self.dofs_count, 1)))

        if fixed_external.any():
            nodal_force = self.k * nodal_disp + fixed_external
        else:
            nodal_force = self.k * nodal_disp

        if self.section.nonlinear.has_axial_yield:
            yield_components_force = np.matrix(np.zeros((4, 1)))
            yield_components_force[0, 0] = nodal_force[0, 0]
            yield_components_force[1, 0] = nodal_force[2, 0]
            yield_components_force[2, 0] = nodal_force[3, 0]
            yield_components_force[3, 0] = nodal_force[5, 0]
        else:
            yield_components_force = np.matrix(np.zeros((2, 1)))
            yield_components_force[0, 0] = nodal_force[2, 0]
            yield_components_force[1, 0] = nodal_force[5, 0]

        response = Response(
            nodal_force=nodal_force,
            yield_components_force=yield_components_force,
        )
        return response

    def get_nodal_forces_from_unit_distortions(self):
        nodal_forces = np.matrix(np.zeros((self.dofs_count, self.yield_specs.components_count)))
        if self.section.nonlinear.has_axial_yield:
            nodal_forces[:, 0] = self.k[:, 0]
            nodal_forces[:, 1] = self.k[:, 2]
            nodal_forces[:, 2] = self.k[:, 3]
            nodal_forces[:, 3] = self.k[:, 5]
        else:
            nodal_forces[:, 0] = self.k[:, 2]
            nodal_forces[:, 1] = self.k[:, 5]
        return nodal_forces
