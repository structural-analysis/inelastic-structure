import numpy as np
from dataclasses import dataclass

from ..points import Node
from ..sections.truss2d import Truss2DSection
from ..yield_models import MemberYieldSpecs


@dataclass
class Response:
    nodal_force: np.array
    yield_components_force: np.array
    nodal_strains: np.array = np.empty(0)
    nodal_stresses: np.array = np.empty(0)
    nodal_moments: np.array = np.empty(0)


class YieldSpecs:
    def __init__(self, section: Truss2DSection):
        self.points_count = 2
        self.components_count = self.points_count * section.yield_specs.components_count
        self.pieces_count = self.points_count * section.yield_specs.pieces_count
        self.section = section


class Mass:
    def __init__(self, magnitude):
        self.magnitude = magnitude


class Truss2DMember:
    def __init__(self, num: int, nodes: tuple[Node, Node], section: Truss2DSection, mass: Mass = None):
        self.num = num
        self.section = section
        self.yield_points_count = 1
        self.nodes = nodes
        self.nodes_count = len(self.nodes)
        self.node_dofs_count = 2
        self.dofs_count = self.node_dofs_count * self.nodes_count
        self.nodal_components_count = self.nodes_count * self.section.yield_specs.components_count
        self.yield_specs = MemberYieldSpecs(self.section, points_count=self.yield_points_count)
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
        # Kassimali A., Matrix Analysis Of Structures, 2nd ed, 2011 page 63
        l = self.l
        a = self.section.geometry.a
        e = self.section.material.e

        k = np.matrix([
            [e * a / l, 0, -e * a / l, 0],
            [ 0, 0, 0, 0],
            [-e * a / l, 0, e * a / l, 0],
            [0, 0, 0, 0]])

        return k

    def _mass(self):
        l = self.l
        mass = self.mass.magnitude
        m = np.matrix(
            [
                [mass * l / 2, 0, 0, 0],
                [0, mass * l / 2, 0, 0],
                [0, 0, mass * l / 2, 0],
                [0, 0, 0, mass * l / 2],
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
            [(b.x - a.x) / l, (b.y - a.y) / l, 0.0, 0.0],
            [-(b.y - a.y) / l, (b.x - a.x) / l, 0.0, 0.0],
            [0.0, 0.0, (b.x - a.x) / l, (b.y - a.y) / l],
            [0.0, 0.0, -(b.y - a.y) / l, (b.x - a.x) / l],
        ])
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

        yield_components_force = np.matrix(np.zeros((1, 1)))
        yield_components_force[0, 0] = nodal_force[2, 0]

        response = Response(
            nodal_force=nodal_force,
            yield_components_force=yield_components_force,
        )
        return response

    def get_nodal_forces_from_unit_distortions(self):
        nodal_forces = np.matrix(np.zeros((self.dofs_count, self.yield_specs.components_count)))
        nodal_forces[:, 0] = self.k[:, 2]
        return nodal_forces
