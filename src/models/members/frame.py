import numpy as np
from dataclasses import dataclass

from ..points import Node
from ..sections.frame import FrameSection


@dataclass
class Response:
    nodal_force: np.matrix
    yield_components_force: np.matrix
    internal_moments: np.matrix = np.matrix(np.zeros([1, 1]))
    top_internal_strains: np.matrix = np.matrix(np.zeros([1, 1]))
    bottom_internal_strains: np.matrix = np.matrix(np.zeros([1, 1]))
    top_internal_stresses: np.matrix = np.matrix(np.zeros([1, 1]))
    bottom_internal_stresses: np.matrix = np.matrix(np.zeros([1, 1]))


class YieldSpecs:
    def __init__(self, section: FrameSection):
        self.points_count = 2
        self.components_count = self.points_count * section.yield_specs.components_count
        self.pieces_count = self.points_count * section.yield_specs.pieces_count


class Mass:
    def __init__(self, magnitude):
        self.magnitude = magnitude


class FrameMember2D:
    def __init__(self, nodes: tuple[Node, Node], ends_fixity, section: FrameSection, mass: Mass = None):
        self.nodes = nodes
        # ends_fixity: one of following: fix_fix, hinge_fix, fix_hinge, hinge_hinge
        self.ends_fixity = ends_fixity
        self.section = section
        self.dofs_count = 6
        self.yield_specs = YieldSpecs(self.section)
        self.start = nodes[0]
        self.end = nodes[1]
        self.l = self._length()
        self.mass = mass if mass else None
        self.m = self._mass() if mass else None
        self.k = self._stiffness()
        self.t = self._transform_matrix()
        # udef: unit distorsions equivalent forces
        self.udefs = self.get_nodal_forces_from_unit_distortions()
        # print(f"{self.udefs=}")

    def _length(self):
        a = self.start
        b = self.end
        l = np.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        return l

    def _stiffness(self):
        # Reference for stiffness formula:
        # Bathe K. J., Finite Element Procedures, 1996, page 151.
        l = self.l
        a = self.section.geometry.a
        i = self.section.geometry.iz
        e = self.section.material.e
        ends_fixity = self.ends_fixity

        if (ends_fixity == "fix_fix"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 12.0 * e * i / (l ** 3.0), -6.0 * e * i / (l ** 2.0), 0.0, -12.0 * e * i / (l ** 3.0), -6.0 * e * i / (l ** 2.0)],
                [0.0, -6.0 * e * i / (l ** 2.0), 4.0 * e * i / (l), 0.0, 6.0 * e * i / (l ** 2.0), 2.0 * e * i / (l)],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -12.0 * e * i / (l ** 3.0), 6.0 * e * i / (l ** 2.0), 0.0, 12.0 * e * i / (l ** 3.0), 6.0 * e * i / (l ** 2.0)],
                [0.0, -6.0 * e * i / (l ** 2.0), 2.0 * e * i / (l), 0.0, 6.0 * e * i / (l ** 2.0), 4.0 * e * i / (l)]])

        elif (ends_fixity == "hinge_fix"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 3.0 * e * i / (l ** 3.0), 0.0, 0.0, -3.0 * e * i / (l ** 3.0), -3.0 * e * i / (l ** 2.0)],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -3.0 * e * i / (l ** 3.0), 0.0, 0.0, 3.0 * e * i / (l ** 3.0), 3.0 * e * i / (l ** 2.0)],
                [0.0, -3.0 * e * i / (l ** 2.0), 0.0, 0.0, 3.0 * e * i / (l ** 2.0), 3.0 * e * i / (l)]])

        elif (ends_fixity == "fix_hinge"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 3.0 * e * i / (l ** 3.0), -3.0 * e * i / (l ** 2.0), 0.0, -3.0 * e * i / (l ** 3.0), 0.0],
                [0.0, -3.0 * e * i / (l ** 2.0), 3.0 * e * i / (l), 0.0, 3.0 * e * i / (l ** 2.0), 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -3.0 * e * i / (l ** 3.0), 3.0 * e * i / (l ** 2.0), 0.0, 3.0 * e * i / (l ** 3.0), 0.0],
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

    def _transform_matrix(self):
        # Reference for transformation formula:
        # Papadrakakis M., Matrix Methods for Advanced Structural Analysis, 2017, page 28
        # Note: the transformation matrix in Kassimali A., Matrix Analysis Of Structures, 2nd ed, 2011 is not correct
        a = self.start
        b = self.end
        l = self.l
        t = np.matrix([
            [(b.x - a.x) / l, (b.y - a.y) / l, 0.0, 0.0, 0.0, 0.0],
            [-(b.y - a.y) / l, (b.x - a.x) / l, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, (b.x - a.x) / l, (b.y - a.y) / l, 0.0],
            [0.0, 0.0, 0.0, -(b.y - a.y) / l, (b.x - a.x) / l, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        return t

    def get_response(self, nodal_disp, fixed_force=None):
        # nodal_disp: numpy matrix
        if fixed_force is None:
            fixed_force = np.matrix(np.zeros((self.dofs_count, 1)))

        if fixed_force.any():
            nodal_force = self.k * nodal_disp + fixed_force
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


class FrameMember3D:
    def __init__(self, nodes: tuple[Node, Node], ends_fixity, section: FrameSection, roll_angle: float, mass: Mass = None):
        self.nodes = nodes
        # ends_fixity: one of following: fix_fix, hinge_fix, fix_hinge, hinge_hinge
        self.ends_fixity = ends_fixity
        self.section = section
        self.dofs_count = 6
        self.yield_specs = YieldSpecs(self.section)
        self.start = nodes[0]
        self.end = nodes[1]
        self.roll_angle = np.deg2rad(roll_angle)
        self.l = self._length()
        self.mass = mass if mass else None
        self.m = self._mass() if mass else None
        self.k = self._stiffness()
        self.t = self._transform_matrix()
        # udef: unit distorsions equivalent forces
        self.udefs = self.get_nodal_forces_from_unit_distortions()
        # print(f"{self.udefs=}")

    def _length(self):
        a = self.start
        b = self.end
        l = np.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)
        return l

    def _stiffness(self):
        # Reference for stiffness formula:
        # Bathe K. J., Finite Element Procedures, 1996, page 151.
        # Papadrakakis M., Matrix Methods for Advanced Structural Analysis, 2017, page 193
        # Kassimali A., Matrix Analysis Of Structures, 2nd ed, 2011 page 464

        l = self.l
        a = self.section.geometry.a
        j = self.section.geometry.j
        iy = self.section.geometry.iy
        iz = self.section.geometry.iz
        e = self.section.material.e
        g = self.section.material.g
        ends_fixity = self.ends_fixity

        if (ends_fixity == "fix_fix"):
            k = np.matrix([
                [e * a / l, 0, 0, 0, 0, 0, -e * a / l, 0, 0, 0, 0, 0],
                [0, 12 * e * iz / (l ** 3), 0, 0, 0, 6 * e * iz / (l ** 2), 0, -12 * e * iz / (l ** 3), 0, 0, 0, 6 * e * iz / (l ** 2)],
                [0, 0, 12 * e * iy / (l ** 3), 0, -6 * e * iy / (l ** 2), 0, 0, 0, -12 * e * iy / (l ** 3), 0, -6 * e * iy / (l ** 2), 0],
                [0, 0, 0, g * j / l, 0, 0, 0, 0, 0, -g * j / l, 0, 0],
                [0, 0, -6 * e * iy / (l ** 2), 0, 4 * e * iy / l, 0, 0, 0, 6 * e * iy / (l ** 2), 0, 2 * e * iy / l, 0],
                [0, 6 * e * iz / (l ** 2), 0, 0, 0, 4 * e * iz / l, 0, -6 * e * iz / (l ** 2), 0, 0, 0, 2 * e * iz / l],
                [-e * a / l, 0, 0, 0, 0, 0, e * a / l, 0, 0, 0, 0, 0],
                [0, -12 * e * iz / (l ** 3), 0, 0, 0, -6 * e * iz / (l ** 2), 0, 12 * e * iz / (l ** 3), 0, 0, 0, -6 * e * iz / (l ** 2)],
                [0, 0, -12 * e * iy / (l ** 3), 0, 6 * e * iy / (l ** 2), 0, 0, 0, 12 * e * iy / (l ** 3), 0, 6 * e * iy / (l ** 2), 0],
                [0, 0, 0, - g * j / l, 0, 0, 0, 0, 0, g * j / l, 0, 0],
                [0, 0, -6 * e * iy / (l ** 2), 0, 2 * e * iy / l, 0, 0, 0, 6 * e * iy / (l ** 2), 0, 4 * e * iy / l, 0],
                [0, 6 * e * iz / (l ** 2), 0, 0, 0, 2 * e * iz / l, 0, -6 * e * iz / (l ** 2), 0, 0, 0, 4 * e * iz / l]
            ])

        elif (ends_fixity == "hinge_fix"):
            k = np.matrix([
                [e * a / l, 0, 0, 0, 0, 0, -e * a / l, 0, 0, 0, 0, 0],
                [0, 3 * e * iz / (l ** 3), 0, 0, 0, 0, 0, -3 * e * iz / (l ** 3), 0, 0, 0, 3 * e * iz / (l ** 2)],
                [0, 0, 3 * e * iy / (l ** 3), 0, 0, 0, 0, 0, -3 * e * iy / (l ** 3), 0, -3 * e * iy / (l ** 2), 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-e * a / l, 0, 0, 0, 0, 0, e * a / l, 0, 0, 0, 0, 0],
                [0, -3 * e * iz / (l ** 3), 0, 0, 0, 0, 0, 3 * e * iz / (l ** 3), 0, 0, 0, -3 * e * iz / (l ** 2)],
                [0, 0, -3 * e * iy / (l ** 3), 0, 0, 0, 0, 0, 3 * e * iy / (l ** 3), 0, 3 * e * iy / (l ** 2), 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -3 * e * iy / (l ** 2), 0, 0, 0, 0, 0, 3 * e * iy / (l ** 2), 0, 3 * e * iy / l, 0],
                [0, 3 * e * iz / (l ** 2), 0, 0, 0, 0, 0, -3 * e * iz / (l ** 2), 0, 0, 0, 3 * e * iz / l]
            ])

        elif (ends_fixity == "fix_hinge"):
            k = np.matrix([
                [e * a / l, 0, 0, 0, 0, 0, -e * a / l, 0, 0, 0, 0, 0],
                [0, 3 * e * iz / (l ** 3), 0, 0, 0, 3 * e * iz / (l ** 2), 0, -3 * e * iz / (l ** 3), 0, 0, 0, 0],
                [0, 0, 3 * e * iy / (l ** 3), 0, -3 * e * iy / (l ** 2), 0, 0, 0, -3 * e * iy / (l ** 3), 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -3 * e * iy / (l ** 2), 0, 3 * e * iy / l, 0, 0, 0, 3 * e * iy / (l ** 2), 0, 0, 0],
                [0, 3 * e * iz / (l ** 2), 0, 0, 0, 3 * e * iz / l, 0, -3 * e * iz / (l ** 2), 0, 0, 0, 0],
                [-e * a / l, 0, 0, 0, 0, 0, e * a / l, 0, 0, 0, 0, 0],
                [0, -3 * e * iz / (l ** 3), 0, 0, 0, -3 * e * iz / (l ** 2), 0, 3 * e * iz / (l ** 3), 0, 0, 0, 0],
                [0, 0, -3 * e * iy / (l ** 3), 0, 3 * e * iy / (l ** 2), 0, 0, 0, 3 * e * iy / (l ** 3), 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])

        elif (ends_fixity == "hinge_hinge"):
            k = np.matrix([
                [e * a / l, 0, 0, 0, 0, 0, -e * a / l, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-e * a / l, 0, 0, 0, 0, 0, e * a / l, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])

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
        # Kassimali A., Matrix Analysis Of Structures, 2nd ed, 2011 page 476
        si = self.roll_angle
        a = self.start
        b = self.end
        l = self.l
        rxx = (b.x - a.x) / l
        rxy = (b.y - a.y) / l
        rxz = (b.z - a.z) / l
        r = np.matrix([
            [rxx, rxy, rxz],
            [(-rxx * rxy * np.cos(si) - rxz * np.sin(si)) / np.sqrt(rxx ** 2 + rxz **2), np.sqrt(rxx ** 2 + rxz ** 2) * np.cos(si), (-rxy * rxz * np.cos(si) + rxx * np.sin(si)) / np.sqrt(rxx ** 2 + rxz ** 2)],
            [(rxx * rxy * np.sin(si) - rxz * np.cos(si)) / np.sqrt(rxx ** 2 + rxz **2), -1 * np.sqrt(rxx ** 2 + rxz ** 2) * np.sin(si), (rxy * rxz * np.sin(si) + rxx * np.cos(si)) / np.sqrt(rxx ** 2 + rxz ** 2)]
        ])
        t = np.matrix(np.zeros((12, 12)))
        t[0:3, 0:3] = r
        t[3:6, 3:6] = r
        t[6:9, 6:9] = r
        t[9:12, 9:12] = r
        return t

    def get_response(self, nodal_disp, fixed_force=None):
        # nodal_disp: numpy matrix
        if fixed_force is None:
            fixed_force = np.matrix(np.zeros((self.dofs_count, 1)))

        if fixed_force.any():
            nodal_force = self.k * nodal_disp + fixed_force
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
