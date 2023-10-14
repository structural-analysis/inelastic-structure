import numpy as np
from dataclasses import dataclass

from ..points import Node
from ..sections.frame3d import Frame3DSection


@dataclass
class Response:
    nodal_force: np.matrix
    yield_components_force: np.matrix
    nodal_strains: np.matrix = np.matrix(np.zeros([1, 1]))
    nodal_stresses: np.matrix = np.matrix(np.zeros([1, 1]))
    nodal_moments: np.matrix = np.matrix(np.zeros([1, 1]))


class YieldSpecs:
    def __init__(self, section: Frame3DSection):
        self.points_count = 2
        self.components_count = self.points_count * section.yield_specs.components_count
        self.pieces_count = self.points_count * section.yield_specs.pieces_count


class Mass:
    def __init__(self, magnitude):
        self.magnitude = magnitude


class Frame3DMember:
    def __init__(self, structure_node_dofs_count: int, structure_type: str, num: int, nodes: tuple[Node, Node], ends_fixity, section: Frame3DSection, roll_angle: float = 0, mass: Mass = None):
        self.structure_type = structure_type
        self.structure_node_dofs_count = structure_node_dofs_count
        self.num = num
        self.nodes = nodes
        self.nodes_count = len(self.nodes)
        self.node_dofs_count = 6
        self.dofs_count = self.node_dofs_count * self.nodes_count
        # ends_fixity: one of following: fix_fix, hinge_fix, fix_hinge, hinge_hinge
        self.ends_fixity = ends_fixity
        self.section = section
        self.yield_specs = YieldSpecs(self.section)
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
        a = self.nodes[0]
        b = self.nodes[1]
        l = np.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)
        return l

    def _stiffness(self):
        # Reference for stiffness formula:
        # Bathe K. J., Finite Element Procedures, 1996, page 151.
        # Papadrakakis M., Matrix Methods for Advanced Structural Analysis, 2017, page 193
        # Kassimali A., Matrix Analysis Of Structures, 2nd ed, 2011 page 464
        # Note: because the coordinate system in sap2000 is different from the reference, to get the results
        # of sap we transform the stiffness matrix of the reference with below transform matrix:
        # r = np.matrix([           t = np.matrix(np.zeros((12, 12)))
        #     [1, 0, 0],            t[0:3, 0:3] = r
        #     [0, 0, -1],           t[3:6, 3:6] = r
        #     [0, 1, 0]             t[6:9, 6:9] = r
        # ])                        t[9:12, 9:12] = r

        l = self.l
        a = self.section.geometry.a
        j = self.section.geometry.j
        i22 = self.section.geometry.i22
        i33 = self.section.geometry.i33
        e = self.section.material.e
        g = self.section.material.g
        ends_fixity = self.ends_fixity

        if (ends_fixity == "fix_fix"):
            k = np.matrix([
                [e * a / l, 0, 0, 0, 0, 0, -e * a / l, 0, 0, 0, 0, 0],
                [0, 12 * e * i22 / l ** 3, 0, 0, 0, 6 * e * i22 / l ** 2, 0, -12 * e * i22 / l ** 3, 0, 0, 0, 6 * e * i22 / l ** 2],
                [0, 0, 12 * e * i33 / l ** 3, 0, -6 * e * i33 / l ** 2, 0, 0, 0, -12 * e * i33 / l ** 3, 0, -6 * e * i33 / l ** 2, 0],
                [0, 0, 0, g * j / l, 0, 0, 0, 0, 0, -g * j / l, 0, 0],
                [0, 0, -6 * e * i33 / l ** 2, 0, 4 * e * i33 / l, 0, 0, 0, 6 * e * i33 / l ** 2, 0, 2 * e * i33 / l, 0],
                [0, 6 * e * i22 / l ** 2, 0, 0, 0, 4 * e * i22 / l, 0, -6 * e * i22 / l ** 2, 0, 0, 0, 2 * e * i22 / l],
                [-e * a / l, 0, 0, 0, 0, 0, e * a / l, 0, 0, 0, 0, 0],
                [0, -12 * e * i22 / l ** 3, 0, 0, 0, -6 * e * i22 / l** 2, 0, 12 * e * i22 / l ** 3, 0, 0, 0, -6 * e * i22 / l ** 2],
                [0, 0, -12 * e * i33 / l ** 3, 0, 6 * e * i33 / l ** 2, 0, 0, 0, 12 * e * i33 / l ** 3, 0, 6 * e * i33 / l ** 2, 0],
                [0, 0, 0, -g * j / l, 0, 0, 0, 0, 0, g * j / l, 0, 0],
                [0, 0, -6 * e * i33 / l ** 2, 0, 2 * e * i33 / l, 0, 0, 0, 6 * e * i33 / l ** 2, 0, 4 * e * i33 / l, 0],
                [0, 6 * e * i22 / l ** 2, 0, 0, 0, 2 * e * i22 / l, 0, -6 * e * i22 / l ** 2, 0, 0, 0, 4 * e * i22 / l],
            ])

        elif (ends_fixity == "hinge_fix"):
            k = np.matrix([
                [e * a / l, 0, 0, 0, 0, 0, -e * a / l, 0, 0, 0, 0, 0],
                [0, 3 * e * i22 / l ** 3, 0, 0, 0, 0, 0, -3 * e * i22 / l ** 3, 0, 0, 0, 3 * e * i22 / l ** 2],
                [0, 0, 3 * e * i33 / l ** 3, 0, 0, 0, 0, 0, -3 * e * i33 / l ** 3, 0, -3 * e * i33 / l ** 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-e * a / l, 0, 0, 0, 0, 0, e * a / l, 0, 0, 0, 0, 0],
                [0, -3 * e * i22 / l ** 3, 0, 0, 0, 0, 0, 3 * e * i22 / l ** 3, 0, 0, 0, -3 * e * i22 / l ** 2],
                [0, 0, -3 * e * i33 / l ** 3, 0, 0, 0, 0, 0, 3 * e * i33 / l ** 3, 0, 3 * e * i33 / l ** 2, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -3 * e * i33 / l ** 2, 0, 0, 0, 0, 0, 3 * e * i33 / l ** 2, 0, 3 * e * i33 / l, 0],
                [0, 3 * e * i22 / l ** 2, 0, 0, 0, 0, 0, -3 * e * i22 / l ** 2, 0, 0, 0, 3 * e * i22 / l],
            ])

        elif (ends_fixity == "fix_hinge"):
            k = np.matrix([
                [e * a / l, 0, 0, 0, 0, 0, -e * a / l, 0, 0, 0, 0, 0],
                [0, 3 * e * i22 / l ** 3, 0, 0, 0, 3 * e * i22 / l ** 2, 0, -3 * e * i22 / l ** 3, 0, 0, 0, 0],
                [0, 0, 3 * e * i33 / l ** 3, 0, -3 * e * i33 / l ** 2, 0, 0, 0, -3 * e * i33 / l ** 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -3 * e * i33 / l ** 2, 0, 3 * e * i33 / l, 0, 0, 0, 3 * e * i33 / l ** 2, 0, 0, 0],
                [0, 3 * e * i22 / l ** 2, 0, 0, 0, 3 * e * i22 / l, 0, -3 * e * i22 / l ** 2, 0, 0, 0, 0],
                [- e * a /l, 0, 0, 0, 0, 0, e * a / l, 0, 0, 0, 0, 0],
                [0, -3 * e * i22 / l ** 3, 0, 0, 0, -3 * e * i22 / l ** 2, 0, 3 * e * i22 / l ** 3, 0, 0, 0, 0],
                [0, 0, -3 * e * i33 / l ** 3, 0, 3 * e * i33 / l ** 2, 0, 0, 0, 3 * e * i33 / l ** 3, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
                [mass * l / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, mass * l / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, mass * l / 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, mass * l / 2, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, mass * l / 2, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, mass * l / 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        return m

    def _transform_matrix(self):
        # Reference for transformation formula:
        # Kassimali A., Matrix Analysis Of Structures, 2nd ed, 2011 page 476
        si = self.roll_angle
        a = self.nodes[0]
        b = self.nodes[1]
        l = self.l
        rxx = (b.x - a.x) / l
        rxy = (b.y - a.y) / l
        rxz = (b.z - a.z) / l
        if rxx == 0 and rxz == 0:
            # Kassimali A., Matrix Analysis Of Structures, 2nd ed, 2011 page 479
            r = np.matrix([
                [0, rxy, 0],
                [-rxy * np.cos(si), 0, np.sin(si)],
                [rxy * np.sin(si), 0, np.cos(si)],
            ])
        else:
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

    def get_response(self, nodal_disp, fixed_force=None, fixed_stress=None):
        # nodal_disp: numpy matrix
        if fixed_force is None:
            fixed_force = np.matrix(np.zeros((self.dofs_count, 1)))

        if fixed_force.any():
            nodal_force = self.k * nodal_disp + fixed_force
        else:
            nodal_force = self.k * nodal_disp

        yield_components_force = np.matrix(np.zeros((6, 1)))
        yield_components_force[0, 0] = nodal_force[0, 0]
        yield_components_force[1, 0] = nodal_force[4, 0]
        yield_components_force[2, 0] = nodal_force[5, 0]
        yield_components_force[3, 0] = nodal_force[6, 0]
        yield_components_force[4, 0] = nodal_force[10, 0]
        yield_components_force[5, 0] = nodal_force[11, 0]

        response = Response(
            nodal_force=nodal_force,
            yield_components_force=yield_components_force,
        )
        return response

    def get_nodal_forces_from_unit_distortions(self):
        nodal_forces = np.matrix(np.zeros((self.dofs_count, self.yield_specs.components_count)))
        nodal_forces[:, 0] = self.k[:, 0]
        nodal_forces[:, 1] = self.k[:, 4]
        nodal_forces[:, 2] = self.k[:, 5]
        nodal_forces[:, 3] = self.k[:, 6]
        nodal_forces[:, 4] = self.k[:, 10]
        nodal_forces[:, 5] = self.k[:, 11]
        return nodal_forces
