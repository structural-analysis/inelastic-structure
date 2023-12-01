import numpy as np
from dataclasses import dataclass
from functools import lru_cache

from ..points import NaturalPoint, GaussPoint
from ..sections.plate import PlateSection
from ..yield_models import YieldPoint, YieldPiece


@dataclass
class Response:
    nodal_force: np.matrix
    yield_components_force: np.matrix
    nodal_moments: np.matrix
    nodal_strains: np.matrix = np.matrix(np.zeros([1, 1]))
    nodal_stresses: np.matrix = np.matrix(np.zeros([1, 1]))


@dataclass
class Moment:
    x: float
    y: float
    xy: float


class YieldSpecs:
    def __init__(self, section: PlateSection, points_count: int):
        self.section = section
        self.points_count = points_count
        self.components_count = self.points_count * section.yield_specs.components_count
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


class PlateMember:
    # calculations is based on four gauss points
    def __init__(self, num: int, section: PlateSection, element_type: str, nodes: tuple):
        self.num = num
        self.section = section
        self.element_type = element_type  # Q4, Q4R, Q8, Q8R
        self.nodes = nodes
        self.nodes_count = len(self.nodes)
        self.node_dofs_count = 3
        self.dofs_count = self.node_dofs_count * self.nodes_count
        self.gauss_points_count = len(self.gauss_points)
        self.yield_specs = YieldSpecs(section=self.section, points_count=self.gauss_points_count)
        self.k = self.get_stiffness()
        self.t = self.get_transform()
        self.m = None
        # udef: unit distorsions equivalent forces (force, moment, ...) in nodes
        # udet: unit distorsions equivalent tractions (stress, force, moment, ...) in gauss points
        self.udefs, self.udets = self.get_nodal_forces_from_unit_distortions()

    @property
    def gauss_points(self):
        if self.element_type == "Q4R":
            gauss_points = [
                GaussPoint(weight=2, r=0, s=0),
            ]
        elif self.element_type in ("Q4", "Q8R"):
            gauss_points = [
                GaussPoint(weight=1, r=-0.57735027, s=-0.57735027),
                GaussPoint(weight=1, r=+0.57735027, s=-0.57735027),
                GaussPoint(weight=1, r=+0.57735027, s=+0.57735027),
                GaussPoint(weight=1, r=-0.57735027, s=+0.57735027),
            ]
        elif self.element_type == "Q8":
            gauss_points = [
                GaussPoint(weight=0.88888888, r=0, s=0),
                GaussPoint(weight=0.55555555, r=-0.77459667, s=-0.77459667),
                GaussPoint(weight=0.55555555, r=0, s=-0.77459667),
                GaussPoint(weight=0.55555555, r=+0.77459667, s=-0.77459667),
                GaussPoint(weight=0.55555555, r=+0.77459667, s=0),
                GaussPoint(weight=0.55555555, r=+0.77459667, s=+0.77459667),
                GaussPoint(weight=0.55555555, r=0, s=+0.77459667),
                GaussPoint(weight=0.55555555, r=-0.77459667, s=+0.77459667),
                GaussPoint(weight=0.55555555, r=-0.77459667, s=0),
            ]
        return gauss_points

    @property
    def natural_nodes(self):
        if self.element_type in ("Q4", "Q4R"):
            natural_nodes = [
                NaturalPoint(r=-1, s=-1),
                NaturalPoint(r=+1, s=-1),
                NaturalPoint(r=+1, s=+1),
                NaturalPoint(r=-1, s=+1),
            ]
        elif self.element_type in ("Q8", "Q8R"):
            natural_nodes = [
                NaturalPoint(r=-1, s=-1),
                NaturalPoint(r=0, s=-1),
                NaturalPoint(r=1, s=-1),
                NaturalPoint(r=1, s=0),
                NaturalPoint(r=1, s=1),
                NaturalPoint(r=0, s=1),
                NaturalPoint(r=-1, s=1),
                NaturalPoint(r=-1, s=0),
            ]
        return natural_nodes

    # REF: Cook (2002), p230.
    def get_extrapolated_natural_point(self, natural_point):
        if self.element_type == "Q4R":
            # TODO: write extrapolations for Q4R and Q8 integration schemes
            extrapolated_point = None
        elif self.element_type in ("Q4", "Q8R"):
            extrapolated_point = NaturalPoint(
                r=np.sqrt(3) * natural_point.r,
                s=np.sqrt(3) * natural_point.s,
            )
        if self.element_type == "Q8":
            extrapolated_point = None
        return extrapolated_point

    def get_extrapolation_shape_functions(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        if self.element_type in ("Q4", "Q8R"):
            n = np.matrix([
                0.25 * (1 - r) * (1 - s),
                0.25 * (1 + r) * (1 - s),
                0.25 * (1 + r) * (1 + s),
                0.25 * (1 - r) * (1 + s),
            ])
        elif self.element_type in ("Q8"):
            # TODO: correct shape functions for Q8 and Q4R gauss points
            n1 = 0.5 * (1 - r ** 2) * (1 - s)
            n3 = 0.5 * (1 + r) * (1 - s ** 2)
            n5 = 0.5 * (1 - r ** 2) * (1 + s)
            n7 = 0.5 * (1 - r) * (1 - s ** 2)

            n0 = 0.25 * (1 - r) * (1 - s) - 0.5 * (n7 + n1)
            n2 = 0.25 * (1 + r) * (1 - s) - 0.5 * (n1 + n3)
            n4 = 0.25 * (1 + r) * (1 + s) - 0.5 * (n3 + n5)
            n6 = 0.25 * (1 - r) * (1 + s) - 0.5 * (n5 + n7)

            n = np.matrix([n0, n1, n2, n3, n4, n5, n6, n7])
        return n

    def get_jacobian(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        nodes = self.nodes
        if self.element_type in ("Q4", "Q4R"):
            j = 0.25 * np.matrix([
                [-(1 - s), (1 - s), (1 + s), -(1 + s)],
                [-(1 - r), -(1 + r), (1 + r), (1 - r)],
            ]) * np.matrix([
                [nodes[0].x, nodes[0].y],
                [nodes[1].x, nodes[1].y],
                [nodes[2].x, nodes[2].y],
                [nodes[3].x, nodes[3].y],
            ])
        elif self.element_type in ("Q8", "Q8R"):
            j = 0.25 * np.matrix([
                [-2 * r * s + 2 * r - s ** 2 + s, 4 * r * (s - 1), -2 * r * s + 2 * r + s ** 2 - s, 2 - 2 * s ** 2, 2 * r * s + 2 * r + s ** 2 + s, -4 * r * (s + 1), 2 * r * s + 2 * r - s ** 2 - s, 2 * s ** 2 - 2],
                [-r ** 2 - 2 * r * s + r + 2 * s, 2 * r ** 2 - 2, -r ** 2 + 2 * r * s - r + 2 * s, -4 * s * (r + 1), r ** 2 + 2 * r * s + r + 2 * s, 2 - 2 * r ** 2, r ** 2 - 2 * r * s - r + 2 * s, 4 * s * (r - 1)],
            ]) * np.matrix([
                [nodes[0].x, nodes[0].y],
                [nodes[1].x, nodes[1].y],
                [nodes[2].x, nodes[2].y],
                [nodes[3].x, nodes[3].y],
                [nodes[4].x, nodes[4].y],
                [nodes[5].x, nodes[5].y],
                [nodes[6].x, nodes[6].y],
                [nodes[7].x, nodes[7].y],
            ])
        return j

    @lru_cache
    def get_shape_derivatives(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        j = self.get_jacobian(natural_point)
        b = np.matrix(np.zeros((5, 3 * self.nodes_count)))

        if self.element_type in ("Q4", "Q4R"):
            n = 0.25 * np.matrix([
                [(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)],
            ])
            dn = 0.25 * np.linalg.inv(j) * np.matrix([
                [-(1 - s), +(1 - s), +(1 + s), -(1 + s)],
                [-(1 - r), -(1 + r), +(1 + r), +(1 - r)],
            ])

        elif self.element_type in ("Q8", "Q8R"):
            n2 = 0.5 * (1 - r ** 2) * (1 - s)
            n4 = 0.5 * (1 + r) * (1 - s ** 2)
            n6 = 0.5 * (1 - r ** 2) * (1 + s)
            n8 = 0.5 * (1 - r) * (1 - s ** 2)
            n1 = 0.25 * (1 - r) * (1 - s) - 0.5 * (n8 + n2)
            n3 = 0.25 * (1 + r) * (1 - s) - 0.5 * (n2 + n4)
            n5 = 0.25 * (1 + r) * (1 + s) - 0.5 * (n4 + n6)
            n7 = 0.25 * (1 - r) * (1 + s) - 0.5 * (n6 + n8)
            n = np.matrix([[n1, n2, n3, n4, n5, n6, n7, n8]])

            dn = 0.25 * np.linalg.inv(j) * np.matrix([
                [-2 * r * s + 2 * r - s ** 2 + s, 4 * r * (s - 1), -2 * r * s + 2 * r + s ** 2 - s, 2 - 2 * s ** 2, 2 * r * s + 2 * r + s ** 2 + s, -4 * r * (s + 1), 2 * r * s + 2 * r - s ** 2 - s, 2 * s ** 2 - 2],
                [-r ** 2 - 2 * r * s + r + 2 * s, 2 * r ** 2 - 2, -r ** 2 + 2 * r * s - r + 2 * s, -4 * s * (r + 1), r ** 2 + 2 * r * s + r + 2 * s, 2 - 2 * r ** 2, r ** 2 - 2 * r * s - r + 2 * s, 4 * s * (r - 1)],
            ])

        for i in range(self.nodes_count):
            b[0, 3 * (i + 1) - 1] = dn[0, i]
            b[1, 3 * (i + 1) - 2] = -dn[1, i]
            b[2, 3 * (i + 1) - 1] = dn[1, i]
            b[2, 3 * (i + 1) - 2] = -dn[0, i]
            b[3, 3 * (i + 1) - 3] = dn[0, i]
            b[3, 3 * (i + 1) - 1] = n[0, i]
            b[4, 3 * (i + 1) - 3] = dn[1, i]
            b[4, 3 * (i + 1) - 2] = -n[0, i]

        return b

    def get_stiffness(self):
        k = np.matrix(np.zeros((self.dofs_count, self.dofs_count)))
        for gauss_point in self.gauss_points:
            b = self.get_shape_derivatives(gauss_point)
            j = self.get_jacobian(gauss_point)
            j_det = np.linalg.det(j)
            gauss_point_k = gauss_point.weight * b.T * self.section.d * b * j_det
            k += gauss_point_k
        return k

    def get_transform(self):
        return np.matrix(np.eye(self.dofs_count))

    def get_nodal_moments(self, nodal_disp, fixed_internal):
        nodal_moments = np.matrix(np.zeros((3 * self.nodes_count, 1)))
        i = 0
        for natural_node in self.natural_nodes:
            natural_point_moment = self.get_natural_point_moment(natural_node, nodal_disp, fixed_internal)
            nodal_moments[i, 0] = natural_point_moment.x
            nodal_moments[i + 1, 0] = natural_point_moment.y
            nodal_moments[i + 2, 0] = natural_point_moment.xy
            i += 3
        return nodal_moments

    def get_natural_point_moment(self, natural_point, nodal_disp, fixed_internal):
        extrapolated_natural_point = self.get_extrapolated_natural_point(natural_point)
        shape_functions = self.get_extrapolation_shape_functions(extrapolated_natural_point)
        gauss_points_moments = self.get_gauss_points_moments(nodal_disp)

        if fixed_internal.any():
            for i in range(self.gauss_points_count):
                gauss_points_moments[i, :] += fixed_internal[3 * i:3 * (i + 1), 0].T

        natural_point_moment = np.dot(gauss_points_moments.T, shape_functions.T)
        return Moment(x=natural_point_moment[0, 0], y=natural_point_moment[1, 0], xy=natural_point_moment[2, 0])

    def get_gauss_points_moments(self, nodal_disp):
        gauss_points_moments = np.matrix(np.zeros((self.gauss_points_count, 3)))
        for i, gauss_point in enumerate(self.gauss_points):
            gauss_points_moments[i, 0] = self.get_gauss_point_moment(gauss_point, nodal_disp).x
            gauss_points_moments[i, 1] = self.get_gauss_point_moment(gauss_point, nodal_disp).y
            gauss_points_moments[i, 2] = self.get_gauss_point_moment(gauss_point, nodal_disp).xy
        return gauss_points_moments

    def get_gauss_point_moment(self, gauss_point, nodal_disp):
        gauss_point_b = self.get_shape_derivatives(gauss_point)
        m = self.section.d * gauss_point_b * nodal_disp
        return Moment(x=m[0, 0], y=m[1, 0], xy=m[2, 0])

    def get_yield_components_force(self, nodal_disp):
        yield_components_force = np.matrix(np.zeros((3 * self.gauss_points_count, 1)))
        i = 0
        for gauss_point in self.gauss_points:
            yield_components_force[i, 0] = self.get_gauss_point_moment(gauss_point, nodal_disp).x
            yield_components_force[i + 1, 0] = self.get_gauss_point_moment(gauss_point, nodal_disp).y
            yield_components_force[i + 2, 0] = self.get_gauss_point_moment(gauss_point, nodal_disp).xy
            i += 3
        return yield_components_force

    def get_unit_distortion(self, gauss_point_component_num):
        distortion = np.matrix(np.zeros((5, 1)))
        distortion[gauss_point_component_num, 0] = 1
        return distortion

    # for element with linear variation of stress
    # REF: Cook (2002), p228.
    def get_nodal_force_from_unit_distortion(self, gauss_point, gauss_point_component_num):
        gauss_point_b = self.get_shape_derivatives(gauss_point)
        distortion = self.get_unit_distortion(gauss_point_component_num)
        j = self.get_jacobian(gauss_point)
        j_det = np.linalg.det(j)
        nodal_force = gauss_point_b.T * self.section.d * distortion * j_det
        gauss_point_moment = self.section.d * distortion
        return nodal_force, gauss_point_moment[0:self.section.yield_specs.components_count]

    def get_nodal_forces_from_unit_distortions(self):
        nodal_forces = np.matrix(np.zeros((self.dofs_count, self.yield_specs.components_count)))
        gauss_points_moments = np.matrix(np.zeros((self.yield_specs.components_count, self.yield_specs.components_count)))
        component_base_num = 0
        for gauss_point in self.gauss_points:
            for j in range(3):
                nodal_forces[:, component_base_num + j] = self.get_nodal_force_from_unit_distortion(gauss_point=gauss_point, gauss_point_component_num=j)[0]
                gauss_points_moments[component_base_num:(component_base_num + 3), component_base_num + j] = self.get_nodal_force_from_unit_distortion(gauss_point=gauss_point, gauss_point_component_num=j)[1]
            component_base_num += 3
        return nodal_forces, gauss_points_moments

    def get_response(self, nodal_disp, fixed_external=None, fixed_internal=None):
        # fixed internal: fixed internal tractions like stress, force, moment, ... in gauss points of a member
        # fixed external: fixed external forces like force, moment, ... nodes of a member

        if fixed_external is None:
            fixed_external = np.matrix(np.zeros((self.dofs_count, 1)))

        if fixed_internal is None:
            fixed_internal = np.matrix(np.zeros((self.yield_specs.components_count, 1)))

        if fixed_external.any():
            nodal_force = self.k * nodal_disp + fixed_external
        else:
            nodal_force = self.k * nodal_disp

        response = Response(
            nodal_force=nodal_force,
            yield_components_force=self.get_yield_components_force(nodal_disp),
            nodal_moments=self.get_nodal_moments(nodal_disp, fixed_internal),
        )
        return response
