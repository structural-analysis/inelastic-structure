import numpy as np
from dataclasses import dataclass

from ..points import Node, GaussPoint
from ..sections.plate import WallSection


@dataclass
class Response:
    nodal_force: np.matrix
    yield_components_force: np.matrix
    internal_strains: np.matrix
    internal_stresses: np.matrix
    internal_moments: np.matrix = np.matrix(np.zeros([1, 1]))
    top_internal_strains: np.matrix = np.matrix(np.zeros([1, 1]))
    bottom_internal_strains: np.matrix = np.matrix(np.zeros([1, 1]))
    top_internal_stresses: np.matrix = np.matrix(np.zeros([1, 1]))
    bottom_internal_stresses: np.matrix = np.matrix(np.zeros([1, 1]))


@dataclass
class Strain:
    x: float
    y: float
    xy: float
    mises: float


@dataclass
class Stress:
    x: float
    y: float
    xy: float
    mises: float


class YieldSpecs:
    def __init__(self, section: WallSection, points_count: int):
        self.points_count = points_count
        self.components_count = self.points_count * section.yield_specs.components_count
        self.pieces_count = self.points_count * section.yield_specs.pieces_count


class WallMember:
    # calculations is based on four gauss points
    def __init__(self, section: WallSection, nodes: tuple[Node, Node, Node, Node]):
        self.section = section
        self.nodes = nodes
        self.nodes_count = len(self.nodes)
        self.dofs_count = 2 * self.nodes_count
        self.gauss_points_count = len(self.gauss_points)
        self.yield_specs = YieldSpecs(section=self.section, points_count=self.gauss_points_count)
        self.gauss_points_shape_derivatives = [self.get_shape_derivatives(gauss_point) for gauss_point in self.gauss_points]
        self.k = self.get_stiffness()
        self.t = self.get_transform()
        self.m = None
        # udef: unit distorsions equivalent forces
        self.udefs = self.get_nodal_forces_from_unit_distortions()

    @property
    def gauss_points(self):
        gauss_points = [
            GaussPoint(r=-0.57735, s=-0.57735),
            GaussPoint(r=+0.57735, s=-0.57735),
            GaussPoint(r=+0.57735, s=+0.57735),
            GaussPoint(r=-0.57735, s=+0.57735),
        ]
        return gauss_points

    def get_shape_functions(self, gauss_point):
        r = gauss_point.r
        s = gauss_point.s
        ax = (self.size_x / 2)
        ay = (self.size_y / 2)
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

    def get_shape_derivatives(self, gauss_point):
        r = gauss_point.r
        s = gauss_point.s
        ax = (self.size_x / 2)
        ay = (self.size_y / 2)
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

    def get_stiffness_integrand(self, gauss_point_b):
        ki = gauss_point_b.T * self.section.de * gauss_point_b
        return ki

    def get_stiffness(self):
        ax = (self.size_x / 2)
        ay = (self.size_y / 2)
        det = (ax * ay)
        kin = np.matrix(np.zeros((self.dofs_count, self.dofs_count)))
        for gauss_point_b in self.gauss_points_shape_derivatives:
            kin += self.get_stiffness_integrand(gauss_point_b)
        k = kin * det
        return k

    def get_transform(self):
        return np.matrix(np.eye(self.dofs_count))

    def get_top_gauss_point_strains(self, gauss_point_b, nodal_disp):
        e = -(self.section.geometry.thickness / 2) * gauss_point_b * nodal_disp
        mises = np.sqrt(e[0, 0] ** 2 + e[1, 0] ** 2 - e[0, 0] * e[1, 0] + 3 * e[2, 0] ** 2)
        return Strain(x=e[0, 0], y=e[1, 0], xy=e[2, 0], mises=mises)

    def get_bottom_gauss_point_strains(self, gauss_point_b, nodal_disp):
        e = (self.section.geometry.thickness / 2) * gauss_point_b * nodal_disp
        mises = np.sqrt(e[0, 0] ** 2 + e[1, 0] ** 2 - e[0, 0] * e[1, 0] + 3 * e[2, 0] ** 2)
        return Strain(x=e[0, 0], y=e[1, 0], xy=e[2, 0], mises=mises)

    def get_top_gauss_point_stresses(self, gauss_point_b, nodal_disp):
        s = -(self.section.geometry.thickness / 2) * self.section.be * gauss_point_b * nodal_disp
        mises = np.sqrt(s[0, 0] ** 2 + s[1, 0] ** 2 - s[0, 0] * s[1, 0] + 3 * s[2, 0] ** 2)
        return Stress(x=s[0, 0], y=s[1, 0], xy=s[2, 0], mises=mises)

    def get_bottom_gauss_point_stresses(self, gauss_point_b, nodal_disp):
        s = (self.section.geometry.thickness / 2) * self.section.be * gauss_point_b * nodal_disp
        mises = np.sqrt(s[0, 0] ** 2 + s[1, 0] ** 2 - s[0, 0] * s[1, 0] + 3 * s[2, 0] ** 2)
        return Stress(x=s[0, 0], y=s[1, 0], xy=s[2, 0], mises=mises)

    def get_yield_components_force(self, nodal_disp):
        yield_components_force = np.matrix(np.zeros((self.yield_specs.components_count, 1)))
        i = 0
        for gauss_point_b in self.gauss_points_shape_derivatives:
            yield_components_force[i, 0] = self.get_gauss_point_moments(gauss_point_b, nodal_disp).x
            yield_components_force[i + 1, 0] = self.get_gauss_point_moments(gauss_point_b, nodal_disp).y
            yield_components_force[i + 2, 0] = self.get_gauss_point_moments(gauss_point_b, nodal_disp).xy
            i += 3
        return yield_components_force

    def get_unit_curvature(self, gauss_point_component_num):
        curvature = np.matrix(np.zeros((3, 1)))
        curvature[gauss_point_component_num, 0] = 1
        return curvature

    # for element with linear variation of moments
    def get_nodal_force_from_unit_curvature(self, gauss_point_b, gauss_point_component_num):
        ax = (self.size_x / 2)
        ay = (self.size_y / 2)
        curvature = self.get_unit_curvature(gauss_point_component_num)
        f = gauss_point_b.T * self.section.de * curvature * ax * ay
        # NOTE: forces are internal, so we must use negative sign:
        return -f

    def get_nodal_forces_from_unit_curvatures(self):
        nodal_forces = np.matrix(np.zeros((self.dofs_count, self.yield_specs.components_count)))
        component_base_num = 0
        for gauss_point_b in self.gauss_points_shape_derivatives:
            for j in range(3):
                nodal_forces[:, component_base_num + j] = self.get_nodal_force_from_unit_curvature(gauss_point_b=gauss_point_b, gauss_point_component_num=j)
            component_base_num += 3
        return nodal_forces

    def get_response(self, nodal_disp, fixed_force=None):
        if fixed_force is None:
            fixed_force = np.matrix(np.zeros((self.dofs_count, 1)))

        if fixed_force.any():
            nodal_force = self.k * nodal_disp + fixed_force
        else:
            nodal_force = self.k * nodal_disp

        elements_nodal_disp = self.get_elements_nodal_disp(nodal_disp)
        yield_components_force = np.matrix(np.zeros((self.yield_specs.components_count, 1)))
        internal_moments = np.matrix(np.zeros((self.yield_specs.components_count + self.gauss_points_count, 1)))
        top_internal_strains = np.matrix(np.zeros((self.yield_specs.components_count + self.gauss_points_count, 1)))
        bottom_internal_strains = np.matrix(np.zeros((self.yield_specs.components_count + self.gauss_points_count, 1)))
        top_internal_stresses = np.matrix(np.zeros((self.yield_specs.components_count + self.gauss_points_count, 1)))
        bottom_internal_stresses = np.matrix(np.zeros((self.yield_specs.components_count + self.gauss_points_count, 1)))

        for i, element in enumerate(self.elements.list):
            yield_components_start_index = i * element.yield_specs.components_count
            yield_components_end_index = (i + 1) * element.yield_specs.components_count
            yield_components_force[yield_components_start_index:yield_components_end_index, 0] = element.get_yield_components_force(elements_nodal_disp[i])

            # these responses have 4 components including mises.
            start_index = i * (element.yield_specs.components_count + element.gauss_points_count)
            end_index = (i + 1) * (element.yield_specs.components_count + element.gauss_points_count)
            internal_moments[start_index:end_index, 0] = element.get_internal_moments(elements_nodal_disp[i])
            top_internal_strains[start_index:end_index, 0] = element.get_top_internal_strains(elements_nodal_disp[i])
            bottom_internal_strains[start_index:end_index, 0] = element.get_bottom_internal_strains(elements_nodal_disp[i])
            top_internal_stresses[start_index:end_index, 0] = element.get_top_internal_stresses(elements_nodal_disp[i])
            bottom_internal_stresses[start_index:end_index, 0] = element.get_bottom_internal_stresses(elements_nodal_disp[i])

        response = Response(
            nodal_force=nodal_force,
            yield_components_force=yield_components_force,
            internal_moments=internal_moments,
            top_internal_strains=top_internal_strains,
            bottom_internal_strains=bottom_internal_strains,
            top_internal_stresses=top_internal_stresses,
            bottom_internal_stresses=bottom_internal_stresses,
        )
        return response
