import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from ..points import Node, GaussPoint
from ..sections.wall import WallSection


@dataclass
class Response:
    nodal_force: np.matrix
    yield_components_force: np.matrix
    internal_strains: np.matrix = np.matrix(np.zeros([1, 1]))
    internal_stresses: np.matrix = np.matrix(np.zeros([1, 1]))
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


@dataclass
class Stress:
    x: float
    y: float
    xy: float


class YieldSpecs:
    def __init__(self, section: WallSection, points_count: int):
        self.points_count = points_count
        self.components_count = self.points_count * section.yield_specs.components_count
        self.pieces_count = self.points_count * section.yield_specs.pieces_count


class WallMember:
    # calculations is based on four gauss points
    def __init__(self, section: WallSection, initial_nodes: tuple[Node, Node, Node, Node]):
        self.section = section
        self.nodes = initial_nodes
        self.nodes_count = len(self.nodes)
        self.dofs_count = 2 * self.nodes_count
        self.gauss_points_count = len(self.gauss_points)
        self.yield_specs = YieldSpecs(section=self.section, points_count=self.gauss_points_count)
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

    def get_gauss_point_shape_functions(self, gauss_point):
        r = gauss_point.r
        s = gauss_point.s
        n = np.matrix([0.25 * (1 - r) * (1 - s),
                       0.25 * (1 + r) * (1 - s),
                       0.25 * (1 + r) * (1 + s),
                       0.25 * (1 - r) * (1 + s),
                       ])
        return n

    @lru_cache
    def get_gauss_point_shape_derivatives(self, gauss_point):
        r = gauss_point.r
        s = gauss_point.s
        nodes = self.nodes
        x0 = nodes[0].x
        x1 = nodes[1].x
        x2 = nodes[2].x
        x3 = nodes[3].x
        y0 = nodes[0].y
        y1 = nodes[1].y
        y2 = nodes[2].y
        y3 = nodes[3].y
        b = np.matrix([[1.0*((r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) -
                        y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0, 1.0*(-(r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0, 1.0*((r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0, 1.0*(-(r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0], [0, 1.0*(-(r - 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1)
                        + y2*(r + 1) - y3*(r - 1))), 0, 1.0*((r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r -
                        1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0, 1.0*(-(r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0, 1.0*((r - 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))], [1.0*(-(r - 1)*(x0*(s
                        - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r
                        - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1)
                        - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1)
                        - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s
                        + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r - 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r +
                        1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))]])
        return b

    def get_det(self, gauss_point):
        r = gauss_point.r
        s = gauss_point.s
        nodes = self.nodes
        x0 = nodes[0].x
        x1 = nodes[1].x
        x2 = nodes[2].x
        x3 = nodes[3].x
        y0 = nodes[0].y
        y1 = nodes[1].y
        y2 = nodes[2].y
        y3 = nodes[3].y
        det = -(0.25 * x0 * (r - 1) + 0.25 * x1 * (-r - 1) + 0.25 * x2 * (r + 1) + 0.25 * x3 * (1 - r)) * (0.25 * y0 * (s - 1) + 0.25 * y1 * (1 - s) + 0.25 * y2 * (s + 1) + 0.25 * y3 * (-s - 1)) + (0.25 * x0 * (s - 1) + 0.25 * x1 * (1 - s) + 0.25 * x2 * (s + 1) + 0.25 * x3 * (-s - 1)) * (0.25 * y0 * (r - 1) + 0.25 * y1 * (-r - 1) + 0.25 * y2 * (r + 1) + 0.25 * y3 * (1 - r))
        return det

    def get_stiffness(self):
        k = np.matrix(np.zeros((self.dofs_count, self.dofs_count)))
        for gauss_point in self.gauss_points:
            gauss_point_b = self.get_gauss_point_shape_derivatives(gauss_point)
            gauss_point_det = self.get_det(gauss_point)
            gauss_point_k = gauss_point_b.T * self.section.ce * gauss_point_b * gauss_point_det * self.section.geometry.thickness
            k += gauss_point_k
        return k

    def get_transform(self):
        return np.matrix(np.eye(self.dofs_count))

    def get_gauss_point_strain(self, gauss_point, nodal_disp):
        gauss_point_b = self.get_gauss_point_shape_derivatives(gauss_point)
        e = gauss_point_b * nodal_disp
        return Strain(x=e[0, 0], y=e[1, 0], xy=e[2, 0])

    def get_gauss_point_stress(self, gauss_point, nodal_disp):
        gauss_point_b = self.get_gauss_point_shape_derivatives(gauss_point)
        s = self.section.ce * gauss_point_b * nodal_disp
        return Stress(x=s[0, 0], y=s[1, 0], xy=s[2, 0])

    def get_strains(self, nodal_disp):
        internal_strains = np.matrix(np.zeros((3 * self.gauss_points_count, 1)))
        i = 0
        for gauss_point in self.gauss_points:
            internal_strains[i, 0] = self.get_gauss_point_strain(gauss_point, nodal_disp).x
            internal_strains[i + 1, 0] = self.get_gauss_point_strain(gauss_point, nodal_disp).y
            internal_strains[i + 2, 0] = self.get_gauss_point_strain(gauss_point, nodal_disp).xy
            i += 3
        return internal_strains

    def get_stresses(self, nodal_disp):
        internal_stresses = np.matrix(np.zeros((3 * self.gauss_points_count, 1)))
        i = 0
        for gauss_point in self.gauss_points:
            internal_stresses[i, 0] = self.get_gauss_point_stress(gauss_point, nodal_disp).x
            internal_stresses[i + 1, 0] = self.get_gauss_point_stress(gauss_point, nodal_disp).y
            internal_stresses[i + 2, 0] = self.get_gauss_point_stress(gauss_point, nodal_disp).xy
            i += 3
        return internal_stresses

    def get_unit_distortion(self, gauss_point_component_num):
        distortion = np.matrix(np.zeros((3, 1)))
        distortion[gauss_point_component_num, 0] = 1
        return distortion

    # for element with linear variation of stress
    def get_nodal_force_from_unit_distortion(self, gauss_point, gauss_point_component_num):
        distortion = self.get_unit_distortion(gauss_point_component_num)
        gauss_point_b = self.get_gauss_point_shape_derivatives(gauss_point)
        f = gauss_point_b.T * self.section.ce * distortion # may need det
        # NOTE: forces are internal, so we must use negative sign:
        return -f

    def get_nodal_forces_from_unit_distortions(self):
        nodal_forces = np.matrix(np.zeros((self.dofs_count, self.yield_specs.components_count)))
        component_base_num = 0
        for gauss_point in self.gauss_points:
            for j in range(3):
                nodal_forces[:, component_base_num + j] = self.get_nodal_force_from_unit_distortion(gauss_point=gauss_point, gauss_point_component_num=j)
            component_base_num += 3
        return nodal_forces

    def get_response(self, nodal_disp, fixed_force=None):
        if fixed_force is None:
            fixed_force = np.matrix(np.zeros((self.dofs_count, 1)))

        if fixed_force.any():
            nodal_force = self.k * nodal_disp + fixed_force
        else:
            nodal_force = self.k * nodal_disp

        response = Response(
            nodal_force=nodal_force,
            yield_components_force=self.get_stresses(nodal_disp),
            internal_strains = self.get_strains(nodal_disp),
            internal_stresses = self.get_stresses(nodal_disp),
        )
        return response
