import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from ..points import Node, NaturalPoint
from ..sections.wall import WallSection


@dataclass
class Response:
    nodal_force: np.matrix
    yield_components_force: np.matrix
    nodal_strains: np.matrix
    nodal_stresses: np.matrix
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
            NaturalPoint(r=-0.57735, s=-0.57735),
            NaturalPoint(r=+0.57735, s=-0.57735),
            NaturalPoint(r=+0.57735, s=+0.57735),
            NaturalPoint(r=-0.57735, s=+0.57735),
        ]
        return gauss_points

    @property
    def natural_nodes(self):
        natural_nodes = [
            NaturalPoint(r=-1, s=-1),
            NaturalPoint(r=+1, s=-1),
            NaturalPoint(r=+1, s=+1),
            NaturalPoint(r=-1, s=+1),
        ]
        return natural_nodes

    # REF: Cook (2002), p230.
    def get_extrapolated_natural_point(self, natural_point):
        return NaturalPoint(
            r = np.sqrt(3) * natural_point.r,
            s = np.sqrt(3) * natural_point.s,
        )

    def get_shape_functions(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        n = np.matrix([0.25 * (1 - r) * (1 - s),
                       0.25 * (1 + r) * (1 - s),
                       0.25 * (1 + r) * (1 + s),
                       0.25 * (1 - r) * (1 + s),
                       ])
        return n

    @lru_cache
    def get_natural_point_shape_derivatives(self, natural_point):
        r = natural_point.r
        s = natural_point.s
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

    def get_jacobian_det(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        nodes = self.nodes
        x0 = nodes[0].x
        x1 = nodes[1].x
        x2 = nodes[2].x
        x3 = nodes[3].x
        y0 = nodes[0].y
        y1 = nodes[1].y
        y2 = nodes[2].y
        y3 = nodes[3].y
        jacobian_det = -(0.25 * x0 * (r - 1) + 0.25 * x1 * (-r - 1) + 0.25 * x2 * (r + 1) + 0.25 * x3 * (1 - r)) * (0.25 * y0 * (s - 1) + 0.25 * y1 * (1 - s) + 0.25 * y2 * (s + 1) + 0.25 * y3 * (-s - 1)) + (0.25 * x0 * (s - 1) + 0.25 * x1 * (1 - s) + 0.25 * x2 * (s + 1) + 0.25 * x3 * (-s - 1)) * (0.25 * y0 * (r - 1) + 0.25 * y1 * (-r - 1) + 0.25 * y2 * (r + 1) + 0.25 * y3 * (1 - r))
        return jacobian_det

    def get_stiffness(self):
        k = np.matrix(np.zeros((self.dofs_count, self.dofs_count)))
        for gauss_point in self.gauss_points:
            gauss_point_b = self.get_natural_point_shape_derivatives(gauss_point)
            gauss_point_det = self.get_jacobian_det(gauss_point)
            gauss_point_k = gauss_point_b.T * self.section.ce * gauss_point_b * gauss_point_det * self.section.geometry.thickness
            k += gauss_point_k
        return k

    def get_transform(self):
        return np.matrix(np.eye(self.dofs_count))

    def get_nodal_strains(self, nodal_disp):
        nodal_strains = np.matrix(np.zeros((3 * self.nodes_count, 1)))
        i = 0
        for natural_node in self.natural_nodes:
            nodal_strains[i, 0] = self.get_natural_point_strain(natural_node, nodal_disp).x
            nodal_strains[i + 1, 0] = self.get_natural_point_strain(natural_node, nodal_disp).y
            nodal_strains[i + 2, 0] = self.get_natural_point_strain(natural_node, nodal_disp).xy
            i += 3
        return nodal_strains

    def get_nodal_stresses(self, nodal_disp):
        nodal_stresses = np.matrix(np.zeros((3 * self.nodes_count, 1)))
        i = 0
        for natural_node in self.natural_nodes:
            nodal_stresses[i, 0] = self.get_natural_point_stress(natural_node, nodal_disp).x
            nodal_stresses[i + 1, 0] = self.get_natural_point_stress(natural_node, nodal_disp).y
            nodal_stresses[i + 2, 0] = self.get_natural_point_stress(natural_node, nodal_disp).xy
            i += 3
        return nodal_stresses

    def get_natural_point_strain(self, natural_point, nodal_disp):
        extrapolated_natural_point = self.get_extrapolated_natural_point(natural_point)
        shape_functions = self.get_shape_functions(extrapolated_natural_point)
        gauss_points_strains = self.get_gauss_points_strains(nodal_disp)
        strain = np.dot(gauss_points_strains.T, shape_functions.T)
        return Strain(x=strain[0, 0], y=strain[1, 0], xy=strain[2, 0])

    def get_natural_point_stress(self, natural_point, nodal_disp):
        extrapolated_natural_point = self.get_extrapolated_natural_point(natural_point)
        shape_functions = self.get_shape_functions(extrapolated_natural_point)
        gauss_points_stresses = self.get_gauss_points_stresses(nodal_disp)
        stress = np.dot(gauss_points_stresses.T, shape_functions.T)
        return Stress(x=stress[0, 0], y=stress[1, 0], xy=stress[2, 0])

    def get_gauss_points_strains(self, nodal_disp):
        gauss_points_strains = np.matrix(np.zeros((self.gauss_points_count, 3)))
        for i, gauss_point in enumerate(self.gauss_points):
            gauss_points_strains[i, 0] = self.get_gauss_point_strain(gauss_point, nodal_disp).x
            gauss_points_strains[i, 1] = self.get_gauss_point_strain(gauss_point, nodal_disp).y
            gauss_points_strains[i, 2] = self.get_gauss_point_strain(gauss_point, nodal_disp).xy
        return gauss_points_strains

    def get_gauss_points_stresses(self, nodal_disp):
        gauss_points_stresses = np.matrix(np.zeros((self.gauss_points_count, 3)))
        for i, gauss_point in enumerate(self.gauss_points):
            gauss_points_stresses[i, 0] = self.get_gauss_point_stress(gauss_point, nodal_disp).x
            gauss_points_stresses[i, 1] = self.get_gauss_point_stress(gauss_point, nodal_disp).y
            gauss_points_stresses[i, 2] = self.get_gauss_point_stress(gauss_point, nodal_disp).xy
        return gauss_points_stresses

    def get_gauss_point_strain(self, gauss_point, nodal_disp):
        gauss_point_b = self.get_natural_point_shape_derivatives(gauss_point)
        e = gauss_point_b * nodal_disp
        return Strain(x=e[0, 0], y=e[1, 0], xy=e[2, 0])

    def get_gauss_point_stress(self, gauss_point, nodal_disp):
        gauss_point_b = self.get_natural_point_shape_derivatives(gauss_point)
        s = self.section.ce * gauss_point_b * nodal_disp
        return Stress(x=s[0, 0], y=s[1, 0], xy=s[2, 0])

    def get_yield_components_force(self, nodal_disp):
        yield_components_force = np.matrix(np.zeros((3 * self.gauss_points_count, 1)))
        i = 0
        for gauss_point in self.gauss_points:
            yield_components_force[i, 0] = self.get_gauss_point_stress(gauss_point, nodal_disp).x
            yield_components_force[i + 1, 0] = self.get_gauss_point_stress(gauss_point, nodal_disp).y
            yield_components_force[i + 2, 0] = self.get_gauss_point_stress(gauss_point, nodal_disp).xy
            i += 3
        return yield_components_force

    def get_unit_distortion(self, gauss_point_component_num):
        distortion = np.matrix(np.zeros((3, 1)))
        distortion[gauss_point_component_num, 0] = 1
        return distortion

    # for element with linear variation of stress
    def get_nodal_force_from_unit_distortion(self, gauss_point, gauss_point_component_num):
        distortion = self.get_unit_distortion(gauss_point_component_num)
        gauss_point_b = self.get_natural_point_shape_derivatives(gauss_point)
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
            yield_components_force=self.get_yield_components_force(nodal_disp),
            nodal_strains = self.get_nodal_strains(nodal_disp),
            nodal_stresses = self.get_nodal_stresses(nodal_disp),
        )
        return response
