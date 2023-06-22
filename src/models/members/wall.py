import numpy as np
from dataclasses import dataclass

from ..points import Node, GaussPoint
from ..sections.wall import WallSection


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
        # self.udefs = self.get_nodal_forces_from_unit_distortions()

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

    def get_gauss_point_strain(self, gauss_point_b, nodal_disp):
        e = -(self.section.geometry.thickness / 2) * gauss_point_b * nodal_disp
        mises = np.sqrt(e[0, 0] ** 2 + e[1, 0] ** 2 - e[0, 0] * e[1, 0] + 3 * e[2, 0] ** 2)
        return Strain(x=e[0, 0], y=e[1, 0], xy=e[2, 0], mises=mises)

    def get_gauss_point_stress(self, gauss_point_b, nodal_disp):
        s = -(self.section.geometry.thickness / 2) * self.section.be * gauss_point_b * nodal_disp
        mises = np.sqrt(s[0, 0] ** 2 + s[1, 0] ** 2 - s[0, 0] * s[1, 0] + 3 * s[2, 0] ** 2)
        return Stress(x=s[0, 0], y=s[1, 0], xy=s[2, 0], mises=mises)

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

    def get_nodal_forces_from_unit_distortions(self):
        nodal_forces = np.matrix(np.zeros((self.dofs_count, self.yield_specs.components_count)))
        base_component_num = 0
        for element in self.elements.list:
            g0 = element.nodes[0].num
            g1 = element.nodes[1].num
            g2 = element.nodes[2].num
            g3 = element.nodes[3].num

            element_global_dofs = np.array([3 * g0, 3 * g0 + 1, 3 * g0 + 2,
                                            3 * g1, 3 * g1 + 1, 3 * g1 + 2,
                                            3 * g2, 3 * g2 + 1, 3 * g2 + 2,
                                            3 * g3, 3 * g3 + 1, 3 * g3 + 2
                                            ])

            for i in range(element.yield_specs.components_count):
                for j in range(element.dofs_count):
                    nodal_forces[element_global_dofs[j], base_component_num + i] = element.udefs[j, i]
            base_component_num += element.yield_specs.components_count
        return nodal_forces
