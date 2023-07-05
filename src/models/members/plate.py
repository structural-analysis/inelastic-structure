import numpy as np
from dataclasses import dataclass

from ..points import Node, NaturalPoint
from ..sections.plate import PlateSection
# from .plate_elements.mkq12_b_simple_new import get_mkq12_simple_new_shape_derivatives
# from .plate_elements.mkq12_b_complicated_new import get_mkq12_complicated_new_shape_derivatives
# from .plate_elements.mkq12_b_complicated_negative import get_mkq12_complicated_negative_shape_derivatives


@dataclass
class Response:
    nodal_force: np.matrix
    yield_components_force: np.matrix
    internal_moments: np.matrix
    top_internal_strains: np.matrix
    bottom_internal_strains: np.matrix
    top_internal_stresses: np.matrix
    bottom_internal_stresses: np.matrix
    nodal_strains: np.matrix = np.matrix(np.zeros([1, 1]))
    nodal_stresses: np.matrix = np.matrix(np.zeros([1, 1]))


@dataclass
class Moment:
    x: float
    y: float
    xy: float
    mises: float


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
    def __init__(self, section: PlateSection, points_count: int):
        self.points_count = points_count
        self.components_count = self.points_count * section.yield_specs.components_count
        self.pieces_count = self.points_count * section.yield_specs.pieces_count


class PlateElement:
    def __init__(self, num, section, size_x, size_y, nodes: tuple[Node, Node, Node, Node]):
        self.num = num
        self.section = section
        self.size_x = size_x
        self.size_y = size_y
        self.nodes = nodes
        self.nodes_count = len(self.nodes)
        self.dofs_count = 3 * self.nodes_count
        self.gauss_points_shape_derivatives = [self.get_shape_derivatives(gauss_point) for gauss_point in self.gauss_points]
        # self.gauss_points_shape_derivatives = [get_mkq12_simple_new_shape_derivatives(gauss_point, self.nodes) for gauss_point in self.gauss_points]
        # self.gauss_points_shape_derivatives = [get_mkq12_complicated_new_shape_derivatives(gauss_point, self.nodes) for gauss_point in self.gauss_points]
        # self.gauss_points_shape_derivatives = [get_mkq12_complicated_negative_shape_derivatives(gauss_point, self.nodes) for gauss_point in self.gauss_points]
        self.gauss_points_count = len(self.gauss_points)
        self.yield_specs = YieldSpecs(section=self.section, points_count=self.gauss_points_count)

        self.k = self.get_stiffness()
        self.t = self.get_transform()
        self.m = None

        self.udefs = self.get_nodal_forces_from_unit_curvatures()

    @property
    def gauss_points(self):
        gauss_points = [
            NaturalPoint(r=-0.57735, s=-0.57735),
            NaturalPoint(r=+0.57735, s=-0.57735),
            NaturalPoint(r=+0.57735, s=+0.57735),
            NaturalPoint(r=-0.57735, s=+0.57735),
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

    def get_gauss_point_moments(self, gauss_point_b, nodal_disp):
        m = self.section.de * gauss_point_b * nodal_disp
        mises = np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2 - m[0, 0] * m[1, 0] + 3 * m[2, 0] ** 2)
        return Moment(x=m[0, 0], y=m[1, 0], xy=m[2, 0], mises=mises)

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

    def get_internal_moments(self, nodal_disp):
        internal_moments = np.matrix(np.zeros((self.yield_specs.components_count + self.gauss_points_count, 1)))
        i = 0
        for gauss_point_b in self.gauss_points_shape_derivatives:
            internal_moments[i, 0] = self.get_gauss_point_moments(gauss_point_b, nodal_disp).x
            internal_moments[i + 1, 0] = self.get_gauss_point_moments(gauss_point_b, nodal_disp).y
            internal_moments[i + 2, 0] = self.get_gauss_point_moments(gauss_point_b, nodal_disp).xy
            internal_moments[i + 3, 0] = self.get_gauss_point_moments(gauss_point_b, nodal_disp).mises
            i += 4
        return internal_moments

    def get_top_internal_strains(self, nodal_disp):
        top_internal_strains = np.matrix(np.zeros((self.yield_specs.components_count + self.gauss_points_count, 1)))
        i = 0
        for gauss_point_b in self.gauss_points_shape_derivatives:
            top_internal_strains[i, 0] = self.get_top_gauss_point_strains(gauss_point_b, nodal_disp).x
            top_internal_strains[i + 1, 0] = self.get_top_gauss_point_strains(gauss_point_b, nodal_disp).y
            top_internal_strains[i + 2, 0] = self.get_top_gauss_point_strains(gauss_point_b, nodal_disp).xy
            top_internal_strains[i + 3, 0] = self.get_top_gauss_point_strains(gauss_point_b, nodal_disp).mises
            i += 4
        return top_internal_strains

    def get_bottom_internal_strains(self, nodal_disp):
        bottom_internal_strains = np.matrix(np.zeros((self.yield_specs.components_count + self.gauss_points_count, 1)))
        i = 0
        for gauss_point_b in self.gauss_points_shape_derivatives:
            bottom_internal_strains[i, 0] = self.get_bottom_gauss_point_strains(gauss_point_b, nodal_disp).x
            bottom_internal_strains[i + 1, 0] = self.get_bottom_gauss_point_strains(gauss_point_b, nodal_disp).y
            bottom_internal_strains[i + 2, 0] = self.get_bottom_gauss_point_strains(gauss_point_b, nodal_disp).xy
            bottom_internal_strains[i + 3, 0] = self.get_bottom_gauss_point_strains(gauss_point_b, nodal_disp).mises
            i += 4
        return bottom_internal_strains

    def get_top_internal_stresses(self, nodal_disp):
        top_internal_stresses = np.matrix(np.zeros((self.yield_specs.components_count + self.gauss_points_count, 1)))
        i = 0
        for gauss_point_b in self.gauss_points_shape_derivatives:
            top_internal_stresses[i, 0] = self.get_top_gauss_point_stresses(gauss_point_b, nodal_disp).x
            top_internal_stresses[i + 1, 0] = self.get_top_gauss_point_stresses(gauss_point_b, nodal_disp).y
            top_internal_stresses[i + 2, 0] = self.get_top_gauss_point_stresses(gauss_point_b, nodal_disp).xy
            top_internal_stresses[i + 3, 0] = self.get_top_gauss_point_stresses(gauss_point_b, nodal_disp).mises
            i += 4
        return top_internal_stresses

    def get_bottom_internal_stresses(self, nodal_disp):
        bottom_internal_stresses = np.matrix(np.zeros((self.yield_specs.components_count + self.gauss_points_count, 1)))
        i = 0
        for gauss_point_b in self.gauss_points_shape_derivatives:
            bottom_internal_stresses[i, 0] = self.get_bottom_gauss_point_stresses(gauss_point_b, nodal_disp).x
            bottom_internal_stresses[i + 1, 0] = self.get_bottom_gauss_point_stresses(gauss_point_b, nodal_disp).y
            bottom_internal_stresses[i + 2, 0] = self.get_bottom_gauss_point_stresses(gauss_point_b, nodal_disp).xy
            bottom_internal_stresses[i + 3, 0] = self.get_bottom_gauss_point_stresses(gauss_point_b, nodal_disp).mises
            i += 4
        return bottom_internal_stresses

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


class PlateElements:
    def __init__(self, z_coordinate, section, member_size: tuple[float, float], mesh_count: tuple[int, int]):
        self.z_coordinate = z_coordinate
        self.section = section
        self.count_x = mesh_count[0]
        self.count_y = mesh_count[1]
        self.count = self.count_x * self.count_y
        self.nodes_count_x = self.count_x + 1
        self.nodes_count_y = self.count_y + 1
        self.element_size_x = member_size[0] / self.count_x
        self.element_size_y = member_size[1] / self.count_y
        self.list = self.get_elements_list()

    def get_elements_list(self):
        elements_list = []
        bottom_node_num_base = 0
        element_num_base = 0
        for j in range(self.count_y):
            for i in range(self.count_x):
                element_num = i + j * element_num_base
                bottom_left_node_num = i + bottom_node_num_base
                bottom_left_node_x = i * self.element_size_x
                bottom_left_node_y = j * self.element_size_y

                bottom_right_node_num = bottom_left_node_num + 1
                bottom_right_node_x = bottom_left_node_x + self.element_size_x
                bottom_right_node_y = bottom_left_node_y

                top_right_node_num = bottom_right_node_num + self.nodes_count_x
                top_right_node_x = bottom_right_node_x
                top_right_node_y = bottom_right_node_y + self.element_size_y

                top_left_node_num = top_right_node_num - 1
                top_left_node_x = bottom_left_node_x
                top_left_node_y = top_right_node_y

                elements_list.append(
                    PlateElement(
                        num=element_num,
                        section=self.section,
                        size_x=self.element_size_x,
                        size_y=self.element_size_y,
                        nodes=(
                            Node(num=bottom_left_node_num, x=bottom_left_node_x, y=bottom_left_node_y, z=self.z_coordinate),
                            Node(num=bottom_right_node_num, x=bottom_right_node_x, y=bottom_right_node_y, z=self.z_coordinate),
                            Node(num=top_right_node_num, x=top_right_node_x, y=top_right_node_y, z=self.z_coordinate),
                            Node(num=top_left_node_num, x=top_left_node_x, y=top_left_node_y, z=self.z_coordinate),
                        )
                    )
                )
            element_num_base += self.count_x
            bottom_node_num_base += self.nodes_count_x
        return elements_list


class PlateMember:
    # calculations is based on four gauss points
    def __init__(self, section: PlateSection, initial_nodes: tuple[Node, Node, Node, Node], mesh_count: tuple[int, int]):
        # assume plate is flat in the 0 height.
        self.z_coordinate = 0
        self.section = section
        self.initial_nodes = initial_nodes
        self.size_x = self.initial_nodes[1].x - self.initial_nodes[0].x
        self.size_y = self.initial_nodes[2].y - self.initial_nodes[1].y
        self.elements = PlateElements(
            z_coordinate=self.z_coordinate,
            section=self.section,
            member_size=(self.size_x, self.size_y),
            mesh_count=mesh_count,
        )

        self.nodes = self.get_nodes()
        self.nodes_count = len(self.nodes)
        self.dofs_count = 3 * self.nodes_count

        self.gauss_points = self.get_gauss_points()
        self.gauss_points_count = len(self.gauss_points)
        self.yield_specs = YieldSpecs(section=self.section, points_count=self.gauss_points_count)

        self.k = self.get_stiffness()
        self.t = self.get_transform()
        self.m = None

        self.udefs = self.get_nodal_forces_from_unit_distortions()

    def get_nodes(self):
        nodes = []
        for element in self.elements.list:
            for node in element.nodes:
                nodes.append(node)
        return sorted(set(nodes))

    def get_gauss_points(self):
        points = []
        for element in self.elements.list:
            for point in element.gauss_points:
                points.append(point)
        return points

    def get_stiffness(self):
        k = np.zeros((self.dofs_count, self.dofs_count))
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

            for i in range(element.dofs_count):
                for j in range(element.dofs_count):
                    k[element_global_dofs[i], element_global_dofs[j]] = k[element_global_dofs[i], element_global_dofs[j]] + element.k[i, j]
        return k

    def get_transform(self):
        return np.matrix(np.eye(self.dofs_count))

    def get_elements_nodal_disp(self, nodal_disp):
        elements_nodal_disp = []
        for element in self.elements.list:
            element_nodal_disp = np.matrix(np.zeros((3 * element.nodes_count, 1)))
            i = 0
            for node in element.nodes:
                element_nodal_disp[i, 0] = nodal_disp[3 * node.num]
                element_nodal_disp[i + 1, 0] = nodal_disp[3 * node.num + 1]
                element_nodal_disp[i + 2, 0] = nodal_disp[3 * node.num + 2]
                i += 3
            elements_nodal_disp.append(element_nodal_disp)
        return elements_nodal_disp

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
