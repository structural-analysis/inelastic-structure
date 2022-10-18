import numpy as np

from ..points import Node, PlateGaussPoint
from ..sections.plate import PlateSection


class PlateElement:
    def __init__(self, num, section, size_x, size_y, nodes = tuple[Node, Node, Node, Node]):
        self.num = num
        self.section = section
        self.size_x = size_x
        self.size_y = size_y
        self.nodes = nodes
        self.nodes_num = len(self.nodes)
        self.gauss_points = self.get_gauss_points()
        self.k = self.get_stiffness()
        self.t = self.get_transform()

    def get_gauss_points():
        gauss_points = [
            PlateGaussPoint(r=-0.57735, s=-0.57735),
            PlateGaussPoint(r=+0.57735, s=-0.57735),
            PlateGaussPoint(r=+0.57735, s=+0.57735),
            PlateGaussPoint(r=-0.57735, s=+0.57735),
        ]
        return gauss_points

    def get_shape_functions(self, r, s):
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

    def get_shape_derivatives(self, r, s):
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

    def get_stiffness_integrand(self, r, s):
        b = self.get_shape_derivatives(r=r, s=s)
        ki = b.T * self.section.de * b
        return ki

    def get_stiffness(self):
        ax = (self.size_x / 2)
        ay = (self.size_y / 2)
        kin = np.matrix(np.zeros(3 * self.nodes_num, 3 * self.nodes_num))
        for point in self.gauss_points:
            kin += self.get_stiffness_integrand(r=point.r, s=point.s)
        k = kin * ax * ay
        return k

    def get_transform(self):
        return np.matrix(np.eye(3 * self.nodes_num))


class PlateElements:
    def __init__(self, z_coordinate, section, member_size: tuple[float, float], mesh_num: tuple[int, int]):
        self.z_coordinate = z_coordinate
        self.section = section
        self.count_x = mesh_num[0]
        self.count_y = mesh_num[1]
        self.count = self.count_x * self.count_y
        self.nodes_num_x = self.count_x + 1
        self.nodes_num_y = self.count_y + 1
        self.nodes_num = self.nodes_num_x * self.nodes_num_y
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

                top_right_node_num = bottom_right_node_num + self.nodes_num_x
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
                        nodes = (
                            Node(num=bottom_left_node_num, x=bottom_left_node_x, y=bottom_left_node_y, z=self.z_coordinate),
                            Node(num=bottom_right_node_num, x=bottom_right_node_x, y=bottom_right_node_y, z=self.z_coordinate),
                            Node(num=top_right_node_num, x=top_right_node_x, y=top_right_node_y, z=self.z_coordinate),
                            Node(num=top_left_node_num, x=top_left_node_x, y=top_left_node_y, z=self.z_coordinate),
                        )
                    )
                )
            element_num_base += self.count_x
            bottom_node_num_base += self.nodes_num_x
        return elements_list


class YieldSpecs:
    def __init__(self, section: PlateSection, member_points_num: int):
        self.points_num = member_points_num
        self.components_num = self.points_num * section.yield_specs.components_num
        self.pieces_num = self.points_num * section.yield_specs.pieces_num


class PlateMember:
    # calculations is based on four gauss points
    def __init__(self, section: PlateSection, initial_nodes: tuple[Node, Node, Node, Node], mesh_num: tuple[int, int]):
        # assume plate is flate in the 0 height.
        self.z_coordinate = 0
        self.section = section
        self.thickness = section.geometry.thickness
        self.initial_nodes = initial_nodes
        self.size_x = self.initial_nodes[1].x - self.initial_nodes[0].x
        self.size_y = self.initial_nodes[2].y - self.initial_nodes[1].y
        self.elements = PlateElements(
            z_coordinate=self.z_coordinate,
            section=self.section,
            member_size=(self.size_x, self.size_y),
            mesh_num=mesh_num,
        )

        self.total_dofs_num = self.nodes_num * 3
        self.nodes = self.get_nodes()
        self.elements_nodes = self.get_elements_nodes()

        self.element_gauss_points = self.get_element_gauss_points()
        self.element_gauss_points_num = len(self.element_gauss_points)
        self.gauss_points_num = self.element_gauss_points_num * self.elements_num
        self.yield_specs = YieldSpecs(section=self.section, member_points_num=self.gauss_points_num)

        self.k = self.get_stiffness()
        self.t = self.get_transform()
        self.m = None

    def get_nodes(self):
        nodes = []
        x = 0
        y = 0
        for i in range(self.nodes_num):
            nodes.append(Node(num=i, x=x, y=y, z=0))
            x = x + self.element_size_x
            if (i + 1) % self.nodes_num_x == 0:
                x = 0
                y = y + self.element_size_y
        return nodes

    def get_stiffness(self):
        k = np.zeros((3 * self.nodes_num, 3 * self.nodes_num))
        klix = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        klix = klix.astype(int)
        for i in range(self.elements_num):
            g1 = self.elements_nodes[i, 0].num
            g2 = self.elements_nodes[i, 1].num
            g3 = self.elements_nodes[i, 2].num
            g4 = self.elements_nodes[i, 3].num

            kgix = np.array([3 * g1, 3 * g1 + 1, 3 * g1 + 2,
                             3 * g2, 3 * g2 + 1, 3 * g2 + 2,
                             3 * g3, 3 * g3 + 1, 3 * g3 + 2,
                             3 * g4, 3 * g4 + 1, 3 * g4 + 2
                             ])

            kgix = kgix.astype(int)
            for i in range(3 * self.element_nodes_num):
                for j in range(3 * self.element_nodes_num):
                    k[kgix[i], kgix[j]] = k[kgix[i], kgix[j]] + self.ke[klix[i], klix[j]]
        return k

    def get_transform(self):
        return np.matrix(np.eye(3 * self.nodes_num))

    # NOTE: these three functions is intuitional and incomplete.
    def get_gauss_point_moments(self, r, s, d):
        # TODO: check for possible too much function call
        b = self.get_element_shape_derivatives(r=r, s=s)
        return self.section.de * b * d

    def get_element_gauss_points_moments(self, d):
        element_gauss_points_moments = np.matrix(np.zeros(3 * self.element_gauss_points_num, 1))
        i = 0
        for point in self.element_gauss_points:
            element_gauss_points_moments[i, 0] = self.get_gauss_point_moments(point.r, point.s, d)[0, 0]
            element_gauss_points_moments[i + 1, 0] = self.get_gauss_point_moments(point.r, point.s, d)[1, 0]
            element_gauss_points_moments[i + 2, 0] = self.get_gauss_point_moments(point.r, point.s, d)[2, 0]
            i +=3
        return element_gauss_points_moments

    def get_gauss_points_moments(self):
        gauss_points_moments = np.matrix(np.zeros(3 * self.gauss_points_num, 1))
        for element in self.elements:
            gauss_points_moments[]
