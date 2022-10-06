import numpy as np

from ..points import Node
from ..sections.plate import PlateSection


class YieldSpecs:
    def __init__(self, section: PlateSection, member_points_num: int):
        self.points_num = member_points_num
        self.components_num = self.points_num * section.yield_specs.components_num
        self.pieces_num = self.points_num * section.yield_specs.pieces_num


class PlateMember:
    # k is calculated based on four integration points
    def __init__(self, section: PlateSection, initial_nodes: tuple[Node, Node, Node, Node], mesh_num: tuple[int, int]):
        self.section = section
        self.thickness = section.geometry.thickness
        self.initial_nodes = initial_nodes
        self.size_x = self.initial_nodes[1].x - self.initial_nodes[0].x
        self.size_y = self.initial_nodes[2].y - self.initial_nodes[1].y
        self.elements_num_x = mesh_num[0]
        self.elements_num_y = mesh_num[1]
        self.elements_num = self.elements_num_x * self.elements_num_y
        self.element_size_x = self.size_x / self.elements_num_x
        self.element_size_y = self.size_y / self.elements_num_y
        self.ke = self.get_element_stiffness()
        self.te = self.get_element_transform()

        self.nodes_num_x = self.elements_num_x + 1
        self.nodes_num_y = self.elements_num_y + 1
        self.nodes_num = self.nodes_num_x * self.nodes_num_y
        self.total_dofs_num = self.nodes_num * 3
        self.nodes = self.get_nodes()
        self.elements_nodes = self.get_elements_nodes()

        self.element_gauss_points_num = 4
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

    def get_elements_nodes(self):
        s = 0
        empty_elements_nodes = np.zeros((self.elements_num, 4))
        elements_nodes = np.matrix(empty_elements_nodes).astype(Node)
        for i in range(self.elements_num):
            botton_left_node_num = i + s
            bottom_right_node_num = botton_left_node_num + 1
            top_right_node_num = bottom_right_node_num + self.nodes_num_x
            top_left_node_num = top_right_node_num - 1
            elements_nodes[i, 0] = self.nodes[botton_left_node_num]
            elements_nodes[i, 1] = self.nodes[bottom_right_node_num]
            elements_nodes[i, 2] = self.nodes[top_right_node_num]
            elements_nodes[i, 3] = self.nodes[top_left_node_num]

            if (i + 1) % self.elements_num_x == 0:
                s += 1
        return elements_nodes

    # def get_nodes_coordinates(self):
    #     x = 0
    #     y = 0
    #     nodes_coordinates = np.zeros((self.self.nodes_num, 2))
    #     for i in range(self.self.nodes_num):
    #         nodes_coordinates[i, 0] = x
    #         nodes_coordinates[i, 1] = y
    #         x = x + self.element_size_x
    #         if (i + 1) % self.nodes_num_x == 0:
    #             x = 0
    #             y = y + self.element_size_y
    #     return nodes_coordinates

    def get_element_shape_functions(self, r, s):
        ax = (self.element_size_x / 2)
        ay = (self.element_size_y / 2)
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

    def get_element_shape_derivatives(self, r, s):
        ax = (self.element_size_x / 2)
        ay = (self.element_size_y / 2)
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

    def get_element_stiffness_integrand(self, r, s):
        b = self.get_element_shape_derivatives(r=r, s=s)
        ki = b.T * self.section.de * b
        return ki

    def get_element_stiffness(self):
        ax = (self.element_size_x / 2)
        ay = (self.element_size_y / 2)
        kin = self.get_element_stiffness_integrand(r=-0.57735, s=-0.57735) + \
            self.get_element_stiffness_integrand(r=+0.57735, s=-0.57735) + \
            self.get_element_stiffness_integrand(r=+0.57735, s=+0.57735) + \
            self.get_element_stiffness_integrand(r=-0.57735, s=+0.57735)
        ke = kin * ax * ay
        return ke

    def get_element_transform(self):
        return np.matrix(np.eye(12))

    def get_stiffness(self):
        k = np.zeros((3 * self.nodes_num, 3 * self.nodes_num))
        klix = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        klix = klix.astype(int)
        for i in range(self.elements_num):
            g1 = self.elements_nodes[i, 0].num
            g2 = self.elements_nodes[i, 1].num
            g3 = self.elements_nodes[i, 2].num
            g4 = self.elements_nodes[i, 3].num

            kgix = np.array([3 * g1 - 3, 3 * g1 - 2, 3 * g1 - 1,
                             3 * g2 - 3, 3 * g2 - 2, 3 * g2 - 1,
                             3 * g3 - 3, 3 * g3 - 2, 3 * g3 - 1,
                             3 * g4 - 3, 3 * g4 - 2, 3 * g4 - 1,
                             ])

            kgix = kgix.astype(int)
            for i in range(12):
                for j in range(12):
                    k[kgix[i], kgix[j]] = k[kgix[i], kgix[j]] + self.ke[klix[i], klix[j]]
        return k

    def get_transform(self):
        return np.matrix(np.eye(3 * self.nodes_num))
