import numpy as np

from ..points import Node
from ..sections.plate import PlateSection


class YieldSpecs:
    def __init__(self, section: PlateSection):
        self.points_num = 4
        self.components_num = self.points_num * section.yield_specs.components_num
        self.pieces_num = self.points_num * section.yield_specs.pieces_num


class PlateMember:
    # k is calculated based on four integration points
    def __init__(self, section: PlateSection, nodes: tuple[Node, Node, Node, Node], mesh_num: tuple[int, int]):
        self.section = section
        self.t = section.geometry.t
        self.nodes = nodes
        self.member_size_x = nodes[1].x - nodes[0].x
        self.member_size_y = nodes[2].y - nodes[1].y
        self.elements_num_x = mesh_num[0]
        self.elements_num_y = mesh_num[1]
        self.elements_num = self.elements_num_x * self.elements_num_y
        self.element_size_x = self.member_size_x / self.elements_num_x
        self.element_size_y = self.member_size_y / self.elements_num_y
        self.ke = self.get_element_stiffness()

        self.member_nodes_num_x = self.elements_num_x + 1
        self.member_nodes_num_y = self.elements_num_y + 1
        self.member_nodes_num = self.member_nodes_num_x * self.member_nodes_num_y
        self.member_nodes = self.get_member_nodes()

        self.element_gauss_points_num = 4
        self.member_gauss_points_num = self.element_gauss_points_num * self.elements_num

        self.km = self.get_member_stiffness()

    def get_member_nodes(self):
        s = 0
        empty_member_nodes = np.zeros((self.elements_num, 4))
        member_nodes = np.matrix(empty_member_nodes)
        for i in range(self.elements_num):
            n1 = i + 1 + s
            n2 = n1 + 1
            n3 = n2 + self.member_nodes_num_x
            n4 = n3 - 1
            member_nodes[i, 0] = n1
            member_nodes[i, 1] = n2
            member_nodes[i, 2] = n3
            member_nodes[i, 3] = n4
            if (i + 1) % self.elements_num_x == 0:
                s += 1
        return member_nodes

    def get_member_nodes_coordinates(self):
        x = 0
        y = 0
        member_nodes_coordinates = np.zeros((self.self.member_nodes_num, 2))
        for i in range(self.self.member_nodes_num):
            member_nodes_coordinates[i, 0] = x
            member_nodes_coordinates[i, 1] = y
            x = x + self.element_size_x
            if (i + 1) % self.member_nodes_num_x == 0:
                x = 0
                y = y + self.element_size_y
        return member_nodes_coordinates

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

    def get_member_stiffness(self):
        km = np.zeros((3 * self.member_nodes_num, 3 * self.member_nodes_num))
        klix = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        klix = klix.astype(int)
        for k in range(self.elements_num):
            g1 = self.member_nodes[k, 0]
            g2 = self.member_nodes[k, 1]
            g3 = self.member_nodes[k, 2]
            g4 = self.member_nodes[k, 3]

            kgix = np.array([3 * g1 - 3, 3 * g1 - 2, 3 * g1 - 1,
                             3 * g2 - 3, 3 * g2 - 2, 3 * g2 - 1,
                             3 * g3 - 3, 3 * g3 - 2, 3 * g3 - 1,
                             3 * g4 - 3, 3 * g4 - 2, 3 * g4 - 1,
                             ])

            kgix = kgix.astype(int)
            for i in range(12):
                for j in range(12):
                    km[kgix[i], kgix[j]] = km[kgix[i], kgix[j]] + kl[klix[i], klix[j]]
        return km
