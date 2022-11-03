import numpy as np

from ..points import Node, PlateGaussPoint
from ..sections.plate import PlateSection


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
        self.gauss_points = self.get_gauss_points()
        self.gauss_points_count = len(self.gauss_points)
        self.yield_specs = YieldSpecs(section=self.section, points_count=self.gauss_points_count)

        self.k = self.get_stiffness()
        self.t = self.get_transform()
        self.m = None

        self.udefs = self.get_nodal_forces_from_unit_curvatures()

    def get_gauss_points(self):
        gauss_points = [
            PlateGaussPoint(r=-0.57735, s=-0.57735),
            PlateGaussPoint(r=+0.57735, s=-0.57735),
            PlateGaussPoint(r=+0.57735, s=+0.57735),
            PlateGaussPoint(r=-0.57735, s=+0.57735),
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

    def get_stiffness_integrand(self, gauss_point):
        b = self.get_shape_derivatives(gauss_point)
        ki = b.T * self.section.de * b
        return ki

    def get_stiffness(self):
        ax = (self.size_x / 2)
        ay = (self.size_y / 2)
        kin = np.matrix(np.zeros((self.dofs_count, self.dofs_count)))
        for gauss_point in self.gauss_points:
            kin += self.get_stiffness_integrand(gauss_point)
        k = kin * ax * ay
        return k

    def get_transform(self):
        return np.matrix(np.eye(self.dofs_count))

    def get_gauss_point_forces(self, gauss_point, nodal_disp):
        b = self.get_shape_derivatives(gauss_point)
        return self.section.de * b * nodal_disp

    def get_yield_components_force(self, nodal_disp):
        yield_components_force = np.matrix(np.zeros((self.yield_specs.components_count, 1)))
        i = 0
        for gauss_point in self.gauss_points:
            yield_components_force[i, 0] = self.get_gauss_point_forces(gauss_point, nodal_disp)[0, 0]
            yield_components_force[i + 1, 0] = self.get_gauss_point_forces(gauss_point, nodal_disp)[1, 0]
            yield_components_force[i + 2, 0] = self.get_gauss_point_forces(gauss_point, nodal_disp)[2, 0]
            i += 3
        return yield_components_force

    def get_unit_curvature(self, gauss_point_component_num):
        curvature = np.matrix(np.zeros((3, 1)))
        curvature[gauss_point_component_num, 0] = 1
        return curvature

    # for element with linear variation of moments
    def get_nodal_force_from_unit_curvature(self, gauss_point, gauss_point_component_num):
        ax = (self.size_x / 2)
        ay = (self.size_y / 2)
        b = self.get_shape_derivatives(gauss_point)
        curvature = self.get_unit_curvature(gauss_point_component_num)
        f = b.T * self.section.de * curvature * ax * ay
        # NOTE: forces are internal, so we must use negative sign:
        return -f

    def get_nodal_forces_from_unit_curvatures(self):
        nodal_forces = np.matrix(np.zeros((self.dofs_count, self.yield_specs.components_count)))
        component_base_num = 0
        for gauss_point in self.gauss_points:
            for j in range(3):
                nodal_forces[:, component_base_num + j] = self.get_nodal_force_from_unit_curvature(gauss_point=gauss_point, gauss_point_component_num=j)
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
        self.thickness = section.geometry.thickness
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

        self.udefs = self.get_nodal_forces_from_unit_curvatures()

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

    def get_elements_nodal_disps(self, nodal_disp):
        elements_nodal_disps = []
        for element in self.elements.list:
            element_nodal_disps = np.matrix(np.zeros((3 * element.nodes_count, 1)))
            i = 0
            for node in element.nodes:
                element_nodal_disps[i, 0] = nodal_disp[3 * node.num]
                element_nodal_disps[i + 1, 0] = nodal_disp[3 * node.num + 1]
                element_nodal_disps[i + 2, 0] = nodal_disp[3 * node.num + 2]
                i += 3
            elements_nodal_disps.append(element_nodal_disps)
        return elements_nodal_disps

    # TODO: fixed_forces?
    def get_nodal_force(self, nodal_disp):
        # nodal_disp: numpy matrix
        nodal_force = self.k * nodal_disp
        return nodal_force

    def get_yield_components_force(self, nodal_disp):
        elements_nodal_disps = self.get_elements_nodal_disps(nodal_disp)
        yield_components_force = np.matrix(np.zeros((self.yield_specs.components_count, 1)))
        for i, element in enumerate(self.elements.list):
            element_yield_components_count = element.yield_specs.components_count
            start_index = i * element_yield_components_count
            end_index = (i + 1) * element_yield_components_count
            yield_components_force[start_index:end_index, 0] = element.get_yield_components_force(elements_nodal_disps[i])
        return yield_components_force

    def get_nodal_forces_from_unit_curvatures(self):
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
