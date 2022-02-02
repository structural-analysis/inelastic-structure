import scipy.linalg
import numpy as np

from src.functions import sqrt


class Node:
    def __init__(self, num, x, y):
        self.num = num
        self.x = x
        self.y = y


class Material:
    def __init__(self, name):
        if name == "steel":
            self.e = 2e11
            self.sy = 240e6
            self.nu = 0.3


class FrameSection:
    def __init__(self, material: Material, a, ix, iy, zp, has_axial_yield: str):
        self.a = a
        self.ix = ix
        self.iy = iy
        self.zp = zp
        self.e = material.e
        self.sy = material.sy
        self.mp = self.zp * self.sy
        self.ap = self.a * self.sy
        self.has_axial_yield = True if has_axial_yield.lower() == "true" else False
        if not self.has_axial_yield:
            self.yield_components_num = 1
            self.phi = np.matrix([-1 / self.mp, 1 / self.mp])
        else:
            self.yield_components_num = 2
            self.phi = np.matrix([
                [1 / self.ap, 0, -1 / self.ap, -1 / self.ap, 0, 1 / self.ap],
                [(1 - 0.15 * self.ap) / self.mp, 1 / self.mp, (1 - 0.15 * self.ap) / self.mp, -(1 - 0.15 * self.ap) / self.mp, -1 / self.mp, (1 - 0.15 * self.ap) / self.mp]
            ])


class FrameElement2D:
    # mp: bending capacity
    # udef: unit distorsions equivalent forces
    # ends_fixity: one of following: fix_fix, hinge_fix, fix_hinge, hinge_hinge
    def __init__(self, nodes: tuple[Node, Node], ends_fixity, section: FrameSection):
        self.nodes = nodes
        # for frame elements yield points coincide on fem nodes
        self.yield_points = self.nodes
        self.start = nodes[0]
        self.end = nodes[1]
        self.ends_fixity = ends_fixity
        self.section = section
        self.a = section.a
        self.i = section.ix
        self.e = section.e
        self.mp = section.mp
        self.has_axial_yield = section.has_axial_yield
        self.total_ycn = 2 * section.yield_components_num
        self.l = self._length()
        self.k = self._stiffness()
        self.t = self._transform_matrix()
        self.udefs = self._udefs()

    def _length(self):
        a = self.start
        b = self.end
        l = sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)
        return l

    def _stiffness(self):
        l = self.l
        a = self.a
        i = self.i
        e = self.e
        ends_fixity = self.ends_fixity

        if (ends_fixity == "fix_fix"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 12.0 * e * i / (l ** 3.0), 6.0 * e * i / (l ** 2.0), 0.0, -12.0 * e * i / (l ** 3.0), 6.0 * e * i / (l ** 2.0)],
                [0.0, 6.0 * e * i / (l ** 2.0), 4.0 * e * i / (l), 0.0, -6.0 * e * i / (l ** 2.0), 2.0 * e * i / (l)],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -12.0 * e * i / (l ** 3.0), -6.0 * e * i / (l ** 2.0), 0.0, 12.0 * e * i / (l ** 3.0), -6.0 * e * i / (l ** 2.0)],
                [0.0, 6.0 * e * i / (l ** 2.0), 2.0 * e * i / (l), 0.0, -6.0 * e * i / (l ** 2.0), 4.0 * e * i / (l)]])

        elif (ends_fixity == "hinge_fix"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 3.0 * e * i / (l ** 3.0), 0.0, 0.0, -3.0 * e * i / (l ** 3.0), 3.0 * e * i / (l ** 2.0)],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -3.0 * e * i / (l ** 3.0), 0.0, 0.0, 3.0 * e * i / (l ** 3.0), -3.0 * e * i / (l ** 2.0)],
                [0.0, 3.0 * e * i / (l ** 2.0), 0.0, 0.0, -3.0 * e * i / (l ** 2.0), 3.0 * e * i / (l)]])

        elif (ends_fixity == "fix_hinge"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 3.0 * e * i / (l ** 3.0), 3.0 * e * i / (l ** 2.0), 0.0, -3.0 * e * i / (l ** 3.0), 0.0],
                [0.0, 3.0 * e * i / (l ** 2.0), 3.0 * e * i / (l), 0.0, -3.0 * e * i / (l ** 2.0), 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -3.0 * e * i / (l ** 3.0), -3.0 * e * i / (l ** 2.0), 0.0, 3.0 * e * i / (l ** 3.0), 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        elif (ends_fixity == "hinge_hinge"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        return k

    def _udefs(self):
        k = self.k
        k_size = k.shape[0]
        if self.has_axial_yield:
            udef_start_empty = np.zeros((k_size, 2))
            udef_end_empty = np.zeros((k_size, 2))
            udef_start = np.matrix(udef_start_empty)
            udef_end = np.matrix(udef_end_empty)
            udef_start[:, 0] = k[:, 0]
            udef_start[:, 1] = k[:, 2]
            udef_end[:, 0] = k[:, 3]
            udef_end[:, 1] = k[:, 5]
        else:
            udef_start_empty = np.zeros((k_size, 1))
            udef_end_empty = np.zeros((k_size, 1))
            udef_start = np.matrix(udef_start_empty)
            udef_end = np.matrix(udef_end_empty)
            udef_start[:, 0] = k[:, 2]
            udef_end[:, 0] = k[:, 5]

        udefs = (udef_start, udef_end)
        return udefs

    def _transform_matrix(self):
        a = self.start
        b = self.end
        l = self.l
        t = np.matrix([
            [(b.x - a.x) / l, (b.y - a.y) / l, 0.0, 0.0, 0.0, 0.0],
            [-(b.y - a.y) / l, (b.x - a.x) / l, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, (b.x - a.x) / l, (b.y - a.y) / l, 0.0],
            [0.0, 0.0, 0.0, -(b.y - a.y) / l, (b.x - a.x) / l, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        return t

    def get_nodal_force(self, displacements, fixed_forces):
        # displacements: numpy matrix
        # fixed_forces: numpy matrix
        k = self.k
        f = (k * displacements + fixed_forces).T
        return f


class PlateSection:
    # nu: poisson ratio
    def __init__(self, material: Material, t):
        e = material.e
        nu = material.nu
        sy = material.sy
        d = np.matrix([[1, nu, 0],
                      [nu, 1, 0],
                      [0, 0, (1 - nu) / 2]])
        self.t = t
        self.mp = 0.25 * t ** 2 * sy
        self.be = (e / (1 - nu ** 2)) * d
        self.de = (e * t ** 3) / (12 * (1 - nu ** 2)) * d


class RectangularThinPlateElement:
    # k is calculated based on four integration points
    def __init__(self, nodes: tuple[Node, Node, Node, Node], section: PlateSection):
        self.t = section.t
        self.nodes = nodes
        self.lx = nodes[1].x - nodes[0].x
        self.ly = nodes[2].y - nodes[1].y
        self.k = self._stiffness(section.de)

    def _shape_functions(self, r, s):
        ax = (self.lx / 2)
        ay = (self.ly / 2)
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

    def _shape_derivatives(self, r, s):
        ax = (self.lx / 2)
        ay = (self.ly / 2)
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

    def _stiffness_integrand(self, r, s, de):
        b = self._shape_derivatives(r=r, s=s)
        ki = b.T * de * b
        return ki

    def _stiffness(self, de):
        ax = (self.lx / 2)
        ay = (self.ly / 2)
        kin = self._stiffness_integrand(r=-0.57735, s=-0.57735, de=de) + \
            self._stiffness_integrand(r=+0.57735, s=-0.57735, de=de) + \
            self._stiffness_integrand(r=+0.57735, s=+0.57735, de=de) + \
            self._stiffness_integrand(r=-0.57735, s=+0.57735, de=de)
        k = kin * ax * ay
        return k


class Structure:
    # ycn: yield components num
    def __init__(self, nodes_num, node_n_dof, elements, boundaries, loads):
        self.nodes_num = nodes_num
        self.node_n_dof = node_n_dof
        self.elements = elements
        self.boundaries = boundaries
        self.loads = loads
        self.k = self.assemble()
        self.reduced_k = self.apply_boundry_conditions()
        self.f = self.apply_loading()
        self.ck = scipy.linalg.cho_factor(self.reduced_k)
        self.disp = self.compute_structure_displacement(self.f)
        self.elements_disps = self.get_elements_disps(self.disp)
        self.ycn = self._ycn()
        self.p0 = self._elastic_internal_force_vector()
        self.pv = self._sensitivity_matrix()
        self.phi = self._create_phi()

    def _ycn(self):
        ycn = 0
        for element in self.elements:
            ycn = ycn + element.total_ycn
        return ycn

    def _transform_loc_2d_matrix_to_glob(self, element_transform, element_stiffness):
        element_global_stiffness = np.dot(np.dot(np.transpose(element_transform), element_stiffness), element_transform)
        return element_global_stiffness

    def assemble(self):
        empty_stiffness = np.zeros((self.node_n_dof * self.nodes_num, self.node_n_dof * self.nodes_num))
        structure_stiffness = np.matrix(empty_stiffness)
        for eln in range(len(self.elements)):
            element_n_nodes = len(self.elements[eln].nodes)
            element_n_dof = self.elements[eln].k.shape[0]
            element_node_n_dof = element_n_dof / element_n_nodes
            element_global_stiffness = self._transform_loc_2d_matrix_to_glob(self.elements[eln].t, self.elements[eln].k)
            for i in range(element_n_dof):
                for j in range(element_n_dof):
                    local_element_node_row = int(j // element_node_n_dof)
                    p = int(element_node_n_dof * self.elements[eln].nodes[local_element_node_row].num + j % element_node_n_dof)
                    local_element_node_column = int(i // element_node_n_dof)
                    q = int(element_node_n_dof * self.elements[eln].nodes[local_element_node_column].num + i % element_node_n_dof)
                    structure_stiffness[p, q] = structure_stiffness[p, q] + element_global_stiffness[j, i]
        return structure_stiffness

    def apply_boundry_conditions(self):
        reduced_matrix = self.k
        deleted_counter = 0
        for i in range(len(self.boundaries)):
            # delete column
            reduced_matrix = np.delete(
                reduced_matrix, 3 * self.boundaries[i, 0] + self.boundaries[i, 1] - deleted_counter, 1
            )
            # delete row
            reduced_matrix = np.delete(
                reduced_matrix, 3 * self.boundaries[i, 0] + self.boundaries[i, 1] - deleted_counter, 0
            )
            deleted_counter += 1
        return reduced_matrix

    def _assemble_join_load(self):
        f_total = np.zeros((self.nodes_num * self.node_n_dof, 1))
        f_total = np.matrix(f_total)
        for joint_load in self.loads["joint_loads"]:
            f_total[self.node_n_dof * int(joint_load[0]) + int(joint_load[1])] = f_total[self.node_n_dof * int(joint_load[0]) + int(joint_load[1])] + joint_load[2]
        return f_total

    def apply_loading(self):
        f_total = np.zeros((self.nodes_num * self.node_n_dof, 1))
        f_total = np.matrix(f_total)
        for load in self.loads:
            if load == "joint_loads":
                f_total = f_total + self._assemble_join_load()
        return f_total

    def apply_load_boundry_conditions(self, force):
        reduced_f = force
        deleted_counter = 0
        for i in range(len(self.boundaries)):
            reduced_f = np.delete(
                reduced_f, 3 * self.boundaries[i, 0] + self.boundaries[i, 1] - deleted_counter, 0
            )
            deleted_counter += 1
        return reduced_f

    def get_elements_disps(self, disp):
        elements_disps = []
        for element in self.elements:
            element_dof_num = element.k.shape[0]
            element_nodes_num = len(element.nodes)
            element_node_dof_num = int(element_dof_num / element_nodes_num)
            v = np.zeros((element_dof_num, 1))
            v = np.matrix(v)
            for i in range(element_dof_num):
                element_node = i // element_node_dof_num
                node_dof = i % element_node_dof_num
                v[i, 0] = disp[element_node_dof_num * element.nodes[element_node].num + node_dof, 0]
            u = element.t * v
            elements_disps.append(u)
        return elements_disps

    def compute_structure_displacement(self, force):
        j = 0
        o = 0
        boundaries_num = len(self.boundaries)
        reduced_forces = self.apply_load_boundry_conditions(force)
        reduced_disp = scipy.linalg.cho_solve(self.ck, reduced_forces)
        disp = np.zeros((self.node_n_dof * self.nodes_num, 1))
        disp = np.matrix(disp)
        for i in range(self.node_n_dof * self.nodes_num):
            if (j != boundaries_num and i == self.node_n_dof * self.boundaries[j, 0] + self.boundaries[j, 1]):
                j += 1
            else:
                disp[i, 0] = reduced_disp[o, 0]
                o += 1
        return disp

    def _elastic_internal_force_vector(self):
        ycn = self.ycn
        elements = self.elements
        elements_disps = self.elements_disps

        fixed_force = np.zeros((self.node_n_dof * 2, 1))
        fixed_force = np.matrix(fixed_force)

        # calculate p0
        empty_p0 = np.zeros((ycn, 1))
        p0 = np.matrix(empty_p0)
        current_p0_row = 0

        for i, element in enumerate(elements):
            if element.__class__.__name__ == "FrameElement2D":
                element_force = element.get_nodal_force(elements_disps[i], fixed_force)
                if not element.has_axial_yield:
                    p0[current_p0_row] = element_force[0, 2]
                    p0[current_p0_row + 1] = element_force[0, 5]
                else:
                    p0[current_p0_row] = element_force[0, 0]
                    p0[current_p0_row + 1] = element_force[0, 2]
                    p0[current_p0_row + 2] = element_force[0, 3]
                    p0[current_p0_row + 3] = element_force[0, 5]
            current_p0_row = current_p0_row + element.total_ycn
        return p0

    def _sensitivity_matrix(self):
        # fv: equivalent global force vector for a yield component's udef
        ycn = self.ycn
        elements = self.elements
        empty_pv = np.zeros((ycn, ycn))
        pv = np.matrix(empty_pv)
        pv_column = 0
        for i_element, element in enumerate(elements):
            if element.__class__.__name__ == "FrameElement2D":
                for yield_point_udef in element.udefs:
                    udef_components_num = yield_point_udef.shape[1]
                    for i_component in range(udef_components_num):
                        fv_size = self.node_n_dof * self.nodes_num
                        fv = np.zeros((fv_size, 1))
                        fv = np.matrix(fv)
                        component_udef_global = element.t.T * yield_point_udef[:, i_component]
                        start_dof = self.node_n_dof * element.nodes[0].num
                        end_dof = self.node_n_dof * element.nodes[1].num

                        fv[start_dof] = component_udef_global[0]
                        fv[start_dof + 1] = component_udef_global[1]
                        fv[start_dof + 2] = component_udef_global[2]

                        fv[end_dof] = component_udef_global[3]
                        fv[end_dof + 1] = component_udef_global[4]
                        fv[end_dof + 2] = component_udef_global[5]

                        affected_struc_disp = self.compute_structure_displacement(fv)
                        affected_elem_disps = self.get_elements_disps(affected_struc_disp)
                        current_affected_element_ycns = 0
                        for i_affected_element, affected_elem_disp in enumerate(affected_elem_disps):
                            if i_element == i_affected_element:
                                fixed_force = -yield_point_udef[:, i_component]
                            else:
                                fixed_force = np.zeros((self.node_n_dof * 2, 1))
                                fixed_force = np.matrix(fixed_force)
                            affected_element_force = self.elements[i_affected_element].get_nodal_force(affected_elem_disp, fixed_force)

                            if not element.has_axial_yield:
                                pv[current_affected_element_ycns, pv_column] = affected_element_force[0, 2]
                                pv[current_affected_element_ycns + 1, pv_column] = affected_element_force[0, 5]
                            else:
                                pv[current_affected_element_ycns, pv_column] = affected_element_force[0, 0]
                                pv[current_affected_element_ycns + 1, pv_column] = affected_element_force[0, 2]
                                pv[current_affected_element_ycns + 2, pv_column] = affected_element_force[0, 3]
                                pv[current_affected_element_ycns + 3, pv_column] = affected_element_force[0, 5]
                            current_affected_element_ycns = current_affected_element_ycns + self.elements[i_affected_element].total_ycn
                        pv_column += 1
        return pv

    def _create_phi(self):
        phi_row_size = 0
        phi_column_size = 0
        for element in self.elements:
            phi_row_size = phi_row_size + element.section.phi.shape[0] * len(element.yield_points)
            phi_column_size = phi_column_size + element.section.phi.shape[1] * len(element.yield_points)

        empty_phi = np.zeros((phi_row_size, phi_column_size))
        phi = np.matrix(empty_phi)
        current_row = 0
        current_column = 0
        for element in self.elements:
            for yield_point in range(len(element.yield_points)):
                for yield_section_row in range(element.section.phi.shape[0]):
                    for yield_section_column in range(element.section.phi.shape[1]):
                        phi[current_row + yield_section_row, current_column + yield_section_column] = element.section.phi[yield_section_row, yield_section_column]
                current_column = current_column + element.section.phi.shape[1]
                current_row = current_row + element.section.phi.shape[0]
        return phi
