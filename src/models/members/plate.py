import numpy as np
from dataclasses import dataclass
from functools import lru_cache

from ..points import NaturalPoint, GaussPoint
from ..sections.plate import PlateSection
from ..yield_models import MemberYieldSpecs


@dataclass
class Response:
    nodal_force: np.array
    yield_components_force: np.array
    nodal_moments: np.array
    nodal_strains: np.array = np.empty(0)
    nodal_stresses: np.array = np.empty(0)


class PlateMember:
    # calculations is based on four gauss points
    def __init__(self, num: int, section: PlateSection, include_softening: bool, element_type: str, nodes: tuple):
        self.num = num
        self.section = section
        self.element_type = element_type  # Q4, Q4R, Q8, Q8R
        self.nodes = nodes
        self.nodes_count = len(self.nodes)
        self.node_dofs_count = 3
        self.dofs_count = self.node_dofs_count * self.nodes_count
        self.nodal_components_count = self.nodes_count * self.section.yield_specs.components_count
        self.gauss_points_count = len(self.gauss_points)
        self.yield_specs = MemberYieldSpecs(
            section=self.section,
            points_count=self.gauss_points_count,
            include_softening=include_softening,
        )
        self.B_all = self._precompute_shape_derivatives()
        self.extrap_all = self._precompute_extrapolation()
        self.k = self.get_stiffness()
        self.t = self.get_transform()
        self.m = None
        # udef: unit distorsions equivalent forces (force, moment, ...) in nodes
        # udet: unit distorsions equivalent tractions (stress, force, moment, ...) in gauss points
        self.udefs, self.udets = self.get_nodal_forces_from_unit_distortions()

    @property
    def gauss_points(self):
        if self.element_type == "Q4R":
            gauss_points = [
                GaussPoint(weight=2, r=0, s=0),
            ]
        elif self.element_type in ("Q4", "Q8R"):
            gauss_points = [
                GaussPoint(weight=1, r=-0.57735027, s=-0.57735027),
                GaussPoint(weight=1, r=+0.57735027, s=-0.57735027),
                GaussPoint(weight=1, r=+0.57735027, s=+0.57735027),
                GaussPoint(weight=1, r=-0.57735027, s=+0.57735027),
            ]
        elif self.element_type == "Q8":
            gauss_points = [
                GaussPoint(weight=0.88888888, r=0, s=0),
                GaussPoint(weight=0.55555555, r=-0.77459667, s=-0.77459667),
                GaussPoint(weight=0.55555555, r=0, s=-0.77459667),
                GaussPoint(weight=0.55555555, r=+0.77459667, s=-0.77459667),
                GaussPoint(weight=0.55555555, r=+0.77459667, s=0),
                GaussPoint(weight=0.55555555, r=+0.77459667, s=+0.77459667),
                GaussPoint(weight=0.55555555, r=0, s=+0.77459667),
                GaussPoint(weight=0.55555555, r=-0.77459667, s=+0.77459667),
                GaussPoint(weight=0.55555555, r=-0.77459667, s=0),
            ]
        return gauss_points

    @property
    def natural_nodes(self):
        if self.element_type in ("Q4", "Q4R"):
            natural_nodes = [
                NaturalPoint(r=-1, s=-1),
                NaturalPoint(r=+1, s=-1),
                NaturalPoint(r=+1, s=+1),
                NaturalPoint(r=-1, s=+1),
            ]
        elif self.element_type in ("Q8", "Q8R"):
            natural_nodes = [
                NaturalPoint(r=-1, s=-1),
                NaturalPoint(r=0, s=-1),
                NaturalPoint(r=1, s=-1),
                NaturalPoint(r=1, s=0),
                NaturalPoint(r=1, s=1),
                NaturalPoint(r=0, s=1),
                NaturalPoint(r=-1, s=1),
                NaturalPoint(r=-1, s=0),
            ]
        return natural_nodes

    def _precompute_shape_derivatives(self):
        """
        Build a 3D array B_all so that B_all[i] == get_shape_derivatives(self.gauss_points[i]).
        Shape: (ng, 5, dofs_count).
        """
        gauss_pts = self.gauss_points
        ng = len(gauss_pts)
        B_all = np.zeros((ng, 5, self.dofs_count), dtype=float)
        for i, gp in enumerate(gauss_pts):
            B_all[i] = self.get_shape_derivatives(gp)  # shape (5, dofs_count)
        return B_all

    def _precompute_extrapolation(self):
        """
        Build an array (node_count, gauss_count) where each row is the shape
        function used to extrapolate from all gauss points to that node.

        For Q4 or Q8R, we have 4 gauss points. Each 'natural_node' gets an
        'extrapolated_natural_point', and we call get_extrapolated_shape_functions(...).

        If your element_type == 'Q8R', you still have 4 gauss points, but 8 nodes.
        So each row i has 4 shape-function values for the i-th node, each col is
        a gauss point.
        """
        node_count = len(self.natural_nodes)
        gp_count = len(self.gauss_points)
        shape_funcs_extrap = np.zeros((node_count, gp_count), dtype=float)

        if self.element_type in ("Q4", "Q8R"):
            for i, node in enumerate(self.natural_nodes):
                extr_pt = self.get_extrapolated_natural_point(node)
                if extr_pt is None:
                    # For Q4R or missing logic, handle carefully
                    shape_funcs_extrap[i, :] = 0.0
                else:
                    # shape_vals shape => (4,)
                    shape_vals = self.get_extrapolated_shape_functions(extr_pt)  
                    # each entry shape_vals[g] = the shape function of that node w.r.t. gauss_point g
                    # but in your code, get_extrapolated_shape_functions() returns them in a certain order.
                    # We'll assume it matches [gp0, gp1, gp2, gp3]
                    shape_funcs_extrap[i] = shape_vals
        else:
            # if "Q8" or something else not implemented, set zero or raise an error
            pass

        return shape_funcs_extrap

    def get_nodal_shape_functions(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        if self.element_type in ("Q4", "Q4R"):
            n = np.array([
                0.25 * (1 - r) * (1 - s),
                0.25 * (1 + r) * (1 - s),
                0.25 * (1 + r) * (1 + s),
                0.25 * (1 - r) * (1 + s),
            ])
        elif self.element_type in ("Q8", "Q8R"):
            n1 = 0.5 * (1 - r ** 2) * (1 - s)
            n3 = 0.5 * (1 + r) * (1 - s ** 2)
            n5 = 0.5 * (1 - r ** 2) * (1 + s)
            n7 = 0.5 * (1 - r) * (1 - s ** 2)

            n0 = 0.25 * (1 - r) * (1 - s) - 0.5 * (n7 + n1)
            n2 = 0.25 * (1 + r) * (1 - s) - 0.5 * (n1 + n3)
            n4 = 0.25 * (1 + r) * (1 + s) - 0.5 * (n3 + n5)
            n6 = 0.25 * (1 - r) * (1 + s) - 0.5 * (n5 + n7)

            n = np.array([n0, n1, n2, n3, n4, n5, n6, n7])
        return n

    def get_extrapolated_shape_functions(self, natural_point):
        # this functions used for stress extrapolations
        r = natural_point.r
        s = natural_point.s
        if self.element_type in ("Q4", "Q8R"):
            n = np.array([
                0.25 * (1 - r) * (1 - s),
                0.25 * (1 + r) * (1 - s),
                0.25 * (1 + r) * (1 + s),
                0.25 * (1 - r) * (1 + s),
            ])
        return n

    # REF: Cook (2002), p230.
    def get_extrapolated_natural_point(self, natural_point):
        if self.element_type == "Q4R":
            # TODO: write extrapolations for Q4R and Q8 integration schemes
            extrapolated_point = None
        elif self.element_type in ("Q4", "Q8R"):
            extrapolated_point = NaturalPoint(
                r=np.sqrt(3) * natural_point.r,
                s=np.sqrt(3) * natural_point.s,
            )
        if self.element_type == "Q8":
            extrapolated_point = None
        return extrapolated_point

    def get_jacobian(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        nodes = self.nodes
        if self.element_type in ("Q4", "Q4R"):
            j = 0.25 * np.array([
                [-(1 - s), (1 - s), (1 + s), -(1 + s)],
                [-(1 - r), -(1 + r), (1 + r), (1 - r)],
            ]) @ np.array([
                [nodes[0].x, nodes[0].y],
                [nodes[1].x, nodes[1].y],
                [nodes[2].x, nodes[2].y],
                [nodes[3].x, nodes[3].y],
            ])
        elif self.element_type in ("Q8", "Q8R"):
            j = 0.25 * np.array([
                [-2 * r * s + 2 * r - s ** 2 + s, 4 * r * (s - 1), -2 * r * s + 2 * r + s ** 2 - s, 2 - 2 * s ** 2, 2 * r * s + 2 * r + s ** 2 + s, -4 * r * (s + 1), 2 * r * s + 2 * r - s ** 2 - s, 2 * s ** 2 - 2],
                [-r ** 2 - 2 * r * s + r + 2 * s, 2 * r ** 2 - 2, -r ** 2 + 2 * r * s - r + 2 * s, -4 * s * (r + 1), r ** 2 + 2 * r * s + r + 2 * s, 2 - 2 * r ** 2, r ** 2 - 2 * r * s - r + 2 * s, 4 * s * (r - 1)],
            ]) @ np.array([
                [nodes[0].x, nodes[0].y],
                [nodes[1].x, nodes[1].y],
                [nodes[2].x, nodes[2].y],
                [nodes[3].x, nodes[3].y],
                [nodes[4].x, nodes[4].y],
                [nodes[5].x, nodes[5].y],
                [nodes[6].x, nodes[6].y],
                [nodes[7].x, nodes[7].y],
            ])
        return j

    def get_shape_derivatives(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        j = self.get_jacobian(natural_point)
        b = np.zeros((5, 3 * self.nodes_count))

        if self.element_type in ("Q4", "Q4R"):
            n = 0.25 * np.array([
                [(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)],
            ])
            dn = 0.25 * np.linalg.inv(j) @ np.array([
                [-(1 - s), +(1 - s), +(1 + s), -(1 + s)],
                [-(1 - r), -(1 + r), +(1 + r), +(1 - r)],
            ])

        elif self.element_type in ("Q8", "Q8R"):
            n2 = 0.5 * (1 - r ** 2) * (1 - s)
            n4 = 0.5 * (1 + r) * (1 - s ** 2)
            n6 = 0.5 * (1 - r ** 2) * (1 + s)
            n8 = 0.5 * (1 - r) * (1 - s ** 2)
            n1 = 0.25 * (1 - r) * (1 - s) - 0.5 * (n8 + n2)
            n3 = 0.25 * (1 + r) * (1 - s) - 0.5 * (n2 + n4)
            n5 = 0.25 * (1 + r) * (1 + s) - 0.5 * (n4 + n6)
            n7 = 0.25 * (1 - r) * (1 + s) - 0.5 * (n6 + n8)
            n = np.array([[n1, n2, n3, n4, n5, n6, n7, n8]])

            dn = 0.25 * np.linalg.inv(j) @ np.array([
                [-2 * r * s + 2 * r - s ** 2 + s, 4 * r * (s - 1), -2 * r * s + 2 * r + s ** 2 - s, 2 - 2 * s ** 2, 2 * r * s + 2 * r + s ** 2 + s, -4 * r * (s + 1), 2 * r * s + 2 * r - s ** 2 - s, 2 * s ** 2 - 2],
                [-r ** 2 - 2 * r * s + r + 2 * s, 2 * r ** 2 - 2, -r ** 2 + 2 * r * s - r + 2 * s, -4 * s * (r + 1), r ** 2 + 2 * r * s + r + 2 * s, 2 - 2 * r ** 2, r ** 2 - 2 * r * s - r + 2 * s, 4 * s * (r - 1)],
            ])

        for i in range(self.nodes_count):
            b[0, 3 * (i + 1) - 1] = dn[0, i]
            b[1, 3 * (i + 1) - 2] = -dn[1, i]
            b[2, 3 * (i + 1) - 1] = dn[1, i]
            b[2, 3 * (i + 1) - 2] = -dn[0, i]
            b[3, 3 * (i + 1) - 3] = dn[0, i]
            b[3, 3 * (i + 1) - 1] = n[0, i]
            b[4, 3 * (i + 1) - 3] = dn[1, i]
            b[4, 3 * (i + 1) - 2] = -n[0, i]

        return b

    def get_stiffness(self):
        k = np.zeros((self.dofs_count, self.dofs_count))
        for gauss_point in self.gauss_points:
            b = self.get_shape_derivatives(gauss_point)
            j = self.get_jacobian(gauss_point)
            j_det = np.linalg.det(j)
            gauss_point_k = gauss_point.weight * b.T @ self.section.d @ b * j_det
            k += gauss_point_k
        return k

    def get_transform(self):
        return np.eye(self.dofs_count)

    def get_nodal_moments_vectorized(self, gauss_points_moments, fixed_internal=None):
        # Only add fixed_internal if it's nonempty and actually has nonzero entries
        if fixed_internal is not None and fixed_internal.size and np.count_nonzero(fixed_internal):
            gauss_points_moments = gauss_points_moments.copy()
            fi_2D = fixed_internal.reshape(-1, 3)  # shape (ng,3)
            gauss_points_moments += fi_2D

        nodal_moms_2D = gauss_points_moments.T @ self.extrap_all.T  # => shape(3, node_count)
        return nodal_moms_2D.T.ravel()

    def get_gauss_points_moments_vectorized(self, nodal_disp):
        result1 = np.einsum('gij,j->gi', self.B_all, nodal_disp)
        temp = (self.section.d @ result1.T).T  # => (ng,5)
        return temp[:, 0:3]

    def get_unit_distortion(self, gauss_point_component_num):
        distortion = np.zeros(5)
        distortion[gauss_point_component_num] = 1
        return distortion

    # for element with linear variation of stress
    # REF: Cook (2002), p228.
    def get_nodal_force_from_unit_distortion(self, gauss_point, gauss_point_component_num):
        gauss_point_b = self.get_shape_derivatives(gauss_point)
        distortion = self.get_unit_distortion(gauss_point_component_num)
        j = self.get_jacobian(gauss_point)
        j_det = np.linalg.det(j)
        nodal_force = gauss_point_b.T @ self.section.d @ distortion * j_det
        gauss_point_moment = self.section.d @ distortion
        return nodal_force, gauss_point_moment[0:self.section.yield_specs.components_count]

    def get_nodal_forces_from_unit_distortions(self):
        nodal_forces = np.zeros((self.dofs_count, self.yield_specs.components_count))
        gauss_points_moments = np.zeros((self.yield_specs.components_count, self.yield_specs.components_count))
        component_base_num = 0
        for gauss_point in self.gauss_points:
            for j in range(3):
                nodal_forces[:, component_base_num + j] = self.get_nodal_force_from_unit_distortion(gauss_point=gauss_point, gauss_point_component_num=j)[0]
                gauss_points_moments[component_base_num:(component_base_num + 3), component_base_num + j] = self.get_nodal_force_from_unit_distortion(gauss_point=gauss_point, gauss_point_component_num=j)[1]
            component_base_num += 3
        return nodal_forces, gauss_points_moments

    def get_response(self, nodal_disp, fixed_external=None, fixed_internal=None):
        if fixed_external is None:
            fixed_external = np.zeros(self.dofs_count)
        if fixed_internal is None:
            fixed_internal = np.zeros(self.yield_specs.components_count)

        if fixed_external.any():
            nodal_force = self.k @ nodal_disp + fixed_external
        else:
            nodal_force = self.k @ nodal_disp

        gp_mom = self.get_gauss_points_moments_vectorized(nodal_disp)
        yield_components_force = gp_mom.copy().ravel()
        nodal_mom = self.get_nodal_moments_vectorized(gp_mom, fixed_internal)

        # 5) Build and return
        return Response(
            nodal_force=nodal_force,
            yield_components_force=yield_components_force,
            nodal_moments=nodal_mom,
        )

    def get_distributed_equivalent_load_vector(self, q):
        rs = np.zeros(self.nodes_count)
        for gauss_point in self.gauss_points:
            n = self.get_nodal_shape_functions(gauss_point)
            j = self.get_jacobian(gauss_point)
            j_det = np.linalg.det(j)
            gauss_point_rs = gauss_point.weight * n.T * j_det
            rs += gauss_point_rs
        return q * rs
