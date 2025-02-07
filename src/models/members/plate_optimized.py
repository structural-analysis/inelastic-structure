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


@dataclass
class Moment:
    x: float
    y: float
    xy: float


class PlateMember:
    """
    An optimized version that caches and reuses Gauss-point data to avoid
    repeated function calls on every `get_response()` invocation.
    """
    def __init__(self, num: int, section: PlateSection, include_softening: bool, element_type: str, nodes: tuple):
        self.num = num
        self.section = section
        self.element_type = element_type  # "Q4", "Q4R", "Q8", "Q8R"
        self.nodes = nodes
        self.nodes_count = len(self.nodes)
        self.node_dofs_count = 3
        self.dofs_count = self.node_dofs_count * self.nodes_count

        # For moment/plastic computations
        self.yield_specs = MemberYieldSpecs(
            section=self.section,
            points_count=len(self.gauss_points),
            include_softening=include_softening,
        )

        # Precompute B matrices, Jacobians, and their determinants at Gauss points
        # so we don't have to recalc them every time in get_response.
        self._gauss_data = self._initialize_gauss_data()

        # Now compute the element stiffness once
        self.k = self._compute_stiffness()

        # Identity transform by default
        self.t = np.eye(self.dofs_count)

        # Precompute the "unit-distortion" results (udefs, udets)
        self.udefs, self.udets = self._compute_nodal_forces_from_unit_distortions()

    # ------------------------------------------------------------
    # 1) Basic data about Gauss and natural points
    # ------------------------------------------------------------
    @property
    def gauss_points(self):
        """Gauss points & weights depending on element type."""
        if self.element_type == "Q4R":
            return [GaussPoint(weight=2, r=0, s=0)]
        elif self.element_type in ("Q4", "Q8R"):
            return [
                GaussPoint(weight=1, r=-0.57735027, s=-0.57735027),
                GaussPoint(weight=1, r=+0.57735027, s=-0.57735027),
                GaussPoint(weight=1, r=+0.57735027, s=+0.57735027),
                GaussPoint(weight=1, r=-0.57735027, s=+0.57735027),
            ]
        elif self.element_type == "Q8":
            return [
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

    @property
    def natural_nodes(self):
        """Corner (and mid-side) natural coordinates depending on element type."""
        if self.element_type in ("Q4", "Q4R"):
            return [
                NaturalPoint(r=-1, s=-1),
                NaturalPoint(r=+1, s=-1),
                NaturalPoint(r=+1, s=+1),
                NaturalPoint(r=-1, s=+1),
            ]
        elif self.element_type in ("Q8", "Q8R"):
            return [
                NaturalPoint(r=-1, s=-1),
                NaturalPoint(r=0,   s=-1),
                NaturalPoint(r=1,   s=-1),
                NaturalPoint(r=1,   s=0),
                NaturalPoint(r=1,   s=1),
                NaturalPoint(r=0,   s=1),
                NaturalPoint(r=-1,  s=1),
                NaturalPoint(r=-1,  s=0),
            ]

    # ------------------------------------------------------------
    # 2) Shape functions and derivatives
    # ------------------------------------------------------------
    def get_nodal_shape_functions(self, natural_point):
        """Nodal shape functions N(r,s) for each node."""
        r = natural_point.r
        s = natural_point.s
        if self.element_type in ("Q4", "Q4R"):
            return np.array([
                0.25 * (1 - r) * (1 - s),
                0.25 * (1 + r) * (1 - s),
                0.25 * (1 + r) * (1 + s),
                0.25 * (1 - r) * (1 + s),
            ])
        elif self.element_type in ("Q8", "Q8R"):
            # mid-side interpolation
            n1 = 0.5 * (1 - r**2) * (1 - s)
            n3 = 0.5 * (1 + r)    * (1 - s**2)
            n5 = 0.5 * (1 - r**2) * (1 + s)
            n7 = 0.5 * (1 - r)    * (1 - s**2)

            n0 = 0.25 * (1 - r) * (1 - s) - 0.5 * (n7 + n1)
            n2 = 0.25 * (1 + r) * (1 - s) - 0.5 * (n1 + n3)
            n4 = 0.25 * (1 + r) * (1 + s) - 0.5 * (n3 + n5)
            n6 = 0.25 * (1 - r) * (1 + s) - 0.5 * (n5 + n7)

            return np.array([n0, n1, n2, n3, n4, n5, n6, n7])

    def _compute_jacobian(self, r, s):
        """
        Jacobian matrix J(r,s).
        We'll do it with a direct array approach to avoid repeated loops.
        """
        nodes_xy = np.array([[nd.x, nd.y] for nd in self.nodes])  # shape: (n_nodes, 2)

        if self.element_type in ("Q4", "Q4R"):
            dN_dxi = 0.25 * np.array([
                [-(1 - s),  (1 - s),  (1 + s), -(1 + s)],
                [-(1 - r), -(1 + r),  (1 + r),  (1 - r)],
            ])
            return dN_dxi @ nodes_xy

        elif self.element_type in ("Q8", "Q8R"):
            # This matches your original partial derivatives block
            dN_dxi = 0.25 * np.array([
                [-2*r*s + 2*r - s**2 + s,   4*r*(s - 1),
                 -2*r*s + 2*r + s**2 - s,   2 - 2*s**2,
                  2*r*s + 2*r + s**2 + s,  -4*r*(s + 1),
                  2*r*s + 2*r - s**2 - s,   2*s**2 - 2],

                [-r**2 - 2*r*s + r + 2*s,   2*r**2 - 2,
                 -r**2 + 2*r*s - r + 2*s,  -4*s*(r + 1),
                  r**2 + 2*r*s + r + 2*s,  2 - 2*r**2,
                  r**2 - 2*r*s - r + 2*s,  4*s*(r - 1)],
            ])
            return dN_dxi @ nodes_xy

    def _compute_b_matrix(self, r, s, inv_j):
        """
        Compute the derivative-of-shape-function matrix B for bending/moment.
        B is 5 x (3*n_nodes).
        """
        # For Q4/Q4R or Q8/Q8R, let's unify the shape function deriv structure
        dof_n = 3 * self.nodes_count
        b = np.zeros((5, dof_n))

        # Evaluate N(r,s) for including w terms in the last rows
        nvals = self.get_nodal_shape_functions(NaturalPoint(r, s))

        # Evaluate dN/dx, dN/dy from dN/dr, dN/ds multiplied by inv(J)
        if self.element_type in ("Q4", "Q4R"):
            dN_dxi = 0.25 * np.array([
                [-(1 - s), (1 - s),  (1 + s), -(1 + s)],
                [-(1 - r), -(1 + r), (1 + r),  (1 - r)],
            ])
        else:  # Q8 or Q8R
            dN_dxi = 0.25 * np.array([
                [-2*r*s + 2*r - s**2 + s,   4*r*(s - 1),
                 -2*r*s + 2*r + s**2 - s,   2 - 2*s**2,
                  2*r*s + 2*r + s**2 + s,  -4*r*(s + 1),
                  2*r*s + 2*r - s**2 - s,   2*s**2 - 2],
                [-r**2 - 2*r*s + r + 2*s,   2*r**2 - 2,
                 -r**2 + 2*r*s - r + 2*s,  -4*s*(r + 1),
                  r**2 + 2*r*s + r + 2*s,  2 - 2*r**2,
                  r**2 - 2*r*s - r + 2*s,  4*s*(r - 1)]
            ])

        # (dN/dx, dN/dy) = inv_j @ dN/dxi
        dN_dx_dy = inv_j @ dN_dxi  # shape: (2, n_nodes)

        # Fill B
        for i in range(self.nodes_count):
            # Because Python indexing starts at 0, offset = i*3
            col = 3 * i

            # dx/dx = dN_dx_dy[0,i], dN_dy_dy = dN_dx_dy[1,i]
            dN_dx = dN_dx_dy[0, i]
            dN_dy = dN_dx_dy[1, i]
            Ni    = nvals[i]

            # B(1,2,3,4,5) forms from the original code
            # 0: Mx = d/dx(θy)
            # 1: My = -d/dy(θx)
            # 2: Mxy
            # 3: Qx = dθy/dx + w*N?
            # 4: Qy = dθx/dy - w*N?

            b[0, col + 2] = dN_dx         # dθy/dx
            b[1, col + 1] = -dN_dy        # -dθx/dy
            b[2, col + 2] = dN_dy         # dθy/dy
            b[2, col + 1] = -dN_dx        # -dθx/dx
            b[3, col + 0] = dN_dx         # dθx/dx
            b[3, col + 2] = Ni            # + w*N
            b[4, col + 0] = dN_dy         # dθx/dy
            b[4, col + 1] = -Ni           # - w*N

        return b

    # ------------------------------------------------------------
    # 3) Precomputation of Gauss data
    # ------------------------------------------------------------
    def _initialize_gauss_data(self):
        """
        For each Gauss point:
          - Compute J(r,s), inv(J), det(J)
          - Compute B(r,s) once
        Store them for reuse. This eliminates repeated calls.
        """
        gauss_data = []
        for gp in self.gauss_points:
            # build the Jacobian & its inverse
            j = self._compute_jacobian(gp.r, gp.s)
            j_det = np.linalg.det(j)
            inv_j = np.linalg.inv(j)

            # compute the B matrix
            b = self._compute_b_matrix(gp.r, gp.s, inv_j)

            gauss_data.append((gp.weight, b, j_det))
        return gauss_data

    # ------------------------------------------------------------
    # 4) Element stiffness
    # ------------------------------------------------------------
    def _compute_stiffness(self):
        """
        Use the precomputed (weight, B, detJ) to build the stiffness K once.
        """
        k = np.zeros((self.dofs_count, self.dofs_count))
        dmat = self.section.d  # material "D" matrix (5x5 or so)

        for (w, b, jdet) in self._gauss_data:
            k += w * (b.T @ dmat @ b) * jdet
        return k

    # ------------------------------------------------------------
    # 5) Unit-distortion approach (udefs, udets) precomputation
    # ------------------------------------------------------------
    def get_unit_distortion(self, comp_idx):
        """Return a 5-component array with 1 in the requested index, 0 otherwise."""
        distortion = np.zeros(5)
        distortion[comp_idx] = 1.0
        return distortion

    def _compute_nodal_forces_from_unit_distortion(self, b, jdet, distortion):
        """
        Single Gauss point: we want b^T * D * distortion * detJ
        for the nodal force. Also the moment portion is (D * distortion).
        """
        dmat = self.section.d
        nodal_force = b.T @ (dmat @ distortion) * jdet
        gauss_moment = dmat @ distortion
        return nodal_force, gauss_moment[: self.yield_specs.components_count]

    def _compute_nodal_forces_from_unit_distortions(self):
        """
        Build the arrays (udefs, udets) over all gauss points for each
        "yield component" (like Mx, My, Mxy, etc., per Gauss point).
        """
        dofs = self.dofs_count
        comps_count = self.yield_specs.components_count  # = 3 * number_of_gauss_points

        nodal_forces = np.zeros((dofs, comps_count))
        gauss_points_moments = np.zeros((comps_count, comps_count))

        # The idea: each gauss point contributes 3 local "yield components" ( Mx, My, Mxy )
        # in your original code you used 3 for each gauss point.
        # We'll loop carefully here and fill them in block slices
        base = 0
        for g_idx, (w, b, jdet) in enumerate(self._gauss_data):
            for local_comp in range(3):
                # global component index for this GP
                comp_idx = base + local_comp
                distortion = self.get_unit_distortion(local_comp)  # shape 5
                nf, gm = self._compute_nodal_forces_from_unit_distortion(b, jdet, distortion)

                nodal_forces[:, comp_idx] = nf
                # fill the sub-block [base:base+3, comp_idx]
                gauss_points_moments[base : base + 3, comp_idx] = gm
            base += 3
        return nodal_forces, gauss_points_moments

    # ------------------------------------------------------------
    # 6) The main "get_response" function in a single pass
    # ------------------------------------------------------------
    def get_response(self, nodal_disp, fixed_external=None, fixed_internal=None):
        """
        Return a Response object with:
          - nodal_force (K*u + any fixed_external)
          - yield_components_force (moments at each Gauss point)
          - nodal_moments (extrapolated to node corners + any fixed_internal)
        """
        if fixed_external is None:
            fixed_external = np.zeros(self.dofs_count)
        if fixed_internal is None:
            fixed_internal = np.zeros(self.yield_specs.components_count)

        # 1) Nodal force from stiffness + external
        nodal_force = self.k @ nodal_disp + fixed_external

        # 2) Gauss-point "moments" ( = D * B * disp ) for each Gauss point
        #    We'll store them in a (n_gauss, 3) array => (Mx, My, Mxy).
        n_gauss = len(self._gauss_data)
        gauss_moments = np.zeros((n_gauss, 3))

        dmat = self.section.d
        for i, (w, b, jdet) in enumerate(self._gauss_data):
            # moment = D * B * disp
            m_local = dmat @ (b @ nodal_disp)
            # slice out the first 3 for Mx, My, Mxy
            gauss_moments[i, 0] = m_local[0]  # Mx
            gauss_moments[i, 1] = m_local[1]  # My
            gauss_moments[i, 2] = m_local[2]  # Mxy

        # Add in any "fixed_internal" piece (like a prescribed moment/stress)
        # Suppose fixed_internal is shaped as 3*g for (Mx,My,Mxy) at each GP
        if fixed_internal.any():
            for gp_i in range(n_gauss):
                # fixed_internal offset = 3 * gp_i
                idx = 3 * gp_i
                gauss_moments[gp_i, 0] += fixed_internal[idx + 0]
                gauss_moments[gp_i, 1] += fixed_internal[idx + 1]
                gauss_moments[gp_i, 2] += fixed_internal[idx + 2]

        # 3) Build yield_components_force from these Gauss moments in a
        #    3*g array (like the original code).
        #    i.e. [ Mx1, My1, Mxy1,  Mx2, My2, Mxy2, ... ]
        yield_components_force = gauss_moments.flatten()

        # 4) Extrapolate to nodal corners => nodal_moments
        #    We'll do Cook p228 style: M_node = sum( shape_extrap * M_gauss ).
        #    For Q4: shape_extrap = N( sqrt(3)*r, sqrt(3)*s ) for each gauss point, etc.
        nodal_moments = np.zeros(3 * self.nodes_count)
        for node_i, nnode in enumerate(self.natural_nodes):
            # compute extrapolated shape function index for each gauss point
            # (r', s') = (sqrt(3)*r, sqrt(3)*s)
            # Then standard 4-node shape function on that (r',s')
            # For Q4 or Q8R
            r_ex = np.sqrt(3) * nnode.r
            s_ex = np.sqrt(3) * nnode.s

            # Evaluate shape functions for that extrap point
            # (Use the same Q4 shape function for "Q8R" Gauss layout if desired)
            nvals = np.array([
                0.25 * (1 - r_ex) * (1 - s_ex),
                0.25 * (1 + r_ex) * (1 - s_ex),
                0.25 * (1 + r_ex) * (1 + s_ex),
                0.25 * (1 - r_ex) * (1 + s_ex),
            ])

            # Now combine the gauss_moments. For Q4 there are 4 gauss points, so
            # we'll multiply each of the 4 gauss_moments by nvals[i]
            # Here, be consistent with your original indexing
            Mx_node = 0.0
            My_node = 0.0
            Mxy_node = 0.0
            for g_i, gp in enumerate(self.gauss_points):
                Mx_node  += gauss_moments[g_i, 0] * nvals[g_i]
                My_node  += gauss_moments[g_i, 1] * nvals[g_i]
                Mxy_node += gauss_moments[g_i, 2] * nvals[g_i]

            # place into nodal_moments
            base_i = 3 * node_i
            nodal_moments[base_i + 0] = Mx_node
            nodal_moments[base_i + 1] = My_node
            nodal_moments[base_i + 2] = Mxy_node

        return Response(
            nodal_force=nodal_force,
            yield_components_force=yield_components_force,
            nodal_moments=nodal_moments,
        )

    # ------------------------------------------------------------
    # 7) Misc utility: distributed loads
    # ------------------------------------------------------------
    def get_distributed_equivalent_load_vector(self, q):
        """
        For a distributed load q (scalar or vector),
        integrate shape functions * q over the domain (using Gauss points).
        Return the resulting (n_nodes) vector.
        """
        rs = np.zeros(self.nodes_count)
        for gp_i, gp in enumerate(self.gauss_points):
            n = self.get_nodal_shape_functions(NaturalPoint(gp.r, gp.s))
            # use the precomputed J for this gp
            # (weight, b, jdet) = self._gauss_data[gp_i]
            (w, _, jdet) = self._gauss_data[gp_i]
            rs += w * n * jdet
        return q * rs
