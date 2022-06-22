import numpy as np
from .materials import Material


class FrameSection:
    def __init__(self, material: Material, a, ix, iy, zp, has_axial_yield: str, abar0, ap=0, mp=0, is_direct_capacity=False, include_softening=False, alpha=0, ep1=0, ep2=0):
        self.a = a
        self.ix = ix
        self.iy = iy
        self.zp = zp
        self.e = material.e
        self.sy = material.sy
        self.is_direct_capacity = is_direct_capacity
        self.mp = mp if is_direct_capacity.lower() == "true" else self.zp * self.sy
        self.ap = ap if is_direct_capacity.lower() == "true" else self.a * self.sy
        self.abar0 = abar0
        self.has_axial_yield = True if has_axial_yield.lower() == "true" else False
        self.phi = self._create_phi()
        self.yield_components_num = 2 if self.has_axial_yield else 1
        self.yield_pieces_num = self.phi.shape[1]

        # softening specifics
        self.include_softening = True if include_softening.lower() == "true" else False
        self.alpha = alpha
        self.ep1 = ep1
        self.ep2 = ep2
        self.h = self._get_softening_slope() if self.include_softening else 0
        self.h_matrix = self._get_h_matrix()
        self.q_matrix = self._get_q_matrix()
        self.w = np.matrix([[-1, -1], [1, 0]])
        self.cs = np.matrix([[self.ep1], [self.ep2 - self.ep1]])

    def _create_phi(self):
        if self.has_axial_yield:
            phi = np.matrix([
                [
                    1 / self.ap,
                    0,
                    -1 / self.ap,
                    -1 / self.ap,
                    0,
                    1 / self.ap,
                ],
                [
                    (1 - self.abar0) / self.mp,
                    1 / self.mp,
                    (1 - self.abar0) / self.mp,
                    -(1 - self.abar0) / self.mp,
                    -1 / self.mp,
                    -(1 - self.abar0) / self.mp,
                ]
            ])
        else:
            phi = np.matrix([-1 / self.mp, 1 / self.mp])
        return phi

    def _get_softening_slope(self):
        # for normalization divided by self.mp:
        return (self.alpha - 1) / (self.ep2 - self.ep1)

    def _get_h_matrix(self):
        h_matrix = np.matrix([
            [self.h, 0],
            [self.h, 0],
            [self.h, 0],
            [self.h, 0],
            [self.h, 0],
            [self.h, 0],
        ])
        return h_matrix

    def _get_q_matrix(self):
        q_matrix = np.matrix(np.zeros([2, self.yield_pieces_num]))
        q_matrix[0, :] = np.linalg.norm(self.phi, axis=0)
        return q_matrix


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
