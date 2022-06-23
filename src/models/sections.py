import numpy as np
from .materials import Material


class FrameSection:
    def __init__(self, material: Material, a, ix, iy, nonlinear: dict):
        self.a = a
        self.ix = ix
        self.iy = iy
        self.e = material.e
        self.sy = material.sy
        self.is_direct_capacity = nonlinear["is_direct_capacity"]
        self.has_axial_yield = nonlinear["has_axial_yield"]
        self.zp = float(nonlinear["zp"])
        self.abar0 = float(nonlinear["abar0"])
        self.mp = float(nonlinear["mp"]) if self.is_direct_capacity else self.zp * self.sy
        self.ap = float(nonlinear["ap"]) if self.is_direct_capacity else self.a * self.sy
        self.phi = self._create_phi()
        self.yield_components_num = 2 if self.has_axial_yield else 1
        self.yield_pieces_num = self.phi.shape[1]

        self.softening = nonlinear["softening"] if nonlinear["softening"] else {}
        self.alpha = float(self.softening.get("alpha", 1))
        self.ep1 = float(self.softening.get("ep1", 1e5))
        self.ep2 = float(self.softening.get("ep2", 1e7))
        self.softening_slope = self._get_softening_slope()
        self.h = self._get_h()
        self.q = self._get_q()
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

    def _get_h(self):
        h = np.matrix([
            [self.softening_slope, 0],
            [self.softening_slope, 0],
            [self.softening_slope, 0],
            [self.softening_slope, 0],
            [self.softening_slope, 0],
            [self.softening_slope, 0],
        ])
        return h

    def _get_q(self):
        q = np.matrix(np.zeros([2, self.yield_pieces_num]))
        q[0, :] = np.linalg.norm(self.phi, axis=0)
        return q


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
