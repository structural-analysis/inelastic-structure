import numpy as np
from .materials import Material


class FrameSection:
    def __init__(self, material: Material, a, ix, iy, zp, has_axial_yield: str, abar0, ap=0, mp=0, is_direct_capacity=False):
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
        if not self.has_axial_yield:
            self.yield_components_num = 1
            self.phi = np.matrix([-1 / self.mp, 1 / self.mp])
        else:
            self.yield_components_num = 2
            self.phi = np.matrix([
                [
                    1 / self.ap,
                    0,
                    -1 / self.ap,
                    -1 / self.ap,
                    0,
                    1 / self.ap,
                ],
                [
                    (1 - abar0) / self.mp,
                    1 / self.mp,
                    (1 - abar0) / self.mp,
                    -(1 - abar0) / self.mp,
                    -1 / self.mp,
                    -(1 - abar0) / self.mp,
                ]
            ])
        self.yield_pieces_num = self.phi.shape[1]


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
