import numpy as np
from functools import lru_cache


class Material:
    def __init__(self, input_material):
        self.e = input_material["e"]
        self.sy = input_material["sy"]
        self.nu = input_material["nu"]


class Geometry:
    def __init__(self, input_geometry):
        self.thickness = input_geometry["t"]


class Nonlinear:
    def __init__(self, material: Material, geometry: Geometry, input_nonlinear):
        self.mp = 0.25 * geometry.thickness ** 2 * material.sy
        self.yield_surface = input_nonlinear["yield_surface"]


class YieldSpecs:
    def __init__(self, nonlinear: Nonlinear):
        self.mp = nonlinear.mp
        self.yield_surface = nonlinear.yield_surface
        self.sifted_pieces_count = 5  # 4 will not run for example plate-semiconfined-inelastic
        self.components_count = self.phi.shape[0]
        self.pieces_count = self.phi.shape[1]

    @property
    def phi(self):
        if self.yield_surface == "simple":
            phi = np.array([
                [1.2143, -0.2143, 2],
                [-0.2143, 1.2143, 2],
                [-1.2143, 0.2143, 2],
                [0.2143, -1.2143, 2],
                [1.2143, -0.2143, -2],
                [-0.2143, 1.2143, -2],
                [-1.2143, 0.2143, -2],
                [0.2143, -1.2143, -2],
            ]).T / self.mp
        elif self.yield_surface == "mises":
            phi = get_von_mises_matrix(mp=self.mp)
        return phi


class Softening:
    def __init__(self, yield_specs: YieldSpecs, input_softening):
        self.data = input_softening if input_softening else {}
        self.alpha = float(self.data.get("alpha", 1))
        self.ep1 = float(self.data.get("ep1", 1e5))
        self.ep2 = float(self.data.get("ep2", 1e7))
        self.slope = self._get_slope()
        self.h = self._get_h(yield_specs)
        self.q = self._get_q(yield_specs)
        self.w = np.matrix([[-1, -1], [1, 0]])
        self.cs = np.matrix([[self.ep1], [self.ep2 - self.ep1]])

    def _get_slope(self):
        # for normalization divided by self.mp:
        return (self.alpha - 1) / (self.ep2 - self.ep1)

    def _get_h(self, yield_specs):
        h = np.matrix(np.zeros([yield_specs.pieces_count, 2]))
        h[:, 0] = self.slope
        return h

    def _get_q(self, yield_specs):
        q = np.matrix(np.zeros([2, yield_specs.pieces_count]))
        q[0, :] = np.linalg.norm(yield_specs.phi, axis=0)
        return q


class PlateSection:
    def __init__(self, input: dict):
        self.material = Material(input["material"])
        self.geometry = Geometry(input["geometry"])
        self.nonlinear = Nonlinear(self.material, self.geometry, input["nonlinear"])
        self.yield_specs = YieldSpecs(self.nonlinear)
        self.softening = Softening(self.yield_specs, input["softening"])

    @property
    def d(self):
        w = 5 / 6 # warping coefficient
        t = self.geometry.thickness
        v = self.material.nu
        e = self.material.e
        d = np.matrix(np.zeros((5, 5)))
        cb = np.matrix([
            [1, v, 0],
            [v, 1, 0],
            [0, 0, (1 - v) / 2]
        ])
        ceb = ((e * t ** 3) / (12 * (1 - v ** 2))) * cb
        cs = np.matrix(np.eye(2))
        ces = ((e * t * w)/(2 * (1 + v))) * cs
        d[0:3, 0:3] = ceb
        d[3:5, 3:5] = ces
        return d


# FIXME: FIX OPTIMIZED NOT WITH CACHING
@lru_cache
def get_von_mises_matrix(mp):
    si = np.array([1.9, 1.7, 1.2, 1, 0.5, 0, -0.5, -1, -1.2, -1.7, -1.9])
    m = 40
    n = si.shape[0]  # -2 & +2 will produce only one plane each
    p_total = m * n + 2  # total number of yield planes
    teta = np.zeros(40)
    pi = np.pi
    for i in range(m):
        teta[i] = 2 * pi * i / m

    # specifying two end planes
    phi = np.zeros((3, p_total))
    phi[:, 0] = np.array([0.5, 0.5, 0]) / mp
    phi[:, p_total - 1] = np.array([-0.5, -0.5, 0]) / mp

    l = 0
    for i in range(n):
        for j in range(m):
            k = j + l + 1
            phi[:, k] = np.array([
                0.25 * (si[i] - 3 * np.cos(teta[j]) * np.sqrt((4 - (si[i]) ** 2) / (3 * (1 + np.sin(teta[j]) ** 2)))),
                0.25 * (si[i] + 3 * np.cos(teta[j]) * np.sqrt((4 - (si[i]) ** 2) / (3 * (1 + np.sin(teta[j]) ** 2)))),
                1.5 * np.sqrt(2) * np.sin(teta[j]) * np.sqrt((4 - (si[i]) ** 2) / (3 * (1 + np.sin(teta[j]) ** 2)))
            ]) / mp
        l += m
    return phi


# def get_von_mises_matrix(mp):
#     si = np.array([-1.9, -1.7, -1.2, -1, -0.5, 0, 0.5, 1, 1.2, 1.7, 1.9])
#     m = 20
#     n = si.shape[0]  # -2 & +2 will produce only one plane each
#     p_total = m * n + 2  # total number of yield planes
#     teta = np.zeros(20)
#     pi = np.pi
#     for i in range(m):
#         teta[i] = 2 * pi * (i + 1) / m

#     # specifying two end planes
#     phi = np.zeros((3, p_total))
#     phi[:, 0] = np.array([-0.5, -0.5, 0]) / mp
#     phi[:, p_total - 1] = np.array([0.5, 0.5, 0]) / mp

#     l = 0
#     for i in range(n):
#         for j in range(m):
#             k = j + l + 1
#             phi[:, k] = np.array([
#                 0.25 * (si[i] + 3 * np.cos(teta[j]) * np.sqrt((4 - (si[i]) ** 2) / (3 * (1 + np.sin(teta[j]) ** 2)))),
#                 0.25 * (si[i] - 3 * np.cos(teta[j]) * np.sqrt((4 - (si[i]) ** 2) / (3 * (1 + np.sin(teta[j]) ** 2)))),
#                 1.5 * np.sqrt(2) * (np.sin(teta[j]) * np.sqrt((4 - (si[i]) ** 2) / (3 * (1 + np.sin(teta[j]) ** 2))))]) / mp
#         l += m
#     return phi
