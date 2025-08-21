import numpy as np
from .functions import get_von_mises_matrix


class Material:
    def __init__(self, input_material):
        self.e = input_material["e"]
        self.sy = input_material["sy"]
        self.nu = input_material["nu"]
        self.rho = input_material.get("rho")


class Geometry:
    def __init__(self, input_geometry):
        self.thickness = input_geometry["t"]


class Nonlinear:
    def __init__(self, material: Material, input_nonlinear):
        self.sy = material.sy
        self.yield_surface = input_nonlinear["yield_surface"]


class YieldSpecs:
    def __init__(self, nonlinear: Nonlinear):
        self.sy = nonlinear.sy
        self.yield_surface = nonlinear.yield_surface
        self.sifted_pieces_count = 4
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
            ]).T / self.sy
        elif self.yield_surface == "mises":
            phi = get_von_mises_matrix(self.sy)
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
        self.w = np.array([[-1, -1], [1, 0]])
        self.cs = np.array([self.ep1, self.ep2 - self.ep1])

    def _get_slope(self):
        # for normalization divided by self.mp:
        return (self.alpha - 1) / (self.ep2 - self.ep1)

    def _get_h(self, yield_specs):
        h = np.zeros([yield_specs.pieces_count, 2])
        h[:, 0] = self.slope
        return h

    def _get_q(self, yield_specs):
        q = np.zeros([2, yield_specs.pieces_count])
        q[0, :] = np.linalg.norm(yield_specs.phi, axis=0)
        return q


class WallSection:
    def __init__(self, input: dict):
        self.material = Material(input["material"])
        self.geometry = Geometry(input["geometry"])
        self.nonlinear = Nonlinear(self.material, input["nonlinear"])
        self.yield_specs = YieldSpecs(self.nonlinear)
        self.softening = Softening(self.yield_specs, input["softening"])
        self.c = np.array([[1, self.material.nu, 0],
                            [self.material.nu, 1, 0],
                            [0, 0, (1 - self.material.nu) / 2]])
        self.ce = (self.material.e / (1 - self.material.nu ** 2)) * self.c
