import numpy as np


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
        self.phi = self.create_phi(nonlinear)
        self.components_count = 3
        self.pieces_count = self.phi.shape[1]

    def create_phi(self, nonlinear):
        if nonlinear.yield_surface == "simple":
            phi = np.array([
                [1.2143, -0.2143, 2],
                [-0.2143, 1.2143, 2],
                [-1.2143, 0.2143, 2],
                [0.2143, -1.2143, 2],
                [1.2143, -0.2143, -2],
                [-0.2143, 1.2143, -2],
                [-1.2143, 0.2143, -2],
                [0.2143, -1.2143, -2],
            ]).T / nonlinear.mp
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
        self.d = np.matrix([[1, self.material.nu, 0],
                            [self.material.nu, 1, 0],
                            [0, 0, (1 - self.material.nu) / 2]])
        self.be = (self.material.e / (1 - self.material.nu ** 2)) * self.d
        self.de = (self.material.e * self.geometry.thickness ** 3) / (12 * (1 - self.material.nu ** 2)) * self.d
