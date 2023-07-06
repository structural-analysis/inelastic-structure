import numpy as np


class Material:
    def __init__(self, input_material):
        self.e = input_material["e"]
        self.sy = input_material["sy"]
        self.nu = input_material["nu"]
        self.g = input_material["g"]


class Geometry:
    def __init__(self, input_geometry):
        self.a = input_geometry["a"]
        self.j = input_geometry["j"]
        self.i22 = input_geometry["i22"]
        self.i33 = input_geometry["i33"]


class Nonlinear:
    def __init__(self, material: Material, geometry: Geometry, input_nonlinear):
        self.is_direct_capacity = input_nonlinear["is_direct_capacity"]
        self.has_axial_yield = input_nonlinear["has_axial_yield"]
        self.zpz = float(input_nonlinear["zpz"])
        self.zpy = float(input_nonlinear["zpy"])
        self.abar0 = float(input_nonlinear["abar0"])
        self.mpz = float(input_nonlinear["mpz"]) if self.is_direct_capacity else self.zpz * material.sy
        self.mpy = float(input_nonlinear["mpz"]) if self.is_direct_capacity else self.zpy * material.sy
        self.ap = float(input_nonlinear["ap"]) if self.is_direct_capacity else geometry.a * material.sy


class YieldSpecs:
    def __init__(self, nonlinear: Nonlinear):
        self.phi = self.create_phi(nonlinear)
        self.components_count = 2 if nonlinear.has_axial_yield else 1
        self.pieces_count = self.phi.shape[1]

    def create_phi(self, nonlinear):
        if nonlinear.has_axial_yield:
            phi = np.matrix([
                [
                    1 / nonlinear.ap,
                    0,
                    -1 / nonlinear.ap,
                    -1 / nonlinear.ap,
                    0,
                    1 / nonlinear.ap,
                ],
                [
                    (1 - nonlinear.abar0) / nonlinear.mpz,
                    1 / nonlinear.mpz,
                    (1 - nonlinear.abar0) / nonlinear.mpz,
                    -(1 - nonlinear.abar0) / nonlinear.mpz,
                    -1 / nonlinear.mpz,
                    -(1 - nonlinear.abar0) / nonlinear.mpz,
                ]
            ])
        else:
            phi = np.matrix([-1 / nonlinear.mpz, 1 / nonlinear.mpz])
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


class Frame3DSection:
    def __init__(self, input: dict):
        self.material = Material(input["material"])
        self.geometry = Geometry(input["geometry"])
        self.nonlinear = Nonlinear(self.material, self.geometry, input["nonlinear"])
        self.yield_specs = YieldSpecs(self.nonlinear)
        self.softening = Softening(self.yield_specs, input["softening"])
