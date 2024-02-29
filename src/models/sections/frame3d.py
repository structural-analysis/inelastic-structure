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
        self.zp33 = float(input_nonlinear["zp33"])
        self.zp22 = float(input_nonlinear["zp22"])
        self.mp33 = float(input_nonlinear["mp33"]) if self.is_direct_capacity else self.zp33 * material.sy
        self.mp22 = float(input_nonlinear["mp22"]) if self.is_direct_capacity else self.zp22 * material.sy
        self.ap = float(input_nonlinear["ap"]) if self.is_direct_capacity else geometry.a * material.sy


class YieldSpecs:
    def __init__(self, nonlinear: Nonlinear):
        # in correct sifting 4 must suffice
        # self.sifted_pieces_count = 4
        self.sifted_pieces_count = 8
        self.phi = self.create_phi(nonlinear)
        self.components_count = self.phi.shape[0]
        self.pieces_count = self.phi.shape[1]

    def create_phi(self, nonlinear):
        ap = nonlinear.ap
        mp33 = nonlinear.mp33
        mp22 = nonlinear.mp22
        phi = np.matrix([
            [1 / (2 * ap), 1 / (2 * ap), 1 / (2 * ap), 1 / (2 * ap), -1 / (2 * ap), -1 / (2 * ap), -1 / (2 * ap), -1 / (2 * ap), 1 / ap, 1 / ap, 1 / ap, 1 / ap, -1 / ap, -1 / ap, -1 / ap, -1 / ap],
            [1 / mp33, -1 / mp33, 1 / mp33, -1 / mp33, 1 / mp33, -1 / mp33, 1 / mp33, -1 / mp33, 8 / (9 * mp33), -8 / (9 * mp33), 8 / (9 * mp33), -8 / (9 * mp33), 8 / (9 * mp33), -8 / (9 * mp33), 8 / (9 * mp33), -8 / (9 * mp33)],
            [1 / mp22, 1 / mp22, -1 / mp22, -1 / mp22, 1 / mp22, 1 / mp22, -1 / mp22, -1 / mp22, 8 / (9 * mp22), 8 / (9 * mp22), -8 / (9 * mp22), -8 / (9 * mp22), 8 / (9 * mp22), 8 / (9 * mp22), -8 / (9 * mp22), -8 / (9 * mp22)],
        ])
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
