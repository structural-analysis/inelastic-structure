import numpy as np


class Material:
    def __init__(self, input_material):
        self.e = input_material["e"]
        self.sy = input_material["sy"]


class Geometry:
    def __init__(self, input_geometry):
        self.a = input_geometry["a"]


class Nonlinear:
    def __init__(self, material: Material, geometry: Geometry, input_nonlinear):
        self.is_direct_capacity = input_nonlinear["is_direct_capacity"]
        self.ap_positive = float(input_nonlinear["ap_positive"]) if self.is_direct_capacity else geometry.a * material.sy
        self.ap_negative = float(input_nonlinear["ap_negative"]) if self.is_direct_capacity else geometry.a * material.sy


class YieldSpecs:
    def __init__(self, nonlinear: Nonlinear):
        self.sifted_pieces_count = 2
        self.phi = self.create_phi(nonlinear)
        self.components_count = 1
        self.pieces_count = self.phi.shape[1]

    def create_phi(self, nonlinear):
        phi = np.matrix([-1 / nonlinear.ap_negative, 1 / nonlinear.ap_positive])
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


class Truss2DSection:
    def __init__(self, input: dict):
        self.material = Material(input["material"])
        self.geometry = Geometry(input["geometry"])
        self.nonlinear = Nonlinear(self.material, self.geometry, input["nonlinear"])
        self.yield_specs = YieldSpecs(self.nonlinear)
        self.softening = Softening(self.yield_specs, input["softening"])
