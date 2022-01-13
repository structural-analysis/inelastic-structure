from src.functions import sqrt
import numpy as np


class Material:
    def __init__(self, name):
        if name == "steel":
            self.e = 2e11
            self.sy = 240e6


class Section:
    def __init__(self, material, a, ix, iy, zp):
        self.e = material.e
        self.sy = material.sy
        self.a = a
        self.ix = ix
        self.iy = iy
        self.zp = zp




class PlateSection:
    # nu: poisson ratio
    def __init__(self, material, t):
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


class PlateElement:
    def __init__(self, nodes: tuple, section):
        self.t = section.t
        self.nodes = nodes
        self.k = self._stiffness(
            a=self.width,
            b=self.height,
            t=section.t,
            de=section.de
        )

    def _stiffness(a, b, t, de):
        return self.de

    def get_nodal_forces(self, displacements):
        k = self.k
        return k * displacements


class FrameElement2D:
    # mp: bending capacity
    # udef: unit distorsions equivalent forces
    # ends_fixity: one of following: fix_fix, hinge_fix, fix_hinge, hinge_hinge
    def __init__(self, section, start, end, nodes, ends_fixity):
        self.start = start
        self.end = end
        self.ends_fixity = ends_fixity
        self.nodes = nodes
        self.a = section.a
        self.i = section.ix
        self.e = section.e
        self.mp = section.zp * section.sy
        self.l = self._length()
        self.k = self._stiffness()["k"]
        self.t = self._transform_matrix()
        self.udef = self._stiffness()["udef"]

    def _length(self):
        a = self.start
        b = self.end
        l = sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
        return l

    def _stiffness(self):
        l = self.l
        a = self.a
        i = self.i
        e = self.e
        ends_fixity = self.ends_fixity

        if (ends_fixity == "fix_fix"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 12.0 * e * i / (l**3.0), 6.0 * e * i / (l**2.0), 0.0, -12.0 * e * i / (l**3.0), 6.0 * e * i / (l**2.0)],
                [0.0, 6.0 * e * i / (l**2.0), 4.0 * e * i / (l), 0.0, -6.0 * e * i / (l**2.0), 2.0 * e * i / (l)],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -12.0 * e * i / (l**3.0), -6.0 * e * i / (l**2.0), 0.0, 12.0 * e * i / (l**3.0), -6.0 * e * i / (l**2.0)],
                [0.0, 6.0 * e * i / (l**2.0), 2.0 * e * i / (l), 0.0, -6.0 * e * i / (l**2.0), 4.0 * e * i / (l)]])

        elif (ends_fixity == "hinge_fix"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 3.0 * e * i / (l**3.0), 0.0, 0.0, -3.0 * e * i / (l**3.0), 3.0 * e * i / (l**2.0)],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -3.0 * e * i / (l**3.0), 0.0, 0.0, 3.0 * e * i / (l**3.0), -3.0 * e * i / (l**2.0)],
                [0.0, 3.0 * e * i / (l**2.0), 0.0, 0.0, -3.0 * e * i / (l**2.0), 3.0 * e * i / (l)]])

        elif (ends_fixity == "fix_hinge"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 3.0 * e * i / (l**3.0), 3.0 * e * i / (l**2.0), 0.0, -3.0 * e * i / (l**3.0), 0.0],
                [0.0, 3.0 * e * i / (l**2.0), 3.0 * e * i / (l), 0.0, -3.0 * e * i / (l**2.0), 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -3.0 * e * i / (l**3.0), -3.0 * e * i / (l**2.0), 0.0, 3.0 * e * i / (l**3.0), 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        elif (ends_fixity == "hinge_hinge"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        udef = k[:, [2, 5]]
        return {"k": k, "udef": udef}

    def _transform_matrix(self):
        a = self.start
        b = self.end
        l = self.l
        xa = a[0]
        ya = a[1]
        xb = b[0]
        yb = b[1]
        t = np.matrix([
            [(xb - xa) / l, (yb - ya) / l, 0.0, 0.0, 0.0, 0.0],
            [-(yb - ya) / l, (xb - xa) / l, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, (xb - xa) / l, (yb - ya) / l, 0.0],
            [0.0, 0.0, 0.0, -(yb - ya) / l, (xb - xa) / l, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        return t

    def get_nodal_forces(self, displacements, fixed_forces):
        # displacements: numpy matrix
        # fixed_forces: numpy matrix
        k = self.k
        f = (k * displacements + fixed_forces).T
        return f
