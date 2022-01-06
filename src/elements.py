from src.functions import sqrt
import numpy as np


class Beam2D():
    # udef: unit distorsions equivalent forces
    def __init__(self, start, end, ends_fixity,
                 section_area, inertia_moment, bending_capacity, elasticity_modulus):
        self.start = start
        self.end = end
        self.ends_fixity = ends_fixity
        self.section_area = section_area
        self.inertia_moment = inertia_moment
        self.elasticity_modulus = elasticity_modulus
        self.bending_capacity = bending_capacity
        self.length = self._length()
        self.stiffness = self._stiffness()["k"]
        self.transform_matrix = self._transform_matrix()
        self.udef = self._stiffness()["udef"]

    def _length(self):
        a = self.start
        b = self.end
        l = sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2 + (b[2] - a[2])**2)
        return l

    def _stiffness(self):
        l = self.length
        a = self.section_area
        i = self.inertia_moment
        e = self.elasticity_modulus
        ends_fixity = self.ends_fixity

        if (ends_fixity == "fixed_fixed"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 12.0 * e * i / (l**3.0), 6.0 * e * i / (l**2.0), 0.0, -12.0 * e * i / (l**3.0), 6.0 * e * i / (l**2.0)],
                [0.0, 6.0 * e * i / (l**2.0), 4.0 * e * i / (l), 0.0, -6.0 * e * i / (l**2.0), 2.0 * e * i / (l)],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -12.0 * e * i / (l**3.0), -6.0 * e * i / (l**2.0), 0.0, 12.0 * e * i / (l**3.0), -6.0 * e * i / (l**2.0)],
                [0.0, 6.0 * e * i / (l**2.0), 2.0 * e * i / (l), 0.0, -6.0 * e * i / (l**2.0), 4.0 * e * i / (l)]])

        elif (ends_fixity == "hinge_fixed"):
            k = np.matrix([
                [e * a / l, 0.0, 0.0, -e * a / l, 0.0, 0.0],
                [0.0, 3.0 * e * i / (l**3.0), 0.0, 0.0, -3.0 * e * i / (l**3.0), 3.0 * e * i / (l**2.0)],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-e * a / l, 0.0, 0.0, e * a / l, 0.0, 0.0],
                [0.0, -3.0 * e * i / (l**3.0), 0.0, 0.0, 3.0 * e * i / (l**3.0), -3.0 * e * i / (l**2.0)],
                [0.0, 3.0 * e * i / (l**2.0), 0.0, 0.0, -3.0 * e * i / (l**2.0), 3.0 * e * i / (l)]])

        elif (ends_fixity == "fixed_hinge"):
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
        l = self.length
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
        k = self.stiffness
        f = (k * displacements + fixed_forces).T
        return f
