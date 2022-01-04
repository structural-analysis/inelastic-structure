from src.functions import sqrt
import numpy as np


class Beam2D():
    def __init__(self, start, end, ends_fixity, section_area, inertia_moment, bending_capacity, elasticity_modulus):
        self.length = sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2 + (end[2]-start[2])**2)
        self.section_area = section_area
        self.inertia_moment = inertia_moment
        self.elasticity_modulus = elasticity_modulus
        self.bending_capacity = bending_capacity
        self.ends_fixity = ends_fixity

    def stiffness(self):
        l = self.length
        a = self.section_area
        i = self.inertia_moment
        e = self.elasticity_modulus
        ends_fixity = self.ends_fixity

        if (ends_fixity == "fixed_fixed"):
            k = np.matrix([
                [e*a/l, 0.0, 0.0, -e*a/l, 0.0, 0.0],
                [0.0, 12.0*e*i/(l**3.0), 6.0*e*i/(l**2.0), 0.0, -12.0*e*i/(l**3.0), 6.0*e*i/(l**2.0)],
                [0.0, 6.0*e*i/(l**2.0), 4.0*e*i/(l), 0.0, -6.0*e*i/(l**2.0), 2.0*e*i/(l)],
                [-e*a/l, 0.0, 0.0, e*a/l, 0.0, 0.0],
                [0.0, -12.0*e*i/(l**3.0), -6.0*e*i/(l**2.0), 0.0, 12.0*e*i/(l**3.0), -6.0*e*i/(l**2.0)],
                [0.0, 6.0*e*i/(l**2.0), 2.0*e*i/(l), 0.0, -6.0*e*i/(l**2.0), 4.0*e*i/(l)]])

        elif (ends_fixity == "hinge_fixed"):
            k = np.matrix([
                [e*a/l, 0.0, 0.0, -e*a/l, 0.0, 0.0],
                [0.0, 3.0*e*i/(l**3.0), 0.0, 0.0, -3.0*e*i/(l**3.0), 3.0*e*i/(l**2.0)],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-e*a/l, 0.0, 0.0, e*a/l, 0.0, 0.0],
                [0.0, -3.0*e*i/(l**3.0), 0.0, 0.0, 3.0*e*i/(l**3.0), -3.0*e*i/(l**2.0)],
                [0.0, 3.0*e*i/(l**2.0), 0.0, 0.0, -3.0*e*i/(l**2.0), 3.0*e*i/(l)]])

        elif (ends_fixity == "fixed_hinge"):
            k = np.matrix([
                [e*a/l, 0.0, 0.0, -e*a/l, 0.0, 0.0],
                [0.0, 3.0*e*i/(l**3.0), 3.0*e*i/(l**2.0), 0.0, -3.0*e*i/(l**3.0), 0.0],
                [0.0, 3.0*e*i/(l**2.0), 3.0*e*i/(l), 0.0, -3.0*e*i/(l**2.0), 0.0],
                [-e*a/l, 0.0, 0.0, e*a/l, 0.0, 0.0],
                [0.0, -3.0*e*i/(l**3.0), -3.0*e*i/(l**2.0), 0.0, 3.0*e*i/(l**3.0), 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        elif (ends_fixity == "hinge_hinge"):
            k = np.matrix([
                [e*a/l, 0.0, 0.0, -e*a/l, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-e*a/l, 0.0, 0.0, e*a/l, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
