import numpy as np
import sympy as sp
from sympy import symbols as syms


class Node:
    def __init__(self, num, x, y, z):
        self.num = num
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.num == other.num

    def __hash__(self):
        return hash(('num', self.num))

    def __gt__(self, other):
        return self.num > other.num


class GaussPoint:
    def __init__(self, r, s):
        self.r = r
        self.s = s

    def __eq__(self, other):
        return self.r == other.r and self.s == other.s

    def __hash__(self):
        return hash(('r', self.r, 's', self.s))


def get_nodes():
    nodes = [
        Node(num=0, x=0, y=0, z=0),
        Node(num=1, x=1, y=0, z=0),
        Node(num=2, x=1, y=1, z=0),
        Node(num=3, x=0, y=1, z=0),
    ]
    return nodes


def get_gauss_points():
    gauss_points = [
        GaussPoint(r=-0.57735, s=-0.57735),
        GaussPoint(r=+0.57735, s=-0.57735),
        GaussPoint(r=+0.57735, s=+0.57735),
        GaussPoint(r=-0.57735, s=+0.57735),
    ]
    return gauss_points


def get_nbar(r, s):
    nbar = [
        0.25 * (1 - r) * (1 - s),
        0.25 * (1 + r) * (1 - s),
        0.25 * (1 + r) * (1 + s),
        0.25 * (1 - r) * (1 + s),
    ]
    return nbar


def calculate_jacobian():
    r, s = syms('r s')
    x0, x1, x2, x3, y0, y1, y2, y3 = syms('x0 x1 x2 x3 y0 y1 y2 y3')
    nbars = get_nbar(r, s)
    # rx/rr
    j1 = sp.diff(nbars[0], r) * x0 + sp.diff(nbars[1], r) * x1 + sp.diff(nbars[2], r) * x2 + sp.diff(nbars[3], r) * x3
    # ry/rr
    j2 = sp.diff(nbars[0], r) * y0 + sp.diff(nbars[1], r) * y1 + sp.diff(nbars[2], r) * y2 + sp.diff(nbars[3], r) * y3
    # rx/rs
    j3 = sp.diff(nbars[0], s) * x0 + sp.diff(nbars[1], s) * x1 + sp.diff(nbars[2], s) * x2 + sp.diff(nbars[3], s) * x3
    # ry/rs
    j4 = sp.diff(nbars[0], s) * y0 + sp.diff(nbars[1], s) * y1 + sp.diff(nbars[2], s) * y2 + sp.diff(nbars[3], s) * y3
    return j1, j2, j3, j4


def get_jacobian():
    r, s = syms('r s')
    x0, x1, x2, x3, y0, y1, y2, y3 = syms('x0 x1 x2 x3 y0 y1 y2 y3')

    # rx/rr
    j1 = 0.25 * (x0 * (s - 1) + x1 * (-s + 1) + x2 * (s + 1) + x3 * (-s - 1))
    # ry/rr
    j2 = 0.25 * (y0 * (s - 1) + y1 * (-s + 1) + y2 * (s + 1) + y3 * (-s - 1))
    # rx/rs
    j3 = 0.25 * (x0 * (r - 1) + x1 * (-r - 1) + x2 * (r + 1) + x3 * (-r + 1))
    # ry/rs
    j4 = 0.25 * (y0 * (r - 1) + y1 * (-r - 1) + y2 * (r + 1) + y3 * (-r + 1))

    return j1, j2, j3, j4


def get_numerical_jacobian(nodes):
    r, s = syms('r s')
    x0 = nodes[0].x
    x1 = nodes[1].x
    x2 = nodes[2].x
    x3 = nodes[3].x

    y0 = nodes[0].y
    y1 = nodes[1].y
    y2 = nodes[2].y
    y3 = nodes[3].y

    # rx/rr
    j1 = 0.25 * (x0 * (s - 1) + x1 * (-s + 1) + x2 * (s + 1) + x3 * (-s - 1))
    # ry/rr
    j2 = 0.25 * (y0 * (s - 1) + y1 * (-s + 1) + y2 * (s + 1) + y3 * (-s - 1))
    # rx/rs
    j3 = 0.25 * (x0 * (r - 1) + x1 * (-r - 1) + x2 * (r + 1) + x3 * (-r + 1))
    # ry/rs
    j4 = 0.25 * (y0 * (r - 1) + y1 * (-r - 1) + y2 * (r + 1) + y3 * (-r + 1))

    return j1, j2, j3, j4


def calculate_det():
    j1, j2, j3, j4 = get_jacobian()
    return j1 * j4 - j2 * j3


def get_det(gauss_point, nodes):
    r = gauss_point.r
    s = gauss_point.s
    x0 = nodes[0].x
    x1 = nodes[1].x
    x2 = nodes[2].x
    x3 = nodes[3].x

    y0 = nodes[0].y
    y1 = nodes[1].y
    y2 = nodes[2].y
    y3 = nodes[3].y
    det = -(0.25 * x0 * (r - 1) + 0.25 * x1 * (-r - 1) + 0.25 * x2 * (r + 1) + 0.25 * x3 * (1 - r)) * (0.25 * y0 * (s - 1) + 0.25 * y1 * (1 - s) + 0.25 * y2 * (s + 1) + 0.25 * y3 * (-s - 1)) + (0.25 * x0 * (s - 1) + 0.25 * x1 * (1 - s) + 0.25 * x2 * (s + 1) + 0.25 * x3 * (-s - 1)) * (0.25 * y0 * (r - 1) + 0.25 * y1 * (-r - 1) + 0.25 * y2 * (r + 1) + 0.25 * y3 * (1 - r))
    return det


# must be written in syms and diffed, later b could be a function of r, s
def calculate_isoparametric_shape_functions():
    r, s = syms('r s')
    n1 = 0.25 * (1 - r) * (1 - s)
    n2 = 0.25 * (1 - r) * (1 - s)
    n3 = 0.25 * (1 + r) * (1 - s)
    n4 = 0.25 * (1 + r) * (1 - s)
    n5 = 0.25 * (1 + r) * (1 + s)
    n6 = 0.25 * (1 + r) * (1 + s)
    n7 = 0.25 * (1 - r) * (1 + s)
    n8 = 0.25 * (1 - r) * (1 + s)
    return n1, n2, n3, n4, n5, n6, n7, n8


def calculate_isoparametric_shape_derivatives():
    r, s = syms('r s')
    n = calculate_isoparametric_shape_functions()
    b = sp.zeros(2, 8)
    for i in range(8):
        b[0, i] = sp.diff(n[i], r)
        b[1, i] = sp.diff(n[i], s)
    return b


def calculate_shape_derivatives():
    j1, j2, j3, j4 = get_jacobian()
    jinv = (1 / (j1 * j4 - j2 * j3)) * np.matrix([
        [j4, -j2],
        [-j3, j1],
    ])
    b = calculate_isoparametric_shape_derivatives()
    bb = np.dot(jinv, b)
    return sp.simplify(bb)


def get_simple_shape_derivatives(gauss_point, nodes):
    r = gauss_point.r
    s = gauss_point.s
    x0 = nodes[0].x
    x1 = nodes[1].x
    x2 = nodes[2].x
    x3 = nodes[3].x

    y0 = nodes[0].y
    y1 = nodes[1].y
    y2 = nodes[2].y
    y3 = nodes[3].y
    return np.matrix([[1.0*((r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))], [1.0*(-(r - 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r - 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r - 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r - 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))]])


# def get_gauss_points_shape_derivatives(gauss_points, nodes):
#     return [get_mkq12_complicated_negative_shape_derivatives(gauss_point, nodes) for gauss_point in gauss_points]


def get_stiffness_integrand(gauss_point_b):
    nu = 0.3
    e = 2e11
    t = 0.05
    d = np.matrix([[1, nu, 0],
                   [nu, 1, 0],
                   [0, 0, (1 - nu) / 2]])
    de = (e * t ** 3) / (12 * (1 - nu ** 2)) * d
    ki = gauss_point_b.T * de * gauss_point_b
    return ki


def get_stiffness(gauss_points):
    nodes = get_nodes()
    size_x = nodes[1].x - nodes[0].x
    size_y = nodes[2].y - nodes[0].y
    ax = (size_x / 2)
    ay = (size_y / 2)
    det = ax * ay
    kin = np.matrix(np.zeros((12, 12)))
    gauss_points_shape_derivatives = get_gauss_points_shape_derivatives(gauss_points, nodes)
    for gauss_point_b in gauss_points_shape_derivatives:
        kin += get_stiffness_integrand(gauss_point_b)
    k = kin * det
    return k


def get_nodal_disp():
    gauss_points = get_gauss_points()
    k = get_stiffness(gauss_points)
    kr = k[0:3, 0:3]
    fr = np.matrix([[-1000000, 0, 0]]).T
    return kr ** -1 * fr


if __name__ == "__main__":
    bb = calculate_shape_derivatives()
    print(bb)
