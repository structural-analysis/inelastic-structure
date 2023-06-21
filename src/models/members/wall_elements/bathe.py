import numpy as np
import sympy as sp
from sympy import symbols as syms


class Node:
    def __init__(self, num, x, y):
        self.num = num
        self.x = x
        self.y = y

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
    j0 = sp.diff(nbars[0], r) * x0 + sp.diff(nbars[1], r) * x1 + sp.diff(nbars[2], r) * x2 + sp.diff(nbars[3], r) * x3
    # ry/rr
    j1 = sp.diff(nbars[0], r) * y0 + sp.diff(nbars[1], r) * y1 + sp.diff(nbars[2], r) * y2 + sp.diff(nbars[3], r) * y3
    # rx/rs
    j2 = sp.diff(nbars[0], s) * x0 + sp.diff(nbars[1], s) * x1 + sp.diff(nbars[2], s) * x2 + sp.diff(nbars[3], s) * x3
    # ry/rs
    j3 = sp.diff(nbars[0], s) * y0 + sp.diff(nbars[1], s) * y1 + sp.diff(nbars[2], s) * y2 + sp.diff(nbars[3], s) * y3
    return j0, j1, j2, j3


def get_jacobian():
    r, s = syms('r s')
    x0, x1, x2, x3, y0, y1, y2, y3 = syms('x0 x1 x2 x3 y0 y1 y2 y3')
    # rx/rr
    j0 = 0.25 * (x0 * (s - 1) + x1 * (-s + 1) + x2 * (s + 1) + x3 * (-s - 1))
    # ry/rr
    j1 = 0.25 * (y0 * (s - 1) + y1 * (-s + 1) + y2 * (s + 1) + y3 * (-s - 1))
    # rx/rs
    j2 = 0.25 * (x0 * (r - 1) + x1 * (-r - 1) + x2 * (r + 1) + x3 * (-r + 1))
    # ry/rs
    j3 = 0.25 * (y0 * (r - 1) + y1 * (-r - 1) + y2 * (r + 1) + y3 * (-r + 1))
    return j0, j1, j2, j3


def calculate_det():
    j0, j1, j2, j3 = get_jacobian()
    return j0 * j3 - j1 * j2


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


def calculate_shape_derivatives():
    r, s = syms('r s')
    n0 = 0.25 * (1 - r) * (1 - s)
    n1 = 0.25 * (1 + r) * (1 - s)
    n2 = 0.25 * (1 + r) * (1 + s)
    n3 = 0.25 * (1 - r) * (1 + s)

    j0, j1, j2, j3 = get_jacobian()
    jinv = (1 / (j0 * j3 - j1 * j2)) * np.matrix([
        [j3, -j1],
        [-j2, j0],
    ])

    u_derivatives = sp.zeros(2, 8)
    v_derivatives = sp.zeros(2, 8)
    b = sp.zeros(3, 8)

    u_derivatives[0, :] = np.array([[sp.diff(n0, r), 0, sp.diff(n1, r), 0, sp.diff(n2, r), 0, sp.diff(n3, r), 0]])
    u_derivatives[1, :] = np.array([[sp.diff(n0, s), 0, sp.diff(n1, s), 0, sp.diff(n2, s), 0, sp.diff(n3, s), 0]])

    v_derivatives[0, :] = np.array([[0, sp.diff(n0, r), 0, sp.diff(n1, r), 0, sp.diff(n2, r), 0, sp.diff(n3, r)]])
    v_derivatives[1, :] = np.array([[0, sp.diff(n0, s), 0, sp.diff(n1, s), 0, sp.diff(n2, s), 0, sp.diff(n3, s)]])

    u_derivatives = np.dot(jinv, u_derivatives)
    v_derivatives = np.dot(jinv, v_derivatives)

    b[0, :] = u_derivatives[0, :]
    b[1, :] = v_derivatives[1, :]
    b[2, :] = u_derivatives[1, :] + v_derivatives[0, :]

    return sp.simplify(b)


def get_gauss_point_shape_derivatives(gauss_point, nodes):
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
    return np.matrix([[1.0*((r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) -
y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0, 1.0*(-(r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0, 1.0*((r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0, 1.0*(-(r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0], [0, 1.0*(-(r - 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1)
+ y2*(r + 1) - y3*(r - 1))), 0, 1.0*((r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r -
1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0, 1.0*(-(r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 0, 1.0*((r - 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))], [1.0*(-(r - 1)*(x0*(s
- 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r
- 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1)
- y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s - 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1)
- x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s
+ 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s - 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r + 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) + (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r + 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*((r - 1)*(x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1)) - (s + 1)*(x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1))), 1.0*(-(r - 1)*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) + (s + 1)*(y0*(r - 1) - y1*(r + 1) + y2*(r +
1) - y3*(r - 1)))/((x0*(r - 1) - x1*(r + 1) + x2*(r + 1) - x3*(r - 1))*(y0*(s - 1) - y1*(s - 1) + y2*(s + 1) - y3*(s + 1)) - (x0*(s - 1) - x1*(s - 1) + x2*(s + 1) - x3*(s + 1))*(y0*(r - 1) - y1*(r + 1) + y2*(r + 1) - y3*(r - 1)))]])


def get_stiffness(gauss_points, nodes, thickness):
    k = np.matrix(np.zeros((8, 8)))
    nu = 0.3
    e = 2e11
    c = np.matrix([[1, nu, 0],
                   [nu, 1, 0],
                   [0, 0, (1 - nu) / 2]])
    ce = (e / (1 - nu ** 2)) * c
    for gauss_point in gauss_points:
        gauss_point_b = get_gauss_point_shape_derivatives(gauss_point, nodes)
        gauss_point_det = get_det(gauss_point, nodes)
        gauss_point_k = gauss_point_b.T * ce * gauss_point_b * gauss_point_det * thickness
        k += gauss_point_k
    return k


def get_nodal_disp():
    thickness = 0.01
    fx = 2000000
    fy = -1500000
    nodes = [
        Node(num=0, x=0, y=3),
        Node(num=1, x=2, y=0),
        Node(num=2, x=3, y=6),
        Node(num=3, x=0.5, y=5),
    ]
    gauss_points = get_gauss_points()
    k = get_stiffness(gauss_points, nodes, thickness)
    kr = k[4:8, 4:8]
    fr = np.matrix([[fx, fy, 0, 0]]).T
    return np.linalg.inv(kr) * fr

if __name__ == "__main__":
    disp = get_nodal_disp()
    print(disp)
