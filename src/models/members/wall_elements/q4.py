import numpy as np

# Reference: Finite Element Analysis - Theory and Programming by Krishnamoorthy (p242)


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


def get_n(r, s):
    n = [
        0.25 * (1 - r) * (1 - s),
        0.25 * (1 + r) * (1 - s),
        0.25 * (1 + r) * (1 + s),
        0.25 * (1 - r) * (1 + s),
    ]
    return n


def get_jacobian(gauss_point, nodes):
    r = gauss_point.r
    s = gauss_point.s

    j_left = 0.25 * np.matrix([
        [-(1 - s), (1 - s), (1 + s), -(1 + s)],
        [-(1 - r), -(1 + r), (1 + r), (1 - r)],
    ])

    j_right = np.matrix([
        [nodes[0].x, nodes[0].y],
        [nodes[1].x, nodes[1].y],
        [nodes[2].x, nodes[2].y],
        [nodes[3].x, nodes[3].y],
    ])

    return np.dot(j_left, j_right)


def get_jacob_det(j):
    return np.linalg.det(j)


def get_b(gauss_point, j):
    r = gauss_point.r
    s = gauss_point.s
    b = np.matrix(np.zeros((3, 8)))
    du = 0.25 * np.linalg.inv(j) * np.matrix([
        [-(1 - s), 0, (1 - s), 0, (1 + s), 0, -(1 + s), 0],
        [-(1 - r), 0, -(1 + r), 0, 1 + r, 0, 1 - r, 0],
    ])
    dv = 0.25 * np.linalg.inv(j) * np.matrix([
        [0, -(1 - s), 0, (1 - s), 0, (1 + s), 0, -(1 + s)],
        [0, -(1 - r), 0, -(1 + r), 0, 1 + r, 0, 1 - r],
    ])
    b[0, :] = du[0, :]
    b[1, :] = dv[1, :]
    b[2, :] = du[1, :] + dv[0, :]
    return b


def get_stiffness(gauss_points, nodes, thickness):
    k = np.matrix(np.zeros((8, 8)))
    nu = 0.3
    e = 2e11
    c = np.matrix([[1, nu, 0],
                   [nu, 1, 0],
                   [0, 0, (1 - nu) / 2]])
    ce = (e / (1 - nu ** 2)) * c
    for gauss_point in gauss_points:
        j = get_jacobian(gauss_point, nodes)
        b = get_b(gauss_point, j)
        det_j = get_jacob_det(j)
        gauss_point_k = b.T * ce * b * det_j * thickness
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
