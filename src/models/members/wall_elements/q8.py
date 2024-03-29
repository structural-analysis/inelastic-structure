import numpy as np


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
    n2 = 0.5 * (1 - r ** 2) * (1 - s)
    n4 = 0.5 * (1 + r) * (1 - s ** 2)
    n6 = 0.5 * (1 - r ** 2) * (1 + s)
    n8 = 0.5 * (1 - r) * (1 - s ** 2)

    n1 = 0.25 * (1 - r) * (1 - s) - 0.5 * (n8 + n2)
    n3 = 0.25 * (1 + r) * (1 - s) - 0.5 * (n2 + n4)
    n5 = 0.25 * (1 + r) * (1 + s) - 0.5 * (n4 + n6)
    n7 = 0.25 * (1 - r) * (1 + s) - 0.5 * (n6 + n8)

    n = np.matrix([n1, n2, n3, n4, n5, n6, n7, n8])
    return n


def get_jacobian(gauss_point, nodes):
    r = gauss_point.r
    s = gauss_point.s

    j_left = 0.25 * np.matrix([
        [-2 * r * s + 2 * r - s ** 2 + s, 4 * r * (s - 1), -2 * r * s + 2 * r + s ** 2 - s, 2 - 2 * s ** 2, 2 * r * s + 2 * r + s ** 2 + s, -4 * r * (s + 1), 2 * r * s + 2 * r - s ** 2 - s, 2 * s ** 2 - 2],
        [-r ** 2 - 2 * r * s + r + 2 * s, 2 * r ** 2 - 2, -r ** 2 + 2 * r * s - r + 2 * s, -4 * s * (r + 1), r ** 2 + 2 * r * s + r + 2 * s, 2 - 2 * r ** 2, r ** 2 - 2 * r * s - r + 2 * s, 4 * s * (r - 1)],
    ])

    j_right = np.matrix([
        [nodes[0].x, nodes[0].y],
        [nodes[1].x, nodes[1].y],
        [nodes[2].x, nodes[2].y],
        [nodes[3].x, nodes[3].y],
        [nodes[4].x, nodes[4].y],
        [nodes[5].x, nodes[5].y],
        [nodes[6].x, nodes[6].y],
        [nodes[7].x, nodes[7].y],
    ])

    return np.dot(j_left, j_right)


def get_jacob_det(j):
    return np.linalg.det(j)


def get_b(gauss_point, j):
    r = gauss_point.r
    s = gauss_point.s
    b = np.matrix(np.zeros((3, 16)))
    du = 0.25 * np.linalg.inv(j) * np.matrix([
        [-2 * r * s + 2 * r - s ** 2 + s, 0, 4 * r * (s - 1), 0, -2 * r * s + 2 * r + s ** 2 - s, 0, 2 - 2 * s ** 2, 0, 2 * r * s + 2 * r + s ** 2 + s, 0, -4 * r * (s + 1), 0, 2 * r * s + 2 * r - s ** 2 - s, 0, 2 * s ** 2 - 2, 0],
        [-r ** 2 - 2 * r * s + r + 2 * s, 0, 2 * r ** 2 - 2, 0, -r ** 2 + 2 * r * s - r + 2 * s, 0, -4 * s * (r + 1), 0, r ** 2 + 2 * r * s + r + 2 * s, 0, 2 - 2 * r ** 2, 0, r ** 2 - 2 * r * s - r + 2 * s, 0, 4 * s * (r - 1), 0],
    ])
    dv = 0.25 * np.linalg.inv(j) * np.matrix([
        [0, -2 * r * s + 2 * r - s ** 2 + s, 0, 4 * r * (s - 1), 0, -2 * r * s + 2 * r + s ** 2 - s, 0, 2 - 2 * s ** 2, 0, 2 * r * s + 2 * r + s ** 2 + s, 0, -4 * r * (s + 1), 0, 2 * r * s + 2 * r - s ** 2 - s, 0, 2 * s ** 2 - 2],
        [0, -r ** 2 - 2 * r * s + r + 2 * s, 0, 2 * r ** 2 - 2, 0, -r ** 2 + 2 * r * s - r + 2 * s, 0, -4 * s * (r + 1), 0, r ** 2 + 2 * r * s + r + 2 * s, 0, 2 - 2 * r ** 2, 0, r ** 2 - 2 * r * s - r + 2 * s, 0, 4 * s * (r - 1)],
    ])
    b[0, :] = du[0, :]
    b[1, :] = dv[1, :]
    b[2, :] = du[1, :] + dv[0, :]
    return b


def get_stiffness(gauss_points, nodes, thickness):
    k = np.matrix(np.zeros((16, 16)))
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
        Node(num=1, x=1, y=1.5),
        Node(num=2, x=2, y=0),
        Node(num=3, x=2.5, y=3),
        Node(num=4, x=3, y=6),
        Node(num=5, x=1.75, y=5.5),
        Node(num=6, x=0.5, y=5),
        Node(num=6, x=0.25, y=4),
    ]
    gauss_points = get_gauss_points()
    k = get_stiffness(gauss_points, nodes, thickness)
    kr = k[6:16, 6:16]
    fr = np.matrix([[0, 0, fx, fy, 0, 0, 0, 0, 0, 0]]).T
    return np.linalg.inv(kr) * fr


if __name__ == "__main__":
    disp = get_nodal_disp()
    print(disp)
