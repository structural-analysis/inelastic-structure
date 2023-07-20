import numpy as np

# Hughes Q4 Plate (Thick and Thin) Element.
# Reference: Finite Element Analysis - Theory and Programming by Krishnamoorthy - 1987 - page: 334


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
    b = np.matrix(np.zeros((5, 12)))
    dn = 0.25 * np.linalg.inv(j) * np.matrix([
        [-(1 - s), +(1 - s), +(1 + s), -(1 + s)],
        [-(1 - r), -(1 + r), +(1 + r), +(1 - r)],
    ])
    n = 0.25 * np.matrix([
        [(1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s)],
    ])
    for i in range(4):
        b[0, 3 * (i + 1) - 1] = dn[0, i]
        b[1, 3 * (i + 1) - 2] = -dn[1, i]
        b[2, 3 * (i + 1) - 1] = dn[1, i]
        b[2, 3 * (i + 1) - 2] = -dn[0, i]
        b[3, 3 * (i + 1) - 3] = dn[0, i]
        b[3, 3 * (i + 1) - 1] = n[0, i]
        b[4, 3 * (i + 1) - 3] = dn[1, i]
        b[4, 3 * (i + 1) - 2] = -n[0, i]
    return b

# incomplete due to different quadrature rules for bending and shear according to krishnamoorthy book
# can we use 1 integration point for both bending and shear? so its become q4r?

# def get_stiffness(gauss_points, nodes, thickness):
#     k = np.matrix(np.zeros((12, 12)))
#     nu = 0.3
#     e = 2e11
#     d = np.matrix([[1, nu, 0],
#                    [nu, 1, 0],
#                    [0, 0, (1 - nu) / 2]])
#     de = (e * thickness ** 3) / (12 * (1 - nu ** 2)) * d
#     for gauss_point in gauss_points:
#         j = get_jacobian(gauss_point, nodes)
#         b = get_b(gauss_point, j)
#         det_j = get_jacob_det(j)
#         gauss_point_k = b.T * de * b * det_j * thickness
#         k += gauss_point_k
#     return k


# def get_nodal_disp():
#     thickness = 0.02
#     fz = - 2000000
#     nodes = [
#         Node(num=0, x=0, y=3),
#         Node(num=1, x=2, y=0),
#         Node(num=2, x=3, y=6),
#         Node(num=3, x=0.5, y=5),
#     ]
#     gauss_points = get_gauss_points()
#     k = get_stiffness(gauss_points, nodes, thickness)
#     kr = k[6:12, 6:12]
#     print(kr)
#     fr = np.matrix([[fz, 0, 0, 0, 0, 0]]).T
#     return np.linalg.inv(kr) * fr


# if __name__ == "__main__":
#     disp = get_nodal_disp()
#     print(disp)
