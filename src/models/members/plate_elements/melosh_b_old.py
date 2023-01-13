import numpy as np


def get_melosh_old_shape_derivatives(gauss_point, nodes):
    r = gauss_point.r
    s = gauss_point.s
    size_x = nodes[1].x - nodes[0].x
    size_y = nodes[2].y - nodes[1].y
    ax = (size_x / 2)
    ay = (size_y / 2)
    b = np.matrix([[((0.125 - 0.125 * r) * (2 * s - 2) - 0.125 * (1 - s) * (-2 * r - 1) + (-2 * r - 1) * (0.125 * s - 0.125)) / ax ** 2,
                    0,
                    (-0.5 * ax * r * (1 - s) + 2 * ax * (0.125 - 0.125 * r) * (1 - s)) / ax ** 2,
                    ((0.125 - 0.125 * s) * (1 - 2 * r) + 0.125 * (1 - 2 * r) * (1 - s) + (0.125 * r + 0.125) * (2 * s - 2)) / ax ** 2,
                    0,
                    (-0.5 * ax * r * (1 - s) - 2 * ax * (1 - s) * (0.125 * r + 0.125)) / ax ** 2,
                    ((1 - 2 * r) * (0.125 * s + 0.125) + 0.125 * (1 - 2 * r) * (s + 1) + (0.125 * r + 0.125) * (-2 * s - 2)) / ax ** 2,
                    0,
                    (-0.5 * ax * r * (s + 1) - 2 * ax * (0.125 * r + 0.125) * (s + 1)) / ax ** 2,
                    ((0.125 - 0.125 * r) * (-2 * s - 2) + (-2 * r - 1) * (-0.125 * s - 0.125) - 0.125 * (-2 * r - 1) * (s + 1)) / ax ** 2,
                    0,
                    (-0.5 * ax * r * (s + 1) + 2 * ax * (0.125 - 0.125 * r) * (s + 1)) / ax ** 2],
                    [((0.125 - 0.125 * r) * (2 * s - 2) + (0.125 - 0.125 * r) * (2 * s + 1) + (0.125 * r - 0.125) * (-2 * s - 1)) / ay ** 2,
                    (4 * ay * s * (0.125 - 0.125 * r) - 2 * ay * (0.125 - 0.125 * r) * (1 - s)) / ay ** 2,
                    0,
                    ((-0.125 * r - 0.125) * (-2 * s - 1) + (0.125 * r + 0.125) * (2 * s - 2) + (0.125 * r + 0.125) * (2 * s + 1)) / ay ** 2,
                    (4 * ay * s * (0.125 * r + 0.125) - 2 * ay * (1 - s) * (0.125 * r + 0.125)) / ay ** 2,
                    0,
                    (2 * (1 - 2 * s) * (0.125 * r + 0.125) + (0.125 * r + 0.125) * (-2 * s - 2)) / ay ** 2,
                    (4 * ay * s * (0.125 * r + 0.125) + 2 * ay * (0.125 * r + 0.125) * (s + 1)) / ay ** 2,
                    0,
                    (2 * (0.125 - 0.125 * r) * (1 - 2 * s) + (0.125 - 0.125 * r) * (-2 * s - 2)) / ay ** 2,
                    (4 * ay * s * (0.125 - 0.125 * r) + 2 * ay * (0.125 - 0.125 * r) * (s + 1)) / ay ** 2,
                    0],
                    [2 * (-0.125 * r ** 2 - 0.125 * r - 0.125 * s ** 2 - 0.125 * s + (0.125 - 0.125 * r) * (2 * r + 1) + (-2 * s - 1) * (0.125 * s - 0.125) + 0.25) / (ax * ay),
                    2 * (0.25 * ay * s * (1 - s) + 0.125 * ay * (1 - s ** 2)) / (ax * ay),
                    2 * (-2 * ax * r * (0.125 - 0.125 * r) - 0.125 * ax * (1 - r ** 2)) / (ax * ay),
                    2 * (0.125 * r ** 2 - 0.125 * r + 0.125 * s ** 2 + 0.125 * s + (0.125 - 0.125 * s) * (-2 * s - 1) + (0.125 * r + 0.125) * (2 * r - 1) - 0.25) / (ax * ay),
                    2 * (-0.25 * ay * s * (1 - s) - 0.125 * ay * (1 - s ** 2)) / (ax * ay),
                    2 * (2 * ax * r * (0.125 * r + 0.125) - 0.125 * ax * (1 - r ** 2)) / (ax * ay),
                    2 * (-0.125 * r ** 2 + 0.125 * r - 0.125 * s ** 2 + 0.125 * s + (1 - 2 * r) * (0.125 * r + 0.125) + (1 - 2 * s) * (0.125 * s + 0.125) + 0.25) / (ax * ay),
                    2 * (0.25 * ay * s * (s + 1) - 0.125 * ay * (1 - s ** 2)) / (ax * ay),
                    2 * (-2 * ax * r * (0.125 * r + 0.125) + 0.125 * ax * (1 - r ** 2)) / (ax * ay),
                    2 * (0.125 * r ** 2 + 0.125 * r + 0.125 * s ** 2 - 0.125 * s + (0.125 - 0.125 * r) * (-2 * r - 1) + (1 - 2 * s) * (-0.125 * s - 0.125) - 0.25) / (ax * ay),
                    2 * (-0.25 * ay * s * (s + 1) + 0.125 * ay * (1 - s ** 2)) / (ax * ay),
                    2 * (2 * ax * r * (0.125 - 0.125 * r) + 0.125 * ax * (1 - r ** 2)) / (ax * ay)]])
    return b