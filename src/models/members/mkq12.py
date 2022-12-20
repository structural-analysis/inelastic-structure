
def get_nbar(r, s):
    nbar = [
        0.25 * (1 - r) * (1 - s),
        0.25 * (1 + r) * (1 - s),
        0.25 * (1 + r) * (1 + s),
        0.25 * (1 - r) * (1 + s),
    ]
    return nbar


def get_jacobian(nodes, r, s):
    j1 = 0.25 * (nodes[0].x * (s - 1) + nodes[1].x * (-s + 1) + nodes[2].x * (s + 1) + nodes[3].x * (-s - 1))
    j2 = 0.25 * (nodes[0].y * (s - 1) + nodes[1].y * (-s + 1) + nodes[2].y * (s + 1) + nodes[3].y * (-s - 1))
    j3 = 0.25 * (nodes[0].x * (r - 1) + nodes[1].x * (-r - 1) + nodes[2].x * (r + 1) + nodes[3].x * (-r + 1))
    j4 = 0.25 * (nodes[0].y * (r - 1) + nodes[1].y * (-r - 1) + nodes[2].y * (r + 1) + nodes[3].y * (-r + 1))
    return [j1, j2, j3, j4]


# must be written in syms and diffed, later b could be a function of r, s
def get_shape_functions(nodes, r, s):
    return []
