# import numpy as np


# class PlateSection:
#     # nu: poisson ratio
#     def __init__(self):
#         e = material.e
#         nu = material.nu
#         sy = material.sy
#         d = np.matrix([[1, nu, 0],
#                       [nu, 1, 0],
#                       [0, 0, (1 - nu) / 2]])
#         self.t = t
#         self.mp = 0.25 * t ** 2 * sy
#         self.be = (e / (1 - nu ** 2)) * d
#         self.de = (e * t ** 3) / (12 * (1 - nu ** 2)) * d
