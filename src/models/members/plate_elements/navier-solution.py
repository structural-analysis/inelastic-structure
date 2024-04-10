import numpy as np


def deflection(a, b, t, v, e, q, x, y, last=5):
    # last is 1, 3, 5, 7, ...
    w = 0
    d = ((e * t ** 3) / (12 * (1 - v ** 2)))
    pmn = (16 * q) / ((np.pi ** 6) * d)
    for m in range(1, last + 1, 2):
        for n in range(1, last + 1, 2):
            w += (np.sin(m * np.pi * x / a) * np.sin(n * np.pi * y / b)) / (m * n * ((m / a) ** 2 + (n / b) ** 2) ** 2)
    return pmn * w

# Example usage:
a = 6  # Length of the plate in x-direction
b = 3  # Length of the plate in y-direction
t = 0.05
v = 0.3
e = 2e11
q = 30000
x = 3  # x-coordinate of the point where deflection is calculated
y = 1.5   # y-coordinate of the point where deflection is calculated
last = 43  # Number of last term in the series

deflection_value = deflection(a, b, t, v, e, q, x, y, last)
print("Deflection at point ({}, {}) is {:.6f}".format(x, y, deflection_value))