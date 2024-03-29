
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


class NaturalPoint:
    def __init__(self, r, s):
        self.r = r
        self.s = s

    def __eq__(self, other):
        return self.r == other.r and self.s == other.s

    def __hash__(self):
        return hash(('r', self.r, 's', self.s))


class GaussPoint:
    def __init__(self, weight, r, s):
        self.weight = weight
        self.r = r
        self.s = s

    def __eq__(self, other):
        return self.weight == other.weight and self.r == other.r and self.s == other.s

    def __hash__(self):
        return hash(('weight', self.weight, 'r', self.r, 's', self.s))
