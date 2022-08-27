class Distributed:
    def __init__(self, element, magnitude):
        self.element = element
        self.magnitude = magnitude


class Dynamic:
    def __init__(self, node, dof, time, load):
        self.node = node
        self.dof = dof
        self.time = time
        self.load = load
