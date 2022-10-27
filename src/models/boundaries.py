from .points import Node


class NodalBoundary:
    def __init__(self, node, dof):
        self.node: Node = node
        self.dof: int = dof

    def __eq__(self, other):
        return self.node.num == other.node.num and self.dof == other.dof

    def __hash__(self):
        return hash(('node', self.node.num, 'dof', self.dof))


class LinearBoundary:
    def __init__(self, start_node, end_node, dof):
        self.start_node: Node = start_node
        self.end_node: Node = end_node
        self.dof: int = dof


class NodeDOFRestrainer:
    def __init__(self, dimension_name, dimension_value, dof):
        self.dimension_name: str = dimension_name
        self.dimension_value: str = dimension_value
        self.dof: int = dof
