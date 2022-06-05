from src.functions import get_elements_max_dof_num


class Element1():
    def __init__(self):
        self.total_dofs_num = 3

class Element2():
    def __init__(self):
        self.total_dofs_num = 6

class Element3():
    def __init__(self):
        self.total_dofs_num = 12

class Element4():
    def __init__(self):
        self.total_dofs_num = 2

class Element5():
    def __init__(self):
        self.total_dofs_num = 9

element1 = Element1()
element2 = Element2()
element3 = Element3()
element4 = Element4()
element5 = Element5()

elements = [element1, element2, element3, element4, element5]
assert(get_elements_max_dof_num(elements)) == 12