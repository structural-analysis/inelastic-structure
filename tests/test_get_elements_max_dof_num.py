from src.functions import get_members_max_dof_num


class Member1():
    def __init__(self):
        self.total_dofs_num = 3


class Member2():
    def __init__(self):
        self.total_dofs_num = 6


class Member3():
    def __init__(self):
        self.total_dofs_num = 12


class Member4():
    def __init__(self):
        self.total_dofs_num = 2


class Member5():
    def __init__(self):
        self.total_dofs_num = 9


member1 = Member1()
member2 = Member2()
member3 = Member3()
member4 = Member4()
member5 = Member5()

members = [member1, member2, member3, member4, member5]
assert(get_members_max_dof_num(members)) == 12
