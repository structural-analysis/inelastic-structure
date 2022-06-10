
class FPM():
    var_num: int
    cost: float


class WillOut():
    row_num: int
    var_num: int


class SlackCandidate():
    def __init__(self, var_num, cost):
        self.var_num = var_num
        self.cost = cost
