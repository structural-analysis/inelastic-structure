class YieldSpecs:
    def __init__(self, yield_specs_dict):
        self.points_num = yield_specs_dict["points_num"]
        self.components_num = yield_specs_dict["components_num"]
        self.pieces_num = yield_specs_dict["pieces_num"]


class Elements:
    def __init__(self, elements_list):
        self.list = elements_list
        self.num = len(elements_list)
        self.yield_specs_dict = self.get_yield_specs_dict()
        self.yield_specs = YieldSpecs(self.yield_specs_dict)

    def get_yield_specs_dict(self):
        points_num = 0
        components_num = 0
        pieces_num = 0

        for element in self.list:
            points_num += element.yield_specs.points_num
            components_num += element.yield_specs.components_num
            pieces_num += element.yield_specs.pieces_num

        yield_specs_dict = {
            "points_num": points_num,
            "components_num": components_num,
            "pieces_num": pieces_num,
        }
        return yield_specs_dict
