
class Material:
    def __init__(self, name):
        if name == "steel":
            self.e = 2e11
            self.sy = 240e6
            self.nu = 0.3
