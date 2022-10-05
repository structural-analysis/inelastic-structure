from src.settings import settings
from src.models.loads import Loads
from src.models.structure import Structure
from src.workshop import get_structure_input, get_loads_input, get_general_info

example_name = settings.example_name

structure_input = get_structure_input(example_name)
loads_input = get_loads_input(example_name)
general_info = get_general_info(example_name)


class Analysis:
    def __init__(self, structure_input, loads_input, general_info):
        self.structure = Structure(structure_input)
        self.loads = Loads(loads_input)
        self.general_info = general_info
        self.analysis_type = self._get_analysis_type()

    def _get_analysis_type(self):
        if self.general_info.get("dynamic_analysis"):
            analysis_type = "dynamic"
        else:
            analysis_type = "static"
        return analysis_type

    def analyze_structure(self):
        structure = self.structure
        loads = self.loads
        if self.analysis_type == "static":
            structure = 1
            f = loads.get_load_vector(structure=structure, loads=loads)
            elastic_nodal_disp = structure.get_nodal_disp(loads)
            elastic_elements_disps = structure.get_elements_disps(elastic_nodal_disp)
            structure.elastic_elements_forces = structure.get_internal_forces()["elements_forces"]
            structure.p0 = structure.get_internal_forces()["p0"]
            structure.d0 = structure.get_nodal_disp_limits()

            structure.pv = structure.get_sensitivity()["pv"]
            structure.elements_forces_sensitivity = structure.get_sensitivity()["elements_forces_sensitivity"]
            structure.elements_disps_sensitivity = structure.get_sensitivity()["elements_disps_sensitivity"]
            structure.nodal_disps_sensitivity = structure.get_sensitivity()["nodal_disps_sensitivity"]
            structure.dv = structure.get_nodal_disp_limits_sensitivity_rows()
        elif analysis_type == "dynamic":
            pass
