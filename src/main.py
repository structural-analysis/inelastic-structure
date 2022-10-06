from src.settings import settings
from src.analysis import Analysis
from src.models.loads import Loads
from src.program.prepare import RawData
from src.program.main import MahiniMethod
from src.models.structure import Structure
from src.response import calculate_responses
from src.workshop import get_structure_input, get_loads_input, get_general_properties


def run(example_name):
    structure_input = get_structure_input(example_name)
    # structure = Structure(input=structure_input)
    loads_input = get_loads_input(example_name)
    general_info = get_general_properties(example_name)
    analysis = Analysis(structure_input=structure_input, loads_input=loads_input, general_info=general_info)

    raw_data = RawData(analysis)
    mahini_method = MahiniMethod(raw_data)
    result = mahini_method.solve()
    calculate_responses(analysis, result, example_name)


if __name__ == "__main__":
    run(example_name=settings.example_name)
