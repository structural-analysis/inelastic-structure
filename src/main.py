from src.settings import settings
from src.program.prepare import RawData
from src.program.main import MahiniMethod
from src.models.structure import Structure
from src.response import calculate_responses
from src.workshop import get_structure_input


def run(example_name):
    structure_input = get_structure_input(example_name)
    structure = Structure(input=structure_input)
    raw_data = RawData(structure)
    mahini_method = MahiniMethod(raw_data)
    result = mahini_method.solve()
    calculate_responses(structure, result, example_name)


if __name__ == "__main__":
    run(example_name=settings.example_name)
