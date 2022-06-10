from src.workshop import create_structure
from src.program.main import MahiniMethod
from src.response import calculate_responses
from src.program.prepare import RawData
from src.settings import settings


def run(example_name):
    structure = create_structure(example_name)
    raw_data = RawData(structure)
    mahini_method = MahiniMethod(raw_data)
    result = mahini_method.solve()
    calculate_responses(structure, result, example_name)


if __name__ == "__main__":
    run(example_name=settings.example_name)
