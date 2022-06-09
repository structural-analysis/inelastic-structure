from src.workshop import create_structure
from src.programming import solve_by_mahini_approach
from src.response import calculate_responses
from src.prepare import get_analysis_data
from src.settings import settings


def run(example_name):
    structure = create_structure(example_name)
    analysis_data = get_analysis_data(structure)
    result = solve_by_mahini_approach(analysis_data)
    calculate_responses(structure, result, example_name)


if __name__ == "__main__":
    run(example_name=settings.example_name)
