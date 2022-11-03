from src.settings import settings
from src.analysis import Analysis
from src.response import calculate_responses, write_responses_to_file
from src.workshop import get_structure_input, get_loads_input, get_general_properties


def run(example_name):
    structure_input = get_structure_input(example_name)
    loads_input = get_loads_input(example_name)
    general_info = get_general_properties(example_name)
    analysis = Analysis(structure_input=structure_input, loads_input=loads_input, general_info=general_info)
    responses = calculate_responses(analysis)
    desired_responses = [
        "nodal_disps",
        "members_forces",
        "members_disps",
        "load_levels",
    ]
    write_responses_to_file(example_name, responses, desired_responses)


if __name__ == "__main__":
    run(example_name=settings.example_name)
