import shutil
from datetime import datetime

from src.settings import settings
from src.analysis import Analysis
from src.aggregate import aggregate_responses
from src.workshop import get_structure_input, get_loads_input, get_general_properties
from src.response import calculate_responses, write_static_responses_to_file, write_dynamic_responses_to_file


def run(example_name):
    start_time = datetime.now()
    example_path = f"output/examples/{example_name}"
    shutil.rmtree(example_path, ignore_errors=True)
    structure_input = get_structure_input(example_name)
    loads_input = get_loads_input(example_name)
    general_info = get_general_properties(example_name)
    analysis = Analysis(structure_input=structure_input, loads_input=loads_input, general_info=general_info)
    end_time = datetime.now()
    analysis_time = end_time - start_time
    print(f"{analysis_time=}")
    print(f"{analysis_time.microseconds=}")
    responses = calculate_responses(analysis)
    structure_type = "inelastic" if analysis.structure.is_inelastic else "elastic"
    desired_responses = [
        "load_levels",
        "nodal_disp",
        "nodal_strains",
        "nodal_stresses",
        "nodal_moments",
        "members_disps",
        "members_nodal_forces",
        "members_nodal_strains",
        "members_nodal_stresses",
        "members_nodal_moments",
    ]
    if analysis.type == "static":
        write_static_responses_to_file(
            example_name=example_name,
            responses=responses,
            desired_responses=desired_responses,
        )
    elif analysis.type == "dynamic":
        write_dynamic_responses_to_file(
            example_name=example_name,
            structure_type=structure_type,
            responses=responses,
            desired_responses=desired_responses,
            time_steps=analysis.time_steps,
        )
        aggregate_responses(example_name)


if __name__ == "__main__":
    run(example_name=settings.example_name)
