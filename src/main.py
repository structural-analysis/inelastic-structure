import shutil
from src.settings import settings
from src.analysis import Analysis
from src.aggregate import aggregate_responses
from src.workshop import get_structure_input, get_loads_input, get_general_properties
from src.response import calculate_responses, write_static_responses_to_file, write_dynamic_responses_to_file


def run(example_name):
    example_path = f"output/examples/{example_name}"
    shutil.rmtree(example_path, ignore_errors=True)
    structure_input = get_structure_input(example_name)
    loads_input = get_loads_input(example_name)
    general_info = get_general_properties(example_name)
    analysis = Analysis(structure_input=structure_input, loads_input=loads_input, general_info=general_info)
    responses = calculate_responses(analysis)
    structure_type = "inelastic" if analysis.structure.is_inelastic else "elastic"
    desired_responses = [
        "load_levels",
        "nodal_disp",
        "members_nodal_forces",
        "members_disps",
        "internal_moments",
        "top_internal_strains",
        "bottom_internal_strains",
        "top_internal_stresses",
        "bottom_internal_stresses",
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
