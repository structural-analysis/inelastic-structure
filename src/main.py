import shutil
from datetime import datetime

from src.analysis.initial_analysis import AnalysisType
from src.analysis.inelastic_analysis import InitialAnalysis, InelasticAnalysis
from src.aggregate import aggregate_responses
from src.workshop import get_structure_input, get_loads_input, get_general_properties
from src.response import calculate_responses, write_static_responses_to_file, write_dynamic_responses_to_file, DesiredResponse
from .models.structure import Structure
from .models.loads import Loads


def run(example_name):
    start_time = datetime.now()
    example_path = f"output/examples/{example_name}"
    shutil.rmtree(example_path, ignore_errors=True)
    structure_input = get_structure_input(example_name)
    loads_input = get_loads_input(example_name)
    general_info = get_general_properties(example_name)
    structure = Structure(structure_input)

    loads = Loads(loads_input)
    analysis_type = get_analysis_type(general_info)
    inelastic_analysis = None

    if analysis_type == AnalysisType.STATIC:
        initial_analysis = InitialAnalysis(structure=structure, loads=loads, analysis_type=analysis_type)
        if structure.is_inelastic:
            inelastic_analysis = InelasticAnalysis(initial_analysis=initial_analysis)
    elif analysis_type == AnalysisType.DYNAMIC:
        initial_analysis = InitialAnalysis(structure=structure, loads=loads, analysis_type=analysis_type)
        inelastic_analysis = InelasticAnalysis(initial_analysis=initial_analysis)
        time_steps = initial_analysis.time_steps
        print(f"{time_steps=}")

        for time_step in range(1, time_steps):
            print(f"{time_step=}")
            initial_analysis.update_dynamic_time_step(time_step)
            if structure.is_inelastic:
                inelastic_analysis.update_dynamic_time_step(analysis_data=initial_analysis.analysis_data)
                inelastic_analysis.update_inelasticity_dependent_variables(time_step=time_step, initial_analysis=initial_analysis)
            print("-------------")
    end_time = datetime.now()
    analysis_time = end_time - start_time
    print(f"{analysis_time=}")
    print(f"{analysis_time.microseconds=}")
    responses = calculate_responses(initial_analysis, inelastic_analysis)

    structure_type = "inelastic" if initial_analysis.structure.is_inelastic else "elastic"
    desired_responses = DesiredResponse[structure.type].value
    if structure_type == "inelastic":
        desired_responses.append("plastic_points")
    else:
        if "plastic_points" in desired_responses:
            desired_responses = [desired_response for desired_response in desired_responses if desired_response != "plastic_points"]
    if initial_analysis.analysis_type == "static":
        write_static_responses_to_file(
            example_name=example_name,
            responses=responses,
            desired_responses=desired_responses,
        )
    elif initial_analysis.analysis_type == "dynamic":
        write_dynamic_responses_to_file(
            example_name=example_name,
            structure_type=structure_type,
            responses=responses,
            desired_responses=desired_responses,
            time_steps=initial_analysis.time_steps,
        )
        aggregate_responses(example_name, desired_responses)


def get_analysis_type(general_info):
    if general_info.get("dynamic_analysis") and general_info["dynamic_analysis"]["enabled"]:
        type = AnalysisType.DYNAMIC
    else:
        type = AnalysisType.STATIC
    return type
