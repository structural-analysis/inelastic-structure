import shutil
import numpy as np
from datetime import datetime

from src.settings import settings
from src.analysis.initial_analysis import AnalysisType
from src.analysis.inelastic_analysis import InitialAnalysis, InelasticAnalysis
from src.aggregate import aggregate_responses
from src.workshop import get_structure_input, get_loads_input, get_general_properties
from src.response import calculate_responses, write_static_responses_to_file, write_dynamic_responses_to_file
from .models.structure import Structure
from .models.loads import Loads
from .functions import get_elastoplastic_response


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

    if structure.is_inelastic:
        if analysis_type == AnalysisType.STATIC:
            initial_analysis = InitialAnalysis(structure=structure, loads=loads, analysis_type=analysis_type)
            inelastic_analysis = InelasticAnalysis(initial_analysis=initial_analysis)
        elif analysis_type == AnalysisType.DYNAMIC:
            initial_analysis = InitialAnalysis(structure=structure, loads=loads, analysis_type=analysis_type)
            time_steps = initial_analysis.time_steps
            # Create initial phi_x_prev by its size
            phi_x_prev = np.matrix(np.zeros((initial_analysis.initial_data.intact_components_count, 1)))
            for time_step in range(1, time_steps):
                initial_analysis.update_dynamic_time_step()
                inelastic_analysis = InelasticAnalysis(initial_analysis=initial_analysis)
                delta_phi_x = inelastic_analysis.plastic_vars["phi_pms_history"][-1]
                load_level = inelastic_analysis.plastic_vars["load_level_history"][-1]
                phi_x = delta_phi_x + phi_x_prev
                phi_x_prev = phi_x

                elastoplastic_a2s = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_x,
                    elastic_response=initial_analysis.a_duhamel[time_step, 0],
                    sensitivity=initial_analysis.a2_sensitivity_history[time_step, 0],
                )

                elastoplastic_b2s = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_x,
                    elastic_response=initial_analysis.b_duhamel[time_step, 0],
                    sensitivity=initial_analysis.b2_sensitivity_history[time_step, 0],
                )

                elastoplastic_modal_loads = get_elastoplastic_response(
                    load_level=load_level,
                    phi_x=phi_x,
                    elastic_response=initial_analysis.modal_loads[time_step, 0],
                    sensitivity=initial_analysis.modal_loads_sensitivity_history[time_step, 0],
                )

                initial_analysis.a_duhamel[time_step, 0] = elastoplastic_a2s
                initial_analysis.b_duhamel[time_step, 0] = elastoplastic_b2s
                initial_analysis.modal_loads[time_step, 0] = elastoplastic_modal_loads

                # elastoplastic_nodal_disp = get_elastoplastic_response(
                #     load_level=load_level,
                #     phi_x=phi_x,
                #     elastic_response=elastic_nodal_disp,
                #     sensitivity=sensitivity.nodal_disp,
                # )

                # elastoplastic_members_disps = get_elastoplastic_response(
                #     load_level=load_level,
                #     phi_x=phi_x,
                #     elastic_response=elastic_members_disps,
                #     sensitivity=sensitivity.members_disps,
                # )

                # elastoplastic_members_nodal_forces = get_elastoplastic_response(
                #     load_level=load_level,
                #     phi_x=phi_x,
                #     elastic_response=internal_responses.members_nodal_forces,
                #     sensitivity=sensitivity.members_nodal_forces,
                # )
    else:
        inelastic_analysis = None

    end_time = datetime.now()
    analysis_time = end_time - start_time
    print(f"{analysis_time=}")
    print(f"{analysis_time.microseconds=}")
    responses = calculate_responses(initial_analysis, inelastic_analysis)
    structure_type = "inelastic" if initial_analysis.structure.is_inelastic else "elastic"
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
            time_steps=inelastic_analysis.time_steps,
        )
        aggregate_responses(example_name)


def get_analysis_type(general_info):
    if general_info.get("dynamic_analysis") and general_info["dynamic_analysis"]["enabled"]:
        type = AnalysisType.DYNAMIC
    else:
        type = AnalysisType.STATIC
    return type


if __name__ == "__main__":
    run(example_name=settings.example_name)
