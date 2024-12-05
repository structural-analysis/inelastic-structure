import numpy as np

from ..program.main import MahiniMethod
from ..functions import get_elastoplastic_response
from .initial_analysis import InitialAnalysis, AnalysisType


class InelasticAnalysis:
    def __init__(self, initial_analysis: InitialAnalysis):
        self.initial_data = initial_analysis.initial_data
        self.analysis_data = initial_analysis.analysis_data
        self.analysis_type = initial_analysis.analysis_type

        if self.analysis_type is AnalysisType.STATIC:
            mahini_method = MahiniMethod(initial_data=self.initial_data, analysis_data=self.analysis_data)
            self.plastic_vars = mahini_method.solve()

        elif self.analysis_type is AnalysisType.DYNAMIC:
            self.final_inc_phi_pms_prev = np.zeros(initial_analysis.initial_data.intact_components_count)
            self.plastic_vars_history = np.zeros((initial_analysis.time_steps, 1), dtype=object)
            self.final_inc_phi_pms_history = np.zeros((initial_analysis.time_steps, initial_analysis.initial_data.intact_components_count))

    def update_dynamic_time_step(self, analysis_data):
        mahini_method = MahiniMethod(
            initial_data=self.initial_data,
            analysis_data=analysis_data,
            final_inc_phi_pms_prev=self.final_inc_phi_pms_prev,
        )
        self.plastic_vars = mahini_method.solve()

    def update_inelasticity_dependent_variables(self, time_step, initial_analysis):
        final_inc_phi_pms = self.plastic_vars["final_inc_phi_pms"]
        final_inc_load_level = self.plastic_vars["load_level_history"][-1]
        self.final_inc_phi_pms_prev = final_inc_phi_pms
        self.plastic_vars_history[time_step, 0] = self.plastic_vars

        self.final_inc_phi_pms_history[time_step, :] = final_inc_phi_pms
        elastoplastic_a2s = get_elastoplastic_response(
            load_level=final_inc_load_level,
            phi_x=final_inc_phi_pms,
            elastic_response=initial_analysis.elastic_a2s,
            sensitivity=initial_analysis.a2_sensitivity,
        )

        elastoplastic_b2s = get_elastoplastic_response(
            load_level=final_inc_load_level,
            phi_x=final_inc_phi_pms,
            elastic_response=initial_analysis.elastic_b2s,
            sensitivity=initial_analysis.b2_sensitivity,
        )

        elastoplastic_modal_loads = get_elastoplastic_response(
            load_level=final_inc_load_level,
            phi_x=final_inc_phi_pms,
            elastic_response=initial_analysis.elastic_modal_loads,
            sensitivity=initial_analysis.modal_loads_sensitivity,
        )

        initial_analysis.previous_a2s = elastoplastic_a2s
        initial_analysis.previous_b2s = elastoplastic_b2s
        initial_analysis.previous_modal_loads = elastoplastic_modal_loads

        # elastoplastic_nodal_disp = get_elastoplastic_response(
        #     load_level=final_inc_load_level,
        #     phi_x=final_inc_phi_pms,
        #     elastic_response=initial_analysis.elastic_nodal_disp_history[time_step, 0],
        #     sensitivity=initial_analysis.nodal_disp_sensitivity_history[time_step, 0],
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
