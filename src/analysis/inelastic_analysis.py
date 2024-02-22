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
            mahini_method = MahiniMethod(initial_data=self.initial_data, analysis_data=self.analysis_data)
            self.plastic_vars = mahini_method.solve_dynamic()
