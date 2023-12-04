import numpy as np
import enum
from dataclasses import dataclass

from .functions import (
    get_nodal_disp_limits_sensitivity_rows,
    get_nodal_disp,
    get_members_disps,
    get_internal_responses,
    get_sensitivity,
    get_nodal_disp_limits,
)
from ..models.structure import Structure
from ..models.loads import Loads


class AnalysisType(str, enum.Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass
class AnalysisData:
    p0: np.matrix
    pv: np.matrix
    d0: np.matrix
    dv: np.matrix


@dataclass
class InitialData:
    load_limit: float
    disp_limits: list
    disp_limits_count: int
    include_softening: bool
    intact_points: np.matrix
    intact_pieces: np.matrix
    intact_points_count: int
    intact_components_count: int
    intact_pieces_count: int
    yield_points_indices: list
    intact_phi: np.matrix
    intact_q: np.matrix
    intact_h: np.matrix
    intact_w: np.matrix
    intact_cs: np.matrix


class InitialAnalysis:
    def __init__(self, structure_input, loads_input, general_info):
        self.structure = Structure(structure_input)
        self.loads = Loads(loads_input)
        self.general_info = general_info
        self.initial_data = InitialData
        self.analysis_data = AnalysisData

        self.initial_data.load_limit = self.structure.limits["load_limit"][0]
        self.initial_data.disp_limits = self.structure.limits["disp_limits"]
        self.initial_data.disp_limits_count = self.initial_data.disp_limits.shape[0]
        self.initial_data.include_softening = self.structure.include_softening

        self.initial_data.intact_points = self.structure.yield_specs.intact_points
        self.initial_data.intact_pieces = self.structure.yield_specs.intact_pieces
        self.initial_data.intact_points_count = self.structure.yield_specs.intact_points_count
        self.initial_data.intact_components_count = self.structure.yield_specs.intact_components_count
        self.initial_data.intact_pieces_count = self.structure.yield_specs.intact_pieces_count
        self.initial_data.yield_points_indices = self.structure.yield_specs.yield_points_indices
        self.initial_data.intact_phi = self.structure.yield_specs.intact_phi
        self.initial_data.intact_q = self.structure.yield_specs.intact_q
        self.initial_data.intact_h = self.structure.yield_specs.intact_h
        self.initial_data.intact_w = self.structure.yield_specs.intact_w
        self.initial_data.intact_cs = self.structure.yield_specs.intact_cs

        if self.analysis_type is AnalysisType.STATIC:
            self.total_load = self.loads.get_total_load(self.structure, self.loads)
            self.elastic_nodal_disp = get_nodal_disp(self.structure, self.loads, self.total_load)
            self.elastic_members_disps = get_members_disps(self.structure, self.elastic_nodal_disp[0, 0])
            internal_responses = get_internal_responses(self.structure, self.elastic_members_disps)
            self.elastic_members_nodal_forces = internal_responses.members_nodal_forces
            self.elastic_members_nodal_strains = internal_responses.members_nodal_strains
            self.elastic_members_nodal_stresses = internal_responses.members_nodal_stresses
            self.elastic_members_nodal_moments = internal_responses.members_nodal_moments

            if self.structure.is_inelastic:
                sensitivity = get_sensitivity(structure=self.structure, loads=self.loads)
                self.nodal_disp_sensitivity = sensitivity.nodal_disp
                self.members_disps_sensitivity = sensitivity.members_disps
                self.members_nodal_forces_sensitivity = sensitivity.members_nodal_forces
                self.members_nodal_strains_sensitivity = sensitivity.members_nodal_strains
                self.members_nodal_stresses_sensitivity = sensitivity.members_nodal_stresses

                self.analysis_data.p0 = internal_responses.p0
                self.analysis_data.d0 = get_nodal_disp_limits(self.structure, self.elastic_nodal_disp[0, 0])
                self.analysis_data.pv = sensitivity.pv
                self.analysis_data.dv = get_nodal_disp_limits_sensitivity_rows(
                    structure=self.structure,
                    nodal_disp_sensitivity=self.nodal_disp_sensitivity
                )

        if self.analysis_type is AnalysisType.DYNAMIC:
            self.damping = self.general_info["dynamic_analysis"]["damping"]
            structure = self.structure
            loads = self.loads
            time_steps = loads.dynamic[0].magnitude.shape[0]
            self.time_steps = time_steps
            self.time = loads.dynamic[0].time
            dt = self.time[1][0, 0] - self.time[0][0, 0]

            modes = np.matrix(structure.modes)
            modes_count = modes.shape[1]
            self.m_modal = structure.get_modal_property(structure.condensed_m, modes)
            self.k_modal = structure.get_modal_property(structure.condensed_k, modes)

            self.modal_loads = np.matrix(np.zeros((time_steps, 1), dtype=object))
            modal_load = np.matrix(np.zeros((modes_count, 1)))
            modal_loads = np.matrix(np.zeros((1, 1)), dtype=object)
            modal_loads[0, 0] = modal_load
            self.modal_loads[0, 0] = modal_loads
            a_duhamel = np.matrix(np.zeros((modes_count, 1)))
            a_duhamels = np.matrix(np.zeros((1, 1)), dtype=object)
            self.a_duhamel = np.matrix(np.zeros((time_steps, 1), dtype=object))
            a_duhamels[0, 0] = a_duhamel
            self.a_duhamel[0, 0] = a_duhamels

            b_duhamel = np.matrix(np.zeros((modes_count, 1)))
            b_duhamels = np.matrix(np.zeros((1, 1)), dtype=object)
            self.b_duhamel = np.matrix(np.zeros((time_steps, 1), dtype=object))
            b_duhamels[0, 0] = b_duhamel
            self.b_duhamel[0, 0] = b_duhamels
            self.modal_disp_history = np.matrix(np.zeros((time_steps, 1), dtype=object))

            self.total_load = np.zeros((structure.dofs_count, 1))
            self.elastic_nodal_disp_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.elastic_members_disps_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.elastic_members_nodal_forces_history = np.matrix(np.zeros((time_steps, 1), dtype=object))
            self.update_b_for_dynamic_analysis(pv_prev, plastic_multipliers_prev)

    @property
    def analysis_type(self):
        if self.general_info.get("dynamic_analysis") and self.general_info["dynamic_analysis"]["enabled"]:
            type = AnalysisType.DYNAMIC
        else:
            type = AnalysisType.STATIC
        return type
