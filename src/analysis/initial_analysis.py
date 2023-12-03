import numpy as np
import enum
from dataclasses import dataclass

from ..settings import settings
from .functions import (
    get_selected_yield_points,
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
    all_points: np.matrix
    all_pieces: np.matrix
    all_components_count: int
    all_points_count: int
    all_pieces_count: int
    disp_limits_count: int
    limits_count: int
    softening_vars_count: int
    plastic_vars_count: int
    primary_vars_count: int
    constraints_count: int
    slack_vars_count: int
    total_vars_count: int
    landa_var: int
    landa_bar_var: int
    limits_slacks: set
    b: np.array
    c: np.array
    intact_phi: np.matrix
    phi: np.matrix
    q: np.matrix
    h: np.matrix
    w: np.matrix
    cs: np.matrix
    yield_points_indices: list


class InitialAnalysis:
    def __init__(self, structure_input, loads_input, general_info):
        self.structure = Structure(structure_input)
        self.loads = Loads(loads_input)
        self.general_info = general_info
        self.initial_data = InitialData
        self.analysis_data = AnalysisData

        self.initial_data.load_limit = self.structure.limits["load_limit"]
        self.initial_data.disp_limits = self.structure.limits["disp_limits"]
        self.initial_data.all_points = self.structure.yield_specs.all_points
        self.initial_data.all_pieces = self.structure.yield_specs.all_pieces
        self.initial_data.all_components_count = self.structure.yield_specs.all_components_count
        self.initial_data.all_points_count = self.structure.yield_specs.all_points_count
        self.initial_data.all_pieces_count = self.structure.yield_specs.all_pieces_count
        self.initial_data.disp_limits_count = self.initial_data.disp_limits.shape[0]
        self.initial_data.limits_count = 1 + self.initial_data.disp_limits_count * 2
        self.initial_data.softening_vars_count = 2 * self.structure.yield_specs.all_points_count if self.structure.include_softening else 0
        self.initial_data.plastic_vars_count = self.structure.yield_specs.all_pieces_count
        self.initial_data.yield_points_indices = self.structure.yield_specs.yield_points_indices
        self.initial_data.primary_vars_count = self.initial_data.plastic_vars_count + self.initial_data.softening_vars_count + 1
        self.initial_data.constraints_count = self.initial_data.plastic_vars_count + self.initial_data.softening_vars_count + self.initial_data.limits_count
        self.initial_data.slack_vars_count = self.initial_data.constraints_count
        self.initial_data.total_vars_count = self.initial_data.primary_vars_count + self.initial_data.slack_vars_count
        self.initial_data.landa_var = self.initial_data.plastic_vars_count + self.initial_data.softening_vars_count
        self.initial_data.landa_bar_var = 2 * self.initial_data.landa_var + 1
        self.initial_data.limits_slacks = set(range(self.initial_data.landa_bar_var, self.initial_data.landa_bar_var + self.initial_data.limits_count))
        self.initial_data.intact_phi = self.structure.yield_specs.intact_phi
        if not settings.sifting_type:
            self.initial_data.phi = self.structure.yield_specs.intact_phi
            self.initial_data.q = self.structure.yield_specs.intact_q
            self.initial_data.h = self.structure.yield_specs.intact_h
            self.initial_data.w = self.structure.yield_specs.intact_w
            self.initial_data.cs = self.structure.yield_specs.intact_cs
        else:
            self.initial_scores = self.load_limit * self.intact_phi.T * self.p0
            self.selected_yield_points = get_selected_yield_points(all_points=self.all_points, scores=self.initial_scores)

        self.initial_data.b = self._get_b_column()
        self.initial_data.c = self._get_costs_row()

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

    def _get_b_column(self):
        yield_pieces_count = self.initial_data.plastic_vars_count
        disp_limits_count = self.initial_data.disp_limits_count

        b = np.ones((self.initial_data.constraints_count))
        b[yield_pieces_count + self.initial_data.softening_vars_count] = self.initial_data.load_limit
        if self.initial_data.softening_vars_count:
            b[yield_pieces_count:(yield_pieces_count + self.initial_data.softening_vars_count)] = np.array(self.initial_data.cs)[:, 0]

        if self.initial_data.disp_limits.any():
            disp_limit_base_num = yield_pieces_count + self.initial_data.softening_vars_count + 1
            b[disp_limit_base_num:(disp_limit_base_num + disp_limits_count)] = abs(self.initial_data.disp_limits[:, 2])
            b[(disp_limit_base_num + disp_limits_count):(disp_limit_base_num + 2 * disp_limits_count)] = abs(self.initial_data.disp_limits[:, 2])

        return b

    def _get_costs_row(self):
        c = np.zeros(self.initial_data.total_vars_count)
        c[0:self.initial_data.plastic_vars_count] = 1.0
        return -1 * c

    def update_b_for_dynamic_analysis(self, pv_prev, plastic_multipliers_prev):
        self.initial_data.b[0:self.initial_data.plastic_vars_count] = (
            self.initial_data.b[0:self.initial_data.plastic_vars_count] -
            np.array(
                self.initial_data.phi.T * pv_prev * self.initial_data.phi * plastic_multipliers_prev
            ).flatten()
        )
