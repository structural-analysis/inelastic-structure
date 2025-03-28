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
    get_dynamic_nodal_disp,
    get_dynamic_sensitivity,
    get_a2s_b2s_sensitivity,
    get_a2s_b2s_sensitivity_constant,
)
from ..models.structure import Structure
from ..models.loads import Loads


class AnalysisType(str, enum.Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass
class AnalysisData:
    p0: np.array
    d0: np.array
    pv: np.array
    dv: np.array


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
    intact_phi: np.array
    intact_q: np.array
    intact_h: np.array
    intact_w: np.array
    intact_cs: np.array


class InitialAnalysis:
    def __init__(self, structure: Structure, loads: Loads, analysis_type):
        self.structure = structure
        self.loads = loads
        self.analysis_type = analysis_type
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
        self.initial_data.intact_phi = self.structure.yield_specs.intact_phi
        self.initial_data.intact_q = self.structure.yield_specs.intact_q
        self.initial_data.intact_h = self.structure.yield_specs.intact_h
        self.initial_data.intact_w = self.structure.yield_specs.intact_w
        self.initial_data.intact_cs = self.structure.yield_specs.intact_cs

        if self.analysis_type is AnalysisType.STATIC:
            self.total_load = self.loads.get_total_load(self.structure, self.loads)
            self.elastic_nodal_disp = get_nodal_disp(self.structure, self.loads, self.total_load)
            self.elastic_members_disps = get_members_disps(self.structure, self.elastic_nodal_disp)
            internal_responses = get_internal_responses(self.structure, self.elastic_members_disps)
            self.elastic_members_nodal_forces = internal_responses.members_nodal_forces
            self.elastic_members_nodal_strains = internal_responses.members_nodal_strains
            self.elastic_members_nodal_stresses = internal_responses.members_nodal_stresses
            self.elastic_members_nodal_moments = internal_responses.members_nodal_moments
            self.elastic_yield_points_forces=internal_responses.p0

            if self.structure.is_inelastic:
                sensitivity = get_sensitivity(structure=self.structure, loads=self.loads)
                self.nodal_disp_sensitivity = sensitivity.nodal_disp
                self.members_disps_sensitivity = sensitivity.members_disps
                self.members_nodal_forces_sensitivity = sensitivity.members_nodal_forces
                self.members_nodal_strains_sensitivity = sensitivity.members_nodal_strains
                self.members_nodal_stresses_sensitivity = sensitivity.members_nodal_stresses
                self.members_nodal_moments_sensitivity = sensitivity.members_nodal_moments
                self.yield_points_forces_sensitivity = sensitivity.pv

                self.analysis_data.p0 = internal_responses.p0
                self.analysis_data.d0 = get_nodal_disp_limits(self.structure, self.elastic_nodal_disp)
                self.analysis_data.pv = sensitivity.pv
                self.analysis_data.dv = get_nodal_disp_limits_sensitivity_rows(
                    structure=self.structure,
                    nodal_disp_sensitivity=self.nodal_disp_sensitivity
                )

        if self.analysis_type is AnalysisType.DYNAMIC:
            structure = self.structure
            self.damping = structure.damping
            loads = self.loads
            self.time_steps = loads.dynamic_loads[0].magnitude.shape[0]
            self.time = loads.dynamic_loads[0].time

            self.m_modal = structure.m_modal
            self.k_modal = structure.k_modal
            modes_count = structure.selected_modes_count

            self.previous_modal_loads = np.zeros(modes_count)
            self.previous_a2s = np.zeros(modes_count)
            self.previous_b2s = np.zeros(modes_count)

            self.total_load = np.zeros((structure.dofs_count, 1))
            self.elastic_nodal_disp_history = np.zeros((self.time_steps, structure.dofs_count))
            self.elastic_members_disps_history = np.zeros((self.time_steps, structure.members_count, structure.max_member_dofs_count))
            self.elastic_members_nodal_forces_history = np.zeros((self.time_steps, structure.members_count, structure.max_member_dofs_count))
            self.elastic_members_nodal_strains_history = np.zeros((self.time_steps, structure.members_count, structure.max_member_nodal_components_count))
            self.elastic_members_nodal_stresses_history = np.zeros((self.time_steps, structure.members_count, structure.max_member_nodal_components_count))
            self.elastic_members_nodal_moments_history = np.zeros((self.time_steps, structure.members_count, structure.max_member_nodal_components_count))

            if self.structure.is_inelastic:
                sensitivity = get_dynamic_sensitivity(
                    structure=self.structure,
                    loads=self.loads,
                    deltat=self.time[1, 0] - self.time[0, 0],
                )

                self.nodal_disp_sensitivity = sensitivity.nodal_disp
                self.members_disps_sensitivity = sensitivity.members_disps
                self.members_nodal_forces_sensitivity = sensitivity.members_nodal_forces

                self.analysis_data.pv = sensitivity.pv
                self.analysis_data.dv = get_nodal_disp_limits_sensitivity_rows(
                    structure=self.structure,
                    nodal_disp_sensitivity=sensitivity.nodal_disp
                )
                self.modal_loads_sensitivity = sensitivity.modal_loads
                self.a2s_b2s_sensitivity_constant = get_a2s_b2s_sensitivity_constant(
                    structure=structure,
                    loads=loads,
                    deltat=self.time[1, 0] - self.time[0, 0],
                    modal_loads_sensitivity=self.modal_loads_sensitivity,
                )

    def update_dynamic_time_step(self, time_step):
        self.total_load = self.loads.get_total_load(self.structure, self.loads, time_step)

        self.elastic_a2s, self.elastic_b2s, a_factor, b_factor, self.elastic_modal_loads, self.elastic_nodal_disp = get_dynamic_nodal_disp(
            structure=self.structure,
            loads=self.loads,
            t1=self.time[time_step - 1, 0],
            t2=self.time[time_step, 0],
            modes=self.structure.selected_modes,
            total_load=self.total_load,
            previous_modal_loads=self.previous_modal_loads,
            previous_a2s=self.previous_a2s,
            previous_b2s=self.previous_b2s,
        )

        self.elastic_nodal_disp_history[time_step, :] = self.elastic_nodal_disp
        self.elastic_members_disps = get_members_disps(self.structure, self.elastic_nodal_disp)
        self.elastic_members_disps_history[time_step, :, :] = self.elastic_members_disps
        internal_responses = get_internal_responses(self.structure, self.elastic_members_disps)

        self.elastic_members_nodal_forces_history[time_step, :, :] = internal_responses.members_nodal_forces
        # self.elastic_members_nodal_strains_history[time_step, :, :] = internal_responses.members_nodal_strains
        # self.elastic_members_nodal_stresses_history[time_step, :, :] = internal_responses.members_nodal_stresses
        # self.elastic_members_nodal_moments_history[time_step, :, :] = internal_responses.members_nodal_moments
        self.previous_modal_loads = self.elastic_modal_loads
        self.previous_a2s = self.elastic_a2s
        self.previous_b2s = self.elastic_b2s

        if self.structure.is_inelastic:
            a2s_b2s_sensitivity = get_a2s_b2s_sensitivity(
                a_factor=a_factor,
                b_factor=b_factor,
                a2s_b2s_sensitivity_constant=self.a2s_b2s_sensitivity_constant,
            )
            self.a2_sensitivity = a2s_b2s_sensitivity.a2s
            self.b2_sensitivity = a2s_b2s_sensitivity.b2s
            self.analysis_data.p0 = internal_responses.p0
            self.analysis_data.d0 = get_nodal_disp_limits(self.structure, self.elastic_nodal_disp)
