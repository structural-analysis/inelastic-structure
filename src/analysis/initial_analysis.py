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
)
from ..models.structure import Structure
from ..models.loads import Loads
from ..functions import create_chunk


class AnalysisType(str, enum.Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass
class AnalysisData:
    p0: np.array
    pv: np.array
    d0: np.array
    dv: np.array
    pv_prev: np.array = np.zeros([1, 1])


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
    # yield_points_indices: list
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

            if self.structure.is_inelastic:
                sensitivity = get_sensitivity(structure=self.structure, loads=self.loads)
                self.nodal_disp_sensitivity = sensitivity.nodal_disp
                self.members_disps_sensitivity = sensitivity.members_disps
                self.members_nodal_forces_sensitivity = sensitivity.members_nodal_forces
                self.members_nodal_strains_sensitivity = sensitivity.members_nodal_strains
                self.members_nodal_stresses_sensitivity = sensitivity.members_nodal_stresses
                self.members_nodal_moments_sensitivity = sensitivity.members_nodal_moments

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

            self.modal_loads = np.zeros((self.time_steps, modes_count))
            self.a_duhamels = np.zeros((self.time_steps, modes_count))
            self.b_duhamels = np.zeros((self.time_steps, modes_count))

            self.total_load = np.zeros((structure.dofs_count, 1))
            self.elastic_nodal_disp_history = np.zeros((self.time_steps, structure.dofs_count))
            self.elastic_members_disps_history = np.zeros((self.time_steps, structure.members_count, structure.max_member_dofs_count))
            self.elastic_members_nodal_forces_history = np.zeros((self.time_steps, structure.members_count, structure.max_member_dofs_count))
            self.elastic_members_nodal_strains_history = np.zeros((self.time_steps, structure.members_count, structure.max_member_nodal_components_count))
            self.elastic_members_nodal_stresses_history = np.zeros((self.time_steps, structure.members_count, structure.max_member_nodal_components_count))
            self.elastic_members_nodal_moments_history = np.zeros((self.time_steps, structure.members_count, structure.max_member_nodal_components_count))

            if self.structure.is_inelastic:
                self.modal_loads_sensitivity_history = np.zeros((self.time_steps, structure.selected_modes_count, structure.yield_specs.intact_components_count))
                self.a2_sensitivity_history = np.zeros((self.time_steps, structure.selected_modes_count, structure.yield_specs.intact_components_count))
                self.b2_sensitivity_history = np.zeros((self.time_steps, structure.selected_modes_count, structure.yield_specs.intact_components_count))
                self.p0_history = np.zeros((self.time_steps, structure.yield_specs.intact_components_count))
                self.d0_history = np.zeros((self.time_steps, structure.limits["disp_limits"].shape[0]))
                self.pv_history = np.zeros((self.time_steps, structure.yield_specs.intact_components_count, structure.yield_specs.intact_components_count))
                self.load_level = 0

                self.plastic_multipliers_prev = np.zeros(self.initial_data.intact_pieces_count)

    def update_dynamic_time_step(self, time_step):
        self.total_load = self.loads.get_total_load(self.structure, self.loads, time_step)

        elastic_a2s, elastic_b2s, elastic_modal_loads, self.elastic_nodal_disp = get_dynamic_nodal_disp(
            structure=self.structure,
            loads=self.loads,
            time=self.time,
            time_step=time_step,
            modes=self.structure.selected_modes,
            previous_modal_loads=self.modal_loads[time_step - 1, :],
            total_load=self.total_load,
            a1s=self.a_duhamels[time_step - 1, :],
            b1s=self.b_duhamels[time_step - 1, :],
        )

        self.a_duhamels[time_step, :] = elastic_a2s
        self.b_duhamels[time_step, :] = elastic_b2s
        self.modal_loads[time_step, :] = elastic_modal_loads

        self.elastic_nodal_disp_history[time_step, :] = self.elastic_nodal_disp
        self.elastic_members_disps = get_members_disps(self.structure, self.elastic_nodal_disp)
        self.elastic_members_disps_history[time_step, :, :] = self.elastic_members_disps
        internal_responses = get_internal_responses(self.structure, self.elastic_members_disps)

        self.elastic_members_nodal_forces_history[time_step, :, :] = internal_responses.members_nodal_forces
        # self.elastic_members_nodal_strains_history[time_step, :, :] = internal_responses.members_nodal_strains
        # self.elastic_members_nodal_stresses_history[time_step, :, :] = internal_responses.members_nodal_stresses
        # self.elastic_members_nodal_moments_history[time_step, :, :] = internal_responses.members_nodal_moments

        if self.structure.is_inelastic:
            self.p0_prev = self.p0_history[time_step - 1]
            self.p0_history[time_step, :] = internal_responses.p0
            self.d0_history[time_step, :] = get_nodal_disp_limits(self.structure, self.elastic_nodal_disp)

            sensitivity = get_dynamic_sensitivity(
                structure=self.structure,
                loads=self.loads,
                time=self.time,
                time_step=time_step,
                modes=self.structure.selected_modes_count,
            )
            self.pv_prev = self.pv_history[time_step - 1, :, :]
            self.pv_history[time_step, :, :] = sensitivity.pv

            create_chunk(time_step=time_step, response="nodal_disp", sensitivity=sensitivity.nodal_disp)
            create_chunk(time_step=time_step, response="members_nodal_forces", sensitivity=sensitivity.members_nodal_forces)
            create_chunk(time_step=time_step, response="members_disps", sensitivity=sensitivity.members_disps)

            self.modal_loads_sensitivity_history[time_step, :, :] = sensitivity.modal_loads
            self.a2_sensitivity_history[time_step, :, :] = sensitivity.a2s
            self.b2_sensitivity_history[time_step, :, :] = sensitivity.b2s

            self.dv = get_nodal_disp_limits_sensitivity_rows(structure=self.structure, nodal_disp_sensitivity=sensitivity.nodal_disp)
            self.load_level_prev = self.load_level

            self.analysis_data.p0 = self.p0_history[time_step]
            self.analysis_data.d0 = self.d0_history[time_step]
            self.analysis_data.pv = self.pv_history[time_step]
            self.analysis_data.dv = self.dv
            self.analysis_data.pv_prev = self.pv_prev
