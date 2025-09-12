import numpy as np
from math import isclose
from dataclasses import dataclass
from scipy.linalg import cho_factor, eigh

from src.models.points import Node
from src.models.boundaries import NodalBoundary, NodeDOFRestrainer
from src.settings import settings
from .yield_models import StructureYieldSpecs


@dataclass
class NodeMapper:
    structure_node: Node
    attached_members: list


@dataclass
class AttachedMember:
    member: object
    node: Node
    member_node_num: int


@dataclass
class VectorizedNodesMap:
    node_ids: np.array
    member_ids: np.array
    local_node_nums: np.array


class Structure:
    # TODO: can't solve truss, fix reduced matrix to model trusses.
    def __init__(self, input):
        self.general_properties = input["general_properties"]
        self.type = input["structure_type"]
        self.dim = self.general_properties["structure_dim"]
        self.initial_nodes = input["initial_nodes"]
        self.initial_nodes_count = len(self.initial_nodes)
        self.members = input["members"]
        self.members_count = len(self.members)
        self.nodes = self.get_nodes()
        self.nodes_count = len(self.nodes)
        self.nodes_map = self.create_nodes_map()
        self.vectorized_nodes_map = self.create_vectorized_nodes_map()
        self.node_dofs_count = input["node_dofs_count"]
        self.analysis_type = self._get_analysis_type()
        self.dofs_count = self.node_dofs_count * self.nodes_count
        self.yield_specs = StructureYieldSpecs(members=self.members, include_softening=self.include_softening)
        self.yield_points_count = self.yield_specs.intact_points_count
        self.intact_components_count = self.yield_specs.intact_components_count
        self.nodal_boundaries = input["nodal_boundaries"]
        self.linear_boundaries = input["linear_boundaries"]
        self.boundaries = self.aggregate_boundaries()
        self.boundaries_dof = self.get_boundaries_dof()
        self.boundaries_dof_mask = self.get_boundaries_dof_mask()
        self.loads = input["loads"]
        self.limits = input["limits"]
        self.k = self.get_stiffness()
        self.reduced_k = self.apply_boundary_conditions(
            row_boundaries_dof=self.boundaries_dof_mask,
            col_boundaries_dof=self.boundaries_dof_mask,
            structure_prop=self.k,
        )
        self.kc = cho_factor(self.reduced_k)
        self.max_member_dofs_count = self.get_max_member_dofs_count()
        self.max_member_nodal_components_count = self.get_max_member_nodal_components_count()
        # assumed that max components count is 3 (must increase in more complicated sections)
        self.max_section_components_count = 3
        self.nodal_components_count = self.nodes_count * self.max_section_components_count

        if self.analysis_type == "dynamic":
            self.m = self.get_mass()
            self.damping = self.general_properties["dynamic_analysis"].get("damping") \
                if self.general_properties["dynamic_analysis"].get("damping") else 0
            self.zero_mass_dofs = self.get_zero_mass_dofs()
            self.mass_dof_mask, self.zero_mass_dof_mask, self.mass_boundaries_mask, self.zero_mass_boundaries_mask = self.get_dynamic_masks()
            condensation_params = self.apply_static_condensation()
            self.condensed_k = condensation_params["condensed_k"]
            self.condensed_m = condensation_params["condensed_m"]
            self.ku0 = condensation_params["ku0"]
            self.reduced_k00_inv = condensation_params["reduced_k00_inv"]
            self.reduced_k00 = condensation_params["reduced_k00"]
            self.reduced_k0t = condensation_params["reduced_k0t"]
            self.wns, self.wds, self.modes, self.modes_count = self.compute_modes_props()
            self.c = self.get_rayleigh_damping(
                damping_ratio=self.damping,
                wns=self.wns,
                m=self.condensed_m,
                k=self.condensed_k,
            )
            self.selected_modes_count = self.get_selected_modes_count()
            self.selected_modes = self.modes[:self.dofs_count, :self.selected_modes_count]
            self.m_modal = self.get_modal_property(self.condensed_m, self.selected_modes)
            self.k_modal = self.get_modal_property(self.condensed_k, self.selected_modes)

    @property
    def is_inelastic(self):
        if self.general_properties["inelastic"]["enabled"]:
            return True
        return False

    @property
    def include_softening(self):
        if self.general_properties["inelastic"]["enabled"]:
            if self.general_properties["inelastic"]["include_softening"]:
                return True
        return False

    def get_nodes(self):
        nodes = self.initial_nodes
        return nodes

    def create_nodes_map(self):
        nodes_map: list(NodeMapper) = []
        for structure_node in self.nodes:
            nodes_map.append(NodeMapper(structure_node=structure_node, attached_members=[]))
            for member in self.members:
                for member_node in member.nodes:
                    if member_node == structure_node:
                        nodes_map[structure_node.num].attached_members.append(
                            AttachedMember(
                                member=member,
                                node=member_node,
                                member_node_num=member.nodes.index(member_node)
                            )
                        )
        return nodes_map

    def create_vectorized_nodes_map(self):
        # 1) Build arrays listing all "links" from (node -> member -> local_node_index).
        #    We'll gather them so we can do a single pass with advanced indexing.
        node_ids = []
        member_ids = []
        local_node_nums = []
        for node in self.nodes:
            n_id = node.num
            attached_list = self.nodes_map[n_id].attached_members
            for attached_member in attached_list:
                node_ids.append(n_id)
                member_ids.append(attached_member.member.num)
                local_node_nums.append(attached_member.member_node_num)

        node_ids = np.array(node_ids, dtype=int)
        member_ids = np.array(member_ids, dtype=int)
        local_node_nums = np.array(local_node_nums, dtype=int)
        return VectorizedNodesMap(node_ids=node_ids, member_ids=member_ids, local_node_nums=local_node_nums,)

    def _get_analysis_type(self):
        if self.general_properties.get("dynamic_analysis") and self.general_properties["dynamic_analysis"]["enabled"]:
            type = "dynamic"
        else:
            type = "static"
        return type

    def _transform_loc_2d_matrix_to_glob(self, member_transform, member_stiffness):
        member_global_stiffness = (np.transpose(member_transform) @ member_stiffness) @ member_transform
        return member_global_stiffness

    def get_stiffness(self):
        # TODO: we must add mapping of element dof to structure dofs.
        structure_stiffness = np.zeros((self.dofs_count, self.dofs_count))
        for member in self.members:
            member_global_stiffness = self._transform_loc_2d_matrix_to_glob(member.t, member.k)
            mapped_member_node_dofs = self.map_member_node_dofs(member)
            mapped_element_dofs = self.map_member_dofs(
                member_nodes_count=member.nodes_count,
                mapped_node_dofs=mapped_member_node_dofs,
            )
            mapped_dofs_count = self.node_dofs_count * member.nodes_count
            mapped_k = np.zeros((mapped_dofs_count, mapped_dofs_count))
            mapped_k[np.ix_(mapped_element_dofs, mapped_element_dofs)] = member_global_stiffness
            structure_stiffness = self._assemble_members(member, mapped_k, structure_stiffness)
        return structure_stiffness

    def apply_boundary_conditions(self, row_boundaries_dof, col_boundaries_dof, structure_prop):
        reduced_structure_prop = structure_prop[row_boundaries_dof][:, col_boundaries_dof]
        return reduced_structure_prop

    def get_mass(self):
        # mass per length is applied in global direction so there is no need to transform.
        # TODO: map nodes of member to nodes of structure
        structure_mass =np.zeros((self.dofs_count, self.dofs_count))
        for member in self.members:
            if member.m is not None:
                mapped_member_node_dofs = self.map_member_node_dofs(member)
                mapped_element_dofs = self.map_member_dofs(
                    member_nodes_count=member.nodes_count,
                    mapped_node_dofs=mapped_member_node_dofs,
                )
                mapped_dofs_count = self.node_dofs_count * member.nodes_count
                mapped_m = np.zeros((mapped_dofs_count, mapped_dofs_count))
                mapped_m[np.ix_(mapped_element_dofs, mapped_element_dofs)] = member.m
                structure_mass = self._assemble_members(member, member.m, structure_mass)
        return structure_mass

    def get_zero_mass_dofs(self):
        return np.sort(np.where(~self.m.any(axis=1))[0])

    def _assemble_members(self, member, member_prop, structure_prop):
        # member_node_dofs_count = member.node_dofs_count
        member_dofs_count = member_prop.shape[0]
        structure_node_dofs_count = self.node_dofs_count
        for i in range(member_dofs_count):
            for j in range(member_dofs_count):
                local_member_node_row = int(j // structure_node_dofs_count)
                p = int(structure_node_dofs_count * member.nodes[local_member_node_row].num + j % structure_node_dofs_count)
                local_member_node_column = int(i // structure_node_dofs_count)
                q = int(structure_node_dofs_count * member.nodes[local_member_node_column].num + i % structure_node_dofs_count)
                structure_prop[p, q] = structure_prop[p, q] + member_prop[j, i]
        return structure_prop

    def _assemble_joint_load(self, loads, time_step=None):
        f_total = np.zeros((self.dofs_count, 1))
        for load in loads:
            load_magnitude = load.magnitude[time_step, 0] if time_step else load.magnitude
            f_total[self.node_dofs_count * load.node + load.dof] = f_total[self.node_dofs_count * load.node + load.dof] + load_magnitude
        return f_total

    def get_load_vector(self, time_step=None):
        f_total = np.zeros((self.dofs_count, 1))
        for load in self.loads:
            if self.loads[load]:
                if load == "joint":
                    f_total = f_total + self._assemble_joint_load(self.loads[load])
                elif load == "dynamic":
                    f_total = f_total + self._assemble_joint_load(self.loads[load], time_step)
        return f_total

    def get_global_dof(self, node_num, dof):
        global_dof = int(self.node_dofs_count * node_num + dof)
        return global_dof

    def aggregate_boundaries(self):
        boundaries = self.nodal_boundaries
        return list(set(boundaries))

    def get_nodal_boundaries_from_linear(self):
        new_nodal_boundaries = []
        for linear_boundary in self.linear_boundaries:
            dof_restrainer = self.get_dof_restrainer_from_linear_boundary(linear_boundary.start_node, linear_boundary.end_node, linear_boundary.dof)
            restrain_dimension_name = dof_restrainer.dimension_name
            restrain_dimension_value = dof_restrainer.dimension_value
            restrain_dof = dof_restrainer.dof
            if restrain_dimension_name == "x":
                for node in self.nodes:
                    if isclose(node.x, restrain_dimension_value, abs_tol=settings.isclose_tolerance):
                        new_nodal_boundaries.append(NodalBoundary(node=node, dof=restrain_dof))
            elif restrain_dimension_name == "y":
                for node in self.nodes:
                    if isclose(node.y, restrain_dimension_value, abs_tol=settings.isclose_tolerance):
                        new_nodal_boundaries.append(NodalBoundary(node=node, dof=restrain_dof))
        return new_nodal_boundaries

    def get_dof_restrainer_from_linear_boundary(self, start_node: Node, end_node: Node, dof):
        x_diff = end_node.x - start_node.x
        y_diff = end_node.y - start_node.y

        if end_node.z != 0 or start_node.z != 0:
            raise Exception("z dimension not supported yet")
        elif x_diff != 0 and y_diff != 0:
            raise Exception("linear boundaries must be straight lines")
        elif x_diff == 0 and y_diff == 0:
            raise Exception("linear boundary start and end nodes overlap")
        elif x_diff == 0 and y_diff != 0:
            restrain_dim = "x"
            restrain_value = end_node.x
        elif x_diff != 0 and y_diff == 0:
            restrain_dim = "y"
            restrain_value = end_node.y
        dof_restrainer = NodeDOFRestrainer(dimension_name=restrain_dim, dimension_value=restrain_value, dof=dof)
        return dof_restrainer

    def get_boundaries_dof(self):
        boundaries_size = len(self.boundaries)
        boundaries_dof = np.zeros(boundaries_size, dtype=int)
        for i in range(boundaries_size):
            boundaries_dof[i] = int(self.node_dofs_count * self.boundaries[i].node.num + self.boundaries[i].dof)
        return np.sort(boundaries_dof)

    def get_boundaries_dof_mask(self):
        boundaries_dof_mask = np.ones(self.dofs_count, dtype=bool)
        boundaries_dof_mask[self.boundaries_dof] = False
        return boundaries_dof_mask

    def get_dynamic_masks(self):
        mass_dof_mask = np.zeros(self.dofs_count, dtype=bool)
        mass_dof_mask[self.zero_mass_dofs] = True
        zero_mass_dof_mask = np.ones(self.dofs_count, dtype=bool)
        zero_mass_dof_mask[self.zero_mass_dofs] = False

        zero_mass_boundaries_mask = zero_mass_dof_mask.copy()
        zero_mass_boundaries_mask[self.boundaries_dof] = False
        mass_boundaries_mask = mass_dof_mask.copy()
        mass_boundaries_mask[self.boundaries_dof] = False
        return mass_dof_mask, zero_mass_dof_mask, mass_boundaries_mask, zero_mass_boundaries_mask

    def apply_static_condensation(self):
        reduced_ktt = self.apply_boundary_conditions(
            row_boundaries_dof=self.zero_mass_boundaries_mask,
            col_boundaries_dof=self.zero_mass_boundaries_mask,
            structure_prop=self.k,
        )
        condensed_m = self.apply_boundary_conditions(
            row_boundaries_dof=self.zero_mass_boundaries_mask,
            col_boundaries_dof=self.zero_mass_boundaries_mask,
            structure_prop=self.m,
        )
        reduced_k00 = self.apply_boundary_conditions(
            row_boundaries_dof=self.mass_boundaries_mask,
            col_boundaries_dof=self.mass_boundaries_mask,
            structure_prop=self.k,
        )
        reduced_k0t = self.apply_boundary_conditions(
            row_boundaries_dof=self.mass_boundaries_mask,
            col_boundaries_dof=self.zero_mass_boundaries_mask,
            structure_prop=self.k,
        )
        reduced_k00_inv = np.linalg.inv(reduced_k00)
        ku0 = -(reduced_k00_inv @ reduced_k0t)
        condensed_k = reduced_ktt - (np.transpose(reduced_k0t) @ reduced_k00_inv) @ reduced_k0t
        condensation_params = {
            "condensed_k": condensed_k,
            "condensed_m": condensed_m,
            "ku0": ku0,
            "reduced_k00_inv": reduced_k00_inv,
            "reduced_k00": reduced_k00,
            "reduced_k0t": reduced_k0t,
        }
        return condensation_params

    def get_modal_property(self, property, modes):
        property_modal = np.transpose(modes) @ (property @ modes)
        return property_modal

    def get_selected_modes_count(self):
        # return 15
        return self.modes_count

    def compute_modes_props(self):
        damping = self.damping
        eigvals, modes = eigh(self.condensed_k, self.condensed_m, eigvals_only=False, lower=False)
        wn = np.sqrt(eigvals)
        wd = np.sqrt(1 - damping ** 2) * wn
        print(f"{eigvals=}")
        print(f"{wn=}")
        modes_count = modes.shape[1]
        return wn, wd, modes, modes_count

    def get_rayleigh_damping(self, damping_ratio, wns, m, k):
        wn0 = wns[0]
        wn1 = wns[1]
        beta = 2 * damping_ratio / (wn0 + wn1)
        alpha = wn0 * wn1 * beta
        c = alpha * m + beta * k
        return c

    def undo_disp_condensation(self, ut, u0):
        disp = np.zeros(self.dofs_count)
        disp[self.mass_boundaries_mask] = u0
        disp[self.zero_mass_boundaries_mask] = ut
        return disp

    def undo_disp_boundaries(self, reduced_disp):
        disp = np.zeros(self.dofs_count)
        disp[self.boundaries_dof_mask] = reduced_disp
        return disp

    def map_member_node_dofs(self, member):
        member_type = member.__class__.__name__
        if self.type == "TRUSS2D":
            if member_type == "Truss2DMember":
                mapped_node_dofs = [0, 1]

        elif self.type == "FRAME2D":
            if member_type == "Truss2DMember":
                mapped_node_dofs = [0, 1]
            if member_type == "WallMember":
                mapped_node_dofs = [0, 1]
            if member_type == "Frame2DMember":
                mapped_node_dofs = [0, 1, 2]

        elif self.type == "FRAME3D":
            if member_type == "Frame3DMember":
                mapped_node_dofs = [0, 1, 2, 3, 4, 5]
            elif member_type == "WallMember":
                mapped_node_dofs = [0, 2]
            elif member_type == "PlateMember":
                # FIXME: check the dofs for plate and fix the values. The values below are not correct.
                mapped_node_dofs = [2, 3, 4]

        elif self.type == "WALL2D":
            if member_type == "WallMember":
                mapped_node_dofs = [0, 1]

        elif self.type == "PLATE2D":
            mapped_node_dofs = [0, 1, 2]

        return mapped_node_dofs

    def map_member_dofs(self, member_nodes_count, mapped_node_dofs):
        element_dofs = []
        for member_node_num in range(member_nodes_count):
            for mapped_node_dof in mapped_node_dofs:
                element_dofs.append(mapped_node_dof + member_node_num * self.node_dofs_count)
        return element_dofs

    def get_max_member_dofs_count(self):
        return np.max([member.dofs_count for member in self.members])

    def get_max_member_nodal_components_count(self):
        return np.max([member.nodal_components_count for member in self.members])
