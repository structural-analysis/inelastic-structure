import numpy as np
from math import isclose
from scipy.linalg import cho_factor

from src.models.points import Node
from src.models.boundaries import NodalBoundary, NodeDOFRestrainer
from src.settings import settings


class YieldSpecs:
    def __init__(self, yield_specs_dict):
        self.points_count = yield_specs_dict["points_count"]
        self.components_count = yield_specs_dict["components_count"]
        self.pieces_count = yield_specs_dict["pieces_count"]


class Members:
    def __init__(self, members_list):
        self.list = members_list
        self.num = len(members_list)
        self.yield_specs = YieldSpecs(self.get_yield_specs_dict())

    def get_yield_specs_dict(self):
        points_count = 0
        components_count = 0
        pieces_count = 0

        for member in self.list:
            points_count += member.yield_specs.points_count
            components_count += member.yield_specs.components_count
            pieces_count += member.yield_specs.pieces_count

        yield_specs_dict = {
            "points_count": points_count,
            "components_count": components_count,
            "pieces_count": pieces_count,
        }
        return yield_specs_dict


class Structure:
    # TODO: can't solve truss, fix reduced matrix to model trusses.
    def __init__(self, input):
        self.general_properties = input["general_properties"]
        self.dim = self.general_properties["structure_dim"]
        self.include_softening = self.general_properties["include_softening"]
        self.initial_nodes = input["initial_nodes"]
        self.initial_nodes_count = len(self.initial_nodes)
        self.members = Members(members_list=input["members"])
        self.nodes = self.get_nodes()
        self.nodes_count = len(self.nodes)
        self.node_dofs_count = 3 if self.dim.lower() == "2d" else 3
        self.analysis_type = self._get_analysis_type()
        self.dofs_count = self.node_dofs_count * self.nodes_count
        self.yield_specs = self.members.yield_specs
        self.nodal_boundaries = input["nodal_boundaries"]
        self.linear_boundaries = input["linear_boundaries"]
        self.boundaries = self.aggregate_boundaries()
        self.boundaries_dof = self.get_boundaries_dof()
        self.loads = input["loads"]
        self.limits = input["limits"]
        self.k = self.get_stiffness()

        self.reduced_k = self.apply_boundary_condition(self.boundaries_dof, self.k)
        self.kc = cho_factor(self.reduced_k)
        self.yield_points_indices = self.get_yield_points_indices()

        self.phi = self.create_phi()
        self.q = self.create_q()
        self.h = self.create_h()
        self.w = self.create_w()
        self.cs = self.create_cs()
        if self.analysis_type == "dynamic":
            self.m = self.get_mass()
            self.damping = self.general_properties["dynamic_analysis"].get("damping") if self.general_properties["dynamic_analysis"].get("damping") else 0
            self.zero_mass_dofs = self.get_zero_mass_dofs()
            self.mass_bounds, self.zero_mass_bounds = self.condense_boundary()
            condensation_params = self.apply_static_condensation()
            self.condensed_k = condensation_params["condensed_k"]
            self.condensed_m = condensation_params["condensed_m"]
            self.ku0 = condensation_params["ku0"]
            self.reduced_k00_inv = condensation_params["reduced_k00_inv"]
            self.reduced_k00 = condensation_params["reduced_k00"]
            self.wns, self.wds, self.modes = self.compute_modes_props()

    def get_nodes(self):
        nodes = self.initial_nodes
        for member in self.members.list:
            if member.__class__.__name__ == "PlateMember":
                nodes = member.nodes
        return nodes

    def _get_analysis_type(self):
        if self.general_properties.get("dynamic_analysis") and self.general_properties["dynamic_analysis"]["enabled"]:
            type = "dynamic"
        else:
            type = "static"
        return type

    def _transform_loc_2d_matrix_to_glob(self, member_transform, member_stiffness):
        member_global_stiffness = np.dot(np.dot(np.transpose(member_transform), member_stiffness), member_transform)
        return member_global_stiffness

    def get_stiffness(self):
        structure_stiffness = np.matrix(np.zeros((self.dofs_count, self.dofs_count)))
        for member in self.members.list:
            member_global_stiffness = self._transform_loc_2d_matrix_to_glob(member.t, member.k)
            structure_stiffness = self._assemble_members(member, member_global_stiffness, structure_stiffness)
        return structure_stiffness

    def apply_boundary_condition(self, boundaries_dof, structure_prop):
        reduced_structure_prop = structure_prop
        row_deleted_counter = 0
        col_deleted_counter = 0
        if structure_prop.shape[0] == structure_prop.shape[1]:
            for boundary in boundaries_dof:
                reduced_structure_prop = np.delete(reduced_structure_prop, boundary - row_deleted_counter, 0)
                reduced_structure_prop = np.delete(reduced_structure_prop, boundary - col_deleted_counter, 1)
                row_deleted_counter += 1
                col_deleted_counter += 1

        else:
            for boundary in self.mass_bounds:
                reduced_structure_prop = np.delete(reduced_structure_prop, boundary - col_deleted_counter, 1)
                col_deleted_counter += 1

            for boundary in self.zero_mass_bounds:
                reduced_structure_prop = np.delete(reduced_structure_prop, boundary - row_deleted_counter, 0)
                row_deleted_counter += 1

        return reduced_structure_prop

    def apply_load_boundry_conditions(self, force):
        reduced_f = force
        deleted_counter = 0
        for i in range(len(self.boundaries_dof)):
            reduced_f = np.delete(
                reduced_f, self.boundaries_dof[i] - deleted_counter, 0
            )
            deleted_counter += 1
        return reduced_f

    def get_mass(self):
        # mass per length is applied in global direction so there is no need to transform.
        structure_mass = np.matrix(np.zeros((self.dofs_count, self.dofs_count)))
        for member in self.members.list:
            if member.m is not None:
                structure_mass = self._assemble_members(member, member.m, structure_mass)
        return structure_mass

    def get_zero_mass_dofs(self):
        return np.sort(np.where(~self.m.any(axis=1))[0])

    def _assemble_members(self, member, member_prop, structure_prop):
        member_nodes_count = len(member.nodes)
        member_dofs_count = member.k.shape[0]
        member_node_dofs_count = int(member_dofs_count / member_nodes_count)
        for i in range(member_dofs_count):
            for j in range(member_dofs_count):
                local_member_node_row = int(j // member_node_dofs_count)
                p = int(member_node_dofs_count * member.nodes[local_member_node_row].num + j % member_node_dofs_count)
                local_member_node_column = int(i // member_node_dofs_count)
                q = int(member_node_dofs_count * member.nodes[local_member_node_column].num + i % member_node_dofs_count)
                structure_prop[p, q] = structure_prop[p, q] + member_prop[j, i]
        return structure_prop

    def _assemble_joint_load(self, loads, time_step=None):
        f_total = np.zeros((self.dofs_count, 1))
        f_total = np.matrix(f_total)
        for load in loads:
            load_magnitude = load.magnitude[time_step, 0] if time_step else load.magnitude
            f_total[self.node_dofs_count * load.node + load.dof] = f_total[self.node_dofs_count * load.node + load.dof] + load_magnitude
        return f_total

    def get_load_vector(self, time_step=None):
        f_total = np.zeros((self.dofs_count, 1))
        f_total = np.matrix(f_total)
        for load in self.loads:
            if self.loads[load]:
                if load == "joint":
                    f_total = f_total + self._assemble_joint_load(self.loads[load])
                elif load == "dynamic":
                    f_total = f_total + self._assemble_joint_load(self.loads[load], time_step)
        return f_total

    def apply_load_boundary_conditions(self, force):
        reduced_f = force
        deleted_counter = 0
        for i in range(len(self.boundaries_dof)):
            reduced_f = np.delete(
                reduced_f, self.boundaries_dof[i] - deleted_counter, 0
            )
            deleted_counter += 1
        return reduced_f

    def get_global_dof(self, node_num, dof):
        global_dof = int(self.node_dofs_count * node_num + dof)
        return global_dof

    def create_phi(self):
        phi = np.matrix(np.zeros((self.yield_specs.components_count, self.yield_specs.pieces_count)))
        current_row = 0
        current_column = 0
        for member in self.members.list:
            for _ in range(member.yield_specs.points_count):
                for yield_section_row in range(member.section.yield_specs.phi.shape[0]):
                    for yield_section_column in range(member.section.yield_specs.phi.shape[1]):
                        phi[current_row + yield_section_row, current_column + yield_section_column] = member.section.yield_specs.phi[yield_section_row, yield_section_column]
                current_column = current_column + member.section.yield_specs.phi.shape[1]
                current_row = current_row + member.section.yield_specs.phi.shape[0]
        return phi

    def create_q(self):
        q = np.matrix(np.zeros((2 * self.yield_specs.points_count, self.yield_specs.pieces_count)))
        yield_point_counter = 0
        yield_pieces_count_counter = 0
        for member in self.members.list:
            for _ in range(member.yield_specs.points_count):
                q[2 * yield_point_counter:2 * yield_point_counter + 2, yield_pieces_count_counter:member.section.yield_specs.pieces_count + yield_pieces_count_counter] = member.section.softening.q
                yield_point_counter += 1
                yield_pieces_count_counter += member.section.yield_specs.pieces_count
        return q

    def create_h(self):
        h = np.matrix(np.zeros((self.yield_specs.pieces_count, 2 * self.yield_specs.points_count)))
        yield_point_counter = 0
        yield_pieces_count_counter = 0
        for member in self.members.list:
            for _ in range(member.yield_specs.points_count):
                h[yield_pieces_count_counter:member.section.yield_specs.pieces_count + yield_pieces_count_counter, 2 * yield_point_counter:2 * yield_point_counter + 2] = member.section.softening.h
                yield_point_counter += 1
                yield_pieces_count_counter += member.section.yield_specs.pieces_count
        return h

    def create_w(self):
        w = np.matrix(np.zeros((2 * self.yield_specs.points_count, 2 * self.yield_specs.points_count)))
        yield_point_counter = 0
        for member in self.members.list:
            for _ in range(member.yield_specs.points_count):
                w[2 * yield_point_counter:2 * yield_point_counter + 2, 2 * yield_point_counter:2 * yield_point_counter + 2] = member.section.softening.w
                yield_point_counter += 1
        return w

    def create_cs(self):
        cs = np.matrix(np.zeros((2 * self.yield_specs.points_count, 1)))
        yield_point_counter = 0
        for member in self.members.list:
            for _ in range(member.yield_specs.points_count):
                cs[2 * yield_point_counter:2 * yield_point_counter + 2, 0] = member.section.softening.cs
                yield_point_counter += 1
        return cs

    def get_yield_points_indices(self):
        yield_points_indices = []
        index_counter = 0
        for member in self.members.list:
            yield_point_pieces = int(member.yield_specs.pieces_count / member.yield_specs.points_count)
            for _ in range(member.yield_specs.points_count):
                yield_points_indices.append(
                    {
                        "begin": index_counter,
                        "end": index_counter + yield_point_pieces - 1,
                    }
                )
                index_counter += yield_point_pieces
        return yield_points_indices

    def aggregate_boundaries(self):
        boundaries = self.nodal_boundaries
        for member in self.members.list:
            if member.__class__.__name__ == "PlateMember":
                boundaries = self.get_nodal_boundaries_from_linear()
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

    def condense_boundary(self):
        zero_mass_dofs = self.zero_mass_dofs

        mass_dof_i = 0
        zero_mass_dof_i = 0

        mass_bounds = self.boundaries_dof.copy()
        zero_mass_bounds = self.boundaries_dof.copy()

        bound_i = 0
        mass_bound_i = 0
        zero_mass_bound_i = 0

        if self.zero_mass_dofs.any():
            for dof in range(self.dofs_count):
                if dof == zero_mass_dofs[zero_mass_dof_i]:
                    if bound_i < self.boundaries_dof.shape[0]:
                        if dof == self.boundaries_dof[bound_i]:
                            mass_bounds = np.delete(mass_bounds, bound_i - mass_bound_i, 0)
                            mass_bound_i += 1

                            zero_mass_bounds[bound_i - zero_mass_bound_i] = zero_mass_bounds[bound_i - zero_mass_bound_i] - mass_dof_i

                            bound_i += 1
                    zero_mass_dof_i += 1
                else:
                    if dof == self.boundaries_dof[bound_i]:
                        mass_bounds[bound_i - mass_bound_i] = mass_bounds[bound_i - mass_bound_i] - zero_mass_dof_i

                        zero_mass_bounds = np.delete(zero_mass_bounds, bound_i - zero_mass_bound_i, 0)
                        zero_mass_bound_i += 1

                        bound_i += 1
                    mass_dof_i += 1
        return mass_bounds, zero_mass_bounds

    def apply_static_condensation(self):
        mtt, ktt, k00, k0t = self.get_zero_and_nonzero_mass_props()

        mass_bounds = self.mass_bounds
        zero_mass_bounds = self.zero_mass_bounds
        reduced_ktt = self.apply_boundary_condition(mass_bounds, ktt)
        condensed_m = self.apply_boundary_condition(mass_bounds, mtt)
        reduced_k00 = self.apply_boundary_condition(zero_mass_bounds, k00)
        reduced_k0t = self.apply_boundary_condition(self.boundaries_dof, k0t)
        reduced_k00_inv = np.linalg.inv(reduced_k00)
        ku0 = -(np.dot(reduced_k00_inv, reduced_k0t))
        condensed_k = reduced_ktt - np.dot(np.dot(np.transpose(reduced_k0t), reduced_k00_inv), reduced_k0t)
        condensation_params = {
            "condensed_k": condensed_k,
            "condensed_m": condensed_m,
            "ku0": ku0,
            "reduced_k00_inv": reduced_k00_inv,
            "reduced_k00": reduced_k00,
        }
        return condensation_params

    def get_zero_and_nonzero_mass_props(self):
        mtt = self.m.copy()
        ktt = self.k.copy()
        k00 = self.k.copy()
        k0t = self.k.copy()
        # for zero mass rows and columns
        mass_i = 0
        # for non-zero mass rows and columns
        zero_i = 0
        zero_mass_dofs_i = 0
        for dof in range(self.dofs_count):
            if dof == self.zero_mass_dofs[zero_mass_dofs_i]:
                mtt = np.delete(mtt, dof - zero_i, 1)
                mtt = np.delete(mtt, dof - zero_i, 0)
                ktt = np.delete(ktt, dof - zero_i, 1)
                ktt = np.delete(ktt, dof - zero_i, 0)
                k0t = np.delete(k0t, dof - zero_i, 1)
                zero_i += 1
                zero_mass_dofs_i += 1
            else:
                k00 = np.delete(k00, dof - mass_i, 1)
                k00 = np.delete(k00, dof - mass_i, 0)
                k0t = np.delete(k0t, dof - mass_i, 0)
                mass_i += 1
        return mtt, ktt, k00, k0t

    def get_modal_property(self, property, modes):
        property_modal = np.dot(np.transpose(modes), np.dot(property, modes))
        return property_modal

    def compute_modes_props(self):
        damping = self.damping
        eigvals, modes = eigh(self.condensed_k, self.condensed_m, eigvals_only=False)
        wn = np.sqrt(eigvals)
        wd = np.sqrt(1 - damping ** 2) * wn
        return wn, wd, modes

    def undo_disp_boundary_condition(self, disp, boundaries_dof):
        free_i = 0
        bound_i = 0
        unrestrianed_dofs = disp.shape[0] + boundaries_dof.shape[0]
        unrestrianed_disp = np.matrix(np.zeros((unrestrianed_dofs, 1)))
        for dof in range(unrestrianed_dofs):
            if bound_i < boundaries_dof.shape[0] and dof == boundaries_dof[bound_i]:
                bound_i += 1
            else:
                unrestrianed_disp[dof, 0] = disp[free_i, 0]
                free_i += 1
        return unrestrianed_disp

    def undo_disp_condensation(self, ut, u0):
        u = np.matrix(np.zeros((self.dofs_count, 1)))
        u0_i = 0
        ut_i = 0
        zero_mass_dofs_i = 0
        for dof in range(self.dofs_count):
            if dof == self.zero_mass_dofs[zero_mass_dofs_i]:
                u[dof, 0] = u0[u0_i, 0]
                u0_i += 1
                zero_mass_dofs_i += 1
            else:
                u[dof, 0] = ut[ut_i, 0]
                ut_i += 1
        return u
