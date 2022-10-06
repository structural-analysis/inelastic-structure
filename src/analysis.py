import numpy as np
from scipy.linalg import cho_solve

from src.models.loads import Loads
from src.models.structure import Structure


class Analysis:
    def __init__(self, structure_input, loads_input, general_info):
        self.structure = Structure(structure_input)
        self.loads = Loads(loads_input)
        self.general_info = general_info
        self.type = self._get_type()

        if self.type == "static":
            self.total_load = self.loads.get_total_load(self.structure, self.loads)
            self.elastic_nodal_disp = self.get_nodal_disp(self.total_load)
            self.elastic_members_disps = self.get_members_disps(self.elastic_nodal_disp)
            internal_forces = self.get_internal_forces()
            self.elastic_members_forces = internal_forces["members_forces"]
            self.p0 = internal_forces["p0"]
            self.d0 = self.get_nodal_disp_limits()
            sensitivity = self.get_sensitivity()
            self.pv = sensitivity["pv"]
            self.members_forces_sensitivity = sensitivity["members_forces_sensitivity"]
            self.members_disps_sensitivity = sensitivity["members_disps_sensitivity"]
            self.nodal_disps_sensitivity = sensitivity["nodal_disps_sensitivity"]
            self.dv = self.get_nodal_disp_limits_sensitivity_rows()
        elif self.type == "dynamic":
            ...

    def _get_type(self):
        if self.general_info.get("dynamic_analysis") and self.general_info["dynamic_analysis"]["enabled"]:
            type = "dynamic"
        else:
            type = "static"
        return type

    def get_nodal_disp(self, total_load):
        j = 0
        o = 0
        structure = self.structure
        reduced_total_load = self.loads.apply_boundry_conditions(structure, total_load)
        reduced_disp = cho_solve(structure.kc, reduced_total_load)
        empty_nodal_disp = np.zeros((structure.total_dofs_num, 1))
        nodal_disp = np.matrix(empty_nodal_disp)
        for i in range(structure.total_dofs_num):
            if (j != structure.boundaries_dof.shape[0] and i == structure.boundaries_dof[j]):
                j += 1
            else:
                nodal_disp[i, 0] = reduced_disp[o, 0]
                o += 1
        return nodal_disp

    def get_members_disps(self, disp):
        structure = self.structure
        empty_members_disps = np.zeros((structure.members.num, 1), dtype=object)
        members_disps = np.matrix(empty_members_disps)
        for i_member, member in enumerate(structure.members.list):
            member_dofs_num = member.total_dofs_num
            member_nodes_num = len(member.nodes)
            member_node_dofs_num = int(member_dofs_num / member_nodes_num)
            v = np.zeros((member_dofs_num, 1))
            v = np.matrix(v)
            for i in range(member_dofs_num):
                member_node = i // member_node_dofs_num
                node_dof = i % member_node_dofs_num
                v[i, 0] = disp[member_node_dofs_num * member.nodes[member_node].num + node_dof, 0]
            u = member.t * v
            members_disps[i_member, 0] = u
        return members_disps

    def get_internal_forces(self):
        structure = self.structure
        members_disps = self.elastic_members_disps

        fixed_force = np.zeros((structure.node_dofs_num * 2, 1))
        fixed_force = np.matrix(fixed_force)

        # calculate p0
        empty_members_forces = np.zeros((structure.members.num, 1), dtype=object)
        members_forces = np.matrix(empty_members_forces)
        empty_p0 = np.zeros((structure.yield_specs.components_num, 1))
        p0 = np.matrix(empty_p0)
        current_p0_row = 0

        for i, member in enumerate(structure.members.list):
            if member.__class__.__name__ == "FrameMember2D":
                member_force = member.get_nodal_force(members_disps[i, 0], fixed_force)
                members_forces[i, 0] = member_force
                if not member.section.nonlinear.has_axial_yield:
                    p0[current_p0_row] = member_force[2, 0]
                    p0[current_p0_row + 1] = member_force[5, 0]
                else:
                    p0[current_p0_row] = member_force[0, 0]
                    p0[current_p0_row + 1] = member_force[2, 0]
                    p0[current_p0_row + 2] = member_force[3, 0]
                    p0[current_p0_row + 3] = member_force[5, 0]
            current_p0_row = current_p0_row + member.yield_specs.components_num
        return {"members_forces": members_forces, "p0": p0}

    def get_nodal_disp_limits(self):
        structure = self.structure
        elastic_nodal_disp = self.elastic_nodal_disp
        disp_limits = structure.limits["disp_limits"]
        disp_limits_num = disp_limits.shape[0]
        empty_d0 = np.zeros((disp_limits_num, 1))
        d0 = np.matrix(empty_d0)
        for i, disp_limit in enumerate(disp_limits):
            node = disp_limit[0]
            node_dof = disp_limit[1]
            dof = structure.get_global_dof(node, node_dof)
            d0[i, 0] = elastic_nodal_disp[dof, 0]
        return d0

    def get_sensitivity(self):
        structure = self.structure
        # fv: equivalent global force vector for a yield component's udef
        members = structure.members.list
        empty_pv = np.zeros((structure.yield_specs.components_num, structure.yield_specs.components_num))
        pv = np.matrix(empty_pv)
        pv_column = 0

        empty_members_forces_sensitivity = np.zeros((structure.members.num, structure.yield_specs.components_num), dtype=object)
        empty_members_disps_sensitivity = np.zeros((structure.members.num, structure.yield_specs.components_num), dtype=object)
        empty_nodal_disps_sensitivity = np.zeros((1, structure.yield_specs.components_num), dtype=object)

        members_forces_sensitivity = np.matrix(empty_members_forces_sensitivity)
        members_disps_sensitivity = np.matrix(empty_members_disps_sensitivity)
        nodal_disps_sensitivity = np.matrix(empty_nodal_disps_sensitivity)

        for i_member, member in enumerate(members):
            if member.__class__.__name__ == "FrameMember2D":
                for yield_point_udef in member.udefs:
                    udef_components_num = yield_point_udef.shape[1]
                    for i_component in range(udef_components_num):
                        fv_size = structure.total_dofs_num
                        fv = np.zeros((fv_size, 1))
                        fv = np.matrix(fv)
                        component_udef_global = member.t.T * yield_point_udef[:, i_component]
                        start_dof = structure.node_dofs_num * member.nodes[0].num
                        end_dof = structure.node_dofs_num * member.nodes[1].num

                        fv[start_dof] = component_udef_global[0]
                        fv[start_dof + 1] = component_udef_global[1]
                        fv[start_dof + 2] = component_udef_global[2]

                        fv[end_dof] = component_udef_global[3]
                        fv[end_dof + 1] = component_udef_global[4]
                        fv[end_dof + 2] = component_udef_global[5]

                        affected_struc_disp = self.get_nodal_disp(fv)
                        nodal_disps_sensitivity[0, pv_column] = affected_struc_disp
                        affected_member_disps = self.get_members_disps(affected_struc_disp)
                        current_affected_member_ycns = 0
                        for i_affected_member, affected_member_disp in enumerate(affected_member_disps):

                            if i_member == i_affected_member:
                                fixed_force = -yield_point_udef[:, i_component]
                            else:
                                fixed_force = np.zeros((structure.node_dofs_num * 2, 1))
                                fixed_force = np.matrix(fixed_force)
                            # FIXME: affected_member_disp[0, 0] is for numpy oskolation when use matrix in matrix and enumerating on it.
                            affected_member_force = structure.members.list[i_affected_member].get_nodal_force(affected_member_disp[0, 0], fixed_force)
                            members_forces_sensitivity[i_affected_member, pv_column] = affected_member_force
                            members_disps_sensitivity[i_affected_member, pv_column] = affected_member_disp[0, 0]

                            if not member.section.nonlinear.has_axial_yield:
                                pv[current_affected_member_ycns, pv_column] = affected_member_force[2, 0]
                                pv[current_affected_member_ycns + 1, pv_column] = affected_member_force[5, 0]
                            else:
                                pv[current_affected_member_ycns, pv_column] = affected_member_force[0, 0]
                                pv[current_affected_member_ycns + 1, pv_column] = affected_member_force[2, 0]
                                pv[current_affected_member_ycns + 2, pv_column] = affected_member_force[3, 0]
                                pv[current_affected_member_ycns + 3, pv_column] = affected_member_force[5, 0]
                            current_affected_member_ycns = current_affected_member_ycns + structure.members.list[i_affected_member].yield_specs.components_num

                        pv_column += 1
        results = {
            "pv": pv,
            "nodal_disps_sensitivity": nodal_disps_sensitivity,
            "members_forces_sensitivity": members_forces_sensitivity,
            "members_disps_sensitivity": members_disps_sensitivity,
        }
        return results

    def get_nodal_disp_limits_sensitivity_rows(self):
        structure = self.structure
        disp_limits = structure.limits["disp_limits"]
        disp_limits_num = disp_limits.shape[0]
        empty_dv = np.zeros((disp_limits_num, structure.yield_specs.components_num))
        dv = np.matrix(empty_dv)
        for i, disp_limit in enumerate(disp_limits):
            node = disp_limit[0]
            node_dof = disp_limit[1]
            dof = structure.get_global_dof(node, node_dof)
            for j in range(structure.yield_specs.components_num):
                dv[i, j] = self.nodal_disps_sensitivity[0, j][dof, 0]
        return dv
