import numpy as np
from dataclasses import dataclass
from functools import lru_cache

from ..points import NaturalPoint, GaussPoint
from ..sections.wall import WallSection
from ..yield_models import MemberYieldSpecs


@dataclass
class Response:
    nodal_force: np.array
    yield_components_force: np.array
    nodal_strains: np.array
    nodal_stresses: np.array
    nodal_moments: np.array = np.empty(0)


@dataclass
class Strain:
    x: float
    y: float
    xy: float


@dataclass
class Stress:
    x: float
    y: float
    xy: float


class WallMember:
    # calculations is based on four gauss points
    def __init__(self, num: int, section: WallSection, include_softening: bool, element_type: str, nodes: tuple):
        self.num = num
        self.section = section
        self.element_type = element_type  # Q4, Q4R, Q8, Q8R
        self.nodes = nodes
        self.nodes_count = len(self.nodes)
        self.node_dofs_count = 2
        self.dofs_count = self.node_dofs_count * self.nodes_count
        self.nodal_components_count = self.nodes_count * self.section.yield_specs.components_count
        self.gauss_points_count = len(self.gauss_points)
        self.yield_specs = MemberYieldSpecs(
            section=self.section,
            points_count=self.gauss_points_count,
            include_softening=include_softening,
        )
        self.k = self.get_stiffness()
        self.t = self.get_transform()
        self.m = self.get_mass() if self.section.material.rho else None
        # udef: unit distorsions equivalent forces (force, moment, ...) in nodes
        # udet: unit distorsions equivalent tractions (stress, force, moment, ...) in gauss points
        self.udefs, self.udets = self.get_nodal_forces_from_unit_distortions()

    @property
    def gauss_points(self):
        if self.element_type == "Q4R":
            gauss_points = [
                GaussPoint(weight=2, r=0, s=0),
            ]
        elif self.element_type in ("Q4", "Q8R"):
            gauss_points = [
                GaussPoint(weight=1, r=-0.57735027, s=-0.57735027),
                GaussPoint(weight=1, r=+0.57735027, s=-0.57735027),
                GaussPoint(weight=1, r=+0.57735027, s=+0.57735027),
                GaussPoint(weight=1, r=-0.57735027, s=+0.57735027),
            ]
        elif self.element_type == "Q8":
            gauss_points = [
                GaussPoint(weight=0.88888888, r=0, s=0),
                GaussPoint(weight=0.55555555, r=-0.77459667, s=-0.77459667),
                GaussPoint(weight=0.55555555, r=0, s=-0.77459667),
                GaussPoint(weight=0.55555555, r=+0.77459667, s=-0.77459667),
                GaussPoint(weight=0.55555555, r=+0.77459667, s=0),
                GaussPoint(weight=0.55555555, r=+0.77459667, s=+0.77459667),
                GaussPoint(weight=0.55555555, r=0, s=+0.77459667),
                GaussPoint(weight=0.55555555, r=-0.77459667, s=+0.77459667),
                GaussPoint(weight=0.55555555, r=-0.77459667, s=0),
            ]
        return gauss_points

    @property
    def natural_nodes(self):
        if self.element_type in ("Q4", "Q4R"):
            natural_nodes = [
                NaturalPoint(r=-1, s=-1),
                NaturalPoint(r=+1, s=-1),
                NaturalPoint(r=+1, s=+1),
                NaturalPoint(r=-1, s=+1),
            ]
        elif self.element_type in ("Q8", "Q8R"):
            natural_nodes = [
                NaturalPoint(r=-1, s=-1),
                NaturalPoint(r=0, s=-1),
                NaturalPoint(r=1, s=-1),
                NaturalPoint(r=1, s=0),
                NaturalPoint(r=1, s=1),
                NaturalPoint(r=0, s=1),
                NaturalPoint(r=-1, s=1),
                NaturalPoint(r=-1, s=0),
            ]
        return natural_nodes

    # REF: Cook (2002), p230.
    def get_extrapolated_natural_point(self, natural_point):
        if self.element_type == "Q4R":
            # TODO: write extrapolations for Q4R and Q8 integration schemes
            extrapolated_point = None
        elif self.element_type in ("Q4", "Q8R"):
            extrapolated_point = NaturalPoint(
                r=np.sqrt(3) * natural_point.r,
                s=np.sqrt(3) * natural_point.s,
            )
        if self.element_type == "Q8":
            extrapolated_point = None
        return extrapolated_point

    def get_extrapolation_shape_functions(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        if self.element_type in ("Q4", "Q8R"):
            n = np.array([
                0.25 * (1 - r) * (1 - s),
                0.25 * (1 + r) * (1 - s),
                0.25 * (1 + r) * (1 + s),
                0.25 * (1 - r) * (1 + s),
            ])
        elif self.element_type in ("Q8"):
            # TODO: correct shape functions for Q8 and Q4R gauss points
            n1 = 0.5 * (1 - r ** 2) * (1 - s)
            n3 = 0.5 * (1 + r) * (1 - s ** 2)
            n5 = 0.5 * (1 - r ** 2) * (1 + s)
            n7 = 0.5 * (1 - r) * (1 - s ** 2)

            n0 = 0.25 * (1 - r) * (1 - s) - 0.5 * (n7 + n1)
            n2 = 0.25 * (1 + r) * (1 - s) - 0.5 * (n1 + n3)
            n4 = 0.25 * (1 + r) * (1 + s) - 0.5 * (n3 + n5)
            n6 = 0.25 * (1 - r) * (1 + s) - 0.5 * (n5 + n7)

            n = np.array([n0, n1, n2, n3, n4, n5, n6, n7])
        return n

    def get_jacobian(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        nodes = self.nodes
        if self.element_type in ("Q4", "Q4R"):
            j = 0.25 * np.array([
                [-(1 - s), (1 - s), (1 + s), -(1 + s)],
                [-(1 - r), -(1 + r), (1 + r), (1 - r)],
            ]) @ np.array([
                [nodes[0].x, nodes[0].y],
                [nodes[1].x, nodes[1].y],
                [nodes[2].x, nodes[2].y],
                [nodes[3].x, nodes[3].y],
            ])
        elif self.element_type in ("Q8", "Q8R"):
            j = 0.25 * np.array([
                [-2 * r * s + 2 * r - s ** 2 + s, 4 * r * (s - 1), -2 * r * s + 2 * r + s ** 2 - s, 2 - 2 * s ** 2, 2 * r * s + 2 * r + s ** 2 + s, -4 * r * (s + 1), 2 * r * s + 2 * r - s ** 2 - s, 2 * s ** 2 - 2],
                [-r ** 2 - 2 * r * s + r + 2 * s, 2 * r ** 2 - 2, -r ** 2 + 2 * r * s - r + 2 * s, -4 * s * (r + 1), r ** 2 + 2 * r * s + r + 2 * s, 2 - 2 * r ** 2, r ** 2 - 2 * r * s - r + 2 * s, 4 * s * (r - 1)],
            ]) @ np.array([
                [nodes[0].x, nodes[0].y],
                [nodes[1].x, nodes[1].y],
                [nodes[2].x, nodes[2].y],
                [nodes[3].x, nodes[3].y],
                [nodes[4].x, nodes[4].y],
                [nodes[5].x, nodes[5].y],
                [nodes[6].x, nodes[6].y],
                [nodes[7].x, nodes[7].y],
            ])
        return j

    @lru_cache(maxsize=192)
    def get_shape_derivatives(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        j = self.get_jacobian(natural_point)
        b = np.zeros((3, 2 * self.nodes_count))

        if self.element_type in ("Q4", "Q4R"):
            du = 0.25 * np.linalg.inv(j) @ np.array([
                [-(1 - s), 0, (1 - s), 0, (1 + s), 0, -(1 + s), 0],
                [-(1 - r), 0, -(1 + r), 0, 1 + r, 0, 1 - r, 0],
            ])
            dv = 0.25 * np.linalg.inv(j) @ np.array([
                [0, -(1 - s), 0, (1 - s), 0, (1 + s), 0, -(1 + s)],
                [0, -(1 - r), 0, -(1 + r), 0, 1 + r, 0, 1 - r],
            ])
            b[0, :] = du[0, :]
            b[1, :] = dv[1, :]
            b[2, :] = du[1, :] + dv[0, :]

        elif self.element_type in ("Q8", "Q8R"):
            du = 0.25 * np.linalg.inv(j) @ np.array([
                [-2 * r * s + 2 * r - s ** 2 + s, 0, 4 * r * (s - 1), 0, -2 * r * s + 2 * r + s ** 2 - s, 0, 2 - 2 * s ** 2, 0, 2 * r * s + 2 * r + s ** 2 + s, 0, -4 * r * (s + 1), 0, 2 * r * s + 2 * r - s ** 2 - s, 0, 2 * s ** 2 - 2, 0],
                [-r ** 2 - 2 * r * s + r + 2 * s, 0, 2 * r ** 2 - 2, 0, -r ** 2 + 2 * r * s - r + 2 * s, 0, -4 * s * (r + 1), 0, r ** 2 + 2 * r * s + r + 2 * s, 0, 2 - 2 * r ** 2, 0, r ** 2 - 2 * r * s - r + 2 * s, 0, 4 * s * (r - 1), 0],
            ])
            dv = 0.25 * np.linalg.inv(j) @ np.array([
                [0, -2 * r * s + 2 * r - s ** 2 + s, 0, 4 * r * (s - 1), 0, -2 * r * s + 2 * r + s ** 2 - s, 0, 2 - 2 * s ** 2, 0, 2 * r * s + 2 * r + s ** 2 + s, 0, -4 * r * (s + 1), 0, 2 * r * s + 2 * r - s ** 2 - s, 0, 2 * s ** 2 - 2],
                [0, -r ** 2 - 2 * r * s + r + 2 * s, 0, 2 * r ** 2 - 2, 0, -r ** 2 + 2 * r * s - r + 2 * s, 0, -4 * s * (r + 1), 0, r ** 2 + 2 * r * s + r + 2 * s, 0, 2 - 2 * r ** 2, 0, r ** 2 - 2 * r * s - r + 2 * s, 0, 4 * s * (r - 1)],
            ])
            b[0, :] = du[0, :]
            b[1, :] = dv[1, :]
            b[2, :] = du[1, :] + dv[0, :]
        return b

    def get_shape_function(self, natural_point):
        r = natural_point.r
        s = natural_point.s
        n = np.zeros((2, 2 * self.nodes_count))
        if self.element_type in ("Q4", "Q4R"):
            n1 = 0.25 * (1 - r) * (1 - s)
            n2 = 0.25 * (1 + r) * (1 - s)
            n3 = 0.25 * (1 + r) * (1 + s)
            n4 = 0.25 * (1 - r) * (1 + s)
            n = np.array([
                [n1, 0, n2, 0, n3, 0, n4, 0],
                [0, n1, 0, n2, 0, n3, 0, n4]
            ])
        elif self.element_type in ("Q8", "Q8R"):
            n1 = 0.25 * (1 - r) * (1 - s) * (-r - s - 1)
            n2 = 0.5 * (1 + r) * (1 - r) * (1 - s)
            n3 = 0.25 * (1 + r) * (1 - s) * (r - s - 1)
            n4 = 0.5 * (1 + r) * (1 + s) * (1 - s)
            n5 = 0.25 * (1 + r) * (1 + s) * (r + s - 1)
            n6 = 0.5 * (1 + r) * (1 - r) * (1 + s)
            n7 = 0.25 * (1 - r) * (1 + s) * (-r + s - 1)
            n8 = 0.5 * (1 - r) * (1 + s) * (1 - s)
            n = np.array([
                [n1, 0, n2, 0, n3, 0, n4, 0, n5, 0, n6, 0, n7, 0, n8, 0],
                [0, n1, 0, n2, 0, n3, 0, n4, 0, n5, 0, n6, 0, n7, 0, n8]
            ])
        return n

    def get_stiffness(self):
        k = np.zeros((self.dofs_count, self.dofs_count))
        for gauss_point in self.gauss_points:
            b = self.get_shape_derivatives(gauss_point)
            j = self.get_jacobian(gauss_point)
            j_det = np.linalg.det(j)
            # gauss_point_k = gauss_point.weight * b.T * self.section.ce * b * j_det * self.section.geometry.thickness * self.section.geometry.thickness * self.section.geometry.thickness
            gauss_point_k = gauss_point.weight * (b.T @ (self.section.ce @ b)) * j_det * self.section.geometry.thickness
            k += gauss_point_k
        return k

    def get_mass(self):
        m = np.zeros((self.dofs_count, self.dofs_count))
        for gauss_point in self.gauss_points:
            n = self.get_shape_function(gauss_point)
            j = self.get_jacobian(gauss_point)
            j_det = np.linalg.det(j)
            gauss_point_m = gauss_point.weight * n.T * self.section.material.rho * n * j_det * self.section.geometry.thickness
            m += gauss_point_m
        diagonal_mass = self.diagonalize_mass(m)
        return diagonal_mass

    def diagonalize_mass(self, m):
        diagonal_m = np.zeros((self.dofs_count, self.dofs_count))
        np.fill_diagonal(diagonal_m, m.diagonal())
        return m.sum() / m.diagonal().sum() * diagonal_m

    def get_transform(self):
        return np.eye(self.dofs_count)

    def get_nodal_strains_and_stresses(self, nodal_disp, fixed_internal):
        nodal_strains = np.zeros(3 * self.nodes_count)
        nodal_stresses = np.zeros(3 * self.nodes_count)
        i = 0
        for natural_node in self.natural_nodes:
            natural_point_strain, natural_point_stress, yield_components_force = self.get_natural_point_strain_and_stress(natural_node, nodal_disp, fixed_internal)
            nodal_strains[i] = natural_point_strain.x
            nodal_strains[i + 1] = natural_point_strain.y
            nodal_strains[i + 2] = natural_point_strain.xy

            nodal_stresses[i] = natural_point_stress.x
            nodal_stresses[i + 1] = natural_point_stress.y
            nodal_stresses[i + 2] = natural_point_stress.xy
            i += 3
        return nodal_strains, nodal_stresses, yield_components_force

    def get_natural_point_strain_and_stress(self, natural_point, nodal_disp, fixed_internal):
        extrapolated_natural_point = self.get_extrapolated_natural_point(natural_point)
        shape_functions = self.get_extrapolation_shape_functions(extrapolated_natural_point)
        gauss_points_strains, gauss_points_stresses, yield_components_force = self.get_gauss_points_responses(nodal_disp)
        if fixed_internal.any():
            for i in range(self.gauss_points_count):
                gauss_points_stresses[i, :] += fixed_internal[3 * i:3 * (i + 1)].T

        natural_point_strain = gauss_points_strains.T @ shape_functions.T
        natural_point_stress = gauss_points_stresses.T @ shape_functions.T
        natural_point_strain_object = Strain(x=natural_point_strain[0], y=natural_point_strain[1], xy=natural_point_strain[2])
        natural_point_stress_object = Stress(x=natural_point_stress[0], y=natural_point_stress[1], xy=natural_point_stress[2])
        return natural_point_strain_object, natural_point_stress_object, yield_components_force

    def get_gauss_points_responses(self, nodal_disp):
        gauss_points_strains = np.zeros((self.gauss_points_count, 3))
        gauss_points_stresses = np.zeros((self.gauss_points_count, 3))
        yield_components_force = np.zeros(3 * self.gauss_points_count)
        j = 0
        for i, gauss_point in enumerate(self.gauss_points):
            gauss_point_strain, gauss_point_stress = self.get_gauss_point_strain_and_stress(gauss_point, nodal_disp)
            gauss_points_strains[i, 0] = gauss_point_strain.x
            gauss_points_strains[i, 1] = gauss_point_strain.y
            gauss_points_strains[i, 2] = gauss_point_strain.xy

            gauss_points_stresses[i, 0] = gauss_point_stress.x
            gauss_points_stresses[i, 1] = gauss_point_stress.y
            gauss_points_stresses[i, 2] = gauss_point_stress.xy

            yield_components_force[j] = gauss_point_stress.x
            yield_components_force[j + 1] = gauss_point_stress.y
            yield_components_force[j + 2] = gauss_point_stress.xy
            j += 3

        return gauss_points_strains, gauss_points_stresses, yield_components_force

    def get_gauss_point_strain_and_stress(self, gauss_point, nodal_disp):
        gauss_point_b = self.get_shape_derivatives(gauss_point)
        e = gauss_point_b @ nodal_disp
        s = self.section.ce @ e
        gauss_point_strain = Strain(x=e[0], y=e[1], xy=e[2])
        gauss_point_stress = Stress(x=s[0], y=s[1], xy=s[2])
        return gauss_point_strain, gauss_point_stress

    def get_unit_distortion(self, gauss_point_component_num):
        distortion = np.zeros(3)
        distortion[gauss_point_component_num] = 1
        return distortion

    # for element with linear variation of stress
    # REF: Cook (2002), p228.
    def get_nodal_force_from_unit_distortion(self, gauss_point, gauss_point_component_num):
        gauss_point_b = self.get_shape_derivatives(gauss_point)
        distortion = self.get_unit_distortion(gauss_point_component_num)
        j = self.get_jacobian(gauss_point)
        j_det = np.linalg.det(j)
        nodal_force = (gauss_point_b.T @ (self.section.ce @ distortion)) * self.section.geometry.thickness * j_det
        gauss_point_stress = self.section.ce @ distortion
        return nodal_force, gauss_point_stress

    def get_nodal_forces_from_unit_distortions(self):
        nodal_forces = np.zeros((self.dofs_count, self.yield_specs.components_count))
        gauss_points_stresses = np.zeros((self.yield_specs.components_count, self.yield_specs.components_count))
        component_base_num = 0
        for gauss_point in self.gauss_points:
            for j in range(3):
                nodal_forces[:, component_base_num + j] = self.get_nodal_force_from_unit_distortion(gauss_point=gauss_point, gauss_point_component_num=j)[0]
                gauss_points_stresses[component_base_num:(component_base_num + 3), component_base_num + j] = self.get_nodal_force_from_unit_distortion(gauss_point=gauss_point, gauss_point_component_num=j)[1]
            component_base_num += 3
        return nodal_forces, gauss_points_stresses

    def get_response(self, nodal_disp, fixed_external=None, fixed_internal=None):
        # fixed internal: fixed internal tractions like stress, force, moment, ... in gauss points of a member
        # fixed external: fixed external forces like force, moment, ... nodes of a member

        if fixed_external is None:
            fixed_external = np.zeros(self.dofs_count)

        if fixed_internal is None:
            fixed_internal = np.zeros(self.yield_specs.components_count)

        if fixed_external.any():
            nodal_force = self.k @ nodal_disp + fixed_external
        else:
            nodal_force = self.k @ nodal_disp

        nodal_strains, nodal_stresses, yield_components_force = self.get_nodal_strains_and_stresses(nodal_disp, fixed_internal)
        response = Response(
            nodal_force=nodal_force,
            yield_components_force=yield_components_force,
            nodal_strains=nodal_strains,
            nodal_stresses=nodal_stresses,
        )
        return response
