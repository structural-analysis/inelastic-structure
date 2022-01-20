import numpy as np

from src.workshop import create_structure
structure = create_structure()
elements = structure.elements
elements_disps = structure.get_elements_disps(structure.disp)

fixed_force = np.zeros((structure.node_n_dof * 2, 1))
fixed_force = np.matrix(fixed_force)

structure_ycn = 0
for element in elements:
    structure_ycn = structure_ycn + element.total_ycn

# calculate p0
elements_forces = []
p0 = np.zeros((structure_ycn, 1))
pv = np.zeros((structure_ycn, structure_ycn))
pv_column = 0
elements_yc = 0
for i, element in enumerate(elements):
    if element.__class__.__name__ == "FrameElement2D":
        fs = element.get_nodal_force(elements_disps[i], fixed_force)
        elements_forces.append(fs)
        if not element.has_axial_yield:
            p0[elements_yc] = fs[0, 2]
            p0[elements_yc + 1] = fs[0, 5]
        else:
            p0[elements_yc] = fs[0, 0]
            p0[elements_yc + 1] = fs[0, 2]
            p0[elements_yc + 2] = fs[0, 3]
            p0[elements_yc + 3] = fs[0, 5]
        for component_udef in element.udefs:
            # calculate pv
            fv_size = structure.node_n_dof * structure.n_nodes
            fv = np.zeros((fv_size, 1))
            fv = np.matrix(fv)
            component_udef_global = element.t.T * component_udef
            start_dof = structure.node_n_dof * element.nodes[0].num
            end_dof = structure.node_n_dof * element.nodes[1].num

            fv[start_dof] = component_udef_global[0]
            fv[start_dof + 1] = component_udef_global[1]
            fv[start_dof + 2] = component_udef_global[2]

            fv[end_dof] = component_udef_global[3]
            fv[end_dof + 1] = component_udef_global[4]
            fv[end_dof + 2] = component_udef_global[5]

            struc_disp = structure.compute_structure_displacement(fv)
            elem_disps = structure.get_elements_disps(struc_disp)
            for elem_disp in elem_disps:
                fs1 = element.get_nodal_force(elem_disp, -component_udef.T)
                print(f"{fs1=}")
                print(f"{elements_yc=}")
                if not element.has_axial_yield:
                    pv[elements_yc, pv_column] = fs1[0, 2]
                    pv[elements_yc + 1, pv_column] = fs1[0, 5]
                else:
                    pv[elements_yc, pv_column] = fs1[0, 0]
                    pv[elements_yc + 1, pv_column] = fs1[0, 2]
                    pv[elements_yc + 2, pv_column] = fs1[0, 3]
                    pv[elements_yc + 3, pv_column] = fs1[0, 5]
            pv_column += 1
    elements_yc = elements_yc + element.total_ycn
