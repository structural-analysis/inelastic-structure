import numpy as np

from src.workshop import create_structure
structure = create_structure()

disp = structure.disp
elements_disps = structure.get_elements_disps(disp)
fixed_force = np.zeros((structure.node_n_dof * 2, 1))
fixed_force = np.matrix(fixed_force)
p0 = np.zeros((2 * len(structure.elements), 1))
elements_forces = []
for i, element in enumerate(structure.elements):
    fs = element.get_nodal_forces(elements_disps[i], fixed_force)
    elements_forces.append(fs)
    p0[2 * i] = fs[0, 2]
    p0[2 * i + 1] = fs[0, 5]
    # disp1 = structure.compute_structure_displacement(structure.)
    for section in range(2):
        fv = np.zeros((structure.node_n_dof * structure.n_nodes, 1))
        fv = np.matrix(fv)
        udef_global = element.t.T * element.udef[section].T
        fv[structure.node_n_dof * element.nodes[0].num] = udef_global[0]
        fv[structure.node_n_dof * element.nodes[0].num + 1] = udef_global[1]
        fv[structure.node_n_dof * element.nodes[0].num + 2] = udef_global[2]

        fv[structure.node_n_dof * element.nodes[1].num] = udef_global[3]
        fv[structure.node_n_dof * element.nodes[1].num + 1] = udef_global[4]
        fv[structure.node_n_dof * element.nodes[1].num + 2] = udef_global[5]
        struc_disp = structure.compute_structure_displacement(fv)
        elem_disps = structure.get_elements_disps(struc_disp)
        for elem_disp in elem_disps:
            fs = element.get_nodal_forces(elem_disp, -element.udef[section].T)
