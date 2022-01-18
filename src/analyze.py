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
    fv = np.zeros((structure.node_n_dof * structure.n_nodes, 1))