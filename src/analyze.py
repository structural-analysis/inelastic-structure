from workshop import create_structure
structure = create_structure()

disp = structure.disp
elements_disps = structure.get_elements_disps(disp)
