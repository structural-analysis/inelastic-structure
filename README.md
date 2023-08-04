# Inelastic Structure
all units are SI (N, m, s)
each member object must have these variables (e.g.):
nodes
total_dofs_num


## TODO:

- visualizations
- include hardening and softening
- write tests for all examples, checking outputs vs inputs.
- separate loading from structure
- add plate element
- dynamic analysis
- add more elements
- optimize code to full usage of numpy (remove for loops, ...)
- improve performance (run every massive calculation with numpy)
- use sifting
- upgrade plate to polygon elements
- multiprocess
- do TODOs


# 3D Frame:
in 3d frame moment33 is dof = 4, 10, moment22 is dof = 5, 11, shear33 is dof 2, 8 and
shear22 is dof 1, 7.