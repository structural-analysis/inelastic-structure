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
- upgrade plate to polygon elements
- multiprocess
- do TODOs


# 3D Frame:
in 3d frame moment33 is dof = 4, 10, moment22 is dof = 5, 11, shear33 is dof 2, 8 and
shear22 is dof 1, 7.

payanname mahini:
صفحه 64:

از سویی به جای محاسبه و ذخیره سازی کل ماتریس می توان تنها ستون های مورد نیاز آن را در زمان مقرر با استفاده از سطرها و ستون های مرتبط از ماتریس های تسلیم و تاثیر محاسبه نمود و به کار برد

ابتدای صفحه 66:
کاهش حجم ذخیره سازی با استفاده از روش سیمپلکس اصلاح شده

pivot optimization:
لازم نیست وقتی میخواهیم پیوت انجام دهیم تمام جدول را به روز کنیم
کد ماهینی را نگاه کنیم و ایده بگیریم
