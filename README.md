# Inelastic Structure
all units are SI (N, m, s)
each member object must have these variables (e.g.):
nodes
total_dofs_num

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

for profiling do the steps below:
in prof.py choose the example:
## to run the example to profile, run the script:
python -m src.prof
## to visualize profiling results run the script:
snakeviz .\profile_data.prof

## to profile line by line
we use line-profiler package
to use:
add a @profile decorator above the function (no need to import anything)
run the code using:
kernprof -l -v run.py

## to run tests
python -m test