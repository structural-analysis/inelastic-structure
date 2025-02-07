## milestones
- hardening
- sifting + hardening
- sifting + softening


## optimizations
- use sparse matrix where applicable
- fix: softening matrices created even if have no softening
- current sensitivity calculations use all yield pieces, in sifting we can use just sifted yield pieces in sensitivity calcs.
- multiprocess in sensitivity calculation
- no need to update all b_matrix_inverse columns, 
mahini pivots instead of update binv,
Cc in mahini code is canonical matrix and bbar column and cbar row and disp rows and landabar row.
see mahini code this line:
Cc(:, [1 nonzeros(BasicsLoc)' + 1 prow + 1]) = pivot(prow, Cprow, Cc(:, [1 nonzeros(BasicsLoc)' + 1 prow + 1]))

- no need to calculate all table, we can just calculate needed a column and fpm cost and slack costs like mahini
- use comprehensive for loops instead of classic for loops for creating and filling lists
- use numpy arrays performant capabilities, it seems np.matrices are gettings deprecated.

# possible case of error:
we selected 4 pieces in sifting for a 3d yield surface
if we have some violations in a specific point
and if a corner happens in that point and all 4 pieces got activated 
or 3 pieces got activated and will in col was in selected pieces
then there is no space to include violated pieces.
so isn't it better to select 6 pieces instead of 4 for good measure?

possible bifurcation:
in examples:
- plate-semiconfined-inelastic
in mahini-sifted type in increment 23 there is multiple row with same b/a and b/c we change the order of piece numbers in final_pieces selection, the selected piece differs from unsifted version and this cause some change in results.
sifted pieces nums was 16 and 18 and num_in_structures was: 2132 and 2131
check b/a in increment 22 b/c will_out of increment 23 computed in inc 22.

-3d-simple beam
sifted and unsifted results was different b/c there is two yield points exactly in one point of structure.


INFINITE LOOP: 
of selected pieces in these examples:
- plate-semiconfined-inelastic:

in one run (on hamed laptop):
sifting with 4 pieces fails in example plate-semiconfined-inelastic.
in increment 72 violated pieces loops between 2268 and 2306 forever.
with 5 pieces example solved 
but there was some changes in responses, probabely b/c of bifurcation.
sifted results are in compare/plate-semiconfined-inelastic/sifted-4sifted_pieces.txt

in another run (in hassan laptop):
even with 5 or 6 pieces not solved. each time stuck in new infinite increment (like 58 or 89)
increased from 4 to 8 and solved. but results were differend and total increaments were 129
compared to previous results that was 117.

- 1story-dynamic-inelastic-ll1.0-ap400k
with 2 selected pieces

- 3d-2side-dynamic-inelastic
with 4 selected pieces. done with 8 pieces.


payanname mahini:
صفحه 64:

از سویی به جای محاسبه و ذخیره سازی کل ماتریس می توان تنها ستون های مورد نیاز آن را در زمان مقرر با استفاده از سطرها و ستون های مرتبط از ماتریس های تسلیم و تاثیر محاسبه نمود و به کار برد

ابتدای صفحه 66:
کاهش حجم ذخیره سازی با استفاده از روش سیمپلکس اصلاح شده

pivot optimization:
لازم نیست وقتی میخواهیم پیوت انجام دهیم تمام جدول را به روز کنیم
کد ماهینی را نگاه کنیم و ایده بگیریم
