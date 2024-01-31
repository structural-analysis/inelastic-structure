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
in plate-semiconfined-inelastic example in mahini-sifted type in increment 23 there is multiple row with same b/a and b/c we change the order of piece numbers in final_pieces selection, the selected piece differs from unsifted version and this cause some change in results.
sifted pieces nums was 16 and 18 and num_in_structures was: 2132 and 2131
check b/a in increment 22 b/c will_out of increment 23 computed in inc 22.

and in example: 3d-simple beam sifted and unsifted results was different b/c there is two yield points exactly in one point of structure.