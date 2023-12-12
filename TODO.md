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

