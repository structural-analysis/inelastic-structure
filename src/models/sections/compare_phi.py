import numpy as np
from .functions import get_hill_matrix, get_von_mises_matrix

def compare_phi(phi_ref, phi_new, try_row_swap=True, name_ref="mises", name_new="hill"):
    """
    Compare two phi matrices element-by-element and print summary stats.

    Args:
        phi_ref: (3, N) reference matrix (e.g., legacy von Mises).
        phi_new: (3, N) new matrix (e.g., Hill with J2 params).
        try_row_swap: if True, also evaluate the case with rows 0 and 1 swapped in phi_new.
        name_ref/name_new: labels for printing.

    Returns:
        A dict with stats for the chosen alignment, plus (optionally) stats for the swapped case.
    """
    def _stats(A, B):
        if A.shape != B.shape:
            raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
        diff = A - B
        abs_diff = np.abs(diff)
        min_diff = float(abs_diff.min())
        max_diff = float(abs_diff.max())
        max_idx = np.unravel_index(abs_diff.argmax(), abs_diff.shape)
        mean_diff = float(abs_diff.mean())
        median_diff = float(np.median(abs_diff))

        # relative diffs (guard tiny ref values)
        denom = np.maximum(np.abs(A), 1e-30)
        rel = abs_diff / denom
        max_rel = float(rel.max())
        mean_rel = float(rel.mean())

        # norms/similarities
        ref_norm = float(np.linalg.norm(A))
        new_norm = float(np.linalg.norm(B))
        diff_norm = float(np.linalg.norm(diff))
        exactness = max(0.0, 1.0 - diff_norm / (ref_norm + 1e-30))
        # cosine similarity in flattened space
        cos_sim = float(np.dot(A.ravel(), B.ravel()) /
                        ((ref_norm * new_norm) + 1e-30))

        return {
            "min_abs_diff": min_diff,
            "max_abs_diff": max_diff,
            "max_abs_diff_at": {"row": int(max_idx[0]), "col": int(max_idx[1])},
            "mean_abs_diff": mean_diff,
            "median_abs_diff": median_diff,
            "max_rel_diff": max_rel,
            "mean_rel_diff": mean_rel,
            "fro_norm_ref": ref_norm,
            "fro_norm_new": new_norm,
            "fro_norm_diff": diff_norm,
            "exactness_index": exactness,   # 1.0 = identical
            "cosine_similarity": cos_sim,   # 1.0 = identical up to scale
            "shape": A.shape
        }

    # direct comparison
    direct = _stats(phi_ref, phi_new)
    chosen = ("direct", direct)
    swapped = None

    if try_row_swap:
        phi_new_swapped = phi_new.copy()
        phi_new_swapped[[0, 1], :] = phi_new_swapped[[1, 0], :]  # swap rows Mx/My
        swapped = _stats(phi_ref, phi_new_swapped)
        # pick the better alignment by exactness_index
        if swapped["exactness_index"] > direct["exactness_index"]:
            chosen = ("row_swap(0↔1)", swapped)

    # pretty print
    label, stats = chosen
    print(f"Comparison {name_new} vs {name_ref}  -> using alignment: {label}")
    print(f"  shape: {stats['shape']}")
    print(f"  lowest |Δ|      : {stats['min_abs_diff']:.6e}")
    print(f"  highest |Δ|     : {stats['max_abs_diff']:.6e} at (row={stats['max_abs_diff_at']['row']}, col={stats['max_abs_diff_at']['col']})")
    print(f"  mean |Δ|        : {stats['mean_abs_diff']:.6e}")
    print(f"  median |Δ|      : {stats['median_abs_diff']:.6e}")
    print(f"  max rel Δ       : {stats['max_rel_diff']:.6e}")
    print(f"  mean rel Δ      : {stats['mean_rel_diff']:.6e}")
    print(f"  ||ref||_F       : {stats['fro_norm_ref']:.6e}")
    print(f"  ||new||_F       : {stats['fro_norm_new']:.6e}")
    print(f"  ||Δ||_F         : {stats['fro_norm_diff']:.6e}")
    print(f"  exactness index : {stats['exactness_index']:.6f}  (1.000 = identical)")
    print(f"  cosine similarity: {stats['cosine_similarity']:.6f}")

    if swapped is not None:
        print("\n(For reference) direct vs swapped exactness:")
        print(f"  direct  exactness: {direct['exactness_index']:.6f}")
        print(f"  swapped exactness: {swapped['exactness_index']:.6f}")

    return {"chosen_alignment": label, "chosen_stats": stats,
            "direct_stats": direct, "swapped_stats": swapped}

# -------------------------
# Example usage (plug yours):
# -------------------------
mp = 150000
phi_mises = get_von_mises_matrix(mp)
phi_hill  = get_hill_matrix(mp, M0x=1, M0y=1, M0xy=1/np.sqrt(3), rho=1.0, gamma=2.0)
results = compare_phi(phi_ref=phi_mises, phi_new=phi_hill, try_row_swap=True,
                      name_ref="mises", name_new="hill(J2)")

print(results)