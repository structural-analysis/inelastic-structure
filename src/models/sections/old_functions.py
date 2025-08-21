
###############################################################################
# Build discretizing points & gradients for Hill's yield surface
###############################################################################

###############################################################################
# Compute the phi matrix (gradients of yield planes) for Hill criterion
###############################################################################
def get_hill_matrix(mp, M0x, M0y, M0xy, rho, gamma):
    """
    Construct the matrix of yield plane normals (phi) for the Hill yield surface.
    Each column phi[:,k] corresponds to a linear yield plane: phi[0]*Mx + phi[1]*My + phi[2]*Mxy = 1.
    The planes include barrel facets (connecting adjacent xi rings) and two caps.
    """
    # Define xi discretization similar to the J2 case (a set of rings between +/- xi_max)
    xi_max = np.sqrt(4.0 / (2.0 - rho)) if (2.0 - rho) > 1e-8 else 2.0  # theoretical max of xi (equi-biaxial yield)
    # Choose xi levels (fractions of xi_max) for discretization
    xi_levels = [0.95, 0.85, 0.60, 0.50, 0.25, 0.0]
    xi_values = sorted({xi_max * lvl for lvl in xi_levels} | {-xi_max * lvl for lvl in xi_levels})
    xi_values = np.array(xi_values)

    # Theta angles (m segments around)
    m = 40
    theta_values = np.linspace(0, 2 * np.pi, m, endpoint=False)

    # Compute all surface points and gradients
    coords, grads = build_discretizing_points_and_gradients_hill(xi_values, theta_values, M0x, M0y, M0xy, rho, gamma)
    n_xi = len(xi_values)
    p_total = (n_xi - 1) * m + 2  # total number of yield planes = barrel facets + 2 caps
    phi = np.zeros((3, p_total))

    # A) Top cap plane (positive equi-biaxial Mx = My)
    # Find yield point for Mx=My (positive) by solving f(M, M, 0) = 1
    lo, hi = 0.0, max(M0x, M0y) * 2.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        # Hill yield function for sx = sy = M (txy=0)
        f_mid = (mid / M0x) ** gamma + (mid / M0y) ** gamma - rho * ((mid * mid) / (M0x * M0y)) ** (gamma / 2.0)
        if f_mid >= 1.0:
            hi = mid
        else:
            lo = mid

    M0b = 0.5 * (lo + hi)  # approximate yield moment under bi-axial bending
    sx_cap = sy_cap = M0b
    # Gradient at the equi-biaxial yield point
    val_P = sx_cap * sy_cap / (M0x * M0y)
    grad_cap_x = gamma * ((sx_cap / M0x) ** (gamma - 1.0)) / M0x \
                 - rho * (gamma / 2.0) * ((val_P) ** (gamma / 2.0 - 1.0) if val_P > 1e-12 else 0.0) * (sy_cap / (M0x * M0y))
    grad_cap_y = gamma * ((sy_cap / M0y) ** (gamma - 1.0)) / M0y \
                 - rho * (gamma / 2.0) * ((val_P) ** (gamma / 2.0 - 1.0) if val_P > 1e-12 else 0.0) * (sx_cap / (M0x * M0y))
    grad_cap_t = 0.0
    # Scale gradient so that phi ⋅ [sx_cap, sy_cap, 0] = 1
    norm_factor = grad_cap_x * sx_cap + grad_cap_y * sy_cap  # dot with the point
    phi[:, 0] = np.array([grad_cap_x, grad_cap_y, grad_cap_t]) / norm_factor / mp

    # B) Bottom cap plane (negative equi-biaxial Mx = My)
    sx_cap_neg = sy_cap_neg = -M0b
    val_P_neg = sx_cap_neg * sy_cap_neg / (M0x * M0y)
    grad_cap_x_b = gamma * ((abs(sx_cap_neg) / M0x) ** (gamma - 1.0)) * (-1 if sx_cap_neg < 0 else 1) / M0x \
                   - rho * (gamma / 2.0) * ((val_P_neg) ** (gamma / 2.0 - 1.0) if val_P_neg > 1e-12 else 0.0) * (sy_cap_neg / (M0x * M0y))
    grad_cap_y_b = gamma * ((abs(sy_cap_neg) / M0y) ** (gamma - 1.0)) * (-1 if sy_cap_neg < 0 else 1) / M0y \
                   - rho * (gamma / 2.0) * ((val_P_neg) ** (gamma / 2.0 - 1.0) if val_P_neg > 1e-12 else 0.0) * (sx_cap_neg / (M0x * M0y))
    grad_cap_t_b = 0.0
    norm_factor_b = grad_cap_x_b * sx_cap_neg + grad_cap_y_b * sy_cap_neg
    phi[:, -1] = np.array([grad_cap_x_b, grad_cap_y_b, grad_cap_t_b]) / norm_factor_b / mp

    # C) Barrel facet planes: for each quadrilateral facet between ring i and i+1
    k = 1  # start filling phi columns after the top cap
    for i in range(n_xi - 1):
        for j in range(m):
            sx, sy, txy = coords[i, j]
            dfd_sx, dfd_sy, dfd_txy = grads[i, j]
            # Plane normal = gradient; scale such that φ⋅[sx,sy,txy] = 1
            norm = dfd_sx * sx + dfd_sy * sy + dfd_txy * txy
            phi[:, k] = np.array([dfd_sx, dfd_sy, dfd_txy]) / norm / mp
            k += 1
    return phi
