import numpy as np
from functools import lru_cache


# FIXME: FIX OPTIMIZED NOT WITH CACHING
@lru_cache(maxsize=192)
def get_von_mises_matrix(mp):
    si = np.array([1.9, 1.7, 1.2, 1, 0.5, 0, -0.5, -1, -1.2, -1.7, -1.9])
    m = 40
    n = si.shape[0]  # -2 & +2 will produce only one plane each
    p_total = m * n + 2  # total number of yield planes
    teta = np.zeros(40)
    pi = np.pi
    for i in range(m):
        teta[i] = 2 * pi * i / m

    # specifying two end planes
    phi = np.zeros((3, p_total))
    phi[:, 0] = np.array([0.5, 0.5, 0]) / mp
    phi[:, p_total - 1] = np.array([-0.5, -0.5, 0]) / mp

    q = 0
    for i in range(n):
        for j in range(m):
            k = j + q + 1
            phi[:, k] = np.array([
                0.25 * (si[i] - 3 * np.cos(teta[j]) * np.sqrt((4 - (si[i]) ** 2) / (3 * (1 + np.sin(teta[j]) ** 2)))),
                0.25 * (si[i] + 3 * np.cos(teta[j]) * np.sqrt((4 - (si[i]) ** 2) / (3 * (1 + np.sin(teta[j]) ** 2)))),
                1.5 * np.sqrt(2) * np.sin(teta[j]) * np.sqrt((4 - (si[i]) ** 2) / (3 * (1 + np.sin(teta[j]) ** 2)))
            ]) / mp
        q += m
    return phi


def get_hill_matrix(mp, M0x, M0y, M0xy, rho, gamma):
    """
    General Hill (power-law) yield planes, ordered like legacy J2:
      - Column 0:  +cap  (Mx=My>0 line)
      - Columns 1..(m*n): barrel (rings i=0..n-1, theta j=0..m-1; theta runs fastest)
      - Last column: -cap  (Mx=My<0 line)
    Each plane is scaled by 1/mp, and normalized so that φ·P = 1 at its construction point P.

    Pass J2 parameters to recover von Mises as a special case (no branching):
      M0x = M0y = mp
      M0xy = mp / sqrt(3)
      rho  = 1.0
      gamma = 2.0
    """
    # ---- legacy-like ring/sector sampling ----
    # same si and m as your old get_von_mises_matrix (keeps order/indexing consistent)
    si = np.array([1.9, 1.7, 1.2, 1.0, 0.5, 0.0, -0.5, -1.0, -1.2, -1.7, -1.9], dtype=float)
    m = 40
    theta_values = np.linspace(0.0, 2.0*np.pi, m, endpoint=False)

    # Map si ∈ [-2,2] to Hill’s xi range using the same D1 used in your builder:
    # D1 = 4/(2 - rho), so |xi| ≤ sqrt(D1). Linear map: xi = si * (sqrt(D1)/2).
    D1 = 4.0 / (2.0 - rho)
    xi_scale = np.sqrt(D1) / 2.0
    xi_values = si * xi_scale

    # ---- get surface points + gradients from your builder ----
    coords, grads = build_discretizing_points_and_gradients_hill(
        xi_values=xi_values,
        theta_values=theta_values,
        M0x=M0x, M0y=M0y, M0xy=M0xy,
        rho=rho, gamma=gamma
    )

    n_xi = len(xi_values)
    p_total = m * n_xi + 2
    phi = np.zeros((3, p_total), dtype=float)

    # ---------- caps (bi-axial Mx=My, txy=0) ----------
    # Solve f(M,M,0)=1 with bisection; use the same power-law form as your builder.
    def _f_biax(M):
        # (txy=0) → term_xy = 0; cross term uses (|Mx*My|/(M0x*M0y))^(gamma/2)
        term_x = (abs(M) / M0x) ** gamma
        term_y = (abs(M) / M0y) ** gamma
        term_cross = rho * ((abs(M*M) / (M0x*M0y)) ** (gamma / 2.0))
        return term_x + term_y - term_cross  # = 1 on the surface

    # positive cap
    lo, hi = 0.0, max(M0x, M0y) * 5.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _f_biax(mid) >= 1.0:
            hi = mid
        else:
            lo = mid
    M_cap = 0.5 * (lo + hi)
    P_cap_pos = np.array([M_cap, M_cap, 0.0], dtype=float)

    # gradient of the power-law Hill at (Mx,My,txy)
    def _grad_power(Mx, My, Txy):
        # Avoid zero^(-something): treat ~zero as zero contribution
        eps = 1e-14
        # cross “product” term exponent base:
        prod = (Mx * My) / (M0x * M0y)
        abs_prod = abs(prod)
        sgn_prod = 0.0 if abs_prod < eps else (1.0 if prod > 0.0 else -1.0)

        dfd_Mx = gamma * ((abs(Mx)/M0x)**(gamma-1.0) if abs(Mx)>eps else 0.0) * np.sign(Mx) / M0x \
                 - rho * (gamma/2.0) * ((abs_prod)**(gamma/2.0 - 1.0) if abs_prod>eps else 0.0) * sgn_prod * (My / (M0x*M0y))
        dfd_My = gamma * ((abs(My)/M0y)**(gamma-1.0) if abs(My)>eps else 0.0) * np.sign(My) / M0y \
                 - rho * (gamma/2.0) * ((abs_prod)**(gamma/2.0 - 1.0) if abs_prod>eps else 0.0) * sgn_prod * (Mx / (M0x*M0y))
        dfd_T  = gamma * ((abs(Txy)/M0xy)**(gamma-1.0) if abs(Txy)>eps else 0.0) * np.sign(Txy) / M0xy
        return np.array([dfd_Mx, dfd_My, dfd_T], dtype=float)

    g_cap_pos = _grad_power(*P_cap_pos)
    # normalize so that φ·P = 1 (then scale by 1/mp)
    norm_cap_pos = float(np.dot(g_cap_pos, P_cap_pos))
    phi[:, 0] = (g_cap_pos / norm_cap_pos) / mp

    # negative cap
    P_cap_neg = -P_cap_pos
    g_cap_neg = _grad_power(*P_cap_neg)
    norm_cap_neg = float(np.dot(g_cap_neg, P_cap_neg))
    phi[:, -1] = (g_cap_neg / norm_cap_neg) / mp

    # ---------- barrel (same order as legacy: ring i outer loop, theta j inner loop) ----------
    col = 1
    for i in range(n_xi):
        for j in range(m):
            P = coords[i, j, :]       # (Mx, My, Txy)
            g = grads[i, j, :]        # gradient at that point (your builder's formula)
            denom = float(np.dot(g, P))
            # Guard against degenerate points extremely close to caps
            if abs(denom) < 1e-14:
                # fallback: skip normalization change (keeps array stable)
                phi[:, col] = phi[:, col-1]
            else:
                phi[:, col] = (g / denom) / mp
            col += 1

    return phi


def build_discretizing_points_and_gradients_hill(xi_values, theta_values, M0x, M0y, M0xy, rho, gamma):
    """
    Computes coordinates (sx, sy, txy) on the Hill yield surface and their gradients.
    Parameters:
      xi_values   -- array of xi parameter values (dimensionless combination of Mx and My).
      theta_values -- array of theta angle values (0 to 2π) for spanning each ring.
      M0x, M0y, M0xy -- yield moment capacities about x, y, and shear (xy).
      rho, gamma   -- Hill yield parameters (rho for cross-coupling, gamma for exponent).
    Returns:
      coords[i,j] = (sx, sy, txy) coordinates on the yield surface for xi=xi_values[i] and θ=theta_values[j].
      grads[i,j]  = gradient vector [df/dsx, df/dsy, df/dtxy] of the yield function at that coordinate.
    """
    n_xi = len(xi_values)
    n_th = len(theta_values)
    coords = np.zeros((n_xi, n_th, 3))
    grads = np.zeros((n_xi, n_th, 3))
    # Precompute constants for initial geometric approximation (for gamma ≈ 2)
    D1 = 4.0 / (2.0 - rho)     # related to equi-biaxial combination
    D2 = 4.0 / (2.0 + rho)     # related to orthogonal combination

    for i, xi in enumerate(xi_values):
        for j, theta in enumerate(theta_values):
            # A) Determine the point on the yield surface for this (xi, theta)
            # Initial guess for lambda (radial parameter) from quadratic (gamma=2) approximation
            radicand = 1.0 - (xi ** 2) / D1
            if radicand < 0.0:
                radicand = 0.0
            lam = np.sqrt(radicand)  # initial λ (when gamma=2, this yields exact surface)

            # If gamma != 2, adjust λ so that the point satisfies f=1 (yield surface equation)
            if abs(gamma - 2.0) > 1e-8:
                # Define yield function value for a given λ along this direction
                def yield_func_val(lmbd):
                    sx = 0.5 * (xi + np.sqrt(D2) * lmbd * np.cos(theta)) * M0x
                    sy = 0.5 * (xi - np.sqrt(D2) * lmbd * np.cos(theta)) * M0y
                    txy = lmbd * np.sin(theta) * M0xy
                    # Hill yield function f(sx,sy,txy) = 1 at the surface
                    term_x = (abs(sx) / M0x) ** gamma
                    term_y = (abs(sy) / M0y) ** gamma
                    term_xy = (abs(txy) / M0xy) ** gamma
                    term_cross = rho * (abs(sx * sy) / (M0x * M0y)) ** (gamma / 2.0)
                    return term_x + term_y + term_xy - term_cross

                # Use bisection to solve yield_func_val(lam) = 1
                lo, hi = 0.0, max(1.0, 1.5 * lam)
                if yield_func_val(hi) < 1.0:
                    # Increase hi until the surface is crossed
                    while yield_func_val(hi) < 1.0:
                        hi *= 2.0
                        if hi > 1e6:
                            break
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if yield_func_val(mid) >= 1.0:
                        hi = mid
                    else:
                        lo = mid
                lam = 0.5 * (lo + hi)

            # Compute the coordinate on the yield surface with final λ
            sx = 0.5 * (xi - np.sqrt(D2) * lam * np.cos(theta)) * M0x
            sy = 0.5 * (xi + np.sqrt(D2) * lam * np.cos(theta)) * M0y
            txy = lam * np.sin(theta) * M0xy
            coords[i, j] = (sx, sy, txy)

            # B) Compute gradient of yield function f at this point (partial derivatives)
            # Gradient components for Hill's criterion:
            # df/dsx = γ * sign(sx) * (|sx|/M0x)^{γ-1} / M0x  -  (ρ * γ/2) * sign(sx*sy) * (|sx*sy|/(M0x M0y))^{γ/2 - 1} * (sy/(M0x M0y))
            # df/dsy = γ * sign(sy) * (|sy|/M0y)^{γ-1} / M0y  -  (ρ * γ/2) * sign(sx*sy) * (|sx*sy|/(M0x M0y))^{γ/2 - 1} * (sx/(M0x M0y))
            # df/dtxy = γ * sign(txy) * (|txy|/M0xy)^{γ-1} / M0xy
            val_P = sx * sy / (M0x * M0y)
            sign_P = 0 if abs(val_P) < 1e-12 else (1 if val_P > 0 else -1)
            # Partial derivatives:
            dfd_sx = gamma * ((abs(sx) / M0x) ** (gamma - 1.0)) * (1 if sx >= 0 else -1) / M0x \
                     - rho * (gamma / 2.0) * ((abs(val_P)) ** (gamma / 2.0 - 1.0) if abs(val_P) > 1e-12 else 0.0) \
                     * sign_P * (sy / (M0x * M0y))
            dfd_sy = gamma * ((abs(sy) / M0y) ** (gamma - 1.0)) * (1 if sy >= 0 else -1) / M0y \
                     - rho * (gamma / 2.0) * ((abs(val_P)) ** (gamma / 2.0 - 1.0) if abs(val_P) > 1e-12 else 0.0) \
                     * sign_P * (sx / (M0x * M0y))
            dfd_txy = gamma * ((abs(txy) / M0xy) ** (gamma - 1.0)) * (1 if txy >= 0 else -1) / M0xy
            grads[i, j] = (dfd_sx, dfd_sy, dfd_txy)
    return coords, grads
