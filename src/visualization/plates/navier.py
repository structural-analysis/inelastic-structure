import numpy as np

def calculate_w_max(a, b, q, h, E, nu, m_max, n_max):
    """
    Calculate the maximum deflection at the center of a simply supported plate.
    
    Parameters:
        a, b   : Plate dimensions (length and width).
        q      : Uniformly distributed load (Pa or N/m²).
        h      : Plate thickness (m).
        E      : Young's modulus (Pa).
        nu     : Poisson's ratio.
        m_max  : Maximum number of terms in the m-series (odd integers).
        n_max  : Maximum number of terms in the n-series (odd integers).
    
    Returns:
        w_max  : Maximum deflection at the center (m).
    """
    D = (E * h**3) / (12 * (1 - nu**2))  # Flexural rigidity
    w_max = 0.0
    
    for m in range(1, m_max + 1, 2):      # Loop over odd m (1, 3, 5...)
        for n in range(1, n_max + 1, 2):  # Loop over odd n (1, 3, 5...)
            term = (16 * q) / (np.pi**6 * D)
            term *= ((-1)**((m + n)/2 - 1)) / (m * n * ((m**2 / a**2) + (n**2 / b**2))**2)
            w_max += term
    
    return w_max

# Example usage
if __name__ == "__main__":
    # Input parameters (example for a square plate)
    a = 6.0      # Length (m)
    b = 4.0      # Width (m)
    q = 100000     # Uniform load (N/m²)
    h = 0.05     # Thickness (m)
    E = 2.0e11   # Young's modulus (Pa, steel)
    nu = 0.3     # Poisson's ratio
    m_max = 100   # Number of terms in m-series (odd)
    n_max = 100   # Number of terms in n-series (odd)
    
    w_max = calculate_w_max(a, b, q, h, E, nu, m_max, n_max)
    print(f"Maximum deflection (w_max): {w_max:.6e} m")