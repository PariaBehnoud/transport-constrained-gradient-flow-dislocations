"""
Values reported for Regime II in Section 7.D, in dimensional form.

Reduced internal force (Gamma_alpha = 0), divided by A_psi:
    F_II(D) = (K/(A_psi eps)) daw(D/2) + L'(D),     D = d/eps,
where daw is the Dawson function and L is the lattice function (7.27).

Prints the zero-stress stationary separations (7.43)-(7.44), the folds
T_- and T_+ with their separations (7.45)-(7.46), Lambda_c (7.49), and
|tau|/tau_lat = |T|/pi.
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import dawsn
from scipy.stats import norm

A = 1.0        # A_psi
K = 0.00625    # K = mu q0^2/(2 pi (1 - nu)); Lambda = K/(A eps) = 0.25
eps = 0.025    # Gaussian core width

K_over_eps = K / eps

u = np.linspace(-30.0, 30.0, 120001)
du = u[1] - u[0]


def g(D):
    """Dimensionless slip gamma/q0 for q0 = b_slip."""
    return norm.cdf(u + D / 2.0) - norm.cdf(u - D / 2.0)


def Lp(D):
    gamma = g(D)
    integrand = (
        np.pi
        * np.sin(2.0 * np.pi * gamma)
        * 0.5
        * (norm.pdf(u + D / 2.0) + norm.pdf(u - D / 2.0))
    )
    return np.trapezoid(integrand, dx=du)


def F_II(D, K_over_eps_value=K_over_eps):
    """Internal force divided by A_psi."""
    return K_over_eps_value * dawsn(D / 2.0) / A + Lp(D)


def folds(K_over_eps_value=K_over_eps):
    """Lower fold (local minimum of F_II) and upper fold (next local maximum)."""
    f = lambda D: F_II(D, K_over_eps_value)
    D_grid = np.linspace(1.5, 11.0, 800)
    F_grid = np.array([f(D) for D in D_grid])
    i_lo = next(i for i in range(1, len(D_grid) - 1)
                if F_grid[i] < F_grid[i - 1] and F_grid[i] < F_grid[i + 1])
    i_hi = next(i for i in range(i_lo + 1, len(D_grid) - 1)
                if F_grid[i] > F_grid[i - 1] and F_grid[i] > F_grid[i + 1])
    lo = minimize_scalar(f, bounds=(D_grid[i_lo - 2], D_grid[i_lo + 2]),
                         method="bounded", options={"xatol": 1e-10})
    hi = minimize_scalar(lambda D: -f(D), bounds=(D_grid[i_hi - 2], D_grid[i_hi + 2]),
                         method="bounded", options={"xatol": 1e-10})
    return lo.fun, -hi.fun, lo.x, hi.x


if __name__ == "__main__":
    print(f"Lambda = K/(A_psi eps) = {K_over_eps / A:.4f}")

    D_scan = np.linspace(0.5, 12.0, 1200)
    F_scan = np.array([F_II(D) for D in D_scan])
    roots = [brentq(F_II, D_scan[i], D_scan[i + 1])
             for i in range(len(D_scan) - 1) if F_scan[i] * F_scan[i + 1] < 0.0]
    for name, D in zip(["D_b ", "D_eq"], roots):
        print(f"{name} = {D:.4f},  d = {D * eps:.5f}")

    T_minus, T_plus, D_minus, D_plus = folds()
    print("T_- = %.5f at d = %.5f" % (T_minus, D_minus * eps))
    print("T_+ = %.5f at d = %.5f" % (T_plus, D_plus * eps))
    print("|tau|/tau_lat = %.4f and %.4f" % (abs(T_minus) / np.pi, abs(T_plus) / np.pi))

    Lambda_c = brentq(lambda k: folds(k * A)[0], 0.3, 0.6, xtol=1e-10)
    print(f"Lambda_c = {Lambda_c:.4f}")
