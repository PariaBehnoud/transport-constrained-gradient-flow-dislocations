import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

A = 1.0
K = 0.01
eps = 0.025
a = 0.035

a_over_eps = a / eps
K_over_eps = K / eps

u = np.linspace(-30.0, 30.0, 120001)
du = u[1] - u[0]


def g(D):
    return norm.cdf(u + D / 2.0) - norm.cdf(u - D / 2.0)


def Gp(D):
    gamma = g(D)
    integrand = (
        np.pi
        * np.sin(2.0 * np.pi * gamma)
        * 0.5
        * (norm.pdf(u + D / 2.0) + norm.pdf(u - D / 2.0))
    )
    return np.trapezoid(integrand, dx=du)


def Gf(D):
    return np.trapezoid(np.sin(np.pi * g(D)) ** 2, dx=du)


def Fred(D):
    return K_over_eps * D / (D**2 + a_over_eps**2) + A * Gp(D)


def Ered(D):
    elastic = 0.5 * K * np.log((D**2 + a_over_eps**2) / a_over_eps**2)
    lattice = A * eps * Gf(D)
    return elastic + lattice


if __name__ == "__main__":
    lower_fold = minimize_scalar(
        Fred,
        bounds=(2.4, 3.4),
        method="bounded",
    )

    upper_fold = minimize_scalar(
        lambda D: -Fred(D),
        bounds=(3.6, 8.0),
        method="bounded",
    )

    print("T_- = %.5f at d = %.5f" % (lower_fold.fun / A, lower_fold.x * eps))
    print("T_+ = %.5f at d = %.5f" % (-upper_fold.fun / A, upper_fold.x * eps))
