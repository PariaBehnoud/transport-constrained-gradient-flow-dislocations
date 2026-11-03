"""
Checks of the glide-layer solver (glide_layer.py).

1. Elastic operator: the discrete principal-value integral for a Gaussian
   density against the exact result (sqrt(2)/eps) daw(x/(sqrt(2) eps)).
2. Peierls-Nabarro core, Eq. (7.11): the residual of dE/dgamma = 0
   for an isolated dislocation without core energy or applied stress.
3. Energy decay, Eq. (7.19), and mirror symmetry q(-x1) = -q(x1).
4. Grid convergence of the width of the single core and of the annihilation time of the pair.
"""

import numpy as np
from scipy.special import dawsn, erf

from glide_layer import (ElasticOperator, Grid, Params, evolve, initial_slip,
                         slip_density)


def check_elastic():
    p = Params(K=1.0, q0=1.0, eps=0.025)
    out = []
    for N in (2001, 4001, 8001):
        grid = Grid(L=1.0, N=N)
        x, dx = grid.x, grid.dx
        edges = x[0] + (np.arange(N + 2) - 1) * dx
        cdf = 0.5 * (1 + erf(edges / (np.sqrt(2) * p.eps)))
        qf = np.diff(cdf) / dx                       # cell averages on the faces
        exact = (np.sqrt(2) / p.eps) * dawsn(x / (np.sqrt(2) * p.eps))
        num = ElasticOperator(grid, p).from_faces(qf)
        out.append((dx, np.abs(num - exact).max() / np.abs(exact).max()))
    print("1. elastic operator, Gaussian density: max relative error: "
          + ", ".join(f"dx = {dx:.1e}: {e:.1e}" for dx, e in out))
    return out


def check_pn_core():
    K, A = 0.00625, 1.0
    zeta = K / (2 * np.pi * A)
    p = Params(K=K, A_psi=A, kappa_h=0.0, q0=1.0, b_slip=1.0)
    out = []
    for N in (8001, 16001, 32001):
        grid = Grid(L=1.0, N=N)
        x, dx = grid.x, grid.dx
        pn = lambda s: 0.5 + np.arctan(s / zeta) / np.pi
        gamma = pn(x)
        qf = np.diff(np.concatenate([[pn(x[0] - dx)], gamma, [pn(x[-1] + dx)]])) / dx
        mu = ElasticOperator(grid, p).from_faces(qf) + np.pi * A * np.sin(2 * np.pi * gamma)
        core = np.abs(x) < 10 * zeta
        out.append((dx, np.abs(mu[core]).max() / (np.pi * A)))
    print(f"2. Peierls-Nabarro core (zeta = {zeta:.3e}): max |dE/dgamma| / (pi A_psi) "
          "within 10 zeta: " + ", ".join(f"dx = {dx:.1e}: {r:.1e}" for dx, r in out))
    return out


def check_decay_and_symmetry():
    p = Params(K=1.0, kappa_h=0.10, A_psi=1.0, eps=0.025, d0=0.2)
    grid = Grid(L=2.0, N=2001)
    out = evolve(p, grid, 0.01, log_every=10, with_energy=True)
    E = out["hist"]["E"]
    rise = np.max(np.diff(E))
    g = out["gamma"]
    asym = np.abs(g - g[::-1]).max()
    print(f"3. Single core, t <= 0.01: largest energy change between logs (negative: decreasing) = {rise:.2e} "
          f"(E from {E[0]:.4f} to {E[-1]:.4f}); max |gamma(x) - gamma(-x)| = {asym:.2e}")
    p2 = Params(K=0.00625, kappa_h=0.0, A_psi=1.0, eps=0.025, d0=0.2)
    out2 = evolve(p2, Grid(L=1.0, N=8001), 0.1, log_every=10, with_energy=True)
    E2, g2 = out2["hist"]["E"], out2["gamma"]
    print(f"   Pair, t <= 0.1: largest energy change between logs = {np.max(np.diff(E2)):.2e}; "
          f"max |gamma(x) - gamma(-x)| = {np.abs(g2 - g2[::-1]).max():.2e}")
    return rise, asym


def check_convergence():
    print("4. grid convergence:")
    from single_core import evolve_single
    p = Params(K=1.0, kappa_h=0.10, A_psi=1.0, eps=0.025)
    for N in (2001, 4001, 8001):
        grid = Grid(L=4.0, N=N)
        out = evolve_single(p, grid, 0.1, log_every=1000)
        w = p.q0 / (np.pi * out["hist"]["q_max"][-1])
        print(f"   Single core, dx = {grid.dx:.1e}: w(0.1)/eps = {w / p.eps:.4f}",
              flush=True)

    p = Params(K=0.00625, kappa_h=0.0, A_psi=1.0, eps=0.025, d0=0.2)
    stop = lambda t, d: t if d["gamma_mid"] < 0.5 * p.q0 else None
    vals = []
    for N in (4001, 8001, 16001):
        out = evolve(p, Grid(L=1.0, N=N), 3.0, stop=stop, log_every=5)
        vals.append((2.0 / (N - 1), out["outcome"]))
        print(f"   ... dx = {2.0 / (N - 1):.1e} done", flush=True)
    print("   Pair, time at which gamma(0, t) = q0/2: "
          + ", ".join(f"dx = {dx:.1e}: {v:.4f}" for dx, v in vals))


if __name__ == "__main__":
    check_elastic()
    check_pn_core()
    check_decay_and_symmetry()
    check_convergence()
