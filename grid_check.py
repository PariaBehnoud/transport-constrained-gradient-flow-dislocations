"""
Grid check for Section 7.C: annihilation time at T = 0, with the parameters of
regime_II.py, on three grids: the grid of the figures (N = 16001, dx = 1.25e-4),
a grid with twice the spacing (N = 8001), and a grid with half the spacing
(N = 32001).
"""
import time

import numpy as np

from glide_layer import Grid, Params, evolve
import regime_II as R

GRIDS = (8001, 16001, 32001)


def t_half(N):
    p = Params(tau=0.0, **R.COMMON)

    def stop(t, d):
        return "annihilation" if d["gamma_mid"] < 0.5 * p.q0 else None

    r = evolve(p, Grid(L=1.0, N=N), 3.0, stop=stop, log_every=10)
    h = r["hist"]
    g, t = h["gamma_mid"], h["t"]
    i = np.flatnonzero(g < 0.5 * p.q0)[0]
    return t[i - 1] + (0.5 * p.q0 - g[i - 1]) * (t[i] - t[i - 1]) / (g[i] - g[i - 1])


if __name__ == "__main__":
    t0 = time.time()
    res = {}
    for N in GRIDS:
        res[N] = t_half(N)
        print(f"N = {N} (dx = {2.0 / (N - 1):.3e}): gamma(0,t) = q0/2 at t = {res[N]:.4f}  "
              f"[{time.time() - t0:.0f} s]", flush=True)
    a, b, c = (res[N] for N in GRIDS)
    print(f"change from N = {GRIDS[0]} to {GRIDS[1]}: {abs(a - b) / b * 100:.2f}%")
    print(f"change from N = {GRIDS[1]} to {GRIDS[2]}: {abs(b - c) / c * 100:.2f}%")
    if (a - b) * (b - c) > 0:
        order = np.log2(abs(a - b) / abs(b - c))
        limit = c - (b - c) / (2.0**order - 1.0)
        print(f"observed order of convergence: {order:.2f}; extrapolated time: {limit:.4f} "
              f"(N = {GRIDS[1]} differs from it by {abs(b - limit) / limit * 100:.2f}%)")
    print(f"wall time {time.time() - t0:.0f} s")
