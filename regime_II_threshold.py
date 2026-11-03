"""
Section 7.C: applied stress that separates the pair instead of annihilating it.

For a given T = tau q0 / A_psi, the evolution of Eq. (7.18) ends either in
annihilation (gamma(0,t) < q0/2) or in unbinding (the centroid separation
exceeds 1.5 d0). Bisection on T gives the threshold T_c, which is compared with
the force balance of two point dislocations at the initial separation,
T = Lambda/D0, i.e., tau q0 = K/d0.
"""

import json
from pathlib import Path

import numpy as np

from glide_layer import Grid, Params, evolve

CACHE = Path(__file__).resolve().parent / "results" / "regime_II_threshold.json"

from regime_II import COMMON  # same parameters as Figs. 3 and 4
T_MAX = 40.0


def outcome(T, N):
    """Annihilation or unbinding for applied stress T; results are cached."""
    CACHE.parent.mkdir(exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = f"{T:.6f}_{N}"
    if key in cache:
        return tuple(cache[key])
    res = _outcome(T, N)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    cache[key] = list(res)
    CACHE.write_text(json.dumps(cache, indent=1))
    return res


def _outcome(T, N):
    p = Params(tau=T, **COMMON)

    def stop(t, d):
        if d["gamma_mid"] < 0.5 * p.q0:
            return "annihilation"
        if d["separation"] > 1.5 * p.d0:
            return "unbinding"
        return None

    r = evolve(p, Grid(L=1.0, N=N), T_MAX, stop=stop, log_every=100)
    if r["outcome"] is None:
        s = r["hist"]["separation"]
        r["outcome"] = "unbinding" if s[-1] > s[-20] else "annihilation"
        r["outcome"] += " (by trend)"
    return r["outcome"], r["t"]


def threshold(N, lo=0.026, hi=0.038, tol=5e-4):
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        res, t = outcome(mid, N)
        print(f"  N = {N}: T = {mid:.5f} -> {res} at t = {t:.2f}", flush=True)
        if res.startswith("annihilation"):
            lo = mid
        else:
            hi = mid
    return lo, hi


if __name__ == "__main__":
    Lam = COMMON["K"] / (COMMON["A_psi"] * COMMON["eps"])
    D0 = COMMON["d0"] / COMMON["eps"]
    print(f"point-dislocation estimate Lambda/D0 = {Lam / D0:.5f}")
    lo, hi = threshold(8001)
    print(f"N = 8001: {lo:.5f} < T_c < {hi:.5f}", flush=True)
    # grid check on the finer grid, at stresses about 4% below and above T_c
    for T in (0.0300, 0.0325):
        res, t = outcome(T, 16001)
        print(f"  N = 16001: T = {T:.5f} -> {res} at t = {t:.2f}", flush=True)
