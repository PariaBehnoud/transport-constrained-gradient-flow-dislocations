"""
Opposite-sign dislocation pair of Section 7.C (Lambda = 0.25, Gamma_alpha = 0.0032).

Evolution of Eq. (7.18) from the Gaussian pair, Eq. (7.4), with a small core
energy and a uniform applied stress tau (T = tau q0 / A_psi).

Outputs (media/Figs/):
  Fig. 3 (T = 0)
  regime_II_profiles.png        q(x1, t) at several times
  regime_II_energies.png        elastic, core, lattice, and total energy against time
  regime_II_core.png            core profile against the Peierls-Nabarro core (7.11)
  Fig. 4 (applied stress)
  regime_II_midpoint_slip.png   midpoint slip gamma(0, t) for several applied stresses
  regime_II_separation.png      centroid separation s(t) for several applied stresses,
                                up to annihilation (x: gamma(0,t) = q0/2)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from glide_layer import Grid, Params, evolve, face_density

COMMON = dict(q0=1.0, b_slip=1.0, m=1.0, K=0.00625, kappa_h=2.0e-6, A_psi=1.0,
              eps=0.025, d0=0.20)
GRID = Grid(L=1.0, N=16001)          # dx = 1.25e-4, about zeta/8
T_END = 4.0
STRESSES = [-0.05, 0.0, 0.02, 0.04]  # T = tau q0 / A_psi (A_psi = q0 = 1)
COLORS = {-0.05: ps.STRESS_NEG, 0.0: ps.STRESS_ZERO,
          0.02: ps.STRESS_POS_LIGHT, 0.04: ps.STRESS_POS}
SNAPSHOTS = [0.0, 0.02, 0.05, 0.5, 0.75, 1.0, 1.25, 1.5, 1.55, 1.6, 1.7]
T_CORE = 0.75
Q_SNAPSHOTS = [0.05, 0.5, 1.0, 1.5, 1.7]

OUT = Path(__file__).resolve().parent / "media" / "Figs"


CACHE = Path(__file__).resolve().parent / "results"


def cached_run(p, T, rec):
    """Run Eq. (7.18), or load the result saved by an earlier run."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"regime_II_T{T:+.4f}_N{GRID.N}.npz"
    if f.exists():
        z = np.load(f)
        hist = {k[2:]: z[k] for k in z.files if k.startswith("h_")}
        prof = {float(k[2:]): z[k] for k in z.files if k.startswith("p_")}
        # reuse a saved run only if it has the energy parts, reaches T_END,
        # and stores every requested profile
        if ("E_el" in hist and hist["t"][-1] >= T_END - 1e-9
                and all(t in prof for t in rec)):
            return {"hist": hist, "profiles": prof}
    r = evolve(p, GRID, T_END, record_times=rec, log_every=50, with_energy=True)
    arrays = {f"h_{k}": v for k, v in r["hist"].items()}
    arrays.update({f"p_{t!r}": g for t, g in r["profiles"].items()})
    np.savez(f, **arrays)
    return r


def plot_energies(h, out=None):
    """Elastic, core, lattice, and total energy along the evolution at T = 0 (Fig. 3b)."""
    out = OUT / "regime_II_energies.png" if out is None else out
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(h["t"], h["E"], color=ps.INK, label=r"$\mathcal{E}_{\rm tot}$")
    ax.plot(h["t"], h["E_el"], "--", color=ps.SERIES[0], label=r"$\mathcal{E}_{\rm el}$")
    ax.plot(h["t"], h["E_lat"], "-.", color=ps.SERIES[1], label=r"$\mathcal{E}_\psi$")
    ax.plot(h["t"], h["E_core"], ":", color=ps.SERIES[2], label=r"$\mathcal{E}_\alpha$")
    ax.set_xlim(-0.03, 3.0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("energy")
    ax.legend(loc="upper right")
    ps.save(fig, out)


def zeta(p):
    return p.K / (2.0 * np.pi * p.A_psi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ps.apply()
    runs = {}
    for T in STRESSES:
        p = Params(tau=T * COMMON["A_psi"] / COMMON["q0"], **COMMON)
        rec = SNAPSHOTS + [T_CORE] if T == 0.0 else []
        runs[T] = (p, cached_run(p, T, rec))
        h = runs[T][1]["hist"]
        half = np.flatnonzero(h["gamma_mid"] < 0.5 * p.q0)
        t_half = h["t"][half[0]] if half.size else np.nan
        print(f"T = {T:+.3f}: gamma(0,t) = q0/2 at t = {t_half:.3f}; "
              f"at t = {h['t'][-1]:.2f}: gamma(0,t)/q0 = {h['gamma_mid'][-1]:.3f}, "
              f"separation = {h['separation'][-1]:.4f}; largest energy increase "
              f"between logs / energy drop = {max(np.diff(h['E']).max(), 0.0) / (h['E'][0] - h['E'].min()):.1e}")

    p0, r0 = runs[0.0]
    z = zeta(p0)
    print(f"Peierls-Nabarro half-width zeta = K/(2 pi A_psi) = {z:.3e} "
          f"(zeta/eps = {z / p0.eps:.4f})")

    xf = np.concatenate([GRID.x - 0.5 * GRID.dx, [GRID.x[-1] + 0.5 * GRID.dx]])

    # Fig. 3a: signed density q(x1, t) at several times, T = 0
    fig, ax = plt.subplots(figsize=(8, 6))
    for k, t in enumerate(Q_SNAPSHOTS):
        ax.plot(xf, face_density(r0["profiles"][t], GRID.dx) / p0.q0,
                color=ps.SERIES[k], lw=1.8, label=fr"$t={t:g}$")
    ax.set_xlim(-0.13, 0.13)
    ax.set_ylim(-380, 380)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$q(x_1,t)/q_0$")
    ax.legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=1.0)
    ps.save(fig, OUT / "regime_II_profiles.png")

    # Fig. 3c: core profile against the Peierls-Nabarro core
    g = r0["profiles"][T_CORE]
    qf = face_density(g, GRID.dx)
    left = xf < 0
    ic = np.argmax(np.where(left, qf, -np.inf))
    u = (xf - xf[ic]) / z
    sel = np.abs(u) <= 8
    uu = np.linspace(-8, 8, 400)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(uu, 1.0 / (1.0 + uu**2), color=ps.INK, label="analytical")
    ax.plot(u[sel], qf[sel] * np.pi * z / p0.q0, "o", color=ps.SERIES[0], ms=6,
            label="simulation")
    ax.set_xlim(-8, 8)
    ax.set_xlabel(r"$(x_1-x_{\rm c})/\zeta$")
    ax.set_ylabel(r"$\pi\zeta\, q/q_0$")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    ps.save(fig, OUT / "regime_II_core.png")
    print(f"core center at t = {T_CORE}: x_c = {xf[ic]:.5f}")

    # Fig. 4a: midpoint slip for several applied stresses
    fig, ax = plt.subplots(figsize=(8, 6))
    for T in STRESSES:
        p, r = runs[T]
        h = r["hist"]
        ax.plot(h["t"], h["gamma_mid"] / p.q0, color=COLORS[T], label=fr"$T={T:g}$")
    ax.set_xlim(-0.03, T_END)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\gamma(0,t)/q_0$")
    ax.legend(loc="center", bbox_to_anchor=(0.585, 0.5), fontsize=16, handlelength=1.4)
    ps.save(fig, OUT / "regime_II_midpoint_slip.png")

    # Fig. 3b: energies along the evolution at T = 0
    plot_energies(r0["hist"])
    h0 = r0["hist"]
    print("energies at T = 0 (elastic, core, lattice, total):")
    for tt in (0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 1.5, 1.55, 1.58, 1.6, 1.62, 1.7, 2.0, 3.0):
        k = np.argmin(np.abs(h0["t"] - tt))
        print(f"  t = {h0['t'][k]:.4f}: {h0['E_el'][k]:.6f} {h0['E_core'][k]:.6f} "
              f"{h0['E_lat'][k]:.6f} {h0['E'][k]:.6f}; gamma(0,t)/q0 = {h0['gamma_mid'][k]:.4f}")

    # Fig. 4b: separation for several applied stresses
    fig, ax = plt.subplots(figsize=(8, 6))
    for T in STRESSES:
        p, r = runs[T]
        h = r["hist"]
        alive = h["gamma_mid"] >= 0.5 * p.q0
        ax.plot(h["t"][alive], h["separation"][alive] / p.d0, color=COLORS[T],
                label=fr"$T={T:g}$")
        if not alive.all():
            i = np.flatnonzero(alive)[-1]
            ax.plot(h["t"][i], h["separation"][i] / p.d0, "x", color=COLORS[T],
                    ms=12, mew=2.5)
    ax.set_xlim(-0.03, T_END)
    smax = max(np.nanmax(r["hist"]["separation"][r["hist"]["gamma_mid"] >= 0.5 * pp.q0]) / pp.d0
               for pp, r in runs.values())
    ax.set_ylim(-0.02, max(1.5, 1.05 * smax))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$s(t)/d_0$")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    ps.save(fig, OUT / "regime_II_separation.png")


if __name__ == "__main__":
    main()
