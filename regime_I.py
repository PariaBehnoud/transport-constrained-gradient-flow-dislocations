"""
Remark 7.1 (Lambda = 40, Gamma_alpha = 160): a single dislocation.

A single Gaussian core of width eps, q(x1, 0) = q0 G_eps(x1), evolves by
Eq. (7.18) with the total energy and no applied stress. Since q >= 0, the
evolution holds on the whole line, with gamma = 0 to the left of the grid and
gamma = q0 to the right of it.

Outputs (media/Figs/):
  regime_I_profiles.png  q(x1, t) at several times, with the Peierls-Nabarro core (7.11)
  regime_I_width.png     core half-width w(t) = q0 / (pi max q) against time
                         (values quoted in Remark 7.1; not shown in the paper)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import plot_style as ps
from glide_layer import Grid, Params
from single_core import evolve_single, pad

P = Params(q0=1.0, b_slip=1.0, m=1.0, K=1.0, kappa_h=0.10, A_psi=1.0, eps=0.025)
GRID = Grid(L=4.0, N=8001)                 # dx = 1e-3 = eps/25
T_END = 1.0
SNAPSHOTS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 1.0]
PLOT_TIMES = [0.003, 0.01, 0.03, 1.0]          # curves shown in Fig. 5

HERE = Path(__file__).resolve().parent
OUT = HERE / "media" / "Figs"
CACHE = HERE / "results"


def run(grid=GRID, t_end=T_END):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"regime_I_single_N{grid.N}.npz"
    if f.exists():
        z = np.load(f)
        return {"t": z["t"], "q_max": z["q_max"],
                "profiles": {float(k[2:]): z[k] for k in z.files if k.startswith("p_")}}
    r = evolve_single(P, grid, t_end, record_times=SNAPSHOTS, log_every=100)
    out = {"t": r["hist"]["t"], "q_max": r["hist"]["q_max"], "profiles": r["profiles"]}
    np.savez(f, t=out["t"], q_max=out["q_max"],
             **{f"p_{k!r}": v for k, v in out["profiles"].items()})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ps.apply()
    r = run()
    zeta = P.K / (2.0 * np.pi * P.A_psi)
    w = P.q0 / (np.pi * r["q_max"])
    for tt in [0.0, 0.01, 0.1, 0.5, 1.0]:
        i = min(np.searchsorted(r["t"], tt), len(r["t"]) - 1)
        print(f"t = {r['t'][i]:.4f}: max q = {r['q_max'][i]:.4f}, "
              f"w = {w[i]:.4f} = {w[i] / P.eps:.2f} eps")
    print(f"zeta = K/(2 pi A_psi) = {zeta:.4f} = {zeta / P.eps:.2f} eps")

    xf = np.concatenate([GRID.x - 0.5 * GRID.dx, [GRID.x[-1] + 0.5 * GRID.dx]])
    fig, ax = plt.subplots(figsize=(8, 6))
    lines = []
    for k, t in enumerate(PLOT_TIMES):
        q = np.diff(pad(r["profiles"][t], P.q0)) / GRID.dx
        lines += ax.plot(xf, q / P.q0, color=ps.SERIES[k], lw=2.2, label=fr"$t={t:g}$")
    # Peierls-Nabarro core (7.11), q = (q0/pi) zeta/(x^2 + zeta^2), drawn on top
    xx = np.linspace(-0.5, 0.5, 2001)
    pn, = ax.plot(xx, zeta / (np.pi * (xx**2 + zeta**2)), "--", color=ps.INK, lw=3.2,
                  dashes=(4, 2), zorder=5, label="Peierls–Nabarro")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.1, 5.6)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$q(x_1,t)/q_0$")
    # times in the empty upper-left corner, the Peierls-Nabarro curve upper right
    ax.add_artist(ax.legend(handles=lines, loc="upper left", handlelength=1.2,
                            borderaxespad=0.3))
    ax.legend(handles=[pn], loc="upper right", handlelength=1.6, borderaxespad=0.3)
    ps.save(fig, OUT / "regime_I_profiles.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sel = r["t"] > 0
    ax.semilogx(r["t"][sel], w[sel] / P.eps, color=ps.INK, label="simulation")
    ax.axhline(zeta / P.eps, color=ps.SERIES[0], ls="--", label="Peierls–Nabarro " + r"$\zeta$")
    ax.set_xlim(1e-4, T_END)
    ax.set_ylim(0, 8.5)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$w(t)/\varepsilon$")
    ax.legend(loc="lower right")
    ps.save(fig, OUT / "regime_I_width.png")


if __name__ == "__main__":
    main()
