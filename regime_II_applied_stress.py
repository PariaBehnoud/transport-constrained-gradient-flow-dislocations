"""
Regime II of Section 7: lattice trapping and stress-driven annihilation or
unbinding of an opposite-sign dislocation pair.

Dimensionless reduced model of Section 7.B with Gamma_alpha = 0:
    D = d/eps,   Lambda = K/(A_psi eps),   T = tau q0/A_psi
    Ehat(D;T) = 2 Lambda int_0^{D/2} daw(s) ds + L(D) - T D      Eq. (7.29)
    F_II(D)   = Lambda daw(D/2) + L'(D)                           Eq. (7.41)
where daw is the Dawson function and L is the lattice function (7.27).
Stationary separations satisfy F_II(D) = T; the folds T_- and T_+ are the
local minimum and the following local maximum of F_II.

Produces Fig. 3: energy landscape (a), force balance (b), and the fold
diagram in the (Lambda, T) plane (c).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.special import dawsn
from scipy.stats import norm


# Plot settings

LABEL_FS = 20
TICK_FS = 17
LEGEND_FS = 15
ANNOT_FS = 15

LINE_W = 3.0
AUX_LINE_W = 2.2

SAVE_FIGURES = True
SHOW_FIGURES = True

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "media" / "Figs"
output_dir.mkdir(parents=True, exist_ok=True)


# Dimensionless parameters (Eq. (7.39)): K = 0.00625, A_psi = 1, eps = 0.025

Lambda = 0.25


# Lattice-energy shape function L(D) and its derivative, Eq. (7.27)

u = np.linspace(-30.0, 30.0, 60001)
du = u[1] - u[0]


def _L_and_Lp(D):

    # Dimensionless slip profile gamma/q0
    g = norm.cdf(u + D / 2) - norm.cdf(u - D / 2)

    L_value = np.trapezoid(np.sin(np.pi * g) ** 2, dx=du)

    Lp_value = np.trapezoid(
        np.pi
        * np.sin(2 * np.pi * g)
        * 0.5
        * (norm.pdf(u + D / 2) + norm.pdf(u - D / 2)),
        dx=du,
    )

    return L_value, Lp_value


# Tabulate L and L' and interpolate with cubic splines

D_tab = np.linspace(0.02, 12.0, 600)
L_tab, Lp_tab = np.array([_L_and_Lp(D) for D in D_tab]).T

L = CubicSpline(D_tab, L_tab)
Lp = CubicSpline(D_tab, Lp_tab)


# Dimensionless elastic energy and force, Eqs. (7.8)-(7.9)

def elastic_energy_hat(D, Lambda_value=Lambda):

    D_arr = np.atleast_1d(np.asarray(D, dtype=float))
    values = np.array(
        [2.0 * Lambda_value * quad(dawsn, 0.0, x / 2.0)[0] for x in D_arr]
    )
    return values[0] if np.ndim(D) == 0 else values


def elastic_force_hat(D, Lambda_value=Lambda):

    return Lambda_value * dawsn(np.asarray(D, dtype=float) / 2.0)


# Dimensionless energy and internal force

def Ehat(D, T):

    return elastic_energy_hat(D) + L(D) - T * np.asarray(D, dtype=float)


def F_II(D, Lambda_value=Lambda):

    return elastic_force_hat(D, Lambda_value) + Lp(D)


def Fhat(D, T=0.0):

    return F_II(D) - T


# Fold stresses: local minimum of F_II and the following local maximum,
# located on a grid and then refined.

def folds(Lambda_value):

    def f(D):
        return F_II(D, Lambda_value)

    D_grid = np.linspace(1.5, 11.0, 4000)
    F_grid = f(D_grid)
    i_lo = next(
        i for i in range(1, len(D_grid) - 1)
        if F_grid[i] < F_grid[i - 1] and F_grid[i] < F_grid[i + 1]
    )
    i_hi = next(
        i for i in range(i_lo + 1, len(D_grid) - 1)
        if F_grid[i] > F_grid[i - 1] and F_grid[i] > F_grid[i + 1]
    )

    lo = minimize_scalar(
        f,
        bounds=(D_grid[i_lo - 2], D_grid[i_lo + 2]),
        method="bounded",
        options={"xatol": 1e-10},
    )
    hi = minimize_scalar(
        lambda D: -f(D),
        bounds=(D_grid[i_hi - 2], D_grid[i_hi + 2]),
        method="bounded",
        options={"xatol": 1e-10},
    )

    return lo.fun, -hi.fun, lo.x, hi.x


# Folds and zero-stress stationary separations for the chosen Lambda

T_minus, T_plus, D_minus, D_plus = folds(Lambda)

D_scan = np.linspace(0.05, 12.0, 6000)
F_scan = F_II(D_scan)
D_stationary = [
    brentq(F_II, D_scan[i], D_scan[i + 1])
    for i in range(len(D_scan) - 1)
    if F_scan[i] * F_scan[i + 1] < 0.0
]

# Lambda_c: zero-stress crossing of the lower fold, Eq. (7.49)
Lambda_c = brentq(lambda k: folds(k)[0], 0.3, 0.6, xtol=1e-10)

print(f"Lambda = {Lambda}")
print("zero-stress stationary separations D:",
      ", ".join(f"{D:.4f}" for D in D_stationary))
print(
    f"decreasing-separation fold  T_- = {T_minus:+.5f} "
    f"at D = {D_minus:.4f}"
)
print(
    f"unbinding fold              T_+ = {T_plus:+.5f} "
    f"at D = {D_plus:.4f}"
)
print(f"zero-stress crossing of the lower fold: Lambda_c = {Lambda_c:.4f}")


def finish_figure(fig, filename):

    fig.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(output_dir / filename, dpi=600, bbox_inches="tight")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# Plotting data

levels = [
    (T_minus, "tab:blue"),
    (-0.03, "tab:cyan"),
    (0.0, "k"),
    (0.03, "tab:orange"),
    (T_plus, "tab:red"),
]

Dp = np.linspace(0.02, 9.0, 900)


# Figure 3(a): tilted reduced-energy landscapes

fig_a, ax_a = plt.subplots(figsize=(7.0, 5.2))

# Reference separation for the energy shift
D_ref = 3.4

for T, c in levels:

    # Shift each curve to zero at the reference separation.
    Ehat_rel = Ehat(Dp, T) - Ehat(D_ref, T)

    ax_a.plot(Dp, Ehat_rel, color=c, lw=LINE_W, label=rf"$T={T:.4f}$")

ax_a.set_xlim(0, 9)
ax_a.set_ylim(-0.35, 0.15)
ax_a.set_xlabel(r"$D=d/\varepsilon$", fontsize=LABEL_FS)
ax_a.set_ylabel(r"$\widehat{\mathcal{E}}(D;T)$", fontsize=LABEL_FS)
ax_a.tick_params(axis="both", which="major", labelsize=TICK_FS)
ax_a.legend(fontsize=LEGEND_FS, loc="lower right")
finish_figure(fig_a, "energ_landscape.png")


# Figure 3(b): force balance

fig_b, ax_b = plt.subplots(figsize=(7.0, 5.2))

# Zero-stress internal force
ax_b.plot(Dp, Fhat(Dp), color="k", lw=LINE_W)

# Applied-stress levels
for T, c in levels:
    ax_b.axhline(T, color=c, ls="--", lw=AUX_LINE_W)

ax_b.axhline(0.0, color="0.8", lw=1.2)

ax_b.set_xlim(0, 9)
ax_b.set_ylim(-0.12, 0.30)
ax_b.set_xlabel(r"$D=d/\varepsilon$", fontsize=LABEL_FS)
ax_b.set_ylabel(r"$\mathcal{F}_{\rm II}(D;\Lambda)$", fontsize=LABEL_FS)
ax_b.tick_params(axis="both", which="major", labelsize=TICK_FS)

ax_b.annotate(
    rf"$T_+={T_plus:.4f}$",
    xy=(6.3, T_plus + 0.008),
    color="tab:red",
    fontsize=ANNOT_FS,
)
ax_b.annotate(
    rf"$T_-={T_minus:.4f}$",
    xy=(6.3, T_minus + 0.007),
    color="tab:blue",
    fontsize=ANNOT_FS,
)
finish_figure(fig_b, "force_balance.png")


# Figure 3(c): fold diagram. The trapping window closes near Lambda = 1.145.

khs = np.linspace(0.05, 1.14, 80)

Tm, Tp = np.array([folds(k)[:2] for k in khs]).T

fig_c, ax_c = plt.subplots(figsize=(7.0, 5.2))

ax_c.fill_between(
    khs, Tm, Tp, color="tab:green", alpha=0.25, label="trapping region"
)
ax_c.plot(khs, Tm, color="tab:blue", lw=LINE_W,
          label=r"$T_-$: decreasing separation")
ax_c.plot(khs, Tp, color="tab:red", lw=LINE_W, label=r"$T_+$: unbinding")
ax_c.axhline(0.0, color="k", lw=1.2)

ax_c.set_xlabel(r"$\Lambda=K/(A_\psi\varepsilon)$", fontsize=LABEL_FS)
ax_c.set_ylabel(r"$T=\tau q_0/A_\psi$", fontsize=LABEL_FS)
ax_c.tick_params(axis="both", which="major", labelsize=TICK_FS)
ax_c.legend(fontsize=LEGEND_FS, loc="upper left")
finish_figure(fig_c, "stability_window.png")
