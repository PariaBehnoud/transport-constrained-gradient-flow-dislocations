import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


# Applied-stress extension of the reduced opposite-pair model
#
# D = d/eps
# T = tau*q0/A_psi
# Lambda = K/(A_psi*eps)
# r = a/eps
#
# Ehat(D,T) = (Lambda/2)*log((D**2+r**2)/r**2) + G(D) - T*D
# Stationary points satisfy F_red/A_psi = T.


# Plot settings

LABEL_FS = 20
TICK_FS = 17
LEGEND_FS = 15
ANNOT_FS = 15

LINE_W = 3.0
AUX_LINE_W = 2.2


# Dimensionless parameters

r = 1.4
Lambda = 0.4


# Lattice-energy shape functions

u = np.linspace(-30.0, 30.0, 60001)

du = u[1] - u[0]


def _G_and_Gp(D):

    # Dimensionless slip profile
    g = (
        norm.cdf(u + D / 2)
        - norm.cdf(u - D / 2)
    )

    G = np.trapezoid(
        np.sin(np.pi * g) ** 2,
        dx=du
    )

    Gp = np.trapezoid(
        np.pi
        * np.sin(2 * np.pi * g)
        * 0.5
        * (
            norm.pdf(u + D / 2)
            + norm.pdf(u - D / 2)
        ),
        dx=du
    )

    return G, Gp


# Tabulate G and G'

D_tab = np.linspace(
    0.02,
    12.0,
    600
)


G_tab, Gp_tab = np.array(
    [_G_and_Gp(D) for D in D_tab]
).T


# Cubic-spline interpolation
G = CubicSpline(
    D_tab,
    G_tab
)

Gp = CubicSpline(
    D_tab,
    Gp_tab
)


# Dimensionless energy

def Ehat(D, T):

    elastic_part = (
        0.5
        * Lambda
        * np.log(
            (D**2 + r**2) / r**2
        )
    )

    lattice_part = G(D)

    stress_part = -T * D

    return (
        elastic_part
        + lattice_part
        + stress_part
    )


# Dimensionless force

def Fhat(D, T=0.0):

    return (
        Lambda * D / (D**2 + r**2)
        + Gp(D)
        - T
    )


# Fold stresses

def folds(Lambda_value):

    def f(D):

        return (
            Lambda_value
            * D
            / (D**2 + r**2)
            + Gp(D)
        )

    # Lower fold
    lo = minimize_scalar(
        f,
        bounds=(1.5, 4.2),
        method="bounded",
        options={
            "xatol": 1e-9
        }
    )

    # Upper fold
    hi = minimize_scalar(
        lambda D: -f(D),
        bounds=(3.0, 11.0),
        method="bounded",
        options={
            "xatol": 1e-9
        }
    )

    return (
        lo.fun,
        -hi.fun,
        lo.x,
        hi.x
    )


# Folds for Lambda = 0.4

T_minus, T_plus, D_minus, D_plus = folds(Lambda)


print(
    f"decreasing-separation fold  T_- = {T_minus:+.5f} "
    f"at D = {D_minus:.4f}"
)

print(
    f"unbinding fold              T_+ = {T_plus:+.5f} "
    f"at D = {D_plus:.4f}"
)


# Plotting data

levels = [

    (T_minus, "tab:blue"),

    (-0.03, "tab:cyan"),

    (0.0, "k"),

    (0.03, "tab:orange"),

    (T_plus, "tab:red")

]


Dp = np.linspace(
    0.02,
    9.0,
    900
)


# Figure (a): tilted reduced-energy landscapes

fig_a, ax_a = plt.subplots(
    figsize=(7.0, 5.2)
)


# Reference separation for the energy shift
D_ref = 3.4


for T, c in levels:

    # Shift each curve to zero at the reference separation.
    Ehat_rel = (
        Ehat(Dp, T)
        - Ehat(D_ref, T)
    )


    ax_a.plot(
        Dp,
        Ehat_rel,
        color=c,
        lw=LINE_W,
        label=rf"$T={T:.4f}$"
    )


ax_a.set_xlim(
    0,
    9
)

ax_a.set_ylim(
    -0.35,
    0.15
)


ax_a.set_xlabel(
    r"$D=d/\varepsilon$",
    fontsize=LABEL_FS
)


ax_a.set_ylabel(
    r"$\widehat{\mathcal{E}}(D;T)$",
    fontsize=LABEL_FS
)


ax_a.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_FS
)


ax_a.legend(
    fontsize=LEGEND_FS,
    loc="lower right"
)


fig_a.tight_layout()


# fig_a.savefig(
#     "superposed_stress_energy_landscape.png",
#     dpi=600,
#     bbox_inches="tight"
# )


plt.show()


# Figure (b): force balance

fig_b, ax_b = plt.subplots(
    figsize=(7.0, 5.2)
)


# Zero-stress reduced force
ax_b.plot(
    Dp,
    Fhat(Dp),
    color="k",
    lw=LINE_W
)


# Applied-stress levels
for T, c in levels:

    ax_b.axhline(
        T,
        color=c,
        ls="--",
        lw=AUX_LINE_W
    )


ax_b.axhline(
    0.0,
    color="0.8",
    lw=1.2
)


ax_b.set_xlim(
    0,
    9
)

ax_b.set_ylim(
    -0.12,
    0.30
)


ax_b.set_xlabel(
    r"$D=d/\varepsilon$",
    fontsize=LABEL_FS
)


ax_b.set_ylabel(
    r"$\mathcal{F}_{\rm II}(D;\Lambda)$",
    fontsize=LABEL_FS
)


ax_b.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_FS
)


ax_b.annotate(
    rf"$T_+={T_plus:.4f}$",
    xy=(
        6.3,
        T_plus + 0.008
    ),
    color="tab:red",
    fontsize=ANNOT_FS
)


ax_b.annotate(
    rf"$T_-={T_minus:.4f}$",
    xy=(
        6.3,
        T_minus + 0.007
    ),
    color="tab:blue",
    fontsize=ANNOT_FS
)


fig_b.tight_layout()


# fig_b.savefig(
#     "superposed_stress_force_balance.png",
#     dpi=600,
#     bbox_inches="tight"
# )


plt.show()


# Figure (c): stability window

khs = np.linspace(
    0.15,
    1.3,
    60
)


Tm, Tp = np.array(
    [
        folds(k)[:2]
        for k in khs
    ]
).T


fig_c, ax_c = plt.subplots(
    figsize=(7.0, 5.2)
)


ax_c.fill_between(
    khs,
    Tm,
    Tp,
    color="tab:green",
    alpha=0.25,
    label="trapped-dipole region"
)


ax_c.plot(
    khs,
    Tm,
    color="tab:blue",
    lw=LINE_W,
    label=r"$T_-$: decreasing separation"
)


ax_c.plot(
    khs,
    Tp,
    color="tab:red",
    lw=LINE_W,
    label=r"$T_+$: unbinding"
)


ax_c.axhline(
    0.0,
    color="k",
    lw=1.2
)


ax_c.set_xlabel(
    r"$\Lambda=K/(A_\psi\varepsilon)$",
    fontsize=LABEL_FS
)


ax_c.set_ylabel(
    r"$T=\tau q_0/A_\psi$",
    fontsize=LABEL_FS
)


ax_c.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_FS
)


ax_c.legend(
    fontsize=LEGEND_FS,
    loc="upper left"
)


fig_c.tight_layout()


# fig_c.savefig(
#     "superposed_stress_stability_window.png",
#     dpi=600,
#     bbox_inches="tight"
# )


plt.show()
