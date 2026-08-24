import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.special import erf


# Parameters
q0 = 1.0
b_slip = q0

M = 1.0
K = 1.0
a = 0.035

kappa = 0.10
eps = 0.025
A_psi = 1.0

d0 = 0.20
dt = 1.0e-5
T = 0.015
d_report = 1.0e-4  # reporting threshold

snapshot_times = [0.0000, 0.0050, 0.0080, 0.0090, T]

# Numerical windows for plotting and quadrature.
x_profile = np.linspace(-0.35, 0.35, 1601)
x_energy = np.linspace(-0.50, 0.50, 4001)

SAVE_FIGURES = True
SHOW_FIGURES = True

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "media" / "Figs"
output_dir.mkdir(parents=True, exist_ok=True)

label_fs = 30
tick_fs = 25
legend_fs = 20
legend_title_fs = 20
line_width = 3.5


# Parameter checks
if d0 <= 0.0:
    raise ValueError("Require d0 > 0.")
if eps <= 0.0:
    raise ValueError("eps must be positive.")
if b_slip <= 0.0:
    raise ValueError("b_slip must be positive.")
if A_psi < 0.0:
    raise ValueError("A_psi must be nonnegative.")
if dt <= 0.0 or T <= 0.0:
    raise ValueError("dt and T must be positive.")
if not 0.0 < d_report < d0:
    raise ValueError("Require 0 < d_report < d0.")
if any(t < 0.0 or t > T for t in snapshot_times):
    raise ValueError("All snapshot times must lie in [0,T].")


# Pair profiles
def gaussian_core(x, center):
    z = (x - center) / eps
    return np.exp(-0.5 * z**2) / (np.sqrt(2.0 * np.pi) * eps)


def normal_cdf(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def pair_centers(d):
    return -0.5 * d, 0.5 * d


def q_profile(d, x=x_profile):
    x_plus, x_minus = pair_centers(d)
    return q0 * (
        gaussian_core(x, x_plus)
        - gaussian_core(x, x_minus)
    )


def beta_p_profile(d, x=x_energy):
    x_plus, x_minus = pair_centers(d)
    return q0 * (
        normal_cdf((x - x_plus) / eps)
        - normal_cdf((x - x_minus) / eps)
    )


def dbeta_p_dd_profile(d, x=x_energy):
    x_plus, x_minus = pair_centers(d)
    return 0.5 * q0 * (
        gaussian_core(x, x_plus)
        + gaussian_core(x, x_minus)
    )


# Lattice energy
def psi_density(beta_p):
    return A_psi * np.sin(np.pi * beta_p / b_slip) ** 2


def dpsi_dbeta(beta_p):
    return (
        A_psi
        * np.pi
        / b_slip
        * np.sin(2.0 * np.pi * beta_p / b_slip)
    )


# Reduced energies
def E_pk(d):
    d = np.asarray(d, dtype=float)
    return 0.5 * K * np.log((d**2 + a**2) / a**2)


def E_alpha(d):
    d = np.asarray(d, dtype=float)
    prefactor = kappa * q0**2 / (2.0 * np.sqrt(np.pi) * eps)
    return prefactor * (1.0 - np.exp(-d**2 / (4.0 * eps**2)))


def E_psi_scalar(d):
    beta_p = beta_p_profile(float(d), x_energy)
    return float(np.trapezoid(psi_density(beta_p), x_energy))


def E_psi(d):
    if np.ndim(d) == 0:
        return E_psi_scalar(float(d))
    d = np.asarray(d, dtype=float)
    return np.asarray([E_psi_scalar(value) for value in d])


def E_total(d):
    return E_pk(d) + E_alpha(d) + E_psi(d)


# Driving terms
def F_pk(d):
    d = np.asarray(d, dtype=float)
    return K * d / (d**2 + a**2)


def F_alpha(d):
    d = np.asarray(d, dtype=float)
    prefactor = kappa * q0**2 / (4.0 * np.sqrt(np.pi) * eps**3)
    return prefactor * d * np.exp(-d**2 / (4.0 * eps**2))


def F_psi_scalar(d):
    beta_p = beta_p_profile(float(d), x_energy)
    dbeta_dd = dbeta_p_dd_profile(float(d), x_energy)
    return float(
        np.trapezoid(
            dpsi_dbeta(beta_p) * dbeta_dd,
            x_energy,
        )
    )


def F_psi(d):
    if np.ndim(d) == 0:
        return F_psi_scalar(float(d))
    d = np.asarray(d, dtype=float)
    return np.asarray([F_psi_scalar(value) for value in d])


def F_total(d):
    return F_pk(d) + F_alpha(d) + F_psi(d)


# Numerical checks
def check_F_psi_derivative():
    h = 1.0e-6
    errors = []

    for d in [0.25 * d0, 0.50 * d0, 0.75 * d0]:
        finite_difference = (
            E_psi_scalar(d + h) - E_psi_scalar(d - h)
        ) / (2.0 * h)
        direct_value = F_psi_scalar(d)
        scale = max(1.0, abs(finite_difference), abs(direct_value))
        errors.append(abs(finite_difference - direct_value) / scale)

    error = max(errors)
    if error > 2.0e-4:
        warnings.warn(f"F_psi derivative check error = {error:.3e}.")
    return error


F_psi_check_error = check_F_psi_derivative()

probe_d = np.linspace(1.0e-6, d0, 300)
minimum_total_force = float(np.min(F_total(probe_d)))
if minimum_total_force <= 0.0:
    warnings.warn(
        "F_total(d) is not positive on the sampled interval; "
        "a finite-separation stationary point may be present."
    )


# Time integration
def distance_rhs(d, closure):
    if closure == "elastic":
        force = float(F_pk(d))
    elif closure == "total":
        force = float(F_total(d))
    else:
        raise ValueError("closure must be 'elastic' or 'total'.")

    return -2.0 * M * force


def rk4_step(d, h, closure):
    k1 = distance_rhs(d, closure)
    k2 = distance_rhs(d + 0.5 * h * k1, closure)
    k3 = distance_rhs(d + 0.5 * h * k2, closure)
    k4 = distance_rhs(d + h * k3, closure)
    return d + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_distance(closure):
    time = 0.0
    distance = d0
    times = [time]
    distances = [distance]

    while time < T:
        h = min(dt, T - time)
        trial_distance = rk4_step(distance, h, closure)

        # Guard against a small negative RK4 overshoot.
        if trial_distance < 0.0:
            if abs(trial_distance) > 1.0e-8:
                warnings.warn("RK4 crossed d=0; reduce dt.")
            trial_distance = 0.0

        time += h
        distance = trial_distance
        times.append(time)
        distances.append(distance)

    return np.asarray(times), np.asarray(distances)


t_el, d_el = integrate_distance("elastic")
t_tot, d_tot = integrate_distance("total")


# Energy along the two trajectories
E0_total = float(E_total(d0))

E_total_el_path = E_total(d_el)
E_total_tot_path = E_total(d_tot)

energy_drop_el = E0_total - E_total_el_path
energy_drop_tot = E0_total - E_total_tot_path

# Dissipation rates evaluated along each trajectory.
dissipation_el = 2.0 * M * F_total(d_el) * F_pk(d_el)
dissipation_tot = 2.0 * M * F_total(d_tot) ** 2

if np.min(dissipation_el) < -1.0e-10:
    warnings.warn("E_total is not monotone along the elastic-only path.")
if np.min(dissipation_tot) < -1.0e-10:
    warnings.warn("E_total is not monotone along the total-energy path.")

dissipation_el = np.maximum(dissipation_el, 0.0)
dissipation_tot = np.maximum(dissipation_tot, 0.0)


# Near-coincidence time
def threshold_crossing_time(times, values, threshold):
    indices = np.flatnonzero(values <= threshold)
    if indices.size == 0:
        return None

    i = int(indices[0])
    if i == 0:
        return float(times[0])

    t0, t1 = float(times[i - 1]), float(times[i])
    v0, v1 = float(values[i - 1]), float(values[i])
    if abs(v0 - v1) < 1.0e-15:
        return t1

    fraction = np.clip((v0 - threshold) / (v0 - v1), 0.0, 1.0)
    return t0 + fraction * (t1 - t0)


t_near_el = threshold_crossing_time(t_el, d_el, d_report)
t_near_tot = threshold_crossing_time(t_tot, d_tot, d_report)


# Summary
print("=" * 82)
print("Opposite-sign pair: same total-energy comparison")
print("=" * 82)
print(f"Initial separation:                   d0       = {d0:.8f}")
print(f"Gaussian width:                       eps      = {eps:.8f}")
print(f"Lattice-energy amplitude:             A_psi    = {A_psi:.8f}")
print(f"Lattice slip period:                  b_slip   = {b_slip:.8f}")
print(f"Final integration time:               T        = {T:.8f}")
print(f"Near-coincidence tolerance only:       d_report = {d_report:.2e}")
print()
print(f"Largest relative F_psi check error:    {F_psi_check_error:.3e}")
print(f"Minimum sampled F_total:                {minimum_total_force:.6e}")
print()
print(f"Initial E_PK:                          {float(E_pk(d0)):.8f}")
print(f"Initial E_alpha:                       {float(E_alpha(d0)):.8f}")
print(f"Initial E_psi:                         {float(E_psi(d0)):.8f}")
print(f"Common initial E_total:                 {E0_total:.8f}")
print()
print(f"Final elastic-only separation:          {d_el[-1]:.8e}")
print(f"Final total-energy separation:          {d_tot[-1]:.8e}")
print(f"Final E_total on elastic-only path:       {E_total_el_path[-1]:.8e}")
print(f"Final E_total on total-energy path:       {E_total_tot_path[-1]:.8e}")
print()

if t_near_el is not None:
    print(
        f"Elastic-only near-coincidence time for d={d_report:.1e}: "
        f"{t_near_el:.8f}"
    )
else:
    print("Elastic-only trajectory did not reach the reporting tolerance.")

if t_near_tot is not None:
    print(
        f"Total-energy near-coincidence time for d={d_report:.1e}: "
        f"{t_near_tot:.8f}"
    )
else:
    print("Total-energy trajectory did not reach the reporting tolerance.")

if t_near_el is not None and t_near_tot is not None:
    reduction = 100.0 * (t_near_el - t_near_tot) / t_near_el
    print(f"Reduction in near-coincidence time:    {reduction:.2f}%")

print()
print("Separations and the same total energy at snapshot times:")
for query_time in snapshot_times:
    d_el_q = float(np.interp(query_time, t_el, d_el))
    d_tot_q = float(np.interp(query_time, t_tot, d_tot))
    E_el_q = float(np.interp(query_time, t_el, E_total_el_path))
    E_tot_q = float(np.interp(query_time, t_tot, E_total_tot_path))

    print(
        f"t={query_time:.4f}:  "
        f"d_el={d_el_q:.7f},  d_tot={d_tot_q:.7f},  "
        f"E_total[d_el]={E_el_q:.7f},  E_total[d_tot]={E_tot_q:.7f}"
    )

print()
print(f"Figures saved in: {output_dir}")
print("=" * 82)


# Plotting
def format_axes(ax, xlabel, ylabel, xlim=None, ylim=None):
    ax.set_xlabel(xlabel, fontsize=label_fs, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=label_fs, labelpad=14)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fs,
        width=1.8,
        length=8,
    )


def compact_time_formatter(x, pos):
    text = f"{x:.3f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


time_formatter = FuncFormatter(compact_time_formatter)


def finish_figure(fig, filename):
    fig.tight_layout()

    if SAVE_FIGURES:
        fig.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# Figure 1: signed density profiles
fig, ax = plt.subplots(figsize=(10.0, 7.0))
time_colors = ["C0", "C1", "C2", "C3", "C4"]

for color, query_time in zip(time_colors, snapshot_times):
    d_el_q = float(np.interp(query_time, t_el, d_el))
    d_tot_q = float(np.interp(query_time, t_tot, d_tot))

    ax.plot(
        x_profile,
        q_profile(d_el_q, x_profile),
        color=color,
        linewidth=line_width,
        linestyle="-",
    )
    ax.plot(
        x_profile,
        q_profile(d_tot_q, x_profile),
        color=color,
        linewidth=line_width,
        linestyle="--",
    )

ax.axhline(0.0, linewidth=1.5)
peak_height = q0 / (np.sqrt(2.0 * np.pi) * eps)
format_axes(
    ax,
    r"$x_1$",
    r"$q(x_1,t)$",
    xlim=(x_profile[0], x_profile[-1]),
    ylim=(-1.10 * peak_height, 1.10 * peak_height),
)

time_handles = [
    Line2D(
        [0],
        [0],
        color=color,
        linewidth=line_width,
        label=rf"$t={time:.4f}$",
    )
    for color, time in zip(time_colors, snapshot_times)
]
legend_times = ax.legend(
    handles=time_handles,
    title="time",
    fontsize=legend_fs,
    title_fontsize=legend_title_fs,
    frameon=True,
    loc="upper right",
)
ax.add_artist(legend_times)

model_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        linewidth=line_width,
        linestyle="-",
        label="elastic-only trajectory",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linewidth=line_width,
        linestyle="--",
        label=r"total-energy trajectory",
    ),
]
ax.legend(
    handles=model_handles,
    title="trajectory",
    fontsize=legend_fs,
    title_fontsize=legend_title_fs,
    frameon=True,
    loc="lower left",
)
finish_figure(fig, "opposite_pair_combined_profiles_with_psi.png")


# Figure 2: separation histories
fig, ax = plt.subplots(figsize=(10.0, 7.0))
ax.plot(t_el, d_el, linewidth=line_width, label="elastic-only trajectory")
ax.plot(
    t_tot,
    d_tot,
    linewidth=line_width,
    label=r"total-energy trajectory",
)
format_axes(ax, r"$t$", r"$d(t)$", xlim=(0.0, T), ylim=(0.0, 1.03 * d0))
ax.xaxis.set_major_locator(MultipleLocator(0.005))
ax.xaxis.set_major_formatter(time_formatter)
ax.legend(fontsize=legend_fs, frameon=True, loc="upper right")
finish_figure(fig, "opposite_pair_distance_all_the_way.png")


# Figure 3: total energy
fig, ax = plt.subplots(figsize=(10.0, 7.0))
ax.plot(
    t_el,
    E_total_el_path,
    linewidth=line_width,
    label=r"$\mathcal{E}_{\rm tot}(d_{\rm el}(t))$",
)
ax.plot(
    t_tot,
    E_total_tot_path,
    linewidth=line_width,
    label=r"$\mathcal{E}_{\rm tot}(d_{\rm tot}(t))$",
)
format_axes(
    ax,
    r"$t$",
    r"$\mathcal{E}_{\rm tot}(t)$",
    xlim=(0.0, T),
    ylim=(0.0, 1.04 * E0_total),
)
ax.xaxis.set_major_locator(MultipleLocator(0.005))
ax.xaxis.set_major_formatter(time_formatter)
ax.legend(fontsize=legend_fs, frameon=True, loc="upper right")
finish_figure(fig, "opposite_pair_common_total_energy.png")


# Figure 4: energy decrease
fig, ax = plt.subplots(figsize=(10.0, 7.0))
ax.plot(t_el, energy_drop_el, linewidth=line_width, label="elastic-only trajectory")
ax.plot(
    t_tot,
    energy_drop_tot,
    linewidth=line_width,
    label=r"total-energy trajectory",
)
format_axes(
    ax,
    r"$t$",
    r"$\mathcal{E}_{\rm tot}(0)-\mathcal{E}_{\rm tot}(t)$",
    xlim=(0.0, T),
    ylim=(0.0, 1.04 * E0_total),
)
ax.xaxis.set_major_locator(MultipleLocator(0.005))
ax.xaxis.set_major_formatter(time_formatter)
ax.legend(fontsize=legend_fs, frameon=True, loc="upper left")
finish_figure(fig, "opposite_pair_common_total_energy_decrease.png")


# Figure 5: dissipation rate
fig, ax = plt.subplots(figsize=(10.0, 7.0))
ax.plot(t_el, dissipation_el, linewidth=line_width, label="elastic-only trajectory")
ax.plot(
    t_tot,
    dissipation_tot,
    linewidth=line_width,
    label=r"total-energy trajectory",
)
format_axes(
    ax,
    r"$t$",
    r"$-\mathrm{d}\mathcal{E}_{\rm tot}/\mathrm{d}t$",
    xlim=(0.0, T),
    ylim=(0.0, None),
)
ax.legend(fontsize=legend_fs, frameon=True, loc="upper right")
finish_figure(fig, "opposite_pair_common_total_energy_dissipation_rate.png")
