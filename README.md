# Transport-Constrained Gradient-Flow Description of Continuum Dislocation Dynamics

This repository contains the Python code accompanying the manuscript

**A Transport-Constrained Gradient-Flow Description of Continuum Dislocation Dynamics**

by **Paria Behnoud, Anthony Rollett, Carlos Mora-Corral, and Kaushik Dayal**.

The code reproduces Section 7: the opposite-sign dislocation pair (Section 7.C) and the spreading of a single dislocation core (Remark 7.1), with the figures and the values reported in the paper.

## Model

Section 7 restricts the continuum model of Sections 2-4 to a single glide layer. The state is the slip `gamma(x1, t)`, with signed Burgers density `q = d gamma/dx1`. With a drag mobility `m/|alpha|`, the transport law and the Onsager principle give one evolution equation. For the pair, `q >= 0` for `x1 < 0`, and (Eq. (7.18))

    d gamma/dt = -m (d gamma/dx1) dE_tot/dgamma,    x1 < 0,

with `gamma` even in `x1`; for a single dislocation, `q >= 0` everywhere and the same equation holds on the whole line. In the code, the equivalent form `d gamma/dt = -m |d gamma/dx1| dE_tot/dgamma` is solved on the whole grid.

where (Eq. (7.14))

    dE_tot/dgamma = (K/q0^2) p.v. int q(x')/(x1 - x') dx'    elastic (Peierls-Nabarro)
                    - kappa_h dq/dx1                         core energy
                    + psi'(gamma)                            lattice energy, psi = A_psi sin^2(pi gamma/b_slip)
                    - tau                                    applied stress

The velocity is `V = m tau_tot` for `x1 < 0` and `V = -m tau_tot` for `x1 > 0` (Eq. (7.17)), with `tau_tot = -dE_tot/dgamma`: it has the same magnitude for both signs of `q` and reverses direction with the sign of `q`. Positive and negative Burgers content therefore move toward each other under attraction and cancel where `q` changes sign, so annihilation follows from the evolution equation. For the pair, the initial state is two Gaussian cores of width `eps` at `x1 = -/+ d0/2` (Eq. (7.4)); for the single dislocation, one Gaussian core at `x1 = 0`. After that, the core shapes evolve freely.

The dimensionless groups are `D0 = d0/eps`, `Lambda = K/(A_psi eps)`, `Gamma_alpha = kappa_h q0^2/(A_psi eps^2)`, and `T = tau q0/A_psi`. The Peierls-Nabarro core half-width is `zeta = K/(2 pi A_psi)`, so `zeta/eps = Lambda/(2 pi)`.

## Numerical method

- Uniform grid on `[-L, L]`, symmetric about `x1 = 0`. Outside the grid, `gamma = 0` for the pair; for the single dislocation, `gamma = 0` on the left and `gamma = q0` on the right.
- Elastic term: a variational discretization. `q` lives on the cell faces, the discrete Peierls-Nabarro energy uses the cell-averaged logarithmic kernel, and its exact derivative with respect to the nodal slips gives the elastic driving force. The convolution over the grid is evaluated exactly with the FFT, after zero-padding to the length of the full linear convolution.
- `|d gamma/dx1|` for the first-order terms: upwind (Godunov) scheme for Hamilton-Jacobi equations (Osher and Sethian, 1988).
- Core term: `m kappa_h |gamma_x| gamma_xx = (m kappa_h/2) d/dx (gamma_x |gamma_x|)`, in conservative form.
- Time stepping: two-stage strong-stability-preserving Runge-Kutta, with an adaptive step limited by the advective and diffusive stability limits and by the stiffness of the elastic term, `m |q| (K/q0^2) |k|`. The last limit matters for the narrow Peierls-Nabarro cores of the pair; without it, the cores develop grid-scale oscillations.

The discrete energy decreases monotonically in all runs, and the pair stays mirror symmetric to round-off. The scripts print these values. The scheme is first-order accurate in the grid spacing; `grid_check.py` measures this for the annihilation time of the pair.

## Repository contents

| File | Purpose |
|---|---|
| `glide_layer.py` | Solver: energies, driving force, time integration, diagnostics |
| `plot_style.py` | Shared figure style and colors |
| `checks.py` | Checks of the solver (see below) |
| `single_core.py` | Evolution of a single dislocation (Remark 7.1) |
| `regime_I.py` | Single core, Remark 7.1: Fig. 5 (`regime_I_profiles.png`); the core half-width `w(t)` quoted there (`regime_I_width.png`, not shown in the paper) |
| `regime_II.py` | Pair, Section 7.C (`kappa_h = 2e-6`): Figs. 3 and 4 (`regime_II_*.png`) and the values in Section 7.C, including the energies of Fig. 3b |
| `regime_II_threshold.py` | Pair: applied stress that separates the pair (bisection) |
| `grid_check.py` | Pair: annihilation time at `T = 0` on three grids, with the parameters of Section 7.C |

Results of the long runs are saved in `results/` and reused when a script is run again; delete that folder to recompute. The repository includes the results used for the paper:

| File | Contents |
|---|---|
| `results/regime_II_T-0.0500_N16001.npz`, `..._T+0.0000_...`, `..._T+0.0200_...`, `..._T+0.0400_...` | Pair, Figs. 3 and 4: histories (`h_t`, `h_gamma_mid`, `h_separation`, `h_q_max`, `h_E`, `h_E_el`, `h_E_core`, `h_E_lat`) and slip profiles `p_<t>` |
| `results/regime_I_single_N8001.npz` | Single core, Fig. 5: `t`, `q_max`, and slip profiles `p_<t>` |
| `results/regime_II_threshold.json` | Outcomes of the bisection on `T` |

All figures are saved in `media/Figs/`.

### `checks.py`

1. Elastic operator for a Gaussian density against the exact result `(sqrt(2)/eps) daw(x/(sqrt(2) eps))` (second-order convergence).
2. Residual of the Peierls-Nabarro core, Eq. (7.11) (second-order convergence).
3. Monotone energy decay, for the single core and the pair, and mirror symmetry `q(-x1, t) = -q(x1, t)` of the pair.
4. Grid convergence of the half-width of the single core, and of the annihilation time of the pair without core energy (the check with the parameters of Section 7.C is `grid_check.py`).

## Expected output

| Quantity | Value |
|---|---|
| Single core, half-width `w = q0/(pi max q)` | `0.80 eps` (`t = 0`), `6.70 eps` (`t = 0.1`), `7.61 eps` (`t = 1`); `zeta = 6.37 eps` |
| Pair, Peierls-Nabarro half-width (without core energy) | `zeta = 9.947e-4`, `zeta/eps = 0.0398`; with `kappa_h = 2e-6` the core half-width at `t = 0.75` is `1.11 zeta` |
| Pair, `gamma(0,t) = q0/2` at | `t = 0.829` (`T = -0.05`), `1.647` (`T = 0`), `3.061` (`T = 0.02`); unbinding for `T = 0.04`, `s = 1.55 d0` at `t = 4` |
| Pair, energies at `T = 0` (elastic, core, lattice, total) | `t = 0`: `0.0147, 0.0000, 0.0729, 0.0876`; `t = 0.05`: `0.0276, 0.0003, 0.0078, 0.0357`; `t = 1.6`: `0.0175, 0.0003, 0.0066, 0.0244` |
| Pair, point-dislocation estimate of the annihilation time at `T = 0` | `q0 d0^2/(4 m K) = 1.6` |
| Pair, threshold (`N = 8001`) | `0.03125 < T_c < 0.03163`; point-dislocation estimate `Lambda/D0 = 0.03125` |
| Pair, threshold check (`N = 16001`) | `T = 0.0300`: annihilation; `T = 0.0325`: unbinding |
| Pair, grid check (`grid_check.py`), `gamma(0,t) = q0/2` at `T = 0` | `t = 1.7076` (`N = 8001`), `1.6467` (`N = 16001`), `1.6185` (`N = 32001`); first-order convergence, extrapolated `t = 1.594` |
| Single core, `w` at `t = 1` with twice the domain or twice the grid spacing | `0.1902`, `0.1903` (`0.1902` on the grid of Fig. 5) |

## Requirements

Python 3 with NumPy, SciPy, and Matplotlib:

    python -m pip install -r requirements.txt

## Running the code

    python checks.py
    python regime_I.py
    python regime_II.py
    python regime_II_threshold.py
    python grid_check.py
    python single_core.py 1e-3 8 1
    python single_core.py 2e-3 4 1

The last two commands are the domain and grid checks of Remark 7.1.

Run times on one core are about 10 minutes for `regime_I.py`, about 1.5 hours for `regime_II.py`, about 30 minutes for `checks.py`, several hours for `regime_II_threshold.py`, and about 2 hours for `grid_check.py`.
