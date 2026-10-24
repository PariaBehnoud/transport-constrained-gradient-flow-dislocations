# Transport-Constrained Gradient-Flow Description of Continuum Dislocation Dynamics

This repository contains the Python scripts accompanying the manuscript

**A Transport-Constrained Gradient-Flow Description of Continuum Dislocation Dynamics**

by **Paria Behnoud, Anthony Rollett, Carlos Mora-Corral, and Kaushik Dayal**.

The scripts reproduce the opposite-sign dislocation-pair examples in Section 7 of the manuscript: the figures and the values reported for Regimes I and II.

## Reduced model

Section 7 restricts the continuum model to two straight edge dislocations of opposite sign. The setup is:

- Slip is confined to a single glide layer.
- Each dislocation is a Gaussian core of width `eps`, centered at `X_+` and `X_-`, with separation `d = X_- - X_+`.
- The two cores are transported as separate dislocation populations with a drag mobility.

Every term of the reduced model comes from the continuum model:

- **Energies:** evaluated from the continuum free energy on the pair.
  - Elastic interaction: `E_el = 2 K int_0^{d/(2 eps)} daw(s) ds`, where `daw` is the Dawson function. For `d >> eps`, it recovers the classical `K log d`.
  - Core energy: `kappa_h q0^2 / (2 sqrt(pi) eps) * (1 - exp(-d^2/(4 eps^2)))`, with `kappa_h = kappa/h`.
  - Lattice energy: `int psi(gamma) dx1`, with `psi(gamma) = A_psi sin^2(pi gamma / b_slip)`.
  - Applied stress: `-tau q0 d`.
- **Evolution law:** restricting the Onsager principle to rigid translations of the two cores gives `dd/dt = -2 M dE_tot/dd`, with `M = m/q0`.

The dimensionless groups are `D = d/eps`, `Lambda = K/(A_psi eps)`, `Gamma_alpha = kappa_h q0^2/(A_psi eps^2)`, and `T = tau q0/A_psi`.

## Repository contents

### `regime_I_opposite_pair.py`

Regime I (`Lambda = 40`, `Gamma_alpha = 160`, `T = 0`). It compares the elastic-only trajectory with the trajectory driven by the total reduced energy. It checks numerically that each driving force is the derivative of its energy, prints the separations and energies at the snapshot times, and saves Fig. 2:

- `opposite_pair_combined_profiles_with_psi.png`
- `opposite_pair_distance_all_the_way.png`
- `opposite_pair_common_total_energy.png`

It also saves two supplementary plots that are not in the manuscript: the energy decrease and its rate.

### `regime_II_applied_stress.py`

Regime II (`Lambda = 0.25`, `Gamma_alpha = 0`) under a superposed uniform resolved shear stress. It prints the zero-stress stationary separations, the fold stresses `T_-` and `T_+`, and `Lambda_c`, and saves Fig. 3:

- `energ_landscape.png`
- `force_balance.png`
- `stability_window.png`

### `regime_II_fold_values.py`

Computes, in dimensional form, the Regime II values reported in Section 7.D:

- the stationary separations
- the folds and their separations
- `Lambda_c`
- `|tau|/tau_lat`

It runs independently of `regime_II_applied_stress.py` and serves as a cross-check.

All figures are saved in `media/Figs/`.

## Expected output

| Quantity | Value |
|---|---|
| Regime I, `t = 0.008` | `d_el = 0.06952`, `d_tot = 0.04926`; `E_tot = 2.17556`, `1.50269` |
| Regime II, stationary separations | `D_b = 2.378`, `D_eq = 3.527` (`d = 0.0595`, `0.0882`) |
| Regime II, folds | `T_- = -0.0745` at `d = 0.0693`; `T_+ = 0.0524` at `d = 0.1198` |
| Regime II, `Lambda_c` | `0.4126` |

## Requirements

Python 3 with NumPy (2.0 or later, for `numpy.trapezoid`), SciPy, and Matplotlib. Install them using:

`python -m pip install -r requirements.txt`

## Running the codes

Regime I:

`python regime_I_opposite_pair.py`

Regime II with applied stress:

`python regime_II_applied_stress.py`

Regime II values:

`python regime_II_fold_values.py`

Set `SHOW_FIGURES = False` in the first two scripts to save the figures without displaying them. `regime_II_fold_values.py` takes about two minutes, mostly to compute `Lambda_c`.

## Scope

These scripts implement the reduced two-core model of Section 7. It is derived from the continuum model under the assumptions stated there. The scripts are not a full-field implementation of the complete transport-constrained continuum model.

## Citation

If you use these codes, please cite the accompanying manuscript:

P. Behnoud, A. Rollett, C. Mora-Corral, and K. Dayal,
*A Transport-Constrained Gradient-Flow Description of Continuum Dislocation Dynamics.*
