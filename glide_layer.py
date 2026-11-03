"""
Glide-layer evolution of Section 7.

The continuum model of Sections 2-4, restricted to a single glide layer
(Section 7.A), gives one evolution equation for the slip gamma(x1, t):

    d gamma/dt = -m |d gamma/dx1| dE_tot/dgamma,

which is Eq. (7.18) where q >= 0 (x1 < 0 for the pair), extended by symmetry,
with the signed Burgers density q = d gamma/dx1 and

    dE_tot/dgamma = (K/q0^2) p.v. int q(x')/(x1 - x') dx'      elastic, Eq. (7.6)
                    - kappa_h dq/dx1                           core,    Eq. (7.7)
                    + psi'(gamma)                              lattice, Eq. (7.10)
                    - tau                                      applied, Eq. (7.12)

where psi(gamma) = A_psi sin^2(pi gamma / b_slip). All energies are per unit
dislocation-line length.

Numerical method (Section 7.B.5)
--------------------------------
* Uniform grid x_i on [-L, L], symmetric about x1 = 0; gamma = 0 outside it.
* Elastic term: q on the cell faces, piecewise constant; the discrete elastic
  energy with the cell-averaged logarithmic kernel is differentiated exactly
  with respect to the nodal slips (a variational discretization), and the
  convolution over the grid is evaluated exactly with the FFT, after
  zero-padding to the length of the full linear convolution (ElasticOperator).
* First-order terms (elastic, lattice, applied): upwind (Godunov) discretization
  of |d gamma/dx1| for Hamilton-Jacobi equations (Osher and Sethian, 1988).
* Core term: m kappa_h |gamma_x| gamma_xx = (m kappa_h / 2) d/dx (gamma_x |gamma_x|)
  in conservative form.
* Time stepping: two-stage strong-stability-preserving Runge-Kutta (Heun),
  with an adaptive step limited by the advective and diffusive stability limits
  and by the stiffness of the elastic term, m |q| (K/q0^2) |k|.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import erf


@dataclass
class Params:
    q0: float = 1.0          # Burgers content of each dislocation
    b_slip: float = 1.0      # lattice-equivalent slip: psi(gamma + b_slip) = psi(gamma)
    m: float = 1.0           # mobility coefficient; drag mobility m/|alpha|
    K: float = 1.0           # K = mu q0^2 / (2 pi (1 - nu))
    kappa_h: float = 0.0     # core-energy coefficient kappa/h
    A_psi: float = 0.0       # lattice-energy amplitude
    tau: float = 0.0         # applied resolved shear stress
    eps: float = 0.025       # initial Gaussian core width
    d0: float = 0.2          # initial separation of the two cores


@dataclass
class Grid:
    L: float
    N: int
    x: np.ndarray = field(init=False)
    dx: float = field(init=False)

    def __post_init__(self):
        if self.N % 2 == 0:
            self.N += 1          # odd N: the grid is symmetric and contains x1 = 0
        self.x = np.linspace(-self.L, self.L, self.N)
        self.dx = self.x[1] - self.x[0]
        self.mid = self.N // 2


def normal_cdf(s):
    return 0.5 * (1.0 + erf(s / np.sqrt(2.0)))


def initial_slip(grid, p):
    """Slip of two Gaussian cores at x1 = -/+ d0/2, the integral of Eq. (7.4)."""
    x = grid.x
    return p.q0 * (normal_cdf((x + 0.5 * p.d0) / p.eps)
                   - normal_cdf((x - 0.5 * p.d0) / p.eps))


def _cell_pair_log(m):
    """Average of log|x - x'| over two unit cells whose centers are m apart."""
    def G(u):
        u = np.asarray(u, dtype=float)
        out = np.zeros_like(u)
        nz = u != 0.0
        out[nz] = 0.5 * u[nz]**2 * np.log(np.abs(u[nz])) - 0.75 * u[nz]**2
        return out
    return G(m + 1.0) - 2.0 * G(m) + G(m - 1.0)


class ElasticOperator:
    """
    Variational discretization of dE_el/dgamma, Eq. (7.6).

    q is taken on the N+1 cell faces, q_a = (gamma_a - gamma_{a-1})/dx with
    gamma = 0 outside the grid, and is piecewise constant on the face cells.
    The discrete elastic energy
        E_el = -(K / 2 q0^2) sum_a sum_b q_a q_b dx^2 w(a - b),
    with w the cell-averaged logarithm, is differentiated exactly with respect
    to the nodal values gamma_i. The result tends to
    (K/q0^2) p.v. int q(x')/(x - x') dx' as dx -> 0.
    """

    def __init__(self, grid, p):
        N = grid.N
        n = np.arange(-N, N)                    # n = i - a, i nodes, a faces
        self.kernel = _cell_pair_log(n + 1.0) - _cell_pair_log(n)
        self.N = N
        # FFT length >= (N + 1) + 2N - 1, the length of the full linear convolution
        # of the N+1 face values with the 2N kernel values: the zero-padded FFT
        # product then gives the sum over the grid exactly, with no wrap-around.
        self.nfft = next_fast_len((N + 1) + 2 * N - 1, real=True)
        self.kernel_hat = rfft(self.kernel, self.nfft)
        self.coef = p.K / p.q0**2
        self.dx = grid.dx
        # largest eigenvalue of the discrete elastic operator (odd-even mode),
        # per unit Burgers density; used for the explicit time-step limit
        v = np.cos(np.pi * np.arange(N))
        self.stiffness = np.abs(self.from_faces(face_density(v, grid.dx))[grid.mid])

    def from_faces(self, qf):
        """Elastic dE/dgamma at the nodes from the N+1 face densities."""
        conv = irfft(rfft(qf, self.nfft) * self.kernel_hat, self.nfft)
        # conv[k] = sum_a qf[a] kernel[k - a]; node i uses n = i - a, index n + N
        return self.coef * conv[self.N:2 * self.N]

    def __call__(self, gamma):
        return self.from_faces(face_density(gamma, self.dx))


def slip_density(gamma, dx):
    """Central-difference q = d gamma/dx1 at the nodes (gamma = 0 outside)."""
    g = np.pad(gamma, 1)
    return (g[2:] - g[:-2]) / (2.0 * dx)


def face_density(gamma, dx):
    """q = d gamma/dx1 on the N+1 cell faces (gamma = 0 outside the grid)."""
    return np.diff(np.pad(gamma, 1)) / dx


def first_order_derivative(gamma, grid, p, elastic):
    """Elastic, lattice, and applied parts of dE_tot/dgamma."""
    mu = elastic(gamma) if p.K != 0.0 else np.zeros_like(gamma)
    if p.A_psi != 0.0:
        mu = mu + (np.pi * p.A_psi / p.b_slip) * np.sin(2.0 * np.pi * gamma / p.b_slip)
    return mu - p.tau


def energy_derivative(gamma, grid, p, elastic):
    """Full dE_tot/dgamma, Eq. (7.14), including the core term."""
    mu = first_order_derivative(gamma, grid, p, elastic)
    if p.kappa_h != 0.0:
        g = np.pad(gamma, 1)
        mu = mu - p.kappa_h * (g[2:] - 2.0 * g[1:-1] + g[:-2]) / grid.dx**2
    return mu


def rate(gamma, grid, p, elastic):
    """Right-hand side of Eq. (7.18) and the stable time step."""
    dx = grid.dx
    g = np.pad(gamma, 1)
    Dm = (g[1:-1] - g[:-2]) / dx          # backward difference
    Dp = (g[2:] - g[1:-1]) / dx           # forward difference

    c = p.m * first_order_derivative(gamma, grid, p, elastic)
    grad_pos = np.sqrt(np.maximum(Dm, 0.0)**2 + np.minimum(Dp, 0.0)**2)
    grad_neg = np.sqrt(np.minimum(Dm, 0.0)**2 + np.maximum(Dp, 0.0)**2)
    out = -c * np.where(c > 0.0, grad_pos, grad_neg)
    dt = 0.4 * dx / max(np.abs(c).max(), 1e-12)
    if p.K != 0.0:
        # stiffness of the elastic term, m |q| (K/q0^2) |k|: keep lambda dt <= 0.5
        qmax = np.abs(np.diff(g) / dx).max()
        dt = min(dt, 0.5 / max(p.m * qmax * elastic.stiffness, 1e-12))

    if p.kappa_h != 0.0:
        face = np.diff(g) / dx                         # gamma_x on the N+1 faces
        flux = 0.5 * p.m * p.kappa_h * face * np.abs(face)
        out = out + np.diff(flux) / dx
        dt = min(dt, 0.4 * dx**2 / (p.m * p.kappa_h * max(np.abs(face).max(), 1e-12)))
    return out, dt


def energies(gamma, grid, p, elastic):
    """Elastic, core, lattice, and applied-stress energies, Eqs. (7.5)-(7.13)."""
    dx = grid.dx
    E_el = 0.5 * np.sum(gamma * elastic(gamma)) * dx if p.K != 0.0 else 0.0
    g = np.pad(gamma, 1)
    face = np.diff(g) / dx
    E_core = 0.5 * p.kappa_h * np.sum(face**2) * dx
    E_lat = p.A_psi * np.sum(np.sin(np.pi * gamma / p.b_slip)**2) * dx
    E_app = -p.tau * np.sum(gamma) * dx
    return {"el": E_el, "core": E_core, "lat": E_lat, "app": E_app,
            "tot": E_el + E_core + E_lat + E_app}


def diagnostics(gamma, grid):
    """Midpoint slip gamma(0,t), centroid separation, and peak density."""
    x, dx = grid.x, grid.dx
    q = slip_density(gamma, dx)
    pos, neg = np.clip(q, 0.0, None), np.clip(-q, 0.0, None)
    Pp, Pn = pos.sum() * dx, neg.sum() * dx
    xp = (x * pos).sum() * dx / Pp if Pp > 1e-12 else np.nan
    xn = (x * neg).sum() * dx / Pn if Pn > 1e-12 else np.nan
    return {"gamma_mid": gamma[grid.mid], "separation": xn - xp,
            "q_max": np.abs(q).max()}


def evolve(p, grid, t_end, record_times=(), stop=None, gamma0=None,
           log_every=20, with_energy=False, energy_params=None):
    """
    Integrate Eq. (7.18) with SSP-RK2 from t = 0 to t_end.

    record_times : times at which gamma is stored.
    stop(t, diag) : optional function, called at each log; integration stops
                    when it returns something other than None.
    log_every : diagnostics (and energies, if with_energy: total E and its
                elastic, core, and lattice parts) are stored every this many steps.
    energy_params : parameters used to evaluate the stored energy (default p);
                    used to evaluate the same total energy along different runs.
    Returns a dict with the history of diagnostics and the stored profiles.
    """
    elastic = ElasticOperator(grid, p)
    pE = p if energy_params is None else energy_params
    elastic_E = elastic if pE.K == p.K and pE.q0 == p.q0 else ElasticOperator(grid, pE)
    gamma = initial_slip(grid, p) if gamma0 is None else gamma0.copy()
    t, step = 0.0, 0
    rec = sorted(record_times)
    ri = 0
    hist = {"t": [], "gamma_mid": [], "separation": [], "q_max": [], "E": [],
            "E_el": [], "E_core": [], "E_lat": []}
    profiles = {}
    outcome = None

    def log(t, gamma):
        d = diagnostics(gamma, grid)
        hist["t"].append(t)
        for key in ("gamma_mid", "separation", "q_max"):
            hist[key].append(d[key])
        if with_energy:
            e = energies(gamma, grid, pE, elastic_E)
            hist["E"].append(e["tot"])
            for key in ("el", "core", "lat"):
                hist["E_" + key].append(e[key])
        return d

    log(t, gamma)
    while t < t_end - 1e-15:
        while ri < len(rec) and rec[ri] <= t + 1e-15:
            profiles[rec[ri]] = gamma.copy()
            ri += 1
        k1, dt = rate(gamma, grid, p, elastic)
        dt = min(dt, t_end - t)
        if ri < len(rec):
            dt = min(dt, rec[ri] - t)
        g1 = gamma + dt * k1
        k2, _ = rate(g1, grid, p, elastic)
        gamma = 0.5 * (gamma + g1 + dt * k2)
        t += dt
        step += 1
        if step % log_every == 0:
            d = log(t, gamma)
            if stop is not None:
                outcome = stop(t, d)
                if outcome is not None:
                    break
    while ri < len(rec) and rec[ri] <= t + 1e-12:
        profiles[rec[ri]] = gamma.copy()
        ri += 1
    log(t, gamma)
    hist = {k: np.array(v) for k, v in hist.items()}
    return {"t": t, "gamma": gamma, "hist": hist, "profiles": profiles,
            "outcome": outcome, "steps": step}
