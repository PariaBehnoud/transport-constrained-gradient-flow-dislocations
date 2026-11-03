"""
Single dislocation (parameters of Remark 7.1): one Gaussian core spreading.

Command line:  python single_core.py dx L t_end
runs one core on [-L, L] with grid spacing dx up to t_end and prints w(t).
The checks of Remark 7.1 use
    python single_core.py 1e-3 8 1     (twice the domain of Fig. 5)
    python single_core.py 2e-3 4 1     (twice the grid spacing of Fig. 5)
"""
import numpy as np
from glide_layer import Grid, Params, ElasticOperator, normal_cdf

def pad(gamma, q0):
    # gamma = 0 to the left of the grid and gamma = q0 to the right of it
    return np.concatenate([[0.0], gamma, [q0]])

def rate_single(gamma, grid, p, elastic):
    dx = grid.dx
    g = pad(gamma, p.q0)
    qf = np.diff(g) / dx                        # q on the N+1 faces
    mu = elastic.from_faces(qf) if p.K != 0.0 else np.zeros_like(gamma)
    if p.A_psi != 0.0:
        mu = mu + (np.pi * p.A_psi / p.b_slip) * np.sin(2 * np.pi * gamma / p.b_slip)
    c = p.m * (mu - p.tau)
    Dm, Dp = qf[:-1], qf[1:]
    grad_pos = np.sqrt(np.maximum(Dm, 0.0)**2 + np.minimum(Dp, 0.0)**2)
    grad_neg = np.sqrt(np.minimum(Dm, 0.0)**2 + np.maximum(Dp, 0.0)**2)
    out = -c * np.where(c > 0.0, grad_pos, grad_neg)
    dt = 0.4 * dx / max(np.abs(c).max(), 1e-12)
    qmax = np.abs(qf).max()
    if p.K != 0.0:
        dt = min(dt, 0.5 / max(p.m * qmax * elastic.stiffness, 1e-12))
    if p.kappa_h != 0.0:
        flux = 0.5 * p.m * p.kappa_h * qf * np.abs(qf)
        out = out + np.diff(flux) / dx
        dt = min(dt, 0.4 * dx**2 / (p.m * p.kappa_h * max(qmax, 1e-12)))
    return out, dt

def evolve_single(p, grid, t_end, record_times=(), log_every=200):
    elastic = ElasticOperator(grid, p)
    gamma = p.q0 * normal_cdf(grid.x / p.eps)
    t, step, rec, ri = 0.0, 0, sorted(record_times), 0
    prof, hist = {}, {"t": [], "q_max": []}
    def log():
        hist["t"].append(t); hist["q_max"].append((np.diff(pad(gamma, p.q0)) / grid.dx).max())
    log()
    while t < t_end - 1e-15:
        while ri < len(rec) and rec[ri] <= t + 1e-15:
            prof[rec[ri]] = gamma.copy(); ri += 1
        k1, dt = rate_single(gamma, grid, p, elastic)
        dt = min(dt, t_end - t)
        if ri < len(rec):
            dt = min(dt, rec[ri] - t)
        g1 = gamma + dt * k1
        k2, _ = rate_single(g1, grid, p, elastic)
        gamma = 0.5 * (gamma + g1 + dt * k2); t += dt; step += 1
        if step % log_every == 0:
            log()
    while ri < len(rec) and rec[ri] <= t + 1e-12:
        prof[rec[ri]] = gamma.copy(); ri += 1
    log()
    return {"gamma": gamma, "profiles": prof, "hist": {k: np.array(v) for k, v in hist.items()}, "steps": step}

if __name__ == "__main__":
    import sys, time
    dx = float(sys.argv[1]); L = float(sys.argv[2]); T_END = float(sys.argv[3])
    p = Params(q0=1, b_slip=1, m=1, K=1.0, kappa_h=0.10, A_psi=1.0, eps=0.025)
    G = Grid(L=L, N=int(round(2 * L / dx)) + 1)
    t0 = time.time()
    r = evolve_single(p, G, T_END, record_times=[0.0, 0.01, 0.03, 0.1, 0.3, 1.0])
    h = r["hist"]
    for tt in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0]:
        i = np.searchsorted(h["t"], tt)
        if i < len(h["t"]):
            print(f"t={h['t'][i]:.4f}  q_max={h['q_max'][i]:.4f}  w=q0/(pi q_max)={1/(np.pi*h['q_max'][i]):.4f}")
    print("PN zeta = K/(2 pi A) =", 1 / (2 * np.pi), " time %.0f s, steps %d" % (time.time() - t0, r["steps"]))
    np.savez(f"single_dx{dx}_L{L}.npz", t=h["t"], q_max=h["q_max"], x=G.x, **{f"p_{k!r}": v for k, v in r["profiles"].items()})
