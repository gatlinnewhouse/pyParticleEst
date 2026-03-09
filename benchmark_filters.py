"""Benchmark: SIS, SIR, APF, and RBPF on a Mixed Linear/Nonlinear Gaussian SSM.

Model (Schön et al. 2005 MLNLG framework, Gordon-Salmond-Smith 1993 nonlinear dynamics):

  Nonlinear sub-state (sampled by all filters, Kalman-filtered conditioned on ξ by RBPF):
    ξ_{t+1} = ξ_t/2 + 25·ξ_t/(1+ξ_t²) + 8·cos(1.2·t) + v_ξ,  v_ξ ~ N(0, Q_ξ)

  Linear sub-state (2 independently fading channel taps, each 2D real/imag):
    z_{t+1} = A_z · z_t + v_z,  v_z ~ N(0, Q_z)

    A_z = block_diag(ρ₁·R(ω₁), ρ₂·R(ω₂))   where R(ω) is a 2×2 rotation matrix
    — models Jakes/Clarke fading: each tap decorrelates at rate ρ with Doppler shift ω

  Observation (linear in z, nonlinear in ξ):
    y_t = C · z_{t+1} + ξ_{t+1}²/20 + e_t,  e_t ~ N(0, R)

The RBPF advantage: SIS/SIR/APF must sample all 5 dimensions (1 ξ + 4 z).
The RBPF samples only ξ (1D) and runs a 4D Kalman filter per particle for z.
As z-dimension grows, the curse of dimensionality hits PF/APF but not RBPF.

References:
  - Gordon, Salmond, Smith (1993): nonlinear ξ dynamics
  - Schön, Gustafsson, Nordlund (2005): MLNLG/MPF framework
  - Hendeby, Karlsson, Gustafsson (2010): RBPF filter bank formulation
"""

from __future__ import annotations

import time
from typing import Any

import matplot2tikz
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import scipy.stats

import pyparticleest.filter as pfilter
from pyparticleest import interfaces
from pyparticleest.models import mlnlg


# ══════════════════════════════════════════════════════════════════════════════
# Model parameters
# ══════════════════════════════════════════════════════════════════════════════
MLNLG_Q_XI = 10.0  # nonlinear process noise variance

# 2 channel taps × (real, imag) = 4D linear sub-state
MLNLG_L = 4

# Per-tap fading: (decay rate ρ, Doppler angular frequency ω)
_TAP_PARAMS = [
    (0.95, 0.05),  # tap 1: slow fade, low Doppler
    (0.88, 0.15),  # tap 2: faster fade, higher Doppler
]
_blocks = [
    rho * np.array([[np.cos(omega), np.sin(omega)], [-np.sin(omega), np.cos(omega)]])
    for rho, omega in _TAP_PARAMS
]
MLNLG_AZ = sla.block_diag(*_blocks)  # (4×4)
MLNLG_QZ = 0.3 * np.eye(MLNLG_L)  # linear process noise
MLNLG_C = np.ones((1, MLNLG_L))  # observation: sum of all taps
MLNLG_R = 1.0  # measurement noise variance
MLNLG_XI0_VAR = 5.0  # initial ξ variance
MLNLG_Z0_COV = 0.5 * np.eye(MLNLG_L)  # initial z covariance


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth simulator
# ══════════════════════════════════════════════════════════════════════════════
def simulate_mlnlg(
    steps: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the MLNLG model and return (ξ, z, y) ground truth."""
    rng = np.random.default_rng(seed)
    L = MLNLG_L

    xis = np.empty(steps + 1)
    zs = np.empty((steps + 1, L))
    ys = np.empty(steps)

    xis[0] = rng.normal(0.0, np.sqrt(MLNLG_XI0_VAR))
    zs[0] = rng.multivariate_normal(np.zeros(L), MLNLG_Z0_COV)

    for t in range(steps):
        xis[t + 1] = (
            xis[t] / 2.0
            + 25.0 * xis[t] / (1.0 + xis[t] ** 2)
            + 8.0 * np.cos(1.2 * t)
            + rng.normal(0.0, np.sqrt(MLNLG_Q_XI))
        )
        zs[t + 1] = MLNLG_AZ @ zs[t] + rng.multivariate_normal(np.zeros(L), MLNLG_QZ)
        ys[t] = (
            (MLNLG_C @ zs[t + 1]).item()
            + xis[t + 1] ** 2 / 20.0
            + rng.normal(0.0, np.sqrt(MLNLG_R))
        )

    return xis, zs, ys


# ══════════════════════════════════════════════════════════════════════════════
# Joint-state PF model (SIS, SIR, APF all use this — samples all 5 dims)
# ══════════════════════════════════════════════════════════════════════════════
class MLNLGModelPF(interfaces.ParticleFiltering, interfaces.AuxiliaryParticleFiltering):
    """5D joint-state PF: particles = [ξ, z₁_re, z₁_im, z₂_re, z₂_im]."""

    def create_initial_estimate(self, N: int) -> np.ndarray:
        L = MLNLG_L
        particles = np.empty((N, 1 + L))
        particles[:, 0] = np.random.normal(0.0, np.sqrt(MLNLG_XI0_VAR), N)
        particles[:, 1:] = np.random.multivariate_normal(np.zeros(L), MLNLG_Z0_COV, N)
        return particles

    def sample_process_noise(self, particles, u, t) -> np.ndarray:
        N = len(particles)
        L = MLNLG_L
        noise = np.empty((N, 1 + L))
        noise[:, 0] = np.random.normal(0.0, np.sqrt(MLNLG_Q_XI), N)
        noise[:, 1:] = np.random.multivariate_normal(np.zeros(L), MLNLG_QZ, N)
        return noise

    def update(self, particles, u, t, noise) -> np.ndarray:
        xi = particles[:, 0]
        particles[:, 0] = (
            xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * np.cos(1.2 * t) + noise[:, 0]
        )
        particles[:, 1:] = (MLNLG_AZ @ particles[:, 1:].T).T + noise[:, 1:]
        return particles

    def measure(self, particles, y, t) -> np.ndarray:
        xi = particles[:, 0]
        z = particles[:, 1:]  # (N, L)
        y_hat = (MLNLG_C @ z.T).ravel() + xi**2 / 20.0
        return scipy.stats.norm.logpdf(float(y), loc=y_hat, scale=np.sqrt(MLNLG_R))

    def eval_1st_stage_weights(self, particles, u, y, t) -> np.ndarray:
        xi = particles[:, 0]
        z = particles[:, 1:]
        xi_pred = xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * np.cos(1.2 * t)
        z_pred = (MLNLG_AZ @ z.T).T
        y_hat = (MLNLG_C @ z_pred.T).ravel() + xi_pred**2 / 20.0
        pred_var = (
            MLNLG_R
            + (MLNLG_C @ MLNLG_QZ @ MLNLG_C.T).item()
            + MLNLG_Q_XI * (xi_pred / 10.0) ** 2
        )
        return scipy.stats.norm.logpdf(float(y), loc=y_hat, scale=np.sqrt(pred_var))


# ══════════════════════════════════════════════════════════════════════════════
# RBPF model (ξ sampled, z Kalman-filtered — samples only 1D)
# ══════════════════════════════════════════════════════════════════════════════
class MLNLGModelRBPF(mlnlg.MixedNLGaussianSampledInitialGaussian):
    """Rao-Blackwellized PF: ξ (1D) sampled, z (4D) analytically tracked."""

    def __init__(self, N: int) -> None:
        L = MLNLG_L
        super().__init__(
            xi0=np.array([[0.0]]),
            z0=np.zeros((L, 1)),
            Pxi0=np.array([[MLNLG_XI0_VAR]]),
            Pz0=np.copy(MLNLG_Z0_COV),
            Az=np.copy(MLNLG_AZ),
            C=np.copy(MLNLG_C),
            Qxi=np.array([[MLNLG_Q_XI]]),
            Qz=np.copy(MLNLG_QZ),
            R=np.array([[MLNLG_R]]),
        )

    def get_nonlin_pred_dynamics(self, particles, u, t):
        """ξ_{t+1} = f_nl(ξ_t, t) + v_ξ, no linear dependence on z."""
        xi = particles[:, 0]
        N = len(particles)
        L = MLNLG_L
        f = xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * np.cos(1.2 * t)
        fxi = f[:, np.newaxis, np.newaxis]  # (N, lxi=1, 1)
        Axi = np.zeros((N, 1, L))  # (N, lxi=1, lz=L) — ξ doesn't depend on z
        return (Axi, fxi, None)  # Qxi=None → uses default

    def get_meas_dynamics(self, y, particles, t):
        """y_t = C·z_t + ξ_t²/20 + e_t.  C is constant, h(ξ) varies."""
        xi = particles[:, 0]
        h = (xi**2 / 20.0)[:, np.newaxis, np.newaxis]  # (N, 1, 1)
        return (np.asarray(y).reshape((-1, 1)), None, h, None)

    # Stubs required by FFBSiRS — not used during forward filtering
    def logp_xnext(self, particles, next_part, u, t):
        return np.zeros(len(particles))

    def logp_xnext_max(self, particles, u, t):
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def weighted_means(
    straj: pfilter.ParticleTrajectory, state_index: int = 0
) -> np.ndarray:
    """Weighted-mean estimate for a given state index across time."""
    means = np.empty(len(straj))
    for k, step in enumerate(straj.traj):
        pa = step.pa
        w = pa.w - np.max(pa.w)
        w = np.exp(w)
        w /= np.sum(w)
        means[k] = np.dot(w, pa.part[:, state_index])
    return means


def weighted_means_z(
    straj: pfilter.ParticleTrajectory, lxi: int, z_index: int
) -> np.ndarray:
    """Weighted-mean z estimate from RBPF trajectory (z starts at index lxi)."""
    means = np.empty(len(straj))
    for k, step in enumerate(straj.traj):
        pa = step.pa
        w = pa.w - np.max(pa.w)
        w = np.exp(w)
        w /= np.sum(w)
        means[k] = np.dot(w, pa.part[:, lxi + z_index])
    return means


def mean_neff(straj: pfilter.ParticleTrajectory) -> float:
    neffs = [step.pa.calc_Neff() / step.pa.num for step in straj.traj]
    return float(np.mean(neffs))


def run_filter(
    model, filter_name: str, ys: np.ndarray, N: int, resample: float = 2.0 / 3.0
) -> tuple:
    """Run a named filter and return (straj, wall_time, resample_count, log_ml)."""
    straj = pfilter.ParticleTrajectory(
        model=model,
        N=N,
        resample=resample,
        filter=filter_name,
        T=len(ys) + 1,
        utype=float,
        ytype=float,
    )
    resample_count = 0
    log_ml = 0.0
    t0 = time.perf_counter()
    for t, y in enumerate(ys):
        resampled = straj.forward(u=float(t), y=float(y))
        if resampled:
            resample_count += 1
        pa = straj.traj[-1].pa
        w = pa.w - np.max(pa.w)
        log_ml += pa.w_offset + np.log(np.sum(np.exp(w)))
    wall = time.perf_counter() - t0
    return straj, wall, resample_count, log_ml


def rmse(estimates: np.ndarray, truth: np.ndarray) -> float:
    n = min(len(estimates), len(truth)) - 1
    return float(np.sqrt(np.mean((estimates[1 : n + 1] - truth[1 : n + 1]) ** 2)))


def rmse_aggregate(est_list: list[np.ndarray], truth_list: list[np.ndarray]) -> float:
    """Aggregate RMSE across multiple state dimensions."""
    total_sq = 0.0
    total_n = 0
    for est, truth in zip(est_list, truth_list):
        n = min(len(est), len(truth)) - 1
        total_sq += np.sum((est[1 : n + 1] - truth[1 : n + 1]) ** 2)
        total_n += n
    return float(np.sqrt(total_sq / total_n))


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {"SIS": "tab:blue", "SIR": "tab:orange", "APF": "tab:green", "RBPF": "tab:red"}
MARKERS = {"SIS": "P", "SIR": "*", "APF": "D", "RBPF": "X"}


def plot_xi_estimates(results, STEPS, xis):
    """All filters' ξ estimates on one figure."""
    t_axis = np.arange(STEPS + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_axis, xis, "k-", lw=1.5, label="Ground truth", zorder=5)
    for name, r in results.items():
        ax.plot(
            t_axis,
            r["est_xi"],
            "--",
            lw=1,
            color=COLORS[name],
            marker=MARKERS[name],
            markersize=4,
            markevery=5,
            label=name,
        )
    ax.set_title(r"MLNLG: $\xi$ estimate vs ground truth")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel(r"$\xi_t$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    matplot2tikz.save("plots/mlnlg_xi.tikz")
    plt.close(fig)


def plot_tap_magnitude(results, STEPS, zs, tap_idx: int):
    """Plot tap envelope |h| = sqrt(re² + im²) for one tap."""
    t_axis = np.arange(STEPS + 1)
    re, im = 2 * tap_idx, 2 * tap_idx + 1
    gt_mag = np.sqrt(zs[:, re] ** 2 + zs[:, im] ** 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_axis, gt_mag, "k-", lw=1.5, label="Ground truth", zorder=5)
    for name, r in results.items():
        est_re = r["est_z"][re]
        est_im = r["est_z"][im]
        est_mag = np.sqrt(est_re**2 + est_im**2)
        ax.plot(
            t_axis,
            est_mag,
            "--",
            lw=1,
            color=COLORS[name],
            marker=MARKERS[name],
            markersize=4,
            markevery=5,
            label=name,
        )
    ax.set_title(f"MLNLG: Tap {tap_idx + 1} envelope $|h_{tap_idx + 1}|$")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel(f"$|h_{tap_idx + 1}|$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    matplot2tikz.save(f"plots/mlnlg_tap{tap_idx + 1}_mag.tikz")
    plt.close(fig)


def plot_neff(results, strajs, STEPS):
    """Neff over time for all filters."""
    t_axis = np.arange(STEPS + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, straj in strajs:
        neffs = [step.pa.calc_Neff() / step.pa.num for step in straj.traj]
        ax.plot(t_axis, neffs, label=name, color=COLORS[name], lw=1)
    ax.axhline(2.0 / 3.0, color="k", ls=":", lw=1, label="Resample threshold")
    ax.set_title(r"Normalised effective sample size ($N_{\mathrm{eff}} / N$)")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel(r"$N_{\mathrm{eff}} / N$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    matplot2tikz.save("plots/mlnlg_neff.tikz")
    plt.close(fig)


def plot_rmse_over_time(results, xis, STEPS):
    """Cumulative RMSE of ξ over time."""
    t_axis = np.arange(1, STEPS + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in results.items():
        est = r["est_xi"]
        cum_rmse = np.sqrt(
            np.cumsum((est[1 : STEPS + 1] - xis[1 : STEPS + 1]) ** 2)
            / np.arange(1, STEPS + 1)
        )
        ax.plot(t_axis, cum_rmse, label=name, color=COLORS[name], lw=1)
    ax.set_title(r"Cumulative RMSE for $\xi$ estimates over time")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("RMSE")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    matplot2tikz.save("plots/mlnlg_xi_rmse_time.tikz")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    np.random.seed(42)
    STEPS = 100
    N = 1000  # particles
    L = MLNLG_L

    xis, zs, ys = simulate_mlnlg(STEPS, seed=42)

    results: dict[str, dict[str, Any]] = {}
    strajs: list[tuple[str, pfilter.ParticleTrajectory]] = []

    # ── SIS (PF with resample=0) ─────────────────────────────────────────────
    print("Running SIS...")
    pf_model = MLNLGModelPF()
    straj_sis, t_sis, res_sis, log_sis = run_filter(pf_model, "pf", ys, N, resample=0)
    est_xi_sis = weighted_means(straj_sis, state_index=0)
    est_z_sis = [weighted_means(straj_sis, state_index=1 + i) for i in range(L)]
    per_tap_sis = [
        rmse_aggregate(
            [est_z_sis[2 * t], est_z_sis[2 * t + 1]], [zs[:, 2 * t], zs[:, 2 * t + 1]]
        )
        for t in range(L // 2)
    ]
    results["SIS"] = {
        "rmse_xi": rmse(est_xi_sis, xis),
        "rmse_z_agg": rmse_aggregate(est_z_sis, [zs[:, i] for i in range(L)]),
        "rmse_tap": per_tap_sis,
        "neff": mean_neff(straj_sis),
        "time_s": t_sis,
        "resamples": res_sis,
        "log_ml": log_sis,
        "est_xi": est_xi_sis,
        "est_z": est_z_sis,
    }
    strajs.append(("SIS", straj_sis))

    # ── SIR (PF with resample=2/3) ───────────────────────────────────────────
    print("Running SIR...")
    pf_model = MLNLGModelPF()
    straj_sir, t_sir, res_sir, log_sir = run_filter(pf_model, "pf", ys, N)
    est_xi_sir = weighted_means(straj_sir, state_index=0)
    est_z_sir = [weighted_means(straj_sir, state_index=1 + i) for i in range(L)]
    per_tap_sir = [
        rmse_aggregate(
            [est_z_sir[2 * t], est_z_sir[2 * t + 1]], [zs[:, 2 * t], zs[:, 2 * t + 1]]
        )
        for t in range(L // 2)
    ]
    results["SIR"] = {
        "rmse_xi": rmse(est_xi_sir, xis),
        "rmse_z_agg": rmse_aggregate(est_z_sir, [zs[:, i] for i in range(L)]),
        "rmse_tap": per_tap_sir,
        "neff": mean_neff(straj_sir),
        "time_s": t_sir,
        "resamples": res_sir,
        "log_ml": log_sir,
        "est_xi": est_xi_sir,
        "est_z": est_z_sir,
    }
    strajs.append(("SIR", straj_sir))

    # ── APF ──────────────────────────────────────────────────────────────────
    print("Running APF...")
    apf_model = MLNLGModelPF()
    straj_apf, t_apf, res_apf, log_apf = run_filter(apf_model, "apf", ys, N)
    est_xi_apf = weighted_means(straj_apf, state_index=0)
    est_z_apf = [weighted_means(straj_apf, state_index=1 + i) for i in range(L)]
    per_tap_apf = [
        rmse_aggregate(
            [est_z_apf[2 * t], est_z_apf[2 * t + 1]], [zs[:, 2 * t], zs[:, 2 * t + 1]]
        )
        for t in range(L // 2)
    ]
    results["APF"] = {
        "rmse_xi": rmse(est_xi_apf, xis),
        "rmse_z_agg": rmse_aggregate(est_z_apf, [zs[:, i] for i in range(L)]),
        "rmse_tap": per_tap_apf,
        "neff": mean_neff(straj_apf),
        "time_s": t_apf,
        "resamples": res_apf,
        "log_ml": log_apf,
        "est_xi": est_xi_apf,
        "est_z": est_z_apf,
    }
    strajs.append(("APF", straj_apf))

    # ── RBPF (ξ sampled, z Kalman-filtered) ──────────────────────────────────
    print("Running RBPF...")
    rb_model = MLNLGModelRBPF(N)
    straj_rb, t_rb, res_rb, log_rb = run_filter(rb_model, "pf", ys, N)
    est_xi_rb = weighted_means(straj_rb, state_index=0)
    est_z_rb = [weighted_means_z(straj_rb, lxi=1, z_index=i) for i in range(L)]
    per_tap_rb = [
        rmse_aggregate(
            [est_z_rb[2 * t], est_z_rb[2 * t + 1]], [zs[:, 2 * t], zs[:, 2 * t + 1]]
        )
        for t in range(L // 2)
    ]
    results["RBPF"] = {
        "rmse_xi": rmse(est_xi_rb, xis),
        "rmse_z_agg": rmse_aggregate(est_z_rb, [zs[:, i] for i in range(L)]),
        "rmse_tap": per_tap_rb,
        "neff": mean_neff(straj_rb),
        "time_s": t_rb,
        "resamples": res_rb,
        "log_ml": log_rb,
        "est_xi": est_xi_rb,
        "est_z": est_z_rb,
    }
    strajs.append(("RBPF", straj_rb))

    # ── Print metrics ────────────────────────────────────────────────────────
    n_taps = L // 2
    tap_headers = "".join(
        f" {'RMSE(tap' + str(i + 1) + ')':>12}" for i in range(n_taps)
    )
    print(
        f"\n{'Filter':<8} {'RMSE(ξ)':>8} {'RMSE(z)':>8}{tap_headers}"
        f" {'Neff':>8} {'Time(s)':>8} {'Resamp':>7}"
    )
    print("-" * (55 + 13 * n_taps))
    for name, r in results.items():
        tap_vals = "".join(f" {t:>12.4f}" for t in r["rmse_tap"])
        print(
            f"{name:<8} {r['rmse_xi']:>8.4f} {r['rmse_z_agg']:>8.4f}{tap_vals}"
            f" {r['neff']:>8.4f} {r['time_s']:>8.4f} {r['resamples']:>7d}"
        )

    # ── Plots ────────────────────────────────────────────────────────────────
    plt.style.use("ggplot")
    plot_xi_estimates(results, STEPS, xis)
    for tap_idx in range(n_taps):
        plot_tap_magnitude(results, STEPS, zs, tap_idx)
    plot_neff(results, strajs, STEPS)
    plot_rmse_over_time(results, xis, STEPS)
    print(f"\nPlots saved to plots/")


if __name__ == "__main__":
    main()
