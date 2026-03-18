"""Benchmark: SIS, SIR, APF, and RBPF on a Mixed Linear/Nonlinear Gaussian SSM."""

from __future__ import annotations
import os
import time
import math
from typing import Any
import matplotlib
import latextable
import matplot2tikz
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import scipy.stats
from texttable import Texttable
import pyparticleest.filter as pfilter
from pyparticleest import interfaces
from pyparticleest.models import mlnlg

matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════════════════════
# Model parameters
# ══════════════════════════════════════════════════════════════════════════════
MLNLG_Q_XI = 10.0
MLNLG_L = 4

_TAP_PARAMS = [
    (0.95, 0.05),
    (0.88, 0.15),
]
_blocks = [
    rho * np.array([[np.cos(omega), np.sin(omega)], [-np.sin(omega), np.cos(omega)]])
    for rho, omega in _TAP_PARAMS
]
MLNLG_AZ = sla.block_diag(*_blocks)
MLNLG_QZ = 0.3 * np.eye(MLNLG_L)
MLNLG_C = np.ones((1, MLNLG_L))
MLNLG_R = 1.0
MLNLG_XI0_VAR = 5.0
MLNLG_Z0_COV = 0.5 * np.eye(MLNLG_L)


# ══════════════════════════════════════════════════════════════════════════════
# Ground-truth simulator
# ══════════════════════════════════════════════════════════════════════════════
def simulate_mlnlg(
    steps: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
# Joint-state PF model
# ══════════════════════════════════════════════════════════════════════════════
class MLNLGModelPF(interfaces.ParticleFiltering, interfaces.AuxiliaryParticleFiltering):
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
        z = particles[:, 1:]
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
# RBPF model
# ══════════════════════════════════════════════════════════════════════════════
class MLNLGModelRBPF(mlnlg.MixedNLGaussianSampledInitialGaussian):
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
        xi = particles[:, 0]
        N = len(particles)
        L = MLNLG_L
        f = xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * np.cos(1.2 * t)
        fxi = f[:, np.newaxis, np.newaxis]
        Axi = np.zeros((N, 1, L))
        return (Axi, fxi, None)

    def get_meas_dynamics(self, y, particles, t):
        xi = particles[:, 0]
        h = (xi**2 / 20.0)[:, np.newaxis, np.newaxis]
        return (np.asarray(y).reshape((-1, 1)), None, h, None)

    def logp_xnext(self, particles, next_part, u, t):
        return np.zeros(len(particles))

    def logp_xnext_max(self, particles, u, t):
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def weighted_means(
    straj: pfilter.ParticleTrajectory,
    state_index: int = 0,
) -> np.ndarray:
    means = np.empty(len(straj))
    for k, step in enumerate(straj.traj):
        pa = step.pa
        w = pa.w - np.max(pa.w)
        w = np.exp(w)
        w /= np.sum(w)
        means[k] = np.dot(w, pa.part[:, state_index])
    return means


def weighted_means_z(
    straj: pfilter.ParticleTrajectory,
    lxi: int,
    z_index: int,
) -> np.ndarray:
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
    model,
    filter_name: str,
    ys: np.ndarray,
    N: int,
    resample: float = 2.0 / 3.0,
) -> tuple:
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
    total_sq = 0.0
    total_n = 0
    for est, truth in zip(est_list, truth_list):
        n = min(len(est), len(truth)) - 1
        total_sq += np.sum((est[1 : n + 1] - truth[1 : n + 1]) ** 2)
        total_n += n
    return float(np.sqrt(total_sq / total_n))


def _make_filter_specs(N: int) -> list[tuple[str, Any, str, float]]:
    """Return a fresh list of (name, model, filter_type, resample) tuples.

    Models are stateful, so this must be called once per run.
    """
    return [
        ("SIS", MLNLGModelPF(), "pf", 0),
        ("SIR", MLNLGModelPF(), "pf", 2.0 / 3.0),
        ("APF", MLNLGModelPF(), "apf", 2.0 / 3.0),
        ("RBPF", MLNLGModelRBPF(N), "pf", 2.0 / 3.0),
    ]


def _run_one_seed(
    seed: int,
    N: int,
    STEPS: int,
    L: int,
) -> dict[str, dict[str, Any]]:
    """Run all filters on one data realisation; return scalar metrics only."""
    xis, zs, ys = simulate_mlnlg(STEPS, seed=seed)
    run_results: dict[str, dict[str, Any]] = {}

    for name, model, filter_type, resample in _make_filter_specs(N):
        straj, t_wall, resamples, log_ml = run_filter(
            model,
            filter_type,
            ys,
            N,
            resample=resample,
        )
        est_xi = weighted_means(straj, state_index=0)
        est_z = (
            [weighted_means_z(straj, lxi=1, z_index=i) for i in range(L)]
            if name == "RBPF"
            else [weighted_means(straj, state_index=1 + i) for i in range(L)]
        )
        per_tap_rmse = [
            rmse_aggregate(
                [est_z[2 * t], est_z[2 * t + 1]],
                [zs[:, 2 * t], zs[:, 2 * t + 1]],
            )
            for t in range(L // 2)
        ]
        run_results[name] = {
            "rmse_xi": rmse(est_xi, xis),
            "rmse_z_agg": rmse_aggregate(est_z, [zs[:, i] for i in range(L)]),
            "rmse_tap": per_tap_rmse,
            "neff": mean_neff(straj),
            "time_s": t_wall,
            "resamples": resamples,
            "log_ml": log_ml,
        }

    return run_results


# ══════════════════════════════════════════════════════════════════════════════
# Output Formatting
# ══════════════════════════════════════════════════════════════════════════════
def save_latex_table(results: dict[str, dict[str, Any]], n_taps: int) -> None:
    filepath = "plots/benchmark_results_table.tex"

    table = Texttable()
    table.set_deco(Texttable.HEADER)

    headers = ["Filter", "RMSE($\\xi$)", "RMSE($z$)"]
    headers.extend([f"RMSE(Tap {i + 1})" for i in range(n_taps)])
    headers.extend(["$N_{\\mathrm{eff}}$", "Time (s)", "Resamples", "Log ML"])

    table.header(headers)

    def pm(mean_std: tuple[float, float], fmt: str = ".4f") -> str:
        m, s = mean_std
        return f"{m:{fmt}} $\\pm$ {s:{fmt}}"

    for name, r in results.items():
        row = [name, pm(r["rmse_xi"]), pm(r["rmse_z_agg"])]
        row.extend([pm(tap) for tap in r["rmse_tap"]])
        row.extend(
            [
                pm(r["neff"]),
                pm(r["time_s"]),
                pm(r["resamples"], fmt=".1f"),
                pm(r["log_ml"]),
            ],
        )
        table.add_row(row)

    latex_output = latextable.draw_latex(
        table,
        caption="Filter Benchmark Results on MLNLG Model (mean $\\pm$ std, $N=100$ runs).",
        label="tab:benchmark_results",
        position="htbp",
    )

    with open(filepath, "w") as f:
        f.write(latex_output)
    print(f"LaTeX table saved to {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {"SIS": "#1f77b4", "SIR": "#ff7f0e", "APF": "#2ca02c", "RBPF": "#d62728"}
MARKERS = {"SIS": "P", "SIR": "*", "APF": "D", "RBPF": "X"}


def setup_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.6, color="#b0b0b0")
    ax.set_facecolor("white")


def finalize_plot(fig, ax, filename):
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.04, 0.5),
        fontsize=11,
        frameon=True,
        facecolor="white",
        edgecolor="black",
    )

    png_path = f"plots/{filename}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")

    tex_path = f"plots/{filename}.tikz"
    matplot2tikz.save(tex_path, strict=True)

    plt.close(fig)


def plot_tap_magnitude(results, STEPS, zs, tap_idx: int):
    t_axis = np.arange(STEPS + 1)
    re, im = 2 * tap_idx, 2 * tap_idx + 1
    gt_mag = np.sqrt(zs[:, re] ** 2 + zs[:, im] ** 2)

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()

    ax.plot(t_axis, gt_mag, "k-", lw=2, label="Ground truth", zorder=5)

    for name, r in results.items():
        est_mag = np.sqrt(r["est_z"][re] ** 2 + r["est_z"][im] ** 2)
        ax.plot(
            t_axis,
            est_mag,
            "--",
            lw=1.5,
            color=COLORS[name],
            marker=MARKERS[name],
            markersize=6,
            markevery=5,
            label=name,
        )

    setup_plot(
        ax,
        f"MLNLG: Tap {tap_idx + 1} Envelope $|h_{tap_idx + 1}|$",
        "Time step $t$",
        f"$|h_{tap_idx + 1}|$",
    )
    finalize_plot(fig, ax, f"mlnlg_tap{tap_idx + 1}_mag")


def plot_neff(results, strajs, STEPS):
    t_axis = np.arange(STEPS + 1)
    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()

    for name, straj in strajs:
        neffs = [step.pa.calc_Neff() / step.pa.num for step in straj.traj]
        ax.plot(t_axis, neffs, label=name, color=COLORS[name], lw=1.5)

    ax.axhline(2.0 / 3.0, color="k", ls=":", lw=2, label="Resample threshold")
    setup_plot(
        ax,
        r"Normalized Effective Sample Size ($N_{\mathrm{eff}} / N$)",
        "Time step $t$",
        r"$N_{\mathrm{eff}} / N$",
    )
    finalize_plot(fig, ax, "mlnlg_neff")


def plot_z_rmse_over_time(results, zs, STEPS):
    """Cumulative RMSE of the aggregate channel state z over time."""
    t_axis = np.arange(1, STEPS + 1)
    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()

    L = zs.shape[1]  # usually 4

    for name, r in results.items():
        est_z = r["est_z"]  # List of L arrays

        # Calculate squared error across all L dimensions at each time step
        sq_err = np.zeros(STEPS)
        for i in range(L):
            sq_err += (est_z[i][1 : STEPS + 1] - zs[1 : STEPS + 1, i]) ** 2

        # Cumulative mean across time AND dimensions
        cum_rmse = np.sqrt(np.cumsum(sq_err) / (np.arange(1, STEPS + 1) * L))

        ax.plot(t_axis, cum_rmse, label=name, color=COLORS[name], lw=1.5)

    setup_plot(
        ax,
        r"Cumulative RMSE for Channel State ($z$) Over Time",
        "Time step $t$",
        "RMSE",
    )
    finalize_plot(fig, ax, "mlnlg_z_rmse_time")


def plot_z_component_estimate(results, STEPS, zs, z_idx: int, component_name: str):
    """Plot a single component of the z state (e.g., Tap 1 Real)."""
    t_axis = np.arange(STEPS + 1)
    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()

    ax.plot(t_axis, zs[:, z_idx], "k-", lw=2, label="Ground truth", zorder=5)

    for name, r in results.items():
        ax.plot(
            t_axis,
            r["est_z"][z_idx],
            "--",
            lw=1.5,
            color=COLORS[name],
            marker=MARKERS[name],
            markersize=6,
            markevery=5,
            label=name,
        )

    setup_plot(
        ax,
        f"MLNLG: {component_name} Estimate vs Ground Truth",
        "Time step $t$",
        f"$z_{{{z_idx}, t}}$",
    )
    finalize_plot(fig, ax, f"mlnlg_z_comp_{z_idx}")


def plot_z_component_estimate_individual(
    name,
    r,
    STEPS,
    zs,
    z_idx: int,
    component_name: str,
):
    """Plot an individual filter's single z state component against ground truth."""
    t_axis = np.arange(STEPS + 1)
    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()

    # Plot ground truth
    ax.plot(t_axis, zs[:, z_idx], "k-", lw=2, label="Ground truth", zorder=5)

    # Plot the specific filter's estimate
    ax.plot(
        t_axis,
        r["est_z"][z_idx],
        "--",
        lw=1.5,
        color=COLORS.get(name, "#1f77b4"),
        marker=MARKERS.get(name, "o"),
        markersize=6,
        markevery=5,
        label=name,
    )

    setup_plot(
        ax,
        f"MLNLG: {name} {component_name} Estimate vs Ground Truth",
        "Time step $t$",
        f"$z_{{{z_idx}, t}}$",
    )
    finalize_plot(fig, ax, f"mlnlg_z_comp_{z_idx}_{name.lower()}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    os.makedirs("plots", exist_ok=True)

    STEPS = 100
    N = 1000
    L = MLNLG_L
    N_RUNS = 100
    PLOT_SEED = 42

    # ── Single representative run for plots ──────────────────────────────────
    print(f"Running representative run (seed={PLOT_SEED}) for plots...")
    np.random.seed(PLOT_SEED)
    xis, zs, ys = simulate_mlnlg(STEPS, seed=PLOT_SEED)

    plot_results: dict[str, dict[str, Any]] = {}
    strajs: list[tuple[str, pfilter.ParticleTrajectory]] = []

    for name, model, filter_type, resample in _make_filter_specs(N):
        straj, t_wall, resamples, log_ml = run_filter(
            model,
            filter_type,
            ys,
            N,
            resample=resample,
        )
        est_xi = weighted_means(straj, state_index=0)
        est_z = (
            [weighted_means_z(straj, lxi=1, z_index=i) for i in range(L)]
            if name == "RBPF"
            else [weighted_means(straj, state_index=1 + i) for i in range(L)]
        )
        per_tap_rmse = [
            rmse_aggregate(
                [est_z[2 * t], est_z[2 * t + 1]],
                [zs[:, 2 * t], zs[:, 2 * t + 1]],
            )
            for t in range(L // 2)
        ]
        plot_results[name] = {
            "rmse_xi": rmse(est_xi, xis),
            "rmse_z_agg": rmse_aggregate(est_z, [zs[:, i] for i in range(L)]),
            "rmse_tap": per_tap_rmse,
            "neff": mean_neff(straj),
            "time_s": t_wall,
            "resamples": resamples,
            "log_ml": log_ml,
            "est_xi": est_xi,
            "est_z": est_z,
        }
        strajs.append((name, straj))

    # ── Multi-run averaging for table ────────────────────────────────────────
    filter_names = [name for name, *_ in _make_filter_specs(N)]
    n_taps = L // 2

    # accumulators: sum of each scalar metric across runs and sum-of-squares
    acc: dict[str, dict[str, Any]] = {
        name: {
            "rmse_xi": {"s": 0.0, "s2": 0.0},
            "rmse_z_agg": {"s": 0.0, "s2": 0.0},
            "rmse_tap": [{"s": 0.0, "s2": 0.0} for _ in range(n_taps)],
            "neff": {"s": 0.0, "s2": 0.0},
            "time_s": {"s": 0.0, "s2": 0.0},
            "resamples": {"s": 0.0, "s2": 0.0},
            "log_ml": {"s": 0.0, "s2": 0.0},
        }
        for name in filter_names
    }

    seeds = range(N_RUNS)
    for i, seed in enumerate(seeds):
        print(
            f"  Averaging run {i + 1}/{N_RUNS} (seed={seed})...", end="\r", flush=True
        )
        run = _run_one_seed(seed, N, STEPS, L)
        for name in filter_names:
            r = run[name]
            a = acc[name]
            for key in (
                "rmse_xi",
                "rmse_z_agg",
                "neff",
                "time_s",
                "resamples",
                "log_ml",
            ):
                v = r[key]
                a[key]["s"] += v
                a[key]["s2"] += v * v
            for t in range(n_taps):
                v = r["rmse_tap"][t]
                a["rmse_tap"][t]["s"] += v
                a["rmse_tap"][t]["s2"] += v * v
    print()  # newline after \r progress

    # divide accumulators by N_RUNS to get means
    def _mean_std(bucket: dict[str, float]) -> tuple[float, float]:
        mean = bucket["s"] / N_RUNS
        # sample variance: E[x^2] - (E[x])^2, corrected with N/(N-1)
        var = (bucket["s2"] / N_RUNS - mean * mean) * N_RUNS / (N_RUNS - 1)
        return mean, math.sqrt(max(var, 0.0))

    avg_results: dict[str, dict[str, Any]] = {
        name: {
            "rmse_xi": _mean_std(acc[name]["rmse_xi"]),
            "rmse_z_agg": _mean_std(acc[name]["rmse_z_agg"]),
            "rmse_tap": [_mean_std(acc[name]["rmse_tap"][t]) for t in range(n_taps)],
            "neff": _mean_std(acc[name]["neff"]),
            "time_s": _mean_std(acc[name]["time_s"]),
            "resamples": _mean_std(acc[name]["resamples"]),
            "log_ml": _mean_std(acc[name]["log_ml"]),
        }
        for name in filter_names
    }

    # ── Console Output ───────────────────────────────────────────────────────
    n_taps = L // 2
    tap_headers = "".join(
        f" {'RMSE(tap' + str(i + 1) + ')':>12}" for i in range(n_taps)
    )
    print(
        f"\n{'Filter':<8} {'RMSE(ξ)':>8} {'RMSE(z)':>8}{tap_headers} {'Neff':>8} {'Time(s)':>8} {'Resamp':>7} {'Log ML':>12}",
    )
    print("-" * (68 + 13 * n_taps))
    for name, r in avg_results.items():
        tap_vals = "".join(f" {m:>8.4f}±{s:<6.4f}" for m, s in r["rmse_tap"])
        rmse_xi_m, rmse_xi_s = r["rmse_xi"]
        rmse_z_m, rmse_z_s = r["rmse_z_agg"]
        neff_m, _ = r["neff"]
        time_m, time_s = r["time_s"]
        resamples_m, _ = r["resamples"]
        log_ml_m, log_ml_s = r["log_ml"]
        print(
            f"{name:<8} {rmse_xi_m:>7.4f}±{rmse_xi_s:<6.4f} "
            f"{rmse_z_m:>7.4f}±{rmse_z_s:<6.4f}"
            f"{tap_vals} "
            f"{neff_m:>8.4f} {time_m:>7.4f}±{time_s:<5.4f} "
            f"{resamples_m:>7.1f} {log_ml_m:>11.4f}±{log_ml_s:<8.4f}",
        )

    # ── Save Outputs ─────────────────────────────────────────────────────────
    plt.style.use("default")

    save_latex_table(avg_results, n_taps)

    # all plots from the single representative run
    plot_z_component_estimate(plot_results, STEPS, zs, z_idx=0, component_name="Tap 1")
    for name, r in plot_results.items():
        plot_z_component_estimate_individual(
            name,
            r,
            STEPS,
            zs,
            z_idx=0,
            component_name="Tap 1 (Real)",
        )
    for tap_idx in range(n_taps):
        plot_tap_magnitude(plot_results, STEPS, zs, tap_idx)
    plot_neff(plot_results, strajs, STEPS)
    plot_z_rmse_over_time(plot_results, zs, STEPS)
    print("\nPlots saved to plots/")


if __name__ == "__main__":
    main()
