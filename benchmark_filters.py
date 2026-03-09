"""Benchmark: SIS, SIR/PF, APF, and RBPF on a non-linear non-Gaussian SSM.

Model (Gordon-Salmond-Smith 1993, stochastic volatility variant):
    x_{t+1} = x_t/2 + 25*x_t/(1+x_t^2) + 8*cos(1.2*t) + v_t,  v_t ~ N(0, Q)
    y_t     = x_t^2/20 + e_t,                                    e_t ~ N(0, R)

For the RBPF the state is written as:
    xi_{t+1} = f_nl(xi_t, t) + v_xi,   (non-linear part, sampled)
    z_{t+1}  = A*z_t + f_z + v_z,      (linear part, Kalman-updated)
    y_t      = C*z_t + h(xi_t) + e_t

We let xi = x (the full state is non-linear) and z be a dummy scalar = 0
kept analytically, so the RBPF degenerates gracefully to a standard PF
while still exercising the RB code path.

Metrics reported per filter:
  - RMSE against ground truth
  - Mean Neff (normalised effective sample size)
  - Wall-clock time
  - Resampling count
"""

import time
from typing import Any

import latextable
import matplot2tikz
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from texttable import Texttable

import pyparticleest.filter as pfilter
from pyparticleest import interfaces
from pyparticleest.models import mlnlg

# ── Model parameters ─────────────────────────────────────────────────────────
Q = 10.0  # process noise variance
R = 1.0  # measurement noise variance
x0_mean = 0.0
x0_var = 5.0


# ── Ground-truth simulator ────────────────────────────────────────────────────
def simulate(steps: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = np.empty(steps + 1)
    ys = np.empty(steps)
    xs[0] = rng.normal(x0_mean, np.sqrt(x0_var))
    for t in range(steps):
        xs[t + 1] = (
            xs[t] / 2.0
            + 25.0 * xs[t] / (1.0 + xs[t] ** 2)
            + 8.0 * np.cos(1.2 * t)
            + rng.normal(0.0, np.sqrt(Q))
        )
        ys[t] = xs[t + 1] ** 2 / 20.0 + rng.normal(0.0, np.sqrt(R))
    return xs, ys


# ── MLNLG model parameters ───────────────────────────────────────────────────
MLNLG_Q_XI = 10.0
MLNLG_AZ = np.array([[0.9, 0.1], [-0.1, 0.85]])
MLNLG_QZ = 0.5 * np.eye(2)
MLNLG_C = np.array([[1.0, 1.0]])
MLNLG_R = 1.0
MLNLG_XI0_VAR = 5.0
MLNLG_Z0_COV = 0.5 * np.eye(2)


def simulate_mlnlg(
    steps: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a mixed linear/nonlinear Gaussian SSM.

    ξ_{t+1} = f_nl(ξ_t, t) + v_ξ       (nonlinear, same as Gordon–Salmond–Smith)
    z_{t+1} = Az·z_t + v_z              (2D linear, damped oscillator)
    y_t     = C·z_{t+1} + ξ_{t+1}²/20 + e_t
    """
    rng = np.random.default_rng(seed)

    xis = np.empty(steps + 1)
    zs = np.empty((steps + 1, 2))
    ys = np.empty(steps)

    xis[0] = rng.normal(0.0, np.sqrt(MLNLG_XI0_VAR))
    zs[0] = rng.multivariate_normal([0.0, 0.0], MLNLG_Z0_COV)

    for t in range(steps):
        xis[t + 1] = (
            xis[t] / 2.0
            + 25.0 * xis[t] / (1.0 + xis[t] ** 2)
            + 8.0 * np.cos(1.2 * t)
            + rng.normal(0.0, np.sqrt(MLNLG_Q_XI))
        )
        zs[t + 1] = MLNLG_AZ @ zs[t] + rng.multivariate_normal([0.0, 0.0], MLNLG_QZ)
        ys[t] = (
            float(MLNLG_C @ zs[t + 1])
            + xis[t + 1] ** 2 / 20.0
            + rng.normal(0.0, np.sqrt(MLNLG_R))
        )

    return xis, zs, ys


# ── Helper: extract weighted mean from a ParticleTrajectory ──────────────────
def weighted_means(
    straj: pfilter.ParticleTrajectory,
    state_index: int = 0,
) -> np.ndarray:
    """Return array of weighted-mean estimates for state_index across time."""
    means = np.empty(len(straj))
    for k, step in enumerate(straj.traj):
        pa = step.pa
        w = pa.w - np.max(pa.w)
        w = np.exp(w)
        w /= np.sum(w)
        means[k] = np.dot(w, pa.part[:, state_index])
    return means


def mean_neff(straj: pfilter.ParticleTrajectory) -> float:
    neffs = [step.pa.calc_Neff() / step.pa.num for step in straj.traj]
    return float(np.mean(neffs))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Standard PF model  (ParticleFiltering + AuxiliaryParticleFiltering)
#     Particles = [x_t]  (scalar, shape (N,1))
# ══════════════════════════════════════════════════════════════════════════════
class NLGSSModel(interfaces.ParticleFiltering, interfaces.AuxiliaryParticleFiltering):
    """Non-linear Gaussian SSM for use with PF and APF.

    Particle state: 1-D np array [x_t].
    """

    def create_initial_estimate(self, N: int) -> np.ndarray:
        return np.random.normal(x0_mean, np.sqrt(x0_var), size=(N, 1))

    def sample_process_noise(
        self,
        particles: np.ndarray,
        u: Any | None,
        t: int,
    ) -> np.ndarray:
        N = len(particles)
        return np.random.normal(0.0, np.sqrt(Q), size=(N, 1))

    def update(
        self,
        particles: np.ndarray,
        u: Any | None,
        t: int,
        noise: np.ndarray,
    ) -> np.ndarray:
        x = particles[:, 0]
        x_next = x / 2.0 + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * t) + noise[:, 0]
        particles[:, 0] = x_next
        return particles

    def measure(self, particles: np.ndarray, y: float, t: int) -> np.ndarray:
        x = particles[:, 0]
        y_hat = x**2 / 20.0
        return scipy.stats.norm.logpdf(float(y), loc=y_hat, scale=np.sqrt(R))

    # ── APF first-stage weights: use propagated mean as predictor ────────────
    def eval_1st_stage_weights(
        self,
        particles: np.ndarray,
        u: Any | None,
        y: float,
        t: int,
    ) -> np.ndarray:
        x = particles[:, 0]
        x_pred = x / 2.0 + 25.0 * x / (1.0 + x**2) + 8.0 * np.cos(1.2 * t)
        y_hat = x_pred**2 / 20.0
        return scipy.stats.norm.logpdf(
            float(y),
            loc=y_hat,
            scale=np.sqrt(R + Q * (x_pred / 10.0) ** 2),
        )


class MLNLGModelPF(interfaces.ParticleFiltering, interfaces.AuxiliaryParticleFiltering):
    """3D joint-state PF for the MLNLG model.

    Particles = [ξ, z1, z2] (shape (N, 3)).
    The PF must sample all three state dimensions — no Rao-Blackwellization.
    """

    def create_initial_estimate(self, N: int) -> np.ndarray:
        particles = np.empty((N, 3))
        particles[:, 0] = np.random.normal(0.0, np.sqrt(MLNLG_XI0_VAR), N)
        particles[:, 1:] = np.random.multivariate_normal([0.0, 0.0], MLNLG_Z0_COV, N)
        return particles

    def sample_process_noise(
        self,
        particles: np.ndarray,
        u: Any | None,
        t: int,
    ) -> np.ndarray:
        N = len(particles)
        noise = np.empty((N, 3))
        noise[:, 0] = np.random.normal(0.0, np.sqrt(MLNLG_Q_XI), N)
        noise[:, 1:] = np.random.multivariate_normal([0.0, 0.0], MLNLG_QZ, N)
        return noise

    def update(
        self,
        particles: np.ndarray,
        u: Any | None,
        t: int,
        noise: np.ndarray,
    ) -> np.ndarray:
        xi = particles[:, 0]
        particles[:, 0] = (
            xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * np.cos(1.2 * t) + noise[:, 0]
        )
        particles[:, 1:] = (MLNLG_AZ @ particles[:, 1:].T).T + noise[:, 1:]
        return particles

    def measure(self, particles: np.ndarray, y: float, t: int) -> np.ndarray:
        xi = particles[:, 0]
        z = particles[:, 1:]  # (N, 2)
        y_hat = (MLNLG_C @ z.T).ravel() + xi**2 / 20.0
        return scipy.stats.norm.logpdf(float(y), loc=y_hat, scale=np.sqrt(MLNLG_R))

    def eval_1st_stage_weights(
        self,
        particles: np.ndarray,
        u: Any | None,
        y: float,
        t: int,
    ) -> np.ndarray:
        xi = particles[:, 0]
        z = particles[:, 1:]
        xi_pred = xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * np.cos(1.2 * t)
        z_pred = (MLNLG_AZ @ z.T).T
        y_hat = (MLNLG_C @ z_pred.T).ravel() + xi_pred**2 / 20.0
        # Predictive variance: R + C·Qz·Cᵀ + nonlinear ξ contribution
        pred_var = (
            MLNLG_R
            + float(MLNLG_C @ MLNLG_QZ @ MLNLG_C.T)
            + MLNLG_Q_XI * (xi_pred / 10.0) ** 2
        )
        return scipy.stats.norm.logpdf(float(y), loc=y_hat, scale=np.sqrt(pred_var))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RBPF model  (MixedNLGaussianSampledInitialGaussian)
#     xi_t = x_t  (non-linear, sampled)
#     z_t  = dummy scalar 0  (linear, Kalman)
#     y_t  = C*z_t + h(xi_t) + e_t  with C=0, h=xi^2/20
# ══════════════════════════════════════════════════════════════════════════════
class RBPFModel(mlnlg.MixedNLGaussianSampledInitialGaussian):
    """Rao-Blackwellized particle filter for the same NLGSSM.

    The 'linear' sub-state z is a dummy scalar kept at 0 with near-zero
    covariance so that the RBPF collapses to standard PF behaviour, while
    still exercising the full RB code path (Kalman measurement update,
    `get_nonlin_pred_dynamics`, `get_meas_dynamics`).
    """

    def __init__(self, N: int) -> None:
        # xi: non-linear scalar state (the actual x_t)
        # z:  linear scalar dummy = 0
        xi0 = np.array([[x0_mean]])
        z0 = np.array([[0.0]])
        Pxi0 = np.array([[x0_var]])
        Pz0 = np.array([[1e-6]])  # near-zero: z is deterministic
        Qxi = np.array([[Q]])
        Qz = np.array([[1e-10]])  # z does not evolve
        Az = np.array([[1.0]])  # z_{t+1} = z_t (trivial)
        C = np.array([[0.0]])  # measurement does not depend on z
        super().__init__(
            z0=z0,
            xi0=xi0,
            Pz0=Pz0,
            Pxi0=Pxi0,
            Az=Az,
            C=C,
            Qxi=Qxi,
            Qz=Qz,
            R=np.array([[R]]),
        )

    # xi_{t+1} = f(xi_t, t) + v_xi
    def get_nonlin_pred_dynamics(
        self,
        particles: np.ndarray,
        u: Any | None,
        t: int,
    ) -> tuple[np.ndarray, np.ndarray, Any | None]:
        xi = particles[:, 0]  # shape (N,)
        N = len(particles)
        f = xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * np.cos(1.2 * t)
        # fxi shape must be (N, lxi, 1)
        fxi = f[:, np.newaxis, np.newaxis]
        # Axi=None means xi_{t+1} has no linear dependence on z
        return (np.zeros((N, 1, 1)), fxi, None)

    # y_t = h(xi_t) + C*z_t + e_t  with C=0, h = xi^2/20
    def get_meas_dynamics(
        self,
        y: float,
        particles: np.ndarray,
        t: int,
    ) -> tuple[np.ndarray, Any | None, np.ndarray, Any | None]:
        xi = particles[:, 0]
        h = (xi**2 / 20.0)[:, np.newaxis, np.newaxis]
        # C=None reuses the stored C=[[0]]
        return (np.asarray(y).reshape((-1, 1)), None, h, None)


def weighted_means_z(
    straj: pfilter.ParticleTrajectory,
    lxi: int,
    z_index: int,
) -> np.ndarray:
    """Return weighted-mean z estimates from RBPF trajectory.

    In the RBPF particle array: [ξ₀..ξ_{lxi-1}, z₀..z_{lz-1}, P_flat...].
    The Kalman-filtered z means start at index lxi.
    """
    means = np.empty(len(straj))
    for k, step in enumerate(straj.traj):
        pa = step.pa
        w = pa.w - np.max(pa.w)
        w = np.exp(w)
        w /= np.sum(w)
        means[k] = np.dot(w, pa.part[:, lxi + z_index])
    return means


class MLNLGModelRBPF(mlnlg.MixedNLGaussianSampledInitialGaussian):
    """Rao-Blackwellized PF for the MLNLG benchmark.

    ξ (scalar, nonlinear) is sampled by particles.
    z (2D, linear|ξ) is analytically Kalman-filtered per particle.
    The RBPF only needs to sample in 1D instead of 3D.
    """

    def __init__(self, N: int) -> None:
        super().__init__(
            xi0=np.array([[0.0]]),
            z0=np.array([[0.0], [0.0]]),
            Pxi0=np.array([[MLNLG_XI0_VAR]]),
            Pz0=np.copy(MLNLG_Z0_COV),
            Az=np.copy(MLNLG_AZ),
            C=np.copy(MLNLG_C),
            Qxi=np.array([[MLNLG_Q_XI]]),
            Qz=np.copy(MLNLG_QZ),
            R=np.array([[MLNLG_R]]),
        )

    def get_nonlin_pred_dynamics(
        self,
        particles: np.ndarray,
        u: Any | None,
        t: int,
    ) -> tuple[np.ndarray, np.ndarray, Any | None]:
        """ξ_{t+1} = f_nl(ξ_t, t) + v_ξ, no linear dependence on z."""
        xi = particles[:, 0]
        N = len(particles)
        f = xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * np.cos(1.2 * t)
        fxi = f[:, np.newaxis, np.newaxis]  # (N, lxi=1, 1)
        Axi = np.zeros((N, 1, 2))  # (N, lxi=1, lz=2) — ξ doesn't depend on z
        return (Axi, fxi, None)  # Qxi=None → uses default

    def get_meas_dynamics(
        self,
        y: float,
        particles: np.ndarray,
        t: int,
    ) -> tuple[np.ndarray, Any | None, np.ndarray, Any | None]:
        """y_t = C·z_t + ξ_t²/20 + e_t.  C is constant, h(ξ) varies."""
        xi = particles[:, 0]
        h = (xi**2 / 20.0)[:, np.newaxis, np.newaxis]  # (N, 1, 1)
        return (np.asarray(y).reshape((-1, 1)), None, h, None)
        #       y preprocessed,                     C=default, h=per-particle, R=default


# ══════════════════════════════════════════════════════════════════════════════
# Runner helpers
# ══════════════════════════════════════════════════════════════════════════════
def run_filter(
    model: Any,
    filter_name: str,
    ys: np.ndarray,
    us: Any | None,
    N: int,
    resample: float = 2.0 / 3.0,
) -> tuple[pfilter.ParticleTrajectory, float, int, float]:
    """Run a named filter and return (straj, wall_time, resample_count)."""
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
    log_ml = 0.0  # log marginal likelihood, aka model evidence
    t0 = time.perf_counter()
    for t, y in enumerate(ys):
        resampled = straj.forward(u=float(t), y=float(y))
        if resampled:
            resample_count += 1
        # Accumulate log p(y_t | y_{1:t-1}) ≈ log(sum(w_unnorm)) - log(N)
        pa = straj.traj[-1].pa
        w = pa.w - np.max(pa.w)
        log_ml += pa.w_offset + np.log(np.sum(np.exp(w)))
    wall = time.perf_counter() - t0
    return straj, wall, resample_count, log_ml


def extract_state(straj: pfilter.ParticleTrajectory, rbpf: bool = False) -> np.ndarray:
    """Return (T,) array of weighted-mean state estimates."""
    if rbpf:
        # For RBPF, xi is the first element of the particle
        return weighted_means(straj, state_index=0)
    return weighted_means(straj, state_index=0)


def rmse(estimates: np.ndarray, truth: np.ndarray) -> float:
    # estimates has length T+1 (includes t=0), truth has length T+1
    # align: skip t=0 (prior), compare t=1..T
    n = min(len(estimates), len(truth)) - 1
    return float(np.sqrt(np.mean((estimates[1 : n + 1] - truth[1 : n + 1]) ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# Metrics and plots
# ══════════════════════════════════════════════════════════════════════════════
def print_metrics(results: dict[str, dict[str, Any]]) -> None:
    print(
        f"\n{'Filter':<20} {'RMSE':>8} {'Mean Neff':>10} {'Time (s)':>10} {'Resamples':>10}",
    )
    print("-" * 62)
    for name, r in results.items():
        print(
            f"{name:<20} {r['rmse']:>8.4f} {r['neff']:>10.4f} "
            f"{r['time_s']:>10.4f} {r['resamples']:>10d}",
        )


def print_latex_table_standard(results: dict[str, dict[str, Any]]) -> None:
    """Generate and print a LaTeX table for the standard benchmark results using latextable."""
    print("\n% --- LaTeX Table: Standard Benchmark ---")
    table = Texttable()
    table.set_cols_align(["l", "r", "r", "r", "r", "r"])
    rows = [
        [
            "Filter",
            "RMSE",
            "Mean $N_{\\mathrm{eff}}$",
            "Time (s)",
            "Resamples",
            "Log ML",
        ],
    ]

    for name, r in results.items():
        rows.append(
            [
                name,
                f"{r['rmse']:.4f}",
                f"{r['neff']:.4f}",
                f"{r['time_s']:.4f}",
                r["resamples"],
                f"{r['log_ml']:.4f}",
            ],
        )

    table.add_rows(rows)
    print(
        latextable.draw_latex(
            table,
            caption="Standard Non-linear Non-Gaussian SSM Benchmark Results",
            label="tab:standard_benchmark",
        ),
    )


colors: dict[str, str] = {
    "SIS": "tab:blue",
    "SIR": "tab:orange",
    "APF": "tab:green",
    "RBPF": "tab:red",
}
markers: dict[str, str] = {
    "SIS": "P",
    "SIR": "*",
    "APF": "D",
    "RBPF": "X",
}


def plot_individual_estimates(
    results: dict[str, dict[str, Any]],
    STEPS: int,
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    """One figure per algorithm, each with ground truth overlaid."""
    t_axis = np.arange(STEPS + 1)
    plt.style.use("ggplot")

    for name, r in results.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_axis, xs, "k-", lw=1.5, label="Ground truth", zorder=5)
        ax.plot(
            t_axis,
            r["estimates"],
            "--",
            lw=1,
            color=colors[name],
            label=name,
        )
        ax.scatter(
            t_axis[1:],
            ys,
            s=6,
            color="gray",
            alpha=0.3,
            label="Observations",
            zorder=1,
        )
        ax.set_title(f"{name}: State estimate vs ground truth")
        ax.set_xlabel("Time step $t$")
        ax.set_ylabel("$x_t$")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(f"plots/estimate_{safe_name}.png", dpi=150)
        matplot2tikz.save(f"plots/estimate_{safe_name}.tikz")
        plt.close(fig)


def plot_combined_estimates(
    results: dict[str, dict[str, Any]],
    STEPS: int,
    xs: np.ndarray,
    ys: np.ndarray,
) -> None:
    """All algorithms on one figure for comparison."""
    t_axis = np.arange(STEPS + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_axis, xs, "k-", lw=1.5, label="Ground truth", zorder=5)
    for name, r in results.items():
        ax.plot(
            t_axis,
            r["estimates"],
            "--",
            lw=1,
            color=colors[name],
            label=name,
            marker=markers[name],
            markersize=4.0,
        )
    ax.set_title("All: State estimate vs ground truth")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("$x_t$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/benchmark_filters_combined.png", dpi=150)
    matplot2tikz.save("plots/benchmark_filters_combined.tikz")
    plt.close(fig)


def plot_neff(
    results: dict[str, dict[str, Any]],
    strajs: list[tuple[str, pfilter.ParticleTrajectory]],
    STEPS: int,
    output_prefix: str = "benchmark",
) -> None:
    """Neff over time for all filters."""
    t_axis = np.arange(STEPS + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, straj in strajs:
        neffs = [step.pa.calc_Neff() / step.pa.num for step in straj.traj]
        ax.plot(t_axis, neffs, label=name, color=colors[name], lw=1)
    ax.axhline(2.0 / 3.0, color="k", ls=":", lw=1, label="Resample threshold")
    ax.set_title("Normalised effective sample size ($N_{\\mathrm{eff}} / N$)")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("$N_{\\mathrm{eff}} / N$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"plots/{output_prefix}_neff.png", dpi=150)
    matplot2tikz.save(f"plots/{output_prefix}_neff.tikz")
    plt.close(fig)


def plot_rmse_over_time(
    results: dict[str, dict[str, Any]],
    xs: np.ndarray,
    STEPS: int,
    output_prefix: str = "benchmark",
    est_key: str = "estimates",
) -> None:
    """Cumulative RMSE over time for each filter."""
    t_axis = np.arange(1, STEPS + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in results.items():
        est = r[est_key]
        cum_rmse = np.sqrt(
            np.cumsum((est[1 : STEPS + 1] - xs[1 : STEPS + 1]) ** 2)
            / np.arange(1, STEPS + 1),
        )
        ax.plot(t_axis, cum_rmse, label=name, color=colors[name], lw=1)
    ax.set_title(f"Cumulative RMSE for {output_prefix} {est_key} over time")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("RMSE")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"plots/{output_prefix}_{est_key}_rmse_time.png", dpi=150)
    matplot2tikz.save(f"plots/{output_prefix}_{est_key}_rmse_time.tikz")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MLNLG Main
# ══════════════════════════════════════════════════════════════════════════════
def run_mlnlg_benchmark() -> None:
    """Run PF, APF, and RBPF on the MLNLG model and compare."""
    np.random.seed(42)
    STEPS = 100
    N = 1000

    xis, zs, ys = simulate_mlnlg(STEPS, seed=42)

    results = {}

    # ── SIS ────────────────────────────────────────
    pf_model = MLNLGModelPF()
    straj_sis, t_sis, res_sis, log_sis = run_filter(
        pf_model,
        "pf",
        ys,
        us=None,
        N=N,
        resample=0,
    )
    est_xi_sis = weighted_means(straj_sis, state_index=0)
    est_z1_sis = weighted_means(straj_sis, state_index=1)
    est_z2_sis = weighted_means(straj_sis, state_index=2)
    results["SIS"] = {
        "rmse_xi": rmse(est_xi_sis, xis),
        "rmse_z1": rmse(est_z1_sis, zs[:, 0]),
        "rmse_z2": rmse(est_z2_sis, zs[:, 1]),
        "neff": mean_neff(straj_sis),
        "time_s": t_sis,
        "resamples": res_sis,
        "est_xi": est_xi_sis,
        "est_z1": est_z1_sis,
        "est_z2": est_z2_sis,
        "log_ml": log_sis,
    }

    # ── SIR ────────────────────────────────────────
    pf_model = MLNLGModelPF()
    straj_pf, t_pf, res_pf, log_pf = run_filter(pf_model, "pf", ys, us=None, N=N)
    est_xi_pf = weighted_means(straj_pf, state_index=0)
    est_z1_pf = weighted_means(straj_pf, state_index=1)
    est_z2_pf = weighted_means(straj_pf, state_index=2)
    results["SIR"] = {
        "rmse_xi": rmse(est_xi_pf, xis),
        "rmse_z1": rmse(est_z1_pf, zs[:, 0]),
        "rmse_z2": rmse(est_z2_pf, zs[:, 1]),
        "neff": mean_neff(straj_pf),
        "time_s": t_pf,
        "resamples": res_pf,
        "est_xi": est_xi_pf,
        "est_z1": est_z1_pf,
        "est_z2": est_z2_pf,
        "log_ml": log_pf,
    }

    # ── APF (3D joint state) ─────────────────────────────────────────────────
    apf_model = MLNLGModelPF()
    straj_apf, t_apf, res_apf, log_apf = run_filter(apf_model, "apf", ys, us=None, N=N)
    est_xi_apf = weighted_means(straj_apf, state_index=0)
    est_z1_apf = weighted_means(straj_apf, state_index=1)
    est_z2_apf = weighted_means(straj_apf, state_index=2)
    results["APF"] = {
        "rmse_xi": rmse(est_xi_apf, xis),
        "rmse_z1": rmse(est_z1_apf, zs[:, 0]),
        "rmse_z2": rmse(est_z2_apf, zs[:, 1]),
        "neff": mean_neff(straj_apf),
        "time_s": t_apf,
        "resamples": res_apf,
        "est_xi": est_xi_apf,
        "est_z1": est_z1_apf,
        "est_z2": est_z2_apf,
        "log_ml": log_apf,
    }

    # ── RBPF (ξ sampled, z Kalman-filtered) ──────────────────────────────────
    rb_model = MLNLGModelRBPF(N)
    straj_rb, t_rb, res_rb, log_rb = run_filter(rb_model, "pf", ys, us=None, N=N)
    est_xi_rb = weighted_means(straj_rb, state_index=0)
    est_z1_rb = weighted_means_z(straj_rb, lxi=1, z_index=0)
    est_z2_rb = weighted_means_z(straj_rb, lxi=1, z_index=1)
    results["RBPF"] = {
        "rmse_xi": rmse(est_xi_rb, xis),
        "rmse_z1": rmse(est_z1_rb, zs[:, 0]),
        "rmse_z2": rmse(est_z2_rb, zs[:, 1]),
        "neff": mean_neff(straj_rb),
        "time_s": t_rb,
        "resamples": res_rb,
        "est_xi": est_xi_rb,
        "est_z1": est_z1_rb,
        "est_z2": est_z2_rb,
        "log_ml": log_rb,
    }

    # ── Print metrics ────────────────────────────────────────────────────────
    print("\n=== MLNLG Benchmark ===")
    print(
        f"\n{'Filter':<20} {'RMSE(ξ)':>8} {'RMSE(z1)':>9} {'RMSE(z2)':>9}"
        f" {'Neff':>8} {'Time(s)':>8} {'Resamp':>7}",
    )
    print("-" * 75)
    for name, r in results.items():
        print(
            f"{name:<20} {r['rmse_xi']:>8.4f} {r['rmse_z1']:>9.4f}"
            f" {r['rmse_z2']:>9.4f} {r['neff']:>8.4f}"
            f" {r['time_s']:>8.4f} {r['resamples']:>7d}",
        )

    # ── LaTeX Table MLNLG using latextable ───────────────────────────────────
    print("\n% --- LaTeX Table: MLNLG Benchmark ---")
    table = Texttable()
    table.set_cols_align(["l", "r", "r", "r", "r", "r", "r", "r"])
    rows = [
        [
            "Filter",
            "RMSE($\\xi$)",
            "RMSE($z_1$)",
            "RMSE($z_2$)",
            "Mean $N_{\\mathrm{eff}}$",
            "Time (s)",
            "Resamples",
            "Log ML",
        ],
    ]

    for name, r in results.items():
        rows.append(
            [
                name,
                f"{r['rmse_xi']:.4f}",
                f"{r['rmse_z1']:.4f}",
                f"{r['rmse_z2']:.4f}",
                f"{r['neff']:.4f}",
                f"{r['time_s']:.4f}",
                r["resamples"],
                f"{r['log_ml']:.4f}",
            ],
        )

    table.add_rows(rows)
    print(
        latextable.draw_latex(
            table,
            caption="Mixed Linear/Non-linear Gaussian SSM Benchmark Results",
            label="tab:mlnlg_benchmark",
        ),
    )

    # ── Plot ξ estimates ─────────────────────────────────────────────────────
    t_axis = np.arange(STEPS + 1)
    plt.style.use("ggplot")

    for state_name, gt, key in [
        ("xi", xis, "est_xi"),
        ("z1", zs[:, 0], "est_z1"),
        ("z2", zs[:, 1], "est_z2"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Pad z ground truth to length STEPS+1 for alignment
        gt_padded = np.empty(STEPS + 1)
        gt_padded[: len(gt)] = gt[: STEPS + 1]

        ax.plot(t_axis, gt_padded, "k-", lw=1.5, label="Ground truth", zorder=5)
        for name, r in results.items():
            ax.plot(
                t_axis,
                r[key],
                "--",
                lw=1,
                color=colors[name],
                marker=markers[name],
                label=name,
                markersize=4,
                markevery=5,
            )
        ax.set_title(f"MLNLG: ${state_name}$ estimate vs ground truth")
        ax.set_xlabel("Time step $t$")
        ax.set_ylabel(f"${state_name}_t$")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"plots/mlnlg_{state_name}.png", dpi=150)
        matplot2tikz.save(f"plots/mlnlg_{state_name}.tikz")
        plt.close(fig)

    print("\nMLNLG plots saved to plots/mlnlg_{xi,z1,z2}.png")
    mlnlg_strajs = [
        ("SIS", straj_sis),
        ("SIR", straj_pf),
        ("APF", straj_apf),
        ("RBPF", straj_rb),
    ]
    plot_neff(results, mlnlg_strajs, STEPS, output_prefix="mlnlg")
    plot_rmse_over_time(results, xis, STEPS, output_prefix="mlnlg", est_key="est_xi")
    plot_rmse_over_time(
        results,
        zs[:, 0],
        STEPS,
        output_prefix="mlnlg",
        est_key="est_z1",
    )
    plot_rmse_over_time(
        results,
        zs[:, 1],
        STEPS,
        output_prefix="mlnlg",
        est_key="est_z2",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    np.random.seed(42)
    STEPS = 100
    N = 1000  # particles

    xs, ys = simulate(STEPS, seed=42)

    results = {}

    # ── Pure SIS ─────────────────────────────────────────────────────────────
    sis_model = NLGSSModel()
    straj_sis, t_sis, res_sis, log_sis = run_filter(
        sis_model,
        "pf",
        ys,
        us=None,
        N=N,
        resample=0,
    )
    est_sis = extract_state(straj_sis)
    results["SIS"] = {
        "rmse": rmse(est_sis, xs),
        "neff": mean_neff(straj_sis),
        "time_s": t_sis,
        "resamples": res_sis,
        "estimates": est_sis,
        "log_ml": log_sis,
    }

    # ── SIR ───────────────────────────────────────────────────
    pf_model = NLGSSModel()
    straj_pf, t_pf, res_pf, log_pf = run_filter(pf_model, "pf", ys, us=None, N=N)
    est_pf = extract_state(straj_pf)
    results["SIR"] = {
        "rmse": rmse(est_pf, xs),
        "neff": mean_neff(straj_pf),
        "time_s": t_pf,
        "resamples": res_pf,
        "estimates": est_pf,
        "log_ml": log_pf,
    }

    # ── APF ──────────────────────────────────────────────────────────────────
    apf_model = NLGSSModel()
    straj_apf, t_apf, res_apf, log_apf = run_filter(apf_model, "apf", ys, us=None, N=N)
    est_apf = extract_state(straj_apf)
    results["APF"] = {
        "rmse": rmse(est_apf, xs),
        "neff": mean_neff(straj_apf),
        "time_s": t_apf,
        "resamples": res_apf,
        "estimates": est_apf,
        "log_ml": log_apf,
    }

    # ── RBPF ─────────────────────────────────────────────────────────────────
    rb_model = RBPFModel(N)
    straj_rb, t_rb, res_rb, log_rbpf = run_filter(rb_model, "pf", ys, us=None, N=N)
    est_rb = extract_state(straj_rb, rbpf=True)
    results["RBPF"] = {
        "rmse": rmse(est_rb, xs),
        "neff": mean_neff(straj_rb),
        "time_s": t_rb,
        "resamples": res_rb,
        "estimates": est_rb,
        "log_ml": log_rbpf,
    }

    # ── Print metrics table ───────────────────────────────────────────────────
    print_metrics(results)
    print_latex_table_standard(results)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_individual_estimates(results, STEPS, xs, ys)
    plot_combined_estimates(results, STEPS, xs, ys)
    strajs = [
        ("SIS", straj_sis),
        ("SIR", straj_pf),
        ("APF", straj_apf),
        ("RBPF", straj_rb),
    ]
    plot_neff(results, strajs, STEPS, output_prefix="benchmark")
    plot_rmse_over_time(results, xs, STEPS, output_prefix="benchmark")
    run_mlnlg_benchmark()


if __name__ == "__main__":
    main()
