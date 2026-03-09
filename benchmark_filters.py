"""
Benchmark: SIS/PF, APF, and RBPF on a non-linear non-Gaussian SSM.

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

import matplotlib.pyplot as plt
import numpy
import scipy.stats
import tikzplotlib

import pyparticleest.filter as pfilter
import pyparticleest.interfaces as interfaces
import pyparticleest.models.mlnlg as mlnlg

# ── Model parameters ─────────────────────────────────────────────────────────
Q = 10.0  # process noise variance
R = 1.0  # measurement noise variance
x0_mean = 0.0
x0_var = 5.0


# ── Ground-truth simulator ────────────────────────────────────────────────────
def simulate(steps, seed=0):
    rng = numpy.random.default_rng(seed)
    xs = numpy.empty(steps + 1)
    ys = numpy.empty(steps)
    xs[0] = rng.normal(x0_mean, numpy.sqrt(x0_var))
    for t in range(steps):
        xs[t + 1] = (
            xs[t] / 2.0
            + 25.0 * xs[t] / (1.0 + xs[t] ** 2)
            + 8.0 * numpy.cos(1.2 * t)
            + rng.normal(0.0, numpy.sqrt(Q))
        )
        ys[t] = xs[t + 1] ** 2 / 20.0 + rng.normal(0.0, numpy.sqrt(R))
    return xs, ys


# ── MLNLG model parameters ───────────────────────────────────────────────────
MLNLG_Q_XI = 10.0
MLNLG_AZ = numpy.array([[0.9, 0.1], [-0.1, 0.85]])
MLNLG_QZ = 0.5 * numpy.eye(2)
MLNLG_C = numpy.array([[1.0, 1.0]])
MLNLG_R = 1.0
MLNLG_XI0_VAR = 5.0
MLNLG_Z0_COV = 0.5 * numpy.eye(2)


def simulate_mlnlg(steps, seed=0):
    """Simulate a mixed linear/nonlinear Gaussian SSM.

    ξ_{t+1} = f_nl(ξ_t, t) + v_ξ       (nonlinear, same as Gordon–Salmond–Smith)
    z_{t+1} = Az·z_t + v_z              (2D linear, damped oscillator)
    y_t     = C·z_{t+1} + ξ_{t+1}²/20 + e_t
    """
    rng = numpy.random.default_rng(seed)

    xis = numpy.empty(steps + 1)
    zs = numpy.empty((steps + 1, 2))
    ys = numpy.empty(steps)

    xis[0] = rng.normal(0.0, numpy.sqrt(MLNLG_XI0_VAR))
    zs[0] = rng.multivariate_normal([0.0, 0.0], MLNLG_Z0_COV)

    for t in range(steps):
        xis[t + 1] = (
            xis[t] / 2.0
            + 25.0 * xis[t] / (1.0 + xis[t] ** 2)
            + 8.0 * numpy.cos(1.2 * t)
            + rng.normal(0.0, numpy.sqrt(MLNLG_Q_XI))
        )
        zs[t + 1] = MLNLG_AZ @ zs[t] + rng.multivariate_normal([0.0, 0.0], MLNLG_QZ)
        ys[t] = (
            float(MLNLG_C @ zs[t + 1])
            + xis[t + 1] ** 2 / 20.0
            + rng.normal(0.0, numpy.sqrt(MLNLG_R))
        )

    return xis, zs, ys


# ── Helper: extract weighted mean from a ParticleTrajectory ──────────────────
def weighted_means(straj, state_index=0):
    """Return array of weighted-mean estimates for state_index across time."""
    means = numpy.empty(len(straj))
    for k, step in enumerate(straj.traj):
        pa = step.pa
        w = pa.w - numpy.max(pa.w)
        w = numpy.exp(w)
        w /= numpy.sum(w)
        means[k] = numpy.dot(w, pa.part[:, state_index])
    return means


def mean_neff(straj):
    neffs = []
    for step in straj.traj:
        neffs.append(step.pa.calc_Neff() / step.pa.num)
    return float(numpy.mean(neffs))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Standard PF model  (ParticleFiltering + AuxiliaryParticleFiltering)
#     Particles = [x_t]  (scalar, shape (N,1))
# ══════════════════════════════════════════════════════════════════════════════
class NLGSSModel(interfaces.ParticleFiltering, interfaces.AuxiliaryParticleFiltering):
    """
    Non-linear Gaussian SSM for use with PF and APF.

    Particle state: 1-D numpy array [x_t].
    """

    def create_initial_estimate(self, N):
        return numpy.random.normal(x0_mean, numpy.sqrt(x0_var), size=(N, 1))

    def sample_process_noise(self, particles, u, t):
        N = len(particles)
        return numpy.random.normal(0.0, numpy.sqrt(Q), size=(N, 1))

    def update(self, particles, u, t, noise):
        x = particles[:, 0]
        x_next = (
            x / 2.0 + 25.0 * x / (1.0 + x**2) + 8.0 * numpy.cos(1.2 * t) + noise[:, 0]
        )
        particles[:, 0] = x_next
        return particles

    def measure(self, particles, y, t):
        x = particles[:, 0]
        y_hat = x**2 / 20.0
        return scipy.stats.norm.logpdf(float(y), loc=y_hat, scale=numpy.sqrt(R))

    # ── APF first-stage weights: use propagated mean as predictor ────────────
    def eval_1st_stage_weights(self, particles, u, y, t):
        x = particles[:, 0]
        x_pred = x / 2.0 + 25.0 * x / (1.0 + x**2) + 8.0 * numpy.cos(1.2 * t)
        y_hat = x_pred**2 / 20.0
        return scipy.stats.norm.logpdf(
            float(y), loc=y_hat, scale=numpy.sqrt(R + Q * (x_pred / 10.0) ** 2)
        )


class MLNLGModelPF(interfaces.ParticleFiltering, interfaces.AuxiliaryParticleFiltering):
    """3D joint-state PF for the MLNLG model.

    Particles = [ξ, z1, z2] (shape (N, 3)).
    The PF must sample all three state dimensions — no Rao-Blackwellization.
    """

    def create_initial_estimate(self, N):
        particles = numpy.empty((N, 3))
        particles[:, 0] = numpy.random.normal(0.0, numpy.sqrt(MLNLG_XI0_VAR), N)
        particles[:, 1:] = numpy.random.multivariate_normal([0.0, 0.0], MLNLG_Z0_COV, N)
        return particles

    def sample_process_noise(self, particles, u, t):
        N = len(particles)
        noise = numpy.empty((N, 3))
        noise[:, 0] = numpy.random.normal(0.0, numpy.sqrt(MLNLG_Q_XI), N)
        noise[:, 1:] = numpy.random.multivariate_normal([0.0, 0.0], MLNLG_QZ, N)
        return noise

    def update(self, particles, u, t, noise):
        xi = particles[:, 0]
        particles[:, 0] = (
            xi / 2.0
            + 25.0 * xi / (1.0 + xi**2)
            + 8.0 * numpy.cos(1.2 * t)
            + noise[:, 0]
        )
        particles[:, 1:] = (MLNLG_AZ @ particles[:, 1:].T).T + noise[:, 1:]
        return particles

    def measure(self, particles, y, t):
        xi = particles[:, 0]
        z = particles[:, 1:]  # (N, 2)
        y_hat = (MLNLG_C @ z.T).ravel() + xi**2 / 20.0
        return scipy.stats.norm.logpdf(float(y), loc=y_hat, scale=numpy.sqrt(MLNLG_R))

    def eval_1st_stage_weights(self, particles, u, y, t):
        xi = particles[:, 0]
        z = particles[:, 1:]
        xi_pred = xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * numpy.cos(1.2 * t)
        z_pred = (MLNLG_AZ @ z.T).T
        y_hat = (MLNLG_C @ z_pred.T).ravel() + xi_pred**2 / 20.0
        # Predictive variance: R + C·Qz·Cᵀ + nonlinear ξ contribution
        pred_var = (
            MLNLG_R
            + float(MLNLG_C @ MLNLG_QZ @ MLNLG_C.T)
            + MLNLG_Q_XI * (xi_pred / 10.0) ** 2
        )
        return scipy.stats.norm.logpdf(float(y), loc=y_hat, scale=numpy.sqrt(pred_var))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RBPF model  (MixedNLGaussianSampledInitialGaussian)
#     xi_t = x_t  (non-linear, sampled)
#     z_t  = dummy scalar 0  (linear, Kalman)
#     y_t  = C*z_t + h(xi_t) + e_t  with C=0, h=xi^2/20
# ══════════════════════════════════════════════════════════════════════════════
class RBPFModel(mlnlg.MixedNLGaussianSampledInitialGaussian):
    """
    Rao-Blackwellized particle filter for the same NLGSSM.

    The 'linear' sub-state z is a dummy scalar kept at 0 with near-zero
    covariance so that the RBPF collapses to standard PF behaviour, while
    still exercising the full RB code path (Kalman measurement update,
    `get_nonlin_pred_dynamics`, `get_meas_dynamics`).
    """

    def __init__(self, N):
        # xi: non-linear scalar state (the actual x_t)
        # z:  linear scalar dummy = 0
        xi0 = numpy.array([[x0_mean]])
        z0 = numpy.array([[0.0]])
        Pxi0 = numpy.array([[x0_var]])
        Pz0 = numpy.array([[1e-6]])  # near-zero: z is deterministic
        Qxi = numpy.array([[Q]])
        Qz = numpy.array([[1e-10]])  # z does not evolve
        Az = numpy.array([[1.0]])  # z_{t+1} = z_t (trivial)
        C = numpy.array([[0.0]])  # measurement does not depend on z
        super().__init__(
            z0=z0,
            xi0=xi0,
            Pz0=Pz0,
            Pxi0=Pxi0,
            Az=Az,
            C=C,
            Qxi=Qxi,
            Qz=Qz,
            R=numpy.array([[R]]),
        )

    # xi_{t+1} = f(xi_t, t) + v_xi
    def get_nonlin_pred_dynamics(self, particles, u, t):
        xi = particles[:, 0]  # shape (N,)
        N = len(particles)
        f = xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * numpy.cos(1.2 * t)
        # fxi shape must be (N, lxi, 1)
        fxi = f[:, numpy.newaxis, numpy.newaxis]
        # Axi=None means xi_{t+1} has no linear dependence on z
        return (numpy.zeros((N, 1, 1)), fxi, None)

    # y_t = h(xi_t) + C*z_t + e_t  with C=0, h = xi^2/20
    def get_meas_dynamics(self, y, particles, t):
        xi = particles[:, 0]
        h = (xi**2 / 20.0)[:, numpy.newaxis, numpy.newaxis]
        # C=None reuses the stored C=[[0]]
        return (numpy.asarray(y).reshape((-1, 1)), None, h, None)


def weighted_means_z(straj, lxi, z_index):
    """Return weighted-mean z estimates from RBPF trajectory.

    In the RBPF particle array: [ξ₀..ξ_{lxi-1}, z₀..z_{lz-1}, P_flat...].
    The Kalman-filtered z means start at index lxi.
    """
    means = numpy.empty(len(straj))
    for k, step in enumerate(straj.traj):
        pa = step.pa
        w = pa.w - numpy.max(pa.w)
        w = numpy.exp(w)
        w /= numpy.sum(w)
        means[k] = numpy.dot(w, pa.part[:, lxi + z_index])
    return means


class MLNLGModelRBPF(mlnlg.MixedNLGaussianSampledInitialGaussian):
    """Rao-Blackwellized PF for the MLNLG benchmark.

    ξ (scalar, nonlinear) is sampled by particles.
    z (2D, linear|ξ) is analytically Kalman-filtered per particle.
    The RBPF only needs to sample in 1D instead of 3D.
    """

    def __init__(self, N):
        super().__init__(
            xi0=numpy.array([[0.0]]),
            z0=numpy.array([[0.0], [0.0]]),
            Pxi0=numpy.array([[MLNLG_XI0_VAR]]),
            Pz0=numpy.copy(MLNLG_Z0_COV),
            Az=numpy.copy(MLNLG_AZ),
            C=numpy.copy(MLNLG_C),
            Qxi=numpy.array([[MLNLG_Q_XI]]),
            Qz=numpy.copy(MLNLG_QZ),
            R=numpy.array([[MLNLG_R]]),
        )

    def get_nonlin_pred_dynamics(self, particles, u, t):
        """ξ_{t+1} = f_nl(ξ_t, t) + v_ξ, no linear dependence on z."""
        xi = particles[:, 0]
        N = len(particles)
        f = xi / 2.0 + 25.0 * xi / (1.0 + xi**2) + 8.0 * numpy.cos(1.2 * t)
        fxi = f[:, numpy.newaxis, numpy.newaxis]  # (N, lxi=1, 1)
        Axi = numpy.zeros((N, 1, 2))  # (N, lxi=1, lz=2) — ξ doesn't depend on z
        return (Axi, fxi, None)  # Qxi=None → uses default

    def get_meas_dynamics(self, y, particles, t):
        """y_t = C·z_t + ξ_t²/20 + e_t.  C is constant, h(ξ) varies."""
        xi = particles[:, 0]
        h = (xi**2 / 20.0)[:, numpy.newaxis, numpy.newaxis]  # (N, 1, 1)
        return (numpy.asarray(y).reshape((-1, 1)), None, h, None)
        #       y preprocessed,                     C=default, h=per-particle, R=default


# ══════════════════════════════════════════════════════════════════════════════
# Runner helpers
# ══════════════════════════════════════════════════════════════════════════════
def run_filter(model, filter_name, ys, us, N, resample=2.0 / 3.0):
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
        w = pa.w - numpy.max(pa.w)
        log_ml += pa.w_offset + numpy.log(numpy.sum(numpy.exp(w)))
    wall = time.perf_counter() - t0
    return straj, wall, resample_count, log_ml


def extract_state(straj, rbpf=False):
    """Return (T,) array of weighted-mean state estimates."""
    if rbpf:
        # For RBPF, xi is the first element of the particle
        return weighted_means(straj, state_index=0)
    return weighted_means(straj, state_index=0)


def rmse(estimates, truth):
    # estimates has length T+1 (includes t=0), truth has length T+1
    # align: skip t=0 (prior), compare t=1..T
    n = min(len(estimates), len(truth)) - 1
    return float(numpy.sqrt(numpy.mean((estimates[1 : n + 1] - truth[1 : n + 1]) ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# Metrics and plots
# ══════════════════════════════════════════════════════════════════════════════
def print_metrics(results):
    print(
        f"\n{'Filter':<20} {'RMSE':>8} {'Mean Neff':>10} {'Time (s)':>10} {'Resamples':>10}"
    )
    print("-" * 62)
    for name, r in results.items():
        print(
            f"{name:<20} {r['rmse']:>8.4f} {r['neff']:>10.4f} "
            f"{r['time_s']:>10.4f} {r['resamples']:>10d}"
        )


colors = {
    "SIS": "tab:blue",
    "SIR (Bootstrap PF)": "tab:orange",
    "APF": "tab:green",
    "RBPF": "tab:red",
}
markers = {
    "SIS": "P",
    "SIR (Bootstrap PF)": "*",
    "APF": "D",
    "RBPF": "X",
}


def plot_individual_estimates(results, STEPS, xs, ys):
    """One figure per algorithm, each with ground truth overlaid."""
    t_axis = numpy.arange(STEPS + 1)
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
        tikzplotlib.save(f"plots/estimate_{safe_name}.tex")
        plt.close(fig)


def plot_combined_estimates(results, STEPS, xs, ys):
    """All algorithms on one figure for comparison."""
    t_axis = numpy.arange(STEPS + 1)
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
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("$x_t$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/benchmark_filters_combined.png", dpi=150)
    tikzplotlib.save("plots/benchmark_filters_combined.tex")
    plt.close(fig)


def plot_neff(results, strajs, STEPS, output_prefix="benchmark"):
    """Neff over time for all filters."""
    t_axis = numpy.arange(STEPS + 1)
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
    tikzplotlib.save(f"plots/{output_prefix}_neff.tex")
    plt.close(fig)


def plot_rmse_over_time(results, xs, STEPS):
    """Cumulative RMSE over time for each filter."""
    t_axis = numpy.arange(1, STEPS + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in results.items():
        est = r["estimates"]
        cum_rmse = numpy.sqrt(
            numpy.cumsum((est[1 : STEPS + 1] - xs[1 : STEPS + 1]) ** 2)
            / numpy.arange(1, STEPS + 1)
        )
        ax.plot(t_axis, cum_rmse, label=name, color=colors[name], lw=1)
    ax.set_title("Cumulative RMSE over time")
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("RMSE")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/benchmark_rmse_time.png", dpi=150)
    tikzplotlib.save("plots/benchmark_rmse_time.tex")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MLNLG Main
# ══════════════════════════════════════════════════════════════════════════════
def run_mlnlg_benchmark():
    """Run PF, APF, and RBPF on the MLNLG model and compare."""
    numpy.random.seed(42)
    STEPS = 100
    N = 1000

    xis, zs, ys = simulate_mlnlg(STEPS, seed=42)

    results = {}

    # ── Bootstrap PF (3D joint state) ────────────────────────────────────────
    pf_model = MLNLGModelPF()
    straj_sis, t_sis, res_sis, log_sis = run_filter(
        pf_model, "pf", ys, us=None, N=N, resample=0
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

    # ── Bootstrap PF (3D joint state) ────────────────────────────────────────
    pf_model = MLNLGModelPF()
    straj_pf, t_pf, res_pf, log_pf = run_filter(pf_model, "pf", ys, us=None, N=N)
    est_xi_pf = weighted_means(straj_pf, state_index=0)
    est_z1_pf = weighted_means(straj_pf, state_index=1)
    est_z2_pf = weighted_means(straj_pf, state_index=2)
    results["SIR (Bootstrap PF)"] = {
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
        f" {'Neff':>8} {'Time(s)':>8} {'Resamp':>7}"
    )
    print("-" * 75)
    for name, r in results.items():
        print(
            f"{name:<20} {r['rmse_xi']:>8.4f} {r['rmse_z1']:>9.4f}"
            f" {r['rmse_z2']:>9.4f} {r['neff']:>8.4f}"
            f" {r['time_s']:>8.4f} {r['resamples']:>7d}"
        )

    # ── Plot ξ estimates ─────────────────────────────────────────────────────
    t_axis = numpy.arange(STEPS + 1)
    plt.style.use("ggplot")

    for state_name, gt, key in [
        ("xi", xis, "est_xi"),
        ("z1", zs[:, 0], "est_z1"),
        ("z2", zs[:, 1], "est_z2"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Pad z ground truth to length STEPS+1 for alignment
        gt_padded = numpy.empty(STEPS + 1)
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
        tikzplotlib.save(f"plots/mlnlg_{state_name}.tex")
        plt.close(fig)

    print("\nMLNLG plots saved to plots/mlnlg_{xi,z1,z2}.png")
    mlnlg_strajs = [
        ("SIR (Bootstrap PF)", straj_pf),
        ("APF", straj_apf),
        ("RBPF", straj_rb),
    ]
    plot_neff(results, mlnlg_strajs, STEPS, output_prefix="mlnlg")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    numpy.random.seed(42)
    STEPS = 100
    N = 1000  # particles

    xs, ys = simulate(STEPS, seed=42)

    results = {}

    # ── Pure SIS ─────────────────────────────────────────────────────────────
    sis_model = NLGSSModel()
    straj_sis, t_sis, res_sis, log_sis = run_filter(
        sis_model, "pf", ys, us=None, N=N, resample=0
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

    # ── Bootstrap PF (SIR) ───────────────────────────────────────────────────
    pf_model = NLGSSModel()
    straj_pf, t_pf, res_pf, log_pf = run_filter(pf_model, "pf", ys, us=None, N=N)
    est_pf = extract_state(straj_pf)
    results["SIR (Bootstrap PF)"] = {
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

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_individual_estimates(results, STEPS, xs, ys)
    plot_combined_estimates(results, STEPS, xs, ys)
    strajs = [
        ("SIS", straj_sis),
        ("SIR (Bootstrap PF)", straj_pf),
        ("APF", straj_apf),
        ("RBPF", straj_rb),
    ]
    plot_neff(results, strajs, STEPS, output_prefix="nlgss")
    plot_rmse_over_time(results, xs, STEPS)
    run_mlnlg_benchmark()


if __name__ == "__main__":
    main()
