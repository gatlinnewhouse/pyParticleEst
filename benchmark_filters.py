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
            marker=markers[name],
            label=name,
            markersize=4.0,
            markevery=5,  # reduce marker clutter
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
        ax.legend(loc="best", fontsize=9)
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
            markevery=10,
        )
    ax.set_xlabel("Time step $t$")
    ax.set_ylabel("$x_t$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/benchmark_filters_combined.png", dpi=150)
    tikzplotlib.save("plots/benchmark_filters_combined.tex")
    plt.close(fig)


def plot_neff(results, strajs, STEPS):
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
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/benchmark_neff.png", dpi=150)
    tikzplotlib.save("plots/benchmark_neff.tex")
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
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("plots/benchmark_rmse_time.png", dpi=150)
    tikzplotlib.save("plots/benchmark_rmse_time.tex")
    plt.close(fig)


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
    plot_neff(results, strajs, STEPS)
    plot_rmse_over_time(results, xs, STEPS)


if __name__ == "__main__":
    main()
