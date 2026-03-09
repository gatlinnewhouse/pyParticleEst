"""Particle filtering for a trivial model
Also illustrates that the"""

import matplotlib.pyplot as plt
import numpy as np

import pyparticleest.interfaces as interfaces
import pyparticleest.simulator as simulator
import pyparticleest.utils.kalman as kalman


def generate_dataset(steps, P0, Q, R):
    x = np.zeros((steps + 1,))
    y = np.zeros((steps,))
    x[0] = 2.0 + 0.0 * np.random.normal(0.0, P0)
    for k in range(1, steps + 1):
        x[k] = x[k - 1] + np.random.normal(0.0, Q)
        y[k - 1] = x[k] + np.random.normal(0.0, R)

    return (x, y)


class Model(interfaces.ParticleFiltering):
    """x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
    y_k = x_k + e_k, e_k ~ N(0,R),
    x(0) ~ N(0,P0)"""

    def __init__(self, P0, Q, R) -> None:
        self.P0 = np.copy(P0)
        self.Q = np.copy(Q)
        self.R = np.copy(R)

    def create_initial_estimate(self, N):
        return np.random.normal(0.0, self.P0, (N,)).reshape((-1, 1))

    def sample_process_noise(self, particles, u, t):
        """Return process noise for input u"""
        N = len(particles)
        return np.random.normal(0.0, self.Q, (N,)).reshape((-1, 1))

    def update(self, particles, u, t, noise) -> None:
        """Update estimate using 'data' as input"""
        particles += noise

    def measure(self, particles, y, t):
        """Return the log-pdf value of the measurement"""
        logyprob = np.empty(len(particles), dtype=float)
        for k in range(len(particles)):
            logyprob[k] = kalman.lognormpdf(particles[k].reshape(-1, 1) - y, self.R)
        return logyprob

    def logp_xnext_full(
        self, part, past_trajs, pind, future_trajs, find, ut, yt, tt, cur_ind,
    ):

        diff = future_trajs[0].pa.part[find] - part

        logpxnext = np.empty(len(diff), dtype=float)
        for k in range(len(logpxnext)):
            logpxnext[k] = kalman.lognormpdf(
                diff[k].reshape(-1, 1), np.asarray(self.Q).reshape(1, 1),
            )
        return logpxnext


if __name__ == "__main__":
    steps = 50
    num = 50
    P0 = 1.0
    Q = 1.0
    R = np.asarray(((1.0,),))

    # Make realization deterministic
    np.random.seed(1)
    (x, y) = generate_dataset(steps, P0, Q, R)

    model = Model(P0, Q, R)
    sim = simulator.Simulator(model, u=None, y=y)
    sim.simulate(num, num, smoother="ancestor")

    plt.plot(range(steps + 1), x, "r-")
    plt.plot(range(1, steps + 1), y, "bx")

    (vals, _) = sim.get_filtered_estimates()

    plt.plot(range(steps + 1), vals[:, :, 0], "k.", markersize=0.8)

    svals = sim.get_smoothed_estimates()

    # Plot "smoothed" trajectories to illustrate that the particle filter
    # suffers from degeneracy when considering the full trajectories
    plt.plot(range(steps + 1), svals[:, :, 0], "b--")
    plt.plot(range(steps + 1), x, "r-")
    plt.xlabel("t")
    plt.ylabel("x")

    plt.show()
