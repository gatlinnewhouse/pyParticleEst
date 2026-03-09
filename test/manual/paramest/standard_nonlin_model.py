import math

import matplotlib.pyplot as plt
import numpy as np

import pyparticleest.paramest.interfaces as pestint
import pyparticleest.paramest.paramest as param_est
from pyparticleest import interfaces
from pyparticleest.utils import kalman


def generate_dataset(steps, P0, Q, R):
    x = np.zeros((steps + 1,))
    y = np.zeros((steps + 1,))
    x[0] = np.random.multivariate_normal((0.0,), P0)
    y[0] = 0.05 * x[0] ** 2 + np.random.multivariate_normal((0.0,), R)
    for k in range(steps):
        x[k + 1] = (
            0.5 * x[k]
            + 25.0 * x[k] / (1 + x[k] ** 2)
            + 8 * math.cos(1.2 * k)
            + np.random.multivariate_normal((0.0,), Q)
        )
        y[k + 1] = 0.05 * x[k + 1] ** 2 + np.random.multivariate_normal((0.0,), R)

    return (x, y)


def wmean(logw, val):
    w = np.exp(logw)
    w = w / sum(w)
    return np.sum(w * val.ravel())


class Model(
    interfaces.FFBSiRS,
    interfaces.ParticleFiltering,
    pestint.ParamEstInterface,
    pestint.ParamEstBaseNumeric,
):
    """x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
    y_k = x_k + e_k, e_k ~ N(0,R),
    x(0) ~ N(0,P0)
    """

    def __init__(self, P0, Q, R) -> None:
        self.P0 = np.copy(P0)
        self.Q = np.copy(Q)
        self.R = np.copy(R)
        self.logxn_max = kalman.lognormpdf_scalar(np.zeros((1,)), self.Q)
        super().__init__()

    def create_initial_estimate(self, N):
        return np.random.normal(0.0, np.sqrt(self.P0).ravel(), (N,))

    def sample_process_noise(self, particles, u, t):
        """Return process noise for input u"""
        N = len(particles)
        return np.random.normal(0.0, np.sqrt(self.Q).ravel(), (N,))

    def update(self, particles, u, noise, t) -> None:
        """Update estimate using 'data' as input"""
        particles[:] = (
            0.5 * particles
            + 25.0 * particles / (1 + particles**2)
            + 8 * math.cos(1.2 * t)
            + noise
        )

    def measure(self, particles, y, t):
        """Return the log-pdf value of the measurement"""
        return kalman.lognormpdf_scalar(0.05 * particles**2 - y, self.R)

    def logp_xnext(self, particles, next_part, u, t):
        """Return the log-pdf value for the possible future state 'next' given input u"""
        pn = (
            0.5 * particles
            + 25.0 * particles / (1 + particles**2)
            + 8 * math.cos(1.2 * t)
        )
        return kalman.lognormpdf_scalar(pn.ravel() - next_part.ravel(), self.Q)

    def logp_xnext_max(self, particles, u, t):
        return self.logxn_max

    def sample_smooth(self, part, ptraj, anc, future_trajs, find, ut, yt, tt, cur_ind):
        """Update ev. Rao-Blackwellized states conditioned on "next_part" """
        return part.reshape((-1, 1))

    def set_params(self, params) -> None:
        """New set of parameters for which the integral approximation terms will be evaluated"""
        self.params = np.copy(params)
        self.Q = math.exp(params[0]) * np.eye(1)
        self.R = math.exp(params[1]) * np.eye(1)

    def eval_logp_x0(self, particles, t):
        """Calculate gradient of a term of the I1 integral approximation
        as specified in [1].
        The gradient is an array where each element is the derivative with
        respect to the corresponding parameter
        """
        return kalman.lognormpdf_scalar(particles, self.P0)

    def copy_ind(self, particles, new_ind=None):
        if new_ind is not None:
            return np.copy(particles[new_ind])
        return np.copy(particles)

    def eval_logp_xnext_fulltraj(self, straj, ut, tt):
        part = straj.get_smoothed_estimates()
        M = part.shape[1]
        cost = 8.0 * np.cos(1.2 * np.asarray(tt, dtype=float))
        xp = (
            0.5 * part
            + 25.0 * part / (1 + part**2)
            + np.repeat(cost.reshape(-1, 1, 1), repeats=M, axis=1)
        )
        diff = part[1:] - xp[:-1]
        logp = kalman.lognormpdf_scalar(diff.ravel(), self.Q)
        return np.sum(logp) / M

    def eval_logp_y_fulltraj(self, straj, yt, tt):
        sest = straj.get_smoothed_estimates()
        M = sest.shape[1]
        yp = 0.05 * sest**2
        diff = yp - np.repeat(
            np.asarray(yt, dtype=float).reshape((-1, 1, 1)),
            repeats=M,
            axis=1,
        )
        return np.sum(kalman.lognormpdf_scalar(diff.ravel(), self.R)) / M


def callback(params, Q, cur_iter) -> None:
    pass


def callback_sim(estimator) -> None:
    # vals = numpy.empty((num, steps+1))

    plt.figure(1)
    plt.clf()
    #    mvals = numpy.empty((steps+1))
    #    for k in range(steps+1):
    #        #vals[:,k] = numpy.copy(estimator.pt.traj[k].pa.part)
    #        mvals[k] = wmean(estimator.pt.traj[k].pa.w,
    #                          estimator.pt.traj[k].pa.part)
    #        #plt.plot((k,)*num, vals[:,k], 'k.', markersize=0.5)
    #    plt.plot(range(steps+1), mvals, 'k-')

    sest = estimator.get_smoothed_estimates()
    for k in range(sest.shape[1]):
        plt.plot(range(steps + 1), sest[:, k], "g-")

    plt.plot(range(steps + 1), x, "r-")
    # plt.plot(range(steps+1), y, 'bx')
    plt.draw()
    plt.show()
    plt.pause(0.0001)


if __name__ == "__main__":
    np.random.seed(1)
    steps = 1499
    iterations = np.arange(1000)
    num = np.ceil(500 + 4500.0 / (iterations[-1] ** 3) * iterations**3).astype(int)
    M = np.ceil(50 + 450.0 / (iterations[-1] ** 3) * iterations**3).astype(int)
    P0 = 5.0 * np.eye(1)
    Q = 1.0 * np.eye(1)
    R = 0.1 * np.eye(1)
    (x, y) = generate_dataset(steps, P0, Q, R)
    theta0 = np.log(np.asarray((2.0, 2.0)))
    model = Model(P0, Q, R)
    estimator = param_est.ParamEstimation(model, u=None, y=y)
    callback(theta0, None, -1)
    estimator.maximize(
        theta0,
        num,
        M,
        smoother="mcmc",
        meas_first=True,
        max_iter=len(iterations),
        callback=callback,
    )
#     plt.ion()
#     estimator.maximize(theta0, num, M, smoother='full', meas_first=True, max_iter=len(iterations),
#                        callback_sim=callback_sim, callback=callback)
#     plt.ioff()
#    traj = pf.ParticleTrajectory(model, num)
#    traj.measure(y[0])
#    for k in range(1,len(y)):
#        traj.forward(u=None, y=y[k])
#
#    straj = traj.perform_smoothing(M, method='rs')
