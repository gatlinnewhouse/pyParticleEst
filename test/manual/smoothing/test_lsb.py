"""
Created on Nov 11, 2013

@author: Jerker Nordh
"""

import math
import sys

import matplotlib.pyplot as plt
import numpy as np

import pyparticleest.models.mlnlg as mlnlg
import pyparticleest.simulator as simulator

C_theta = np.array(
    [
        [0.0, 0.04, 0.044, 0.008],
    ],
)


def calc_Ae_fe(eta, t):
    Ae = eta / (1 + eta**2) * C_theta
    fe = 0.5 * eta + 25 * eta / (1 + eta**2) + 8 * math.cos(1.2 * t)
    return (Ae, fe)


def calc_h(eta):
    return 0.05 * eta**2


def generate_dataset(length):
    Az = np.array(
        [
            [3.0, -1.691, 0.849, -0.3201],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
        ],
    )

    C = np.array([[0.0, 0.0, 0.0, 0.0]])

    Qe = np.diag([0.005])
    Qz = np.diag([0.01, 0.01, 0.01, 0.01])
    R = np.diag(
        [
            0.1,
        ],
    )

    e_vec = np.zeros((1, length + 1))
    z_vec = np.zeros((4, length + 1))

    e = np.array(
        [
            [
                0.0,
            ],
        ],
    )
    z = np.zeros((4, 1))

    e_vec[:, 0] = e.ravel()
    z_vec[:, 0] = z.ravel()

    y = np.zeros((1, length))
    t = 0
    h = calc_h(e)
    # y[:,0] = (h + C.dot(z)).ravel()

    for i in range(1, length + 1):
        (Ae, fe) = calc_Ae_fe(e, t)

        e = fe + Ae.dot(z) + np.random.multivariate_normal(np.zeros((1,)), Qe)

        wz = (
            np.random.multivariate_normal(np.zeros((4,)), Qz)
            .ravel()
            .reshape((-1, 1))
        )

        z = Az.dot(z) + wz
        t = t + 1
        h = calc_h(e)
        y[:, i - 1] = (
            h + C.dot(z) + np.random.multivariate_normal(np.zeros((1,)), R)
        ).ravel()
        e_vec[:, i] = e.ravel()
        z_vec[:, i] = z.ravel()

    return (y.T.tolist(), e_vec, z_vec)


class ParticleLSB(mlnlg.MixedNLGaussianMarginalizedInitialGaussian):
    """Model 60 & 61 from Lindsten & Schon (2011)"""

    def __init__(self) -> None:
        """Define all model variables"""

        # No uncertainty in initial state
        xi0 = np.array(
            [
                [0.0],
            ],
        )
        z0 = np.array([[0.0], [0.0], [0.0], [0.0]])
        P0 = 0.0 * np.eye(4)

        Az = np.array(
            [
                [3.0, -1.691, 0.849, -0.3201],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0],
            ],
        )

        Qxi = np.diag([0.005])
        Qz = np.diag([0.01, 0.01, 0.01, 0.01])
        R = np.diag(
            [
                0.1,
            ],
        )

        super().__init__(
            xi0=xi0,
            z0=z0,
            Pz0=P0,
            Az=Az,
            R=R,
            Qxi=Qxi,
            Qz=Qz,
        )

    def get_nonlin_pred_dynamics(self, particles, u, t):
        tmp = np.vstack(particles)[:, np.newaxis, :]
        xi = tmp[:, :, 0]
        Axi = (xi / (1 + xi**2)).dot(C_theta)
        Axi = Axi[:, np.newaxis, :]
        fxi = 0.5 * xi + 25 * xi / (1 + xi**2) + 8 * math.cos(1.2 * t)
        fxi = fxi[:, np.newaxis, :]
        return (Axi, fxi, None)

    def get_meas_dynamics(self, particles, y, t):
        if y is None:
            return (y, None, None, None)
        tmp = np.vstack(particles)
        h = 0.05 * tmp[:, 0] ** 2
        h = h[:, np.newaxis, np.newaxis]

        return (np.asarray(y).reshape((-1, 1)), None, h, None)


if __name__ == "__main__":
    num = 300
    nums = 50

    # How many steps forward in time should our simulation run
    steps = 100

    if len(sys.argv) > 1:
        if sys.argv[1] == "nogui":
            sims = 1000
            sqr_err_eta = np.zeros((sims, steps + 1))
            sqr_err_theta = np.zeros((sims, steps + 1))

            for k in range(sims):
                # Create reference
                np.random.seed(k)
                (y, e, z) = generate_dataset(steps)

                model = ParticleLSB()

                # Create an array for our particles
                sim = simulator.Simulator(model=model, u=None, y=y)
                sim.simulate(num, nums, res=0.67, filter="PF", smoother="mcmc")

                smean = sim.get_smoothed_mean()
                theta_mean = 25.0 + C_theta.dot(smean[:, 1:5].T).T
                theta = 25.0 + C_theta.dot(z.reshape((4, -1)))
                sqr_err_eta[k, :] = (smean[:, 0] - e[0, :]) ** 2
                sqr_err_theta[k, :] = (theta_mean[:, 0] - theta) ** 2

                rmse_eta = np.sqrt(np.mean(sqr_err_eta[k, :]))
                rmse_theta = np.sqrt(np.mean(sqr_err_theta[k, :]))

    else:
        # Create arrays for storing some values for later plotting
        vals = np.zeros((2, num + 1, steps + 1))

        plt.ion()

        # Create reference
        np.random.seed(14)
        # numpy.random.seed(86)
        (y, e, z) = generate_dataset(steps)
        # Store values for last time-step aswell

        x = np.asarray(range(steps + 1))
        model = ParticleLSB()
        # Create an array for our particles
        sim = simulator.Simulator(model=model, u=None, y=y)
        sim.simulate(num, nums, res=0.67, filter="PF", smoother="mcmc")

        sest = sim.get_smoothed_estimates()
        (est, _) = sim.get_filtered_estimates()

        ftheta = 25.0 + np.tensordot(
            C_theta.ravel(),
            est[:, :, 1:5],
            axes=(
                [
                    0,
                ],
                [
                    2,
                ],
            ),
        )
        stheta = 25.0 + np.tensordot(
            C_theta.ravel(),
            sest[:, :, 1:5],
            axes=(
                [
                    0,
                ],
                [
                    2,
                ],
            ),
        )

        svals = np.zeros((2, nums, steps + 1))
        vals = np.zeros((2, num, steps + 1))

        for j in range(num):
            plt.plot(
                range(steps + 1), est[:, j, 0], ".", markersize=3.0, color="#BBBBBB",
            )
        plt.plot(x, e.T, "k-", markersize=1.0)
        for j in range(nums):
            plt.plot(
                range(steps + 1),
                sest[:, j, 0],
                "--",
                markersize=2.0,
                color="#999999",
                dashes=(7, 50),
            )

        plt.show()

        plt.figure()
        for j in range(num):
            plt.plot(
                range(steps + 1), ftheta[:, j], ".", markersize=3.0, color="#BBBBBB",
            )
        plt.plot(x, (25.0 + C_theta.dot(z)).ravel(), "k-", markersize=1.0)
        for j in range(nums):
            plt.plot(
                range(steps + 1),
                stheta[:, j],
                "--",
                markersize=2.0,
                color="#999999",
                dashes=(10, 25),
            )

        plt.show()
        plt.draw()

        plt.ioff()
        plt.show()
        plt.draw()
