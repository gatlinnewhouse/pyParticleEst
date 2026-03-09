"""Created on Nov 11, 2013

@author: Jerker Nordh
"""

import math
import sys

import matplotlib.pyplot as plt
import numpy as np

import pyparticleest.paramest.interfaces as pestinf
import pyparticleest.paramest.paramest as param_est
from pyparticleest.models import mlnlg
from pyparticleest.paramest import gradienttest


def sign(x) -> float:
    if x < 0.0:
        return -1.0
    return 1.0


def calc_h(eta):
    return np.asarray(((0.1 * eta[0, 0] * math.fabs(eta[0, 0])), 0.0)).reshape(
        (-1, 1),
    )


def generate_dataset(params, length):
    Ae = np.array([[params[1], 0.0, 0.0]])
    Az = np.asarray(
        (
            (1.0, params[2], 0.0),
            (0.0, params[3] * math.cos(params[4]), -params[3] * math.sin(params[4])),
            (0.0, params[3] * math.sin(params[4]), params[3] * math.cos(params[4])),
        ),
    )

    C = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0]])

    e_vec = np.zeros((1, length))
    z_vec = np.zeros((3, length))

    e = np.asarray(((np.random.normal(0.0, 1.0),),))
    z = np.zeros((3, 1))

    e_vec[:, 0] = e.ravel()
    z_vec[:, 0] = z.ravel()

    y = np.zeros((2, length))
    h = calc_h(e)
    y[:, 0] = (h + C.dot(z)).ravel()

    for i in range(1, length):
        e = (
            params[0] * np.arctan(e)
            + Ae.dot(z)
            + np.random.normal(0.0, math.sqrt(0.01))
        )

        wz = (
            np.random.multivariate_normal(np.zeros((3,)), 0.01 * np.eye(3, 3))
            .ravel()
            .reshape((-1, 1))
        )

        z = Az.dot(z) + wz
        h = calc_h(e)
        y[:, i] = (h + C.dot(z)).ravel()
        e_vec[:, i] = e.ravel()
        z_vec[:, i] = z.ravel()

    return (y.T.tolist(), e_vec, z_vec)


class ParticleLS2(
    mlnlg.MixedNLGaussianSampledInitialGaussian,
    pestinf.ParamEstBaseNumericGrad,
    pestinf.ParamEstInterface_GradientSearch,
):
    """Implement a simple system by extending the MixedNLGaussian class"""

    def __init__(self, params) -> None:
        """Define all model variables"""
        Axi = np.array([[params[1], 0.0, 0.0]])
        Az = np.asarray(
            (
                (1.0, params[2], 0.0),
                (
                    0.0,
                    params[3] * math.cos(params[4]),
                    -params[3] * math.sin(params[4]),
                ),
                (0.0, params[3] * math.sin(params[4]), params[3] * math.cos(params[4])),
            ),
        )

        C = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 1.0]])
        Qxi = np.diag(
            [
                0.01,
            ],
        )
        Qz = np.diag([0.01, 0.01, 0.01])
        R = np.diag([0.1, 0.1])
        xi0 = np.asarray((0.0,)).reshape((-1, 1))
        Pxi0 = np.eye(1)
        z0 = np.zeros((3,))
        Pz0 = 0.0 * np.eye(3)

        # Linear states handled by base-class
        super().__init__(
            xi0=xi0,
            z0=z0,
            Pz0=Pz0,
            Pxi0=Pxi0,
            Az=Az,
            C=C,
            Axi=Axi,
            R=R,
            Qxi=Qxi,
            Qz=Qz,
            params=params,
        )

    def get_nonlin_pred_dynamics(self, particles, u, t):
        xil = particles[:, 0]
        fxil = self.params[0] * np.arctan(xil)
        return (None, fxil[:, np.newaxis, np.newaxis], None)

    def get_meas_dynamics(self, particles, y, t):
        N = len(particles)
        xil = np.vstack(particles)[:, 0]
        h = np.zeros((N, 2, 1))
        h[:, 0, 0] = 0.1 * np.fabs(xil) * xil
        return (np.asarray(y).reshape((-1, 1)), None, h, None)

    # Override this method since there is no uncertainty in z0
    def eval_logp_x0(self, particles, t):
        return self.eval_logp_xi0(particles[:, : self.lxi])

    def eval_logp_x0_val_grad(self, particles, t):
        return (
            self.eval_logp_xi0(particles[:, : self.lxi]),
            self.eval_logp_xi0_grad(particles[:, : self.lxi]),
        )

    def get_pred_dynamics_grad(self, particles, u, t):
        N = len(particles)
        xil = particles[:, 0]
        f_grad = np.zeros((N, 5, 4, 1))
        f_grad[:, 0, 0, 0] = np.arctan(xil)

        return (np.repeat(self.A_grad[np.newaxis], N, 0), f_grad, None)

    def set_params(self, params) -> None:
        """New set of parameters"""
        # Update all needed matrices and derivates with respect
        # to the new parameter set
        self.params = np.copy(params)
        Axi = np.array([[params[1], 0.0, 0.0]])

        Az = np.asarray(
            (
                (1.0, params[2], 0.0),
                (
                    0.0,
                    params[3] * math.cos(params[4]),
                    -params[3] * math.sin(params[4]),
                ),
                (0.0, params[3] * math.sin(params[4]), params[3] * math.cos(params[4])),
            ),
        )

        self.A_grad = np.vstack(
            (
                np.zeros((4, 3))[np.newaxis],
                np.asarray(
                    (
                        (1.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                )[np.newaxis],
                np.asarray(
                    (
                        (0.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                )[np.newaxis],
                np.asarray(
                    (
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, math.cos(params[4]), -math.sin(params[4])),
                        (0.0, math.sin(params[4]), math.cos(params[4])),
                    ),
                )[np.newaxis],
                np.asarray(
                    (
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (
                            0.0,
                            -params[3] * math.sin(params[4]),
                            -params[3] * math.cos(params[4]),
                        ),
                        (
                            0.0,
                            params[3] * math.cos(params[4]),
                            -params[3] * math.sin(params[4]),
                        ),
                    ),
                )[np.newaxis],
            ),
        )
        self.set_dynamics(Axi=Axi, Az=Az)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1].lower() == "gui":
        num = 50
        nums = 5

        theta_true = np.array((1.0, 1.0, 0.3, 0.968, 0.315))

        # How many steps forward in time should our simulation run
        steps = 200
        sims = 1

        # Create arrays for storing some values for later plotting
        vals = np.zeros((2, num + 1, steps + 1))

        estimate = np.zeros((5, sims))

        plt.ion()
        fig1 = plt.figure()
        fig2 = plt.figure()

        max_iter = 1000

        for k in range(sims):
            theta_guess = np.array(
                (
                    np.random.uniform(0.0, 2.0),
                    np.random.uniform(0.0, 2.0),
                    np.random.uniform(0.0, 0.6),
                    np.random.uniform(0.0, 1.0),
                    np.random.uniform(0.0, math.pi / 2.0),
                ),
            )

            # theta_guess = numpy.copy(theta_true)

            # Create reference
            (y, e, z) = generate_dataset(theta_true, steps)
            # Store values for last time-step aswell

            plt.figure(fig1.number)
            plt.clf()
            x = np.asarray(range(steps + 1))
            plt.plot(x[1:], np.asarray(y)[:, :], ".")
            fig1.show()
            plt.draw()

            params_it = np.zeros((max_iter, len(theta_guess)))
            Q_it = np.zeros(max_iter)
            it = 0

            def callback(params, Q, cur_iter):
                global it
                params_it[it, :] = params
                Q_it[it] = Q
                it = it + 1
                plt.figure(fig2.number)
                plt.clf()
                plt.plot(range(it), params_it[:it, 0], "b-")
                plt.plot((0.0, it), (theta_true[0], theta_true[0]), "b--")
                plt.plot(range(it), params_it[:it, 1], "r-")
                plt.plot((0.0, it), (theta_true[1], theta_true[1]), "r--")
                plt.plot(range(it), params_it[:it, 2], "g-")
                plt.plot((0.0, it), (theta_true[2], theta_true[2]), "g--")
                plt.plot(range(it), params_it[:it, 3], "c-")
                plt.plot((0.0, it), (theta_true[3], theta_true[3]), "c--")
                plt.plot(range(it), params_it[:it, 4], "k-")
                plt.plot((0.0, it), (theta_true[4], theta_true[4]), "k--")
                plt.show()
                plt.draw()
                plt.pause(0.0001)
                return cur_iter > max_iter

            # Create an array for our particles
            model = ParticleLS2(theta_guess)
            ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=y)
            ParamEstimator.set_params(theta_guess)
            # ParamEstimator.simulate(num, nums, False)

            (param, Q) = ParamEstimator.maximize(
                param0=theta_guess,
                num_part=num,
                num_traj=nums,
                max_iter=max_iter,
                callback=callback,
                smoother="rsas",
                tol=0.0,
            )

            svals = np.zeros((4, nums, steps + 1))

            fig3 = plt.figure()
            fig4 = plt.figure()
            fig5 = plt.figure()
            fig6 = plt.figure()

            sest = ParamEstimator.straj.get_smoothed_estimates()

            for i in range(steps + 1):
                for j in range(nums):
                    svals[0, j, i] = sest[i, j, 0]
                    svals[1:, j, i] = sest[i, j, 1:4]

            plt.figure(fig3.number)
            plt.clf()
            # TODO, does these smoothed estimates really look ok??
            for j in range(nums):
                plt.plot(range(steps + 1), svals[0, j, :], "g-")
                # plt.plot(range(steps+1),svals[1,j,:],'r-')
            plt.plot(x[:-1], e.T, "rx")
            # plt.plot(x[:-1], e,'r-')
            fig3.show()

            plt.figure(fig4.number)
            plt.clf()
            # TODO, does these smoothed estimates really look ok??
            for j in range(nums):
                plt.plot(range(steps + 1), svals[1, j, :], "g-")
                # plt.plot(range(steps+1),svals[1,j,:],'r-')
            plt.plot(x[:-1], z[0, :], "rx")
            # plt.plot(x[:-1], e,'r-')
            fig4.show()

            plt.figure(fig5.number)
            plt.clf()
            # TODO, does these smoothed estimates really look ok??
            for j in range(nums):
                plt.plot(range(steps + 1), svals[2, j, :], "g-")
                # plt.plot(range(steps+1),svals[1,j,:],'r-')
            plt.plot(x[:-1], z[1, :], "rx")
            # plt.plot(x[:-1], e,'r-')
            fig5.show()

            plt.figure(fig6.number)
            plt.clf()
            # TODO, does these smoothed estimates really look ok??
            for j in range(nums):
                plt.plot(range(steps + 1), svals[2, j, :], "g-")
                # plt.plot(range(steps+1),svals[1,j,:],'r-')
            plt.plot(x[:-1], z[2, :], "rx")
            # plt.plot(x[:-1], e,'r-')
            fig6.show()

            plt.draw()

            estimate[:, k] = param

        #        plt.figure(fig2.number)
        #        plt.clf()
        #        plt.hist(estimate[:,:(k+1)].T)
        #        fig2.show()
        #        plt.show()
        #        plt.draw()

        #    plt.hist(estimate.T)
        plt.ioff()
        plt.show()
        plt.draw()
    elif sys.argv[1].lower() == "nogui":
        num = 50
        nums = 10

        theta_true = np.array((1.0, 1.0, 0.3, 0.968, 0.315))

        # How many steps forward in time should our simulation run
        steps = 200
        sims = 20

        # Create arrays for storing some values for later plotting
        vals = np.zeros((2, num + 1, steps + 1))

        estimate = np.zeros((5, sims))

        max_iter = 1000

        for k in range(sims):
            theta_guess = np.array(
                (
                    np.random.uniform(0.0, 2.0),
                    np.random.uniform(0.0, 2.0),
                    np.random.uniform(0.0, 0.6),
                    np.random.uniform(0.0, 1.0),
                    np.random.uniform(0.0, math.pi / 2.0),
                ),
            )

            # Create reference
            # numpy.random.seed(k)
            (y, e, z) = generate_dataset(theta_true, steps)

            # Create an array for our particles
            model = ParticleLS2(theta_guess)
            ParamEstimator = param_est.ParamEstimation(model=model, u=None, y=y)
            ParamEstimator.set_params(theta_guess)
            # ParamEstimator.simulate(num, nums, False)

            (param, Q) = ParamEstimator.maximize(
                param0=theta_guess,
                num_part=num,
                num_traj=nums,
                max_iter=max_iter,
                callback=None,
                smoother="rsas",
                tol=0.0,
            )

            print("{} {} {} {} {}") % tuple(round(param, 4))

    elif sys.argv[1].lower() == "gradient":
        num = 50
        nums = 5
        np.random.seed(4)  # 3
        theta_true = np.array((1.0, 1.0, 0.3, 0.968, 0.315))

        # How many steps forward in time should our simulation run
        steps = 50
        sims = 1

        # Create arrays for storing some values for later plotting
        vals = np.zeros((2, num + 1, steps + 1))

        estimate = np.zeros((5, sims))

        (y, e, z) = generate_dataset(theta_true, steps)

        model = ParticleLS2(theta_true)
        # Create an array for our particles
        gt = gradienttest.GradientTest(model=model, u=None, y=y)
        gt.set_params(theta_true)

        param_id = 4
        param_steps = 101
        tval = theta_true[param_id]
        param_vals = np.linspace(
            tval - math.fabs(tval),
            tval + math.fabs(tval),
            param_steps,
        )
        gt.test(param_id, param_vals, nums=nums)

        gt.plot_y.plot(1)
        gt.plot_xn.plot(2)
        gt.plot_x0.plot(3)
        plt.show()
    else:
        pass
