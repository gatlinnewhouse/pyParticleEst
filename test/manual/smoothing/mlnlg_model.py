import matplotlib.pyplot as plt
import numpy as np

from pyparticleest import simulator
from pyparticleest.models import mlnlg


def generate_dataset(steps, P0_xi, P0_z, Qxi, Qz, Qxiz, R):
    Q = np.vstack((np.hstack((Qxi, Qxiz)), np.hstack((Qxiz.T, Qz))))
    xi = np.zeros((1, steps + 1))
    z = np.zeros((1, steps + 1))
    y = np.zeros((steps, 1))
    xi[:, 0] = np.random.normal(0.0, np.sqrt(P0_xi))
    z[:, 0] = np.random.normal(0.0, np.sqrt(P0_z))
    for k in range(1, steps + 1):
        noise = np.random.multivariate_normal(np.zeros((2,)), Q)
        xi[:, k] = xi[:, k - 1] + z[:, k - 1] + noise[0]
        z[:, k] = z[:, k - 1] + noise[1]
        y[k - 1, 0] = xi[:, k] + np.random.normal(0.0, np.sqrt(R))

    x = np.vstack((xi, z))
    return (x, y)


class Model(mlnlg.MixedNLGaussianMarginalizedInitialGaussian):
    # class Model(mlnlg.MixedNLGaussianSampledInitialGaussian):
    """xi_{k+1} = xi_k + z_k + v_xi_k, v_xi ~ N(0,Q_xi)
    z_{k+1} = z_{k} + v_z, v_z_k ~ N(0, Q_z)
    y_k = xi_k + +z_k + e_k, e_k ~ N(0,R_z),
    (v_xi v_z).T ~ N(0, ((Q_xi, Qxiz), (Qxiz.T Qz))
    """

    def __init__(self, P0_xi, P0_z, Q_xi, Q_z, Q_xiz, R) -> None:
        Axi = np.eye(1)
        Az = np.eye(1)
        self.pn_count = 0
        P0_xi = np.copy(P0_xi)
        P0_z = np.copy(P0_z)
        z0 = np.zeros((1,))
        xi0 = np.zeros((1,))
        super().__init__(
            z0=z0,
            Pz0=P0_z,
            Pxi0=P0_xi,
            xi0=xi0,
            Axi=Axi,
            Az=Az,
            Qxi=Q_xi,
            Qxiz=Q_xiz,
            Qz=Q_z,
            R=R,
        )

    def get_nonlin_pred_dynamics(self, particles, u, t):
        tmp = np.vstack(particles)
        fxi = tmp[:, 0].tolist()
        return (None, fxi, None)

    def get_meas_dynamics(self, particles, y, t):
        N = len(particles)
        tmp = np.vstack(particles)
        h = tmp[:, 0].reshape(N, 1, 1)
        return (y, None, h, None)


if __name__ == "__main__":
    steps = 30
    num = 100
    nums = 10
    P0_xi = np.eye(1)
    P0_z = np.eye(1)
    Q_xi = 1.0 * np.eye(1)
    Q_z = 1.0 * np.eye(1)
    Q_xiz = 0.0 * np.eye(1)
    R = 0.1 * np.eye(1)
    np.random.seed(0)
    (x, y) = generate_dataset(steps, P0_xi, P0_z, Q_xi, Q_z, Q_xiz, R)

    model = Model(P0_xi, P0_z, Q_xi, Q_z, Q_xiz, R)
    sim = simulator.Simulator(model, None, y)
    sim.simulate(
        num_part=num,
        num_traj=nums,
        filter="PF",
        smoother="mhips_reduced",
        meas_first=False,
    )

    if True:
        plt.plot(range(steps + 1), x[0, :], "r-")
        plt.plot(range(steps + 1), x[1, :], "b-")
        plt.plot(range(1, steps + 1), y, "rx")

        fmean = sim.get_filtered_mean()
        (fest, _) = sim.get_filtered_estimates()
        for k in range(steps + 1):
            plt.plot((k,) * num, fest[k, :, 0], "r.", markersize=1.0)
            plt.plot((k,) * num, fest[k, :, 1], "b.", markersize=1.0)

        sest = sim.get_smoothed_estimates()
        smean = sim.get_smoothed_mean()

        for j in range(nums):
            plt.plot(range(steps + 1), sest[:, j, 0], "r--")
            plt.plot(range(steps + 1), sest[:, j, 1], "b--")

        plt.plot(range(steps + 1), smean[:, 0], "ro", markersize=3.0)
        plt.plot(range(steps + 1), smean[:, 1], "bo", markersize=3.0)

        plt.show()
