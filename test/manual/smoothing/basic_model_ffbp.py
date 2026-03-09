import matplotlib.pyplot as plt
import numpy as np

import pyparticleest.models.nlg as nlg
import pyparticleest.simulator as simulator


def generate_dataset(steps, P0, Q, R):
    x = np.zeros((steps + 1,))
    y = np.zeros((steps,))
    x[0] = 2.0 + 0.0 * np.random.normal(0.0, P0)
    for k in range(1, steps + 1):
        x[k] = x[k - 1] + np.random.normal(0.0, Q)
        y[k - 1] = x[k] + np.random.normal(0.0, R)

    return (x, y)


class Model(nlg.NonlinearGaussianInitialGaussian):
    """x_{k+1} = x_k + v_k, v_k ~ N(0,Q)
    y_k = x_k + e_k, e_k ~ N(0,R),
    x(0) ~ N(0,P0)"""

    def __init__(self, P0, Q, R) -> None:
        x0 = np.zeros((1, 1))
        super().__init__(
            x0=x0,
            Px0=np.asarray(P0).reshape((1, 1)),
            Q=np.asarray(Q).reshape((1, 1)),
            R=np.asarray(R).reshape((1, 1)),
        )

    def calc_f(self, particles, u, t):
        return particles

    def calc_g(self, particles, t):
        return particles


if __name__ == "__main__":
    steps = 80
    num = 40
    M = 20
    P0 = 1.0
    Q = 1.0
    R = np.asarray(((1.0,),))
    np.random.seed(0)
    (x, y) = generate_dataset(steps, P0, Q, R)

    model = Model(P0, Q, R)
    sim = simulator.Simulator(model, None, y)
    sim.simulate(
        num,
        M,
        filter="PF",
        smoother="mhbp",
        smoother_options={"R": 50},
        meas_first=False,
    )
    plt.plot(range(steps + 1), x, "r-")
    plt.plot(range(1, steps + 1), y, "bx")
    plt.plot(range(steps + 1), sim.get_smoothed_estimates()[:, :, 0], "g.")
    plt.show()
