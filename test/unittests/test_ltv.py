"""
Created on Jul 23, 2015

@author: ajn
"""

import unittest

import numpy as np
import numpy.testing as npt

import pyparticleest.models.ltv as ltv


class Model(ltv.LTV):
    """x_{k+1} = sin(x_k) + v_k, v_k ~ N(0,Q)
    y_k = x_k + e_k, e_k ~ N(0,R),
    x(0) ~ N(0,P0)"""

    def __init__(self, x0, P0, A, C, f, Q, R) -> None:
        super().__init__(
            np.asarray(x0).reshape((1, 1)),
            np.asarray(P0).reshape((1, 1)),
            A=np.asarray(A).reshape((1, 1)),
            C=np.asarray(C).reshape((1, 1)),
            f=np.asarray(f).reshape((1, 1)),
            Q=np.asarray(Q).reshape((1, 1)),
            R=np.asarray(R).reshape((1, 1)),
        )


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.A = 0.9
        self.C = 1.5
        self.f = 0.5
        self.R = 0.5
        self.Q = 2.0
        self.P0 = 3.0
        self.z0 = -0.5

        self.model = Model(self.z0, self.P0, self.A, self.C, self.f, self.Q, self.R)

    def tearDown(self) -> None:
        pass

    def testUpdate(self) -> None:
        particles = self.model.create_initial_estimate(1)
        nextp = self.model.update(np.copy(particles), None, None, None)

        (zl, Pl) = self.model.get_states(particles)
        (nzl, nPl) = self.model.get_states(nextp)
        npt.assert_array_equal(np.asarray(nzl), self.A * np.asarray(zl) + self.f)
        npt.assert_array_equal(
            np.asarray(nPl), (self.A**2) * np.asarray(Pl) + self.Q,
        )

    def testMeasure(self) -> None:
        particles = self.model.create_initial_estimate(1)
        (zl, Pl) = self.model.get_states(particles)
        y = 1.0
        # https://en.wikipedia.org/wiki/Kalman_filter
        S = self.C * self.P0 * self.C + self.R
        K = self.P0 * self.C / S
        xn = zl[0] + K * (y - self.C * zl[0])
        Pn = Pl[0] - K * self.C * Pl[0]

        partn = np.copy(particles)
        _ = self.model.measure(partn, np.asarray(y).reshape((-1, 1)), None)
        (nzl, nPl) = self.model.get_states(partn)

        npt.assert_array_almost_equal(xn[0].ravel(), nzl[0].ravel(), 10)
        npt.assert_array_almost_equal(Pn[0].ravel(), nPl[0].ravel(), 10)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
