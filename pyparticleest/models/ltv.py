"""Model definition for base class for Linear Time-varying systems
@author: Jerker Nordh
"""

from typing import Any

import numpy
import numba as nb
import scipy.linalg

import pyparticleest.utils.kalman as kalman
import pyparticleest.utils.mlnlg_compute as mlnlg_compute
from pyparticleest.interfaces import FFBSi, ParticleFiltering


@nb.njit(cache=True)
def _nb_calc_l1(z: numpy.ndarray, P: numpy.ndarray, z0: numpy.ndarray) -> numpy.ndarray:
    z0_diff = z - z0
    l1 = numpy.dot(z0_diff, z0_diff.T) + P
    return l1


@nb.njit(cache=True)
def _nb_calc_l1_grad(
    z: numpy.ndarray,
    P: numpy.ndarray,
    z0: numpy.ndarray,
    z0_grad: numpy.ndarray | None,
    lparams: int,
    lz: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    z0_diff = z - z0
    l1 = numpy.dot(z0_diff, z0_diff.T) + P
    l1_diff = numpy.zeros((lparams, lz, lz))

    if z0_grad is not None:
        for j in range(lparams):
            tmp = -numpy.dot(z0_grad[j], z0_diff.T)
            l1_diff[j] += tmp + tmp.T

    return l1, l1_diff


@nb.njit(cache=True)
def _nb_calc_l2(
    zn: numpy.ndarray,
    Pn: numpy.ndarray,
    z: numpy.ndarray,
    P: numpy.ndarray,
    A: numpy.ndarray,
    f: numpy.ndarray,
    M: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    predict_err = zn - f - numpy.dot(A, z)
    AM = numpy.dot(A, M)
    l2 = numpy.dot(predict_err, predict_err.T)
    l2 += Pn + numpy.dot(A, numpy.dot(P, A.T)) - AM.T - AM
    return l2, A, M, predict_err


@nb.njit(cache=True)
def _nb_calc_l2_grad(
    zn: numpy.ndarray,
    Pn: numpy.ndarray,
    z: numpy.ndarray,
    P: numpy.ndarray,
    A: numpy.ndarray,
    f: numpy.ndarray,
    M: numpy.ndarray,
    A_grad: numpy.ndarray | None,
    f_grad: numpy.ndarray | None,
    lparam: int,
    lz: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    predict_err = zn - f - numpy.dot(A, z)
    AM = numpy.dot(A, M)
    l2 = numpy.dot(predict_err, predict_err.T)
    l2 += Pn + numpy.dot(A, numpy.dot(P, A.T)) - AM.T - AM

    l2_grad = numpy.zeros((lparam, lz, lz))
    if f_grad is not None:
        for j in range(lparam):
            tmp = -numpy.dot(f_grad[j], predict_err.T)
            l2_grad[j] += tmp + tmp.T

    if A_grad is not None:
        for j in range(lparam):
            tmp = -numpy.dot(numpy.dot(A_grad[j], z), predict_err.T)
            l2_grad[j] += tmp + tmp.T
            tmp = numpy.dot(numpy.dot(A_grad[j], P), A.T)
            l2_grad[j] += tmp + tmp.T
            tmp = -numpy.dot(A_grad[j], M)
            l2_grad[j] += tmp + tmp.T

    return l2, l2_grad


@nb.njit(cache=True)
def _nb_calc_l3(
    y: numpy.ndarray,
    z: numpy.ndarray,
    P: numpy.ndarray,
    C: numpy.ndarray,
    h_k: numpy.ndarray | None,
) -> numpy.ndarray:
    if h_k is not None:
        meas_diff = y - (numpy.dot(C, z) + h_k)
    else:
        meas_diff = y - numpy.dot(C, z)

    l3 = numpy.dot(meas_diff, meas_diff.T)
    l3 += numpy.dot(C, numpy.dot(P, C.T))
    return l3


@nb.njit(cache=True)
def _nb_calc_l3_grad(
    y: numpy.ndarray,
    z: numpy.ndarray,
    P: numpy.ndarray,
    C: numpy.ndarray,
    h_k: numpy.ndarray | None,
    C_grad: numpy.ndarray | None,
    h_grad: numpy.ndarray | None,
    lparam: int,
    len_y: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    if h_k is not None:
        meas_diff = y - (numpy.dot(C, z) + h_k)
    else:
        meas_diff = y - numpy.dot(C, z)

    l3 = numpy.dot(meas_diff, meas_diff.T)
    l3 += numpy.dot(C, numpy.dot(P, C.T))
    l3_grad = numpy.zeros((lparam, len_y, len_y))

    if h_grad is not None:
        for j in range(lparam):
            tmp = -numpy.dot(h_grad[j], meas_diff)
            l3_grad[j] += tmp + tmp.T

    if C_grad is not None:
        for j in range(lparam):
            tmp = -numpy.dot(numpy.dot(C_grad[j], z), meas_diff)
            l3_grad[j] += tmp + tmp.T
            tmp = numpy.dot(numpy.dot(C_grad[j], P), C)
            l3_grad[j] += tmp + tmp.T

    return l3, l3_grad


class LTV(FFBSi, ParticleFiltering):
    """
    Base class for particles of the type linear time varying with additive gaussian noise.

    Implement this type of system by extending this class and provide the methods for returning
    the system matrices at each time instant

    z_{t+1} = A*z_t + f + v, v ~ N(0, Q)
    y_t = C*z_t + h + e, e ~ N(0,R)

    Args:
     - z0: Initial mean value of the state estimate
     - P0: Coviariance of initial z estimate
     - A (array-like): A matrix (if constant)
     - C (array-like): C matrix (if constant)
     - Q (array-like): Q matrix (if constant)
     - R (array-like): R matrix (if constant)
     - f (array-like): f vector (if constant)
     - h (array-like): h vector (if constant)
     - params (array-like): model parameters (if any)
    """

    def __init__(
        self,
        z0: Any,
        P0: Any,
        A: Any = None,
        C: Any = None,
        Q: Any = None,
        R: Any = None,
        f: Any = None,
        h: Any = None,
        params: Any = None,
        **kwargs: Any,
    ) -> None:
        self.z0 = numpy.copy(z0).reshape((-1, 1))
        self.P0 = numpy.copy(P0)
        if f is None:
            f = numpy.zeros_like(self.z0)
        self.kf = kalman.KalmanSmoother(
            lz=len(self.z0),
            A=A,
            C=C,
            Q=Q,
            R=R,
            f_k=f,
            h_k=h,
        )
        super().__init__(**kwargs)

    def create_initial_estimate(self, N: int) -> numpy.ndarray:
        """Sample particles from initial distribution

        Args:
         - N (int): Number of particles to sample, since the estimate is
           deterministic there is no reason for N > 1

        Returns:
         (array-like) with first dimension = N, model specific representation
         of all particles"""

        if N > 1:
            print(
                f"N > 1 redundamt for LTV system (N={N})",
            )
        lz = len(self.z0)
        dim = lz + lz * lz
        particles = numpy.empty((N, dim))

        for i in range(N):
            particles[i, :lz] = numpy.copy(self.z0).ravel()
            particles[i, lz:] = numpy.copy(self.P0).ravel()
        return particles

    def set_states(
        self,
        particles: numpy.ndarray,
        z_list: list[numpy.ndarray] | numpy.ndarray,
        P_list: list[numpy.ndarray] | numpy.ndarray,
    ) -> None:
        """
        Set the estimate of the states

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - z_list (list): list of mean values for z for each particle
         - P_list (list): list of covariance matrices for z for each particle
        """
        lz = len(self.z0)
        N = len(particles)
        for i in range(N):
            particles[i, :lz] = z_list[i].ravel()
            lzP = lz + lz * lz
            particles[i, lz:lzP] = P_list[i].ravel()

    def get_states(
        self,
        particles: numpy.ndarray,
    ) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """
        Return the estimates contained in the particles array

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)

        Returns
            (zl, Pl):
             - zl: list of mean values for z
             - Pl: list of covariance matrices for z
        """
        N = len(particles)
        zl = []
        Pl = []
        lz = len(self.z0)
        for i in range(N):
            zl.append(particles[i, :lz].reshape(-1, 1))
            lzP = lz + lz * lz
            Pl.append(particles[i, lz:lzP].reshape(self.P0.shape))

        return (zl, Pl)

    def get_pred_dynamics(self, u: Any, t: float) -> tuple[Any, Any, Any]:
        """
        Return matrices describing affine relation of next
        nonlinear state conditioned on the current time and input signal

        z_{t+1} = A*z_t + f + v, v ~ N(0, Q)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (A, f, Q) where each element is a list
         with the corresponding matrix for each particle. None indicates
         that the matrix is identical for all particles and the value stored
         in this class should be used instead
        """
        return (None, None, None)

    def update(
        self,
        particles: numpy.ndarray,
        u: Any,
        t: float,
        noise: Any,
    ) -> numpy.ndarray:
        """Propagate estimate forward in time

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like):  input signal
         - t (float): time-stamp
         - noise: Unused for this type of model

        Returns:
         (array-like) with first dimension = N, particle estimate at time t+1
        """
        # Update linear estimate with data from measurement of next non-linear
        # state
        (zl, Pl) = self.get_states(particles)
        (A, f, Q) = self.get_pred_dynamics(u=u, t=t)
        self.kf.set_dynamics(A=A, Q=Q, f_k=f)
        for i in range(len(zl)):
            # Predict z_{t+1}
            (zl[i], Pl[i]) = self.kf.predict(zl[i], Pl[i])

        # Predict next states conditioned on eta_next
        self.set_states(particles, zl, Pl)
        return particles

    def get_meas_dynamics(self, y: Any, t: float) -> tuple[Any, Any, Any, Any]:
        """
        Return matrices describing affine relation of measurement and current
        state estimates

        y_t = C*z_t + h + e, e ~ N(0,R)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns:
         (y, C, h, R): y is a preprocessed measurement, the rest are lists
         with the corresponding matrix for each particle. None indicates
         that the matrix is identical for all particles and the value stored
         in this class should be used instead
        """
        return (y, None, None, None)

    def measure(self, particles: numpy.ndarray, y: Any, t: float) -> numpy.ndarray:
        """
        Return the log-pdf value of the measurement and update the statistics
        for the states

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like):  measurement
         - t (float): time-stamp

        Returns:
         (array-like) with first dimension = N, logp(y|x^i)
        """

        (zl, Pl) = self.get_states(particles)
        (y, C, h, R) = self.get_meas_dynamics(y=y, t=t)
        self.kf.set_dynamics(C=C, R=R, h_k=h)
        lyz = numpy.empty(len(particles))
        for i in range(len(zl)):
            # Predict z_{t+1}
            lyz[i] = self.kf.measure(y, zl[i], Pl[i])

        self.set_states(particles, zl, Pl)
        return lyz

    def logp_xnext(
        self,
        particles: numpy.ndarray,
        next_part: Any,
        u: Any,
        t: float,
    ) -> numpy.ndarray:
        """
        Return the log-pdf value for the possible future state 'next'
        given input u.

        Always returns zeros since all particles are always equivalent for this
        type of model

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - next_part: Unused
         - u: Unused
         - t: Unused

        Returns:
         (array-like) with first dimension = N, numpu.zeros((N,))
        """
        # Not needed for Linear Gaussian models, always return 0 (all particles will be identical anyhow)
        N = len(particles)
        return numpy.zeros((N,))

    def sample_process_noise(self, particles: numpy.ndarray, u: Any, t: float) -> None:
        """
        There is no need to sample noise for this type of model

        Args:

         - particles: Unused
         - next_part: Unused
         - u: Unused
         - t: Unused

        Returns:
         None
        """
        return

    def sample_smooth(
        self,
        part: numpy.ndarray,
        ptraj: list[Any] | None,
        anc: numpy.ndarray,
        future_trajs: list[Any] | None,
        find: numpy.ndarray | None,
        ut: numpy.ndarray,
        yt: numpy.ndarray,
        tt: numpy.ndarray,
        cur_ind: int,
    ) -> numpy.ndarray:
        """
        Update sufficient statistics based on the future states

        Args:
         - part  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - ptraj: array of trajectory step objects from previous time-steps,
           last index is step just before the current
         - anc (array-like): index of the ancestor of each particle in part
         - future_trajs (array-like): particle estimate for {t+1:T}
         - find (array-like): index in future_trajs corresponding to each
           particle in part
         - ut (array-like): input signals for {0:T}
         - yt (array-like): measurements for {0:T}
         - tt (array-like): time stamps for {0:T}
         - cur_ind (int): index of current timestep (in ut, yt and tt)

        Returns:
         (array-like) with first dimension = N
        """

        (zl, Pl) = self.get_states(part)
        M = len(part)
        lz = len(self.z0)
        lzP = lz + lz * lz
        res = numpy.empty((M, lz + 2 * lz**2))
        for j in range(M):
            if future_trajs is not None:
                zn = future_trajs[0].pa.part[j, :lz].reshape((lz, 1))
                Pn = future_trajs[0].pa.part[j, lz:lzP].reshape((lz, lz))
                (A, f, Q) = self.get_pred_dynamics(u=ut[0], t=tt[0])
                self.kf.set_dynamics(A=A, Q=Q, f_k=f)
                (zs, Ps, Ms) = self.kf.smooth(
                    zl[0],
                    Pl[0],
                    zn,
                    Pn,
                    self.kf.A,
                    self.kf.f_k,
                    self.kf.Q,
                )
            else:
                zs = zl[j]
                Ps = Pl[j]
                Ms = numpy.zeros_like(Ps)
            res[j] = numpy.hstack((zs.ravel(), Ps.ravel(), Ms.ravel()))

        return res

    def fwd_peak_density(self, u: Any, t: float) -> float:
        """
        No need for rejections sampling for this type of model, always returns
        0.0 since all particles are equivalent

        Args:
         - u: Unused
         - t: Unused

        Returns
         (float) 0.0
        """
        return 0.0

    def eval_logp_x0(self, particles: numpy.ndarray, t: float) -> numpy.ndarray:
        """
        Evaluate sum log p(x_0)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - t (float): time stamp
        """
        # Calculate l1 according to (19a)
        N = len(particles)
        (zl, Pl) = self.get_states(particles)
        lpz0 = numpy.empty(N)
        for i in range(N):
            l1 = self.calc_l1(zl[i], Pl[i], self.z0, self.P0)
            (_tmp, ld) = numpy.linalg.slogdet(self.P0)
            tmp = numpy.linalg.solve(self.P0, l1)
            lpz0[i] = -0.5 * (ld + numpy.trace(tmp))
        return lpz0

    def eval_logp_x0_val_grad(
        self,
        particles: numpy.ndarray,
        t: float,
    ) -> tuple[float | numpy.ndarray, numpy.ndarray]:
        """
        Evaluate gradient of sum log p(x_0)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - t (float): time stamp
        """
        # Calculate l1 according to (19a)
        N = len(particles)
        lparam = len(self.params)
        lpz0_grad = numpy.zeros(lparam)
        (zl, Pl) = self.get_states(particles)
        (z0_grad, P0_grad) = self.get_initial_grad()
        if z0_grad is None and P0_grad is None:
            lpz0 = self.eval_logp_x0(particles, t)
        else:
            lpz0 = 0.0
            P0cho = scipy.linalg.cho_factor(self.P0)
            ld = numpy.sum(numpy.log(numpy.diagonal(P0cho[0]))) * 2
            for i in range(N):
                (l1, l1_grad) = self.calc_l1_grad(
                    zl[i],
                    Pl[i],
                    self.z0,
                    self.P0,
                    z0_grad,
                )
                tmp = scipy.linalg.cho_solve(P0cho, l1)
                lpz0 += -0.5 * (ld + numpy.trace(tmp))
                for j in range(len(self.params)):
                    lpz0_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(
                        P0cho,
                        P0_grad[j],
                        l1,
                        l1_grad[j],
                    )
        return (lpz0, lpz0_grad)

    def eval_logp_xnext(
        self,
        particles: numpy.ndarray,
        x_next: numpy.ndarray,
        u: Any,
        t: float,
    ) -> numpy.ndarray:
        """
        Evaluate log p(x_{t+1}|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - x_next (array-like): future states
         - t (float): time stamp

        Returns: (array-like)
        """
        # Calculate l2 according to (16)
        N = len(particles)
        (zl, Pl) = self.get_states(particles)
        (zn, Pn) = self.get_states(x_next)
        (A, f, Q) = self.get_pred_dynamics(u=u, t=t)
        self.kf.set_dynamics(A=A, Q=Q, f_k=f)
        self.t = t
        lpxn = numpy.empty(N)

        for k in range(N):
            lz = len(self.z0)
            lzP = lz + lz * lz
            Mz = particles[k][lzP:].reshape((lz, lz))
            (l2, _A, _M_ext, _predict_err) = self.calc_l2(
                zn[k],
                Pn[k],
                zl[k],
                Pl[k],
                self.kf.A,
                self.kf.f_k,
                Mz,
            )
            (_tmp, ld) = numpy.linalg.slogdet(self.kf.Q)
            tmp = numpy.linalg.solve(self.kf.Q, l2)
            lpxn[k] = -0.5 * (ld + numpy.trace(tmp))

        return lpxn

    def eval_logp_xnext_val_grad(
        self,
        particles: numpy.ndarray,
        x_next: numpy.ndarray,
        u: Any,
        t: float,
    ) -> tuple[float | numpy.ndarray, numpy.ndarray]:
        """
        Evaluate value and gradient of log p(x_{t+1}|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - x_next (array-like): future states
         - t (float): time stamp

        Returns: ((array-like), (array-like))
        """
        # Calculate l2 according to (16)
        N = len(particles)
        lparam = len(self.params)
        (zl, Pl) = self.get_states(particles)
        (zn, Pn) = self.get_states(x_next)
        (A, f, Q) = self.get_pred_dynamics(u=u, t=t)
        (A_grad, f_grad, Q_grad) = self.get_pred_dynamics_grad(u=u, t=t)
        lpxn_grad = numpy.zeros(lparam)
        if A_grad is None and f_grad is None and Q_grad is None:
            lpxn = self.eval_logp_xnext(particles, x_next, u, t)
        else:
            self.kf.set_dynamics(A=A, Q=Q, f_k=f)
            lpxn = 0.0
            Qcho = scipy.linalg.cho_factor(self.kf.Q, check_finite=False)
            ld = numpy.sum(numpy.log(numpy.diagonal(Qcho[0]))) * 2

            if Q_grad is None:
                Q_grad = numpy.zeros((len(self.params), self.kf.lz, self.kf.lz))

            for k in range(N):
                lz = len(self.z0)
                lzP = lz + lz * lz
                Mz = particles[k][lzP:].reshape((lz, lz))
                (l2, l2_grad) = self.calc_l2_grad(
                    zn[k],
                    Pn[k],
                    zl[k],
                    Pl[k],
                    self.kf.A,
                    self.kf.f_k,
                    Mz,
                    A_grad,
                    f_grad,
                )
                tmp = scipy.linalg.cho_solve(Qcho, l2)
                lpxn += -0.5 * (ld + numpy.trace(tmp))

                for j in range(len(self.params)):
                    lpxn_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(
                        Qcho,
                        Q_grad[j],
                        l2,
                        l2_grad[j],
                    )

        return (lpxn, lpxn_grad)

    def eval_logp_y(self, particles: numpy.ndarray, y: Any, t: float) -> numpy.ndarray:
        """
        Evaluate value of log p(y_t|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns: (array-like)
        """
        N = len(particles)
        self.t = t
        (y, C, h, R) = self.get_meas_dynamics(y=y, t=t)
        self.kf.set_dynamics(C=C, R=R, h_k=h)
        (zl, Pl) = self.get_states(particles)
        logpy = numpy.empty(N)
        for i in range(N):
            # Calculate l3 according to (19b)
            l3 = self.calc_l3(y, zl[i], Pl[i])
            (_tmp, ld) = numpy.linalg.slogdet(self.kf.R)
            tmp = numpy.linalg.solve(self.kf.R, l3)
            logpy[i] = -0.5 * (ld + numpy.trace(tmp))

        return logpy

    def eval_logp_y_val_grad(
        self,
        particles: numpy.ndarray,
        y: Any,
        t: float,
    ) -> tuple[float | numpy.ndarray, numpy.ndarray]:
        """
        Evaluate value and gradient of log p(y_t|x_t)

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns: ((array-like), (array-like))
        """
        N = len(particles)
        lparam = len(self.params)
        (y, C, h, R) = self.get_meas_dynamics(y=y, t=t)
        (C_grad, h_grad, R_grad) = self.get_meas_dynamics_grad(y=y, t=t)
        logpy_grad = numpy.zeros(lparam)
        if C_grad is None and h_grad is None and R_grad is None:
            logpy = self.eval_logp_y(particles, y, t)
        else:
            self.kf.set_dynamics(C=C, R=R, h_k=h)
            Rcho = scipy.linalg.cho_factor(self.kf.R, check_finite=False)
            ld = numpy.sum(numpy.log(numpy.diagonal(Rcho[0]))) * 2
            (zl, Pl) = self.get_states(particles)
            logpy = 0.0

            if R_grad is None:
                R_grad = numpy.zeros((len(self.params), len(y), len(y)))

            for i in range(N):
                # Calculate l3 according to (19b)
                (l3, l3_grad) = self.calc_l3_grad(y, zl[i], Pl[i])
                tmp = scipy.linalg.cho_solve(Rcho, l3)
                logpy += -0.5 * (ld + numpy.trace(tmp))

                for j in range(len(self.params)):
                    logpy_grad[j] -= 0.5 * mlnlg_compute.compute_logprod_derivative(
                        Rcho,
                        R_grad[j],
                        l3,
                        l3_grad[j],
                    )

        return (logpy, logpy_grad)

    def get_pred_dynamics_grad(self, u: Any, t: float) -> tuple[Any, Any, Any]:
        """
        Override this method if (A, f, Q) depends on the parameters

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamps

        Returns:
         (A_grad, f_grad, Q_grad): Element-wise gradients with respect to all
         the parameters for the system matrices
        """
        return (None, None, None)

    def get_meas_dynamics_grad(self, y: Any, t: float) -> tuple[Any, Any, Any]:
        """
        Override this method if (C, h, R) depends on the parameters

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurment
         - t (float): time stamps

        Returns:
         (C_grad, h_grad, R_grad): Element-wise gradients with respect to all
         the parameters for the system matrices
        """
        return (None, None, None)

    def get_initial_grad(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Default implementation has no dependence on xi, override if needed

        Calculate gradient estimate of initial state for linear state condition on the
        nonlinear estimate

        Args:
         - xi0 (array-like): Initial xi states

        Returns:
         (z,P): z is a list of element-wise gradients for the inital mean values,
         P is a list of element-wise gradients for the covariance matrices
        """
        lparam = len(self.params)
        return (
            numpy.zeros((lparam, self.kf.lz, 1)),
            numpy.zeros((lparam, self.kf.lz, self.kf.lz)),
        )

    def calc_l1(
        self,
        z: numpy.ndarray,
        P: numpy.ndarray,
        z0: numpy.ndarray,
        P0: numpy.ndarray,
    ) -> numpy.ndarray:
        """internal helper function"""
        return _nb_calc_l1(z, P, z0)

    def calc_l1_grad(
        self,
        z: numpy.ndarray,
        P: numpy.ndarray,
        z0: numpy.ndarray,
        P0: numpy.ndarray,
        z0_grad: numpy.ndarray | None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """internal helper function"""
        return _nb_calc_l1_grad(z, P, z0, z0_grad, len(self.params), self.kf.lz)

    def calc_l2(
        self,
        zn: numpy.ndarray,
        Pn: numpy.ndarray,
        z: numpy.ndarray,
        P: numpy.ndarray,
        A: numpy.ndarray,
        f: numpy.ndarray,
        M: numpy.ndarray,
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """internal helper function"""
        return _nb_calc_l2(zn, Pn, z, P, A, f, M)

    def calc_l2_grad(
        self,
        zn: numpy.ndarray,
        Pn: numpy.ndarray,
        z: numpy.ndarray,
        P: numpy.ndarray,
        A: numpy.ndarray,
        f: numpy.ndarray,
        M: numpy.ndarray,
        A_grad: numpy.ndarray | None,
        f_grad: numpy.ndarray | None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """internal helper function"""
        return _nb_calc_l2_grad(
            zn, Pn, z, P, A, f, M, A_grad, f_grad, len(self.params), self.kf.lz
        )

    def calc_l3(
        self,
        y: numpy.ndarray,
        z: numpy.ndarray,
        P: numpy.ndarray,
    ) -> numpy.ndarray:
        """internal helper function"""
        return _nb_calc_l3(y.reshape((-1, 1)), z, P, self.kf.C, self.kf.h_k)

    def calc_l3_grad(
        self,
        y: numpy.ndarray,
        z: numpy.ndarray,
        P: numpy.ndarray,
        C_grad: numpy.ndarray | None,
        h_grad: numpy.ndarray | None,
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """internal helper function"""
        return _nb_calc_l3_grad(
            y.reshape((-1, 1)),
            z,
            P,
            self.kf.C,
            self.kf.h_k,
            C_grad,
            h_grad,
            len(self.params),
            len(y),
        )
