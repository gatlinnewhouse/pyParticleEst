"""Model definition for base class for hierarchical systems

@author: Jerker Nordh
"""

import abc
import copy
import math
from typing import Any

import numpy as np

import pyparticleest.utils.kalman as kalman
from pyparticleest.interfaces import FFBSiRS
from pyparticleest.models.rbpf import RBPSBase


class HierarchicalBase(RBPSBase, abc.ABC):
    """
    Base class for Rao-Blackwellization of hierarchical models

    Args:
     - len_xi (int): number of nonlinear states
     - len_z (int): number of linear states
    """

    def __init__(self, len_xi: int, len_z: int, **kwargs: Any) -> None:
        self.lxi = len_xi
        super().__init__(lz=len_z, **kwargs)

    def measure(self, particles: np.ndarray, y: Any, t: float) -> np.ndarray:
        """
        Return the log-pdf value of the measurement and update the statistics
        for the linear states

        Args:

         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like):  measurement
         - t (float): time-stamp

        Returns:
         (array-like) with first dimension = N, logp(y|x^i)
        """

        lyxi = self.measure_nonlin(particles, y, t)
        (xil, zl, Pl) = self.get_states(particles)
        N = len(particles)
        (y, Cz, hz, Rz) = self.get_lin_meas_dynamics(particles, y, t)
        if Cz is None:
            Cz = np.repeat(self.kf.C[np.newaxis, :, :], N, axis=0)
        if hz is None:
            hz = np.repeat(self.kf.h_k[np.newaxis, :, :], N, axis=0)
        if Rz is None:
            Rz = np.repeat(self.kf.R[np.newaxis, :, :], N, axis=0)

        lyz = np.empty_like(lyxi)
        for i in range(len(zl)):
            lyz[i] = self.kf.measure_full(
                np.asarray(y).reshape((-1, 1)),
                zl[i],
                Pl[i],
                Cz[i],
                hz[i],
                Rz[i],
            )

        self.set_states(particles, xil, zl, Pl)
        return lyxi + lyz

    def calc_cond_dynamics(
        self,
        particles: np.ndarray,
        xi_next: np.ndarray,
        u: Any,
        t: float,
    ) -> tuple[Any, Any, Any]:
        """
        Calculates the linear dynamics for each particle

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - xi_next (array-like): next non linear state
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (Az, fz, Qz):
          - Az (array-like): Az matrix for each particle
          - fz (array-like): fz vector for each particle
          - Qz (array-lie): Noise covariance for each particle

        """
        (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles, u, t)
        return (Az, fz, Qz)

    def meas_xi_next(
        self,
        particles: np.ndarray,
        xi_next: np.ndarray,
        u: Any,
        t: float,
    ) -> np.ndarray:
        # There is no information in the next nonlinear state about the
        # current linear states for this class of models
        return particles

    def logp_xnext(
        self,
        particles: np.ndarray,
        next_part: np.ndarray,
        u: Any,
        t: float,
    ) -> np.ndarray:
        """
        Return the log-pdf value for the possible future state 'next_part' given
        input u

        If Nn = 1 all particle are evaluated against the same future state,
        otherwise N must equal Nn and each particle is only evaluated against
        the future state with the same index.

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - next_part (array-like): future states, with first dimension = Nn
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (array-like):
          log-probability of the future state for each particle

        """
        N = len(particles)
        Nn = len(next_part)
        if N > 1 and Nn == 1:
            next_part = np.repeat(next_part, N, 0)

        lpz = np.empty(N)

        (Az, fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles, u, t)
        (_xil, zl, Pl) = self.get_states(particles)
        zln = np.empty_like(zl)
        Pln = np.empty_like(Pl)

        lpxi = self.logp_xnext_xi(particles, next_part[:, : self.lxi], u, t).ravel()

        for i in range(N):
            # Predict z_{t+1}
            (zln[i], Pln[i]) = self.kf.predict_full(zl[i], Pl[i], Az[i], fz[i], Qz[i])

            lpz[i] = kalman.lognormpdf(
                next_part[i][self.lxi : (self.lxi + self.kf.lz)].reshape((-1, 1))
                - zln[i],
                Pln[i],
            )

        return lpxi + lpz

    def sample_smooth(
        self,
        part: np.ndarray,
        ptraj: list[Any] | None,
        anc: np.ndarray,
        future_trajs: list[Any] | None,
        find: np.ndarray | None,
        ut: np.ndarray,
        yt: np.ndarray,
        tt: np.ndarray,
        cur_ind: int,
    ) -> np.ndarray:
        """
        Sampled linear state conditioned on future_trajs

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
        M = len(part)
        res = np.zeros((M, self.lxi + self.kf.lz + 2 * self.kf.lz**2))
        for j in range(M):
            partj = np.copy(part[j : j + 1])
            (xil, zl, Pl) = self.get_states(
                part,
            )
            if future_trajs is not None:
                (A, f, Q, _, _, _) = self.get_lin_pred_dynamics_int(
                    partj,
                    ut[cur_ind],
                    tt[cur_ind],
                )
                # Measure the sampled next state,
                self.kf.measure_full(
                    future_trajs[0]
                    .pa.part[find[j], self.lxi : (self.lxi + self.kf.lz)]
                    .reshape((-1, 1)),
                    zl[0],
                    Pl[0],
                    C=A[0],
                    h_k=f[0],
                    R=Q[0],
                )

            xi = copy.copy(xil[0]).ravel()
            # Sample the linear variables, the full conditional density
            # is recovred later in the post_smoothing step
            rng = np.random.default_rng()
            z = rng.multivariate_normal(zl[0].ravel(), Pl[0]).ravel()
            res[j, : (self.lxi + self.kf.lz)] = np.hstack((xi, z))
        return res

    @abc.abstractmethod
    def logp_xnext_xi(
        self,
        particles: np.ndarray,
        next_xi: np.ndarray,
        u: Any,
        t: float,
    ) -> np.ndarray:
        """
        Evaluate the log-probability of the next nonlinear state

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - next_xi (array-like): future nonlinear state
         - u (array-like): input signal
         - t (float): time stamp

        Returns:
         (array-like):
          log-probability of the future nonlinear state for each particle
        """
        pass

    @abc.abstractmethod
    def calc_xi_next(
        self,
        particles: np.ndarray,
        u: Any,
        t: float,
        noise: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the next nonlinear state given the input and noise
        realization

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp
         - noise (array-like): noise realization for each particle

        Returns:
         (array-like): xi values for future particles
        """
        pass

    @abc.abstractmethod
    def measure_nonlin(
        self,
        particles: np.ndarray,
        y: Any,
        t: float,
    ) -> np.ndarray:
        """
        Measurement probability for the nonlinear parts of the measurement
        equations

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - y (array-like): measurement
         - t (float): time stamp

        Returns:
         (array-like):
          log-probability of the measurement for each particle
        """
        pass


class HierarchicalRSBase(HierarchicalBase, FFBSiRS):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def logp_xnext_max(self, particles: np.ndarray, u: Any, t: float) -> float:
        """
        Calculate maximum value of the logp_xnext function, used for
        rejection sampling


        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

         Returns:
          (float) max value over all particle and possible future states
        """
        N = len(particles)
        lpxi = self.logp_xnext_xi_max(particles, u, t)
        (Az, _fz, Qz, _, _, _) = self.get_lin_pred_dynamics_int(particles, u, t)
        lpz = np.empty_like(lpxi)
        (_xil, _zl, Pl) = self.get_states(particles)
        nx = len(Qz[0])
        for i in range(N):
            # Predict z_{t+1}
            Pn = Az[i].dot(Pl[i]).dot(Az[i].T) + Qz[i]
            lpz[i] = -0.5 * nx * math.log(2 * math.pi) + np.linalg.slogdet(Pn)[1]
        return np.max(lpxi + lpz)

    @abc.abstractmethod
    def logp_xnext_xi_max(
        self,
        particles: np.ndarray,
        u: Any,
        t: float,
    ) -> np.ndarray:
        """
        Maximum for nonlinear part of the logp_xnext, called from
        logp_xnext_max

        Args:
         - particles  (array-like): Model specific representation
           of all particles, with first dimension = N (number of particles)
         - u (array-like): input signal
         - t (float): time stamp

         Returns:
          (array-like) max value for each particle
        """
        pass
