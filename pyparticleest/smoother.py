"""Collection of smoothing algorithms and support classes for using them

@author: Jerker Nordh
"""

from typing import Any

import numba as nb
import numpy as np

from . import filter as pf
from .filter import ParticleApproximation, TrajectoryStep


@nb.njit(cache=True)
def _nb_adaptive_stop_update(
    mk: int, ak: int, pk: float, Pk: float, sv: float, sw: float
):
    # meas update for adaptive stop
    mk2 = mk * mk
    sw2 = sw * sw
    pk = pk + (mk * Pk) / (mk2 * Pk + sw2) * (ak - mk * pk)
    Pk = (1 - (mk2 * Pk) / (mk2 * Pk + sw2)) * Pk
    # predict
    pk = (1 - ak / mk) * pk
    Pk = (1 - ak / mk) ** 2 * Pk + sv * sv
    return pk, Pk


def bsi_full(
    model: Any,
    pa: ParticleApproximation,
    ptraj: list[Any],
    pind: np.ndarray,
    future_trajs: list[Any],
    find: np.ndarray,
    ut: np.ndarray,
    yt: np.ndarray,
    tt: np.ndarray,
    cur_ind: int,
) -> np.ndarray:
    r"""Perform backward simulation by drawing particles from
    the categorical distribution with weights given by
    \omega_{t|T}^i = \omega_{t|t}^i*p(x_{t+1}|x^i)

    Args:
    - pa (ParticleApproximation): particles approximation from which to sample
    - model (FFBSi): model defining probability density function
    - future_trajs (array-like): trajectory estimate of {t+1:T}
    - ut (array-like): inputs signal for {t:T}
    - yt (array-like): measurements for {t:T}
    - tt (array-like): time stamps for {t:T}

    """
    M = len(find)
    N = len(pa.w)
    res = np.empty(M, dtype=int)

    for j in range(M):
        currfind = find[j] * np.ones((N,), dtype=int)
        p_next = model.logp_xnext_full(
            pa.part,
            ptraj,
            pind,
            future_trajs,
            currfind,
            ut=ut,
            yt=yt,
            tt=tt,
            cur_ind=cur_ind,
        )

        w = pa.w + p_next
        w = w - np.max(w)
        w_norm = np.exp(w)
        w_norm /= np.sum(w_norm)
        res[j] = pf.sample(w_norm, 1)
    return res


def bsi_rs(
    model: Any,
    pa: ParticleApproximation,
    ptraj: list[Any],
    pind: np.ndarray,
    future_trajs: list[Any],
    find: np.ndarray,
    ut: np.ndarray,
    yt: np.ndarray,
    tt: np.ndarray,
    cur_ind: int,
    maxpdf: float,
    max_iter: int,
) -> np.ndarray:
    r"""Perform backward simulation by using rejection sampling to draw particles
    from the categorical distribution with weights given by
    \omega_{t|T}^i = \omega_{t|t}^i*p(x_{t+1}|x^i)

    Args:
     - pa (ParticleApproximation): particles approximation from which to sample
     - model (FFBSi): model defining probability density function
     - future_trajs (array-like): trajectory estimate of {t+1:T}
     - ut (array-like): inputs signal for {t:T}
     - yt (array-like): measurements for {t:T}
     - tt (array-like): time stamps for {t:T}
     - maxpdf (float): argmax p(x_{t+1:T}|x_t)
     - max_iter (int): number of attempts before falling back to bsi_full

    """
    M = len(find)
    todo = np.arange(M)
    res = np.empty(M, dtype=int)
    weights = np.copy(pa.w)
    weights -= np.max(weights)
    weights = np.exp(weights)
    weights /= np.sum(weights)
    rng = np.random.default_rng()
    for _i in range(max_iter):
        ind = rng.permutation(pf.sample(weights, len(todo)))
        pn = model.logp_xnext_full(
            pa.part[ind],
            ptraj,
            pind[ind],
            future_trajs,
            todo,
            ut=ut,
            yt=yt,
            tt=tt,
            cur_ind=cur_ind,
        )
        test = np.log(rng.uniform(size=len(todo)))
        accept = test < pn - maxpdf
        res[todo[accept]] = ind[accept]
        todo = todo[~accept]
        if len(todo) == 0:
            return res

    # TODO, is there an efficient way to store those weights
    # already calculated to avoid double work, or will that
    # take more time than simply evaulating them all again?
    res[todo] = bsi_full(
        model,
        pa,
        ptraj,
        pind,
        future_trajs,
        todo,
        ut=ut,
        yt=yt,
        tt=tt,
        cur_ind=cur_ind,
    )
    return res


def bsi_rsas(
    model: Any,
    pa: ParticleApproximation,
    ptraj: list[Any],
    pind: np.ndarray,
    future_trajs: list[Any],
    find: np.ndarray,
    ut: np.ndarray,
    yt: np.ndarray,
    tt: np.ndarray,
    cur_ind: int,
    maxpdf: float,
    x1: float,
    P1: float,
    sv: float,
    sw: float,
    ratio: float,
) -> np.ndarray:
    """Perform backward simulation by using rejection sampling to draw particles
    from the categorical distribution with weights given by
    \\omega_{t|T}^i = \\omega_{t|t}^i*p(x_{t+1}|x^i)

    Adaptively determine when to to fallback to bsi_full by using a Kalman
    filter to track the prediceted acceptance rate of the rejection sampler

    Based on "Adaptive Stopping for Fast Particle Smoothing" by
    Taghavi, Lindsten, Svensson and Sch\"{o}n. See orignal article for details
    about the meaning of the Kalman filter paramters

    Args:
     - pa (ParticleApproximation): particles approximation from which to sample
     - model (FFBSi): model defining probability density function
     - future_trajs (array-like): trajectory estimate of {t+1:T}
     - ut (array-like): inputs signal for {t:T}
     - yt (array-like): measurements for {t:T}
     - tt (array-like): time stamps for {t:T}
     - maxpdf (float): argmax p(x_{t+1:T}|x_t)
     - x1 (float): initial state of Kalman filter
     - P1 (float): initial covariance of Kalman filter estimate
     - sv (float): process noise (for Kalman filter)
     - sw (float): measurement noise (for Kalman filter)
     - ratio (float): cost ration of running rejection sampling compared to
       switching to the full bsi (D_0 / D_1)

    """
    M = len(find)
    todo = np.arange(M)
    res = np.empty(M, dtype=int)
    weights = np.copy(pa.w)
    weights -= np.max(weights)
    weights = np.exp(weights)
    weights /= np.sum(weights)
    pk = x1
    Pk = P1
    stop_criteria = ratio / len(pa)
    rng = np.random.default_rng()
    while True:
        ind = rng.permutation(pf.sample(weights, len(todo)))
        pn = model.logp_xnext_full(
            pa.part[ind],
            ptraj,
            pind[ind],
            future_trajs,
            todo,
            ut=ut,
            yt=yt,
            tt=tt,
            cur_ind=cur_ind,
        )
        test = np.log(rng.uniform(size=len(todo)))
        accept = test < pn - maxpdf
        ak = np.sum(accept)
        mk = len(todo)
        res[todo[accept]] = ind[accept]
        todo = todo[~accept]
        if len(todo) == 0:
            return res
        pk, Pk = _nb_adaptive_stop_update(mk, ak, pk, Pk, sv, sw)
        if pk < stop_criteria:
            break

    res[todo] = bsi_full(
        model,
        pa,
        ptraj,
        pind,
        future_trajs,
        todo,
        ut=ut,
        yt=yt,
        tt=tt,
        cur_ind=cur_ind,
    )
    return res


def bsi_mcmc(
    model: Any,
    pa: ParticleApproximation,
    ptraj: list[Any],
    pind: np.ndarray,
    future_trajs: list[Any],
    find: np.ndarray,
    ut: np.ndarray,
    yt: np.ndarray,
    tt: np.ndarray,
    cur_ind: int,
    R: int,
    ancestors: np.ndarray,
) -> np.ndarray:
    r"""Perform backward simulation by using Metropolis-Hastings to draw particles
    from the categorical distribution with weights given by
    \omega_{t|T}^i = \omega_{t|t}^i*p(x_{t+1}|x^i)

    Args:
     - pa (ParticleApproximation): particles approximation from which to sample
     - model (FFBSi): model defining probability density function
     - future_trajs (array-like): trajectory estimate of {t+1:T}
     - ut (array-like): inputs signal for {t:T}
     - yt (array-like): measurements for {t:T}
     - tt (array-like): time stamps for {t:T}
     - R (int): number of iterations to run the markov chain
     - ancestor (array-like): ancestor of each particle from the particle filter

    """
    # Perform backward simulation using an MCMC sampler proposing new
    # backward particles, initialized with the filtered trajectory

    M = len(find)
    ind = ancestors
    weights = np.copy(pa.w)
    weights -= np.max(weights)
    weights = np.exp(weights)
    weights /= np.sum(weights)

    pcurr = model.logp_xnext_full(
        pa.part[ind],
        ptraj,
        pind[ind],
        future_trajs,
        find,
        ut=ut,
        yt=yt,
        tt=tt,
        cur_ind=cur_ind,
    )
    rng = np.random.default_rng()
    for _j in range(R):
        propind = rng.permutation(pf.sample(weights, M))
        pprop = model.logp_xnext_full(
            pa.part[propind],
            ptraj,
            pind[propind],
            future_trajs,
            find,
            ut=ut,
            yt=yt,
            tt=tt,
            cur_ind=cur_ind,
        )
        diff = pprop - pcurr
        diff[diff > 0.0] = 0.0
        test = np.log(rng.uniform(size=M))
        accept = test < diff
        ind[accept] = propind[accept]
        pcurr[accept] = pprop[accept]

    return ind


class SmoothTrajectory:
    """Create smoothed trajectory from filtered trajectory

    Args:
     - pt (ParticleTrajectory): Forward estimates (typically
       generated by a ParticleFilter), combined with inputs and measurements
     - M (int): Number of smoothed trajectories to create
     - method (string): Smoothing method to use
     - options (dict): options to pass on to the smoothing algorithm

    """

    def __init__(
        self,
        pt: Any,
        M: int = 1,
        method: str = "full",
        options: dict[str, Any] | None = None,
    ) -> None:

        self.traj = None

        self.u = np.copy(pt.uvec)
        self.y = np.copy(pt.yvec)
        self.t = np.copy(pt.tvec)
        self.M = M

        self.model = pt.pf.model
        if method == "full" or method == "mcmc" or method == "rs" or method == "rsas":
            self.perform_bsi(pt=pt, M=M, method=method, options=options)
        elif method == "ancestor":
            self.perform_ancestors(pt=pt, M=M)
        elif method == "mhips" or method == "mhips_reduced":
            reduced = method != "mhips"
            # Initialize using forward trajectories
            self.traj = self.perform_ancestors_int(pt=pt, M=M)

            R = options.get("R", 10)
            for _i in range(R):
                # Recover filtering statistics for linear states
                if hasattr(self.model, "pre_mhips_pass"):
                    self.traj = self.model.pre_mhips_pass(self)
                self.traj = self.perform_mhips_pass(options=options, reduced=reduced)

        elif method == "mhbp":
            R = options.get("R", 10)
            self.perform_mhbp(pt=pt, M=M, R=R)
        else:
            msg = f"Unknown smoother: {method}"
            raise ValueError(msg)

        if hasattr(self.model, "post_smoothing"):
            self.traj = self.model.post_smoothing(self)

    def __len__(self) -> int:
        if self.traj is None:
            return 0
        return len(self.traj)

    def perform_ancestors(self, pt: Any, M: int) -> None:
        """Create smoothed trajectories by taking the forward trajectories

        Args:
         - pt (ParticleTrajectory): forward trajetories
         - M (int): number of trajectories to createa

        """
        self.traj = self.perform_ancestors_int(pt, M)

        if hasattr(self.model, "post_smoothing"):
            # Do e.g. constrained smoothing for RBPS models
            self.traj = self.model.post_smoothing(self)

    def calculate_ancestors(self, pt: Any, ind: np.ndarray) -> np.ndarray:
        T = len(pt)
        M = len(ind)
        ancestors = pt[T - 1].ancestors[ind]
        find = np.arange(M, dtype=int)
        last_part = self.model.sample_smooth(
            part=pt[T - 1].pa.part[ind],
            ptraj=pt[: (T - 1)],
            anc=ancestors,
            future_trajs=None,
            find=None,
            ut=self.u,
            yt=self.y,
            tt=self.t,
            cur_ind=T - 1,
        )

        traj = np.empty((len(pt),), dtype=object)
        traj[T - 1] = TrajectoryStep(
            ParticleApproximation(last_part),
            np.arange(M, dtype=int),
        )

        for t in reversed(range(T - 1)):
            ind = ancestors
            ancestors = pt[t].ancestors[ind]
            # Select 'previous' particle
            traj[t] = TrajectoryStep(
                ParticleApproximation(
                    self.model.sample_smooth(
                        part=pt[t].pa.part[ind],
                        ptraj=pt[:t],
                        anc=ancestors,
                        future_trajs=traj[(t + 1) :],
                        find=find,
                        ut=self.u,
                        yt=self.y,
                        tt=self.t,
                        cur_ind=t,
                    ),
                ),
                ancestors=find,
            )
        return traj

    def perform_ancestors_int(self, pt: Any, M: int) -> np.ndarray:
        """Create smoothed trajectories by taking the forward trajectories, don't
        perform post processing

        Args:
         - pt (ParticleTrajectory): forward trajetories
         - M (int): number of trajectories to createa

        """
        tmp = np.copy(pt[-1].pa.w)
        tmp -= np.max(tmp)
        tmp = np.exp(tmp)
        tmp = tmp / np.sum(tmp)
        ind = pf.sample(tmp, M)

        return self.calculate_ancestors(pt, ind)

    def perform_bsi(
        self,
        pt: Any,
        M: int,
        method: str,
        options: dict[str, Any],
    ) -> None:
        """Create smoothed trajectories using Backward Simulation

        Args:
         - pt (ParticleTrajectory): forward trajetories
         - M (int): number of trajectories to createa
         - method (string): Type of backward simulation to use
         - optiones (dict): Parameters to the backward simulator

        """
        # Sample from end time estimates
        tmp = np.copy(pt[-1].pa.w)
        tmp -= np.max(tmp)
        tmp = np.exp(tmp)
        tmp = tmp / np.sum(tmp)
        ind = pf.sample(tmp, M)
        ancestors = pt[-1].ancestors[ind]
        last_part = self.model.sample_smooth(
            part=pt[-1].pa.part[ind],
            ptraj=pt[:-1],
            anc=ancestors,
            future_trajs=None,
            find=None,
            ut=self.u,
            yt=self.y,
            tt=self.t,
            cur_ind=len(pt) - 1,
        )
        self.traj = np.empty((len(pt),), dtype=object)
        self.traj[-1] = TrajectoryStep(
            ParticleApproximation(last_part),
            np.arange(M, dtype=int),
        )

        if (
            method == "full"
            or method == "mcmc"
            or method == "ancestor"
            or method == "mhips"
        ):
            pass
        elif method == "rs":
            max_iter = options["R"]
        elif method == "rsas":
            x1 = options["x1"]
            P1 = options["P1"]
            sv = options["sv"]
            sw = options["sw"]
            ratio = options["ratio"]
        else:
            msg = f"Unknown sampler: {method}"
            raise ValueError(msg)

        find = np.arange(M, dtype=int)

        for cur_ind in reversed(range(len(pt) - 1)):
            ft = self.traj[(cur_ind + 1) :]
            ut = self.u
            yt = self.y
            tt = self.t

            if method == "rs":
                ind = bsi_rs(
                    self.model,
                    pt[cur_ind].pa,
                    pt[:cur_ind],
                    pt[cur_ind].ancestors,
                    ft,
                    find,
                    ut=ut,
                    yt=yt,
                    tt=tt,
                    cur_ind=cur_ind,
                    maxpdf=options["maxpdf"][cur_ind],
                    max_iter=int(max_iter),
                )
            elif method == "rsas":
                ind = bsi_rsas(
                    self.model,
                    pt[cur_ind].pa,
                    pt[:cur_ind],
                    pt[cur_ind].ancestors,
                    ft,
                    find,
                    ut=ut,
                    yt=yt,
                    tt=tt,
                    cur_ind=cur_ind,
                    maxpdf=options["maxpdf"][cur_ind],
                    x1=x1,
                    P1=P1,
                    sv=sv,
                    sw=sw,
                    ratio=ratio,
                )
            elif method == "mcmc":
                ind = bsi_mcmc(
                    self.model,
                    pt[cur_ind].pa,
                    pt[:cur_ind],
                    pt[cur_ind].ancestors,
                    ft,
                    find,
                    ut=ut,
                    yt=yt,
                    tt=tt,
                    cur_ind=cur_ind,
                    R=options["R"],
                    ancestors=ancestors,
                )
                ancestors = pt[cur_ind].ancestors[ind]
            elif method == "full":
                ind = bsi_full(
                    self.model,
                    pt[cur_ind].pa,
                    pt[:cur_ind],
                    pt[cur_ind].ancestors,
                    ft,
                    find,
                    ut=ut,
                    yt=yt,
                    tt=tt,
                    cur_ind=cur_ind,
                )
            elif method == "ancestor":
                ind = ancestors

            ancestors = pt[cur_ind].ancestors[ind]
            # Select 'previous' particle
            find = np.arange(M, dtype=int)
            tmp = self.model.sample_smooth(
                part=pt[cur_ind].pa.part[ind],
                ptraj=pt[:cur_ind],
                anc=ancestors,
                future_trajs=ft,
                find=find,
                ut=ut,
                yt=yt,
                tt=tt,
                cur_ind=cur_ind,
            )
            self.traj[cur_ind] = TrajectoryStep(
                ParticleApproximation(tmp),
                np.arange(M, dtype=int),
            )

    #        if hasattr(self.model, 'post_smoothing'):
    #            # Do e.g. constrained smoothing for RBPS models
    #            self.traj = self.model.post_smoothing(self)

    def perform_mhbp(self, pt: Any, M: int, R: int, reduced: bool = False) -> None:
        """Create smoothed trajectories using Metropolis-Hastings Backward Propeser

        Args:
         - pt (ParticleTrajectory): forward trajetories
         - M (int): number of trajectories to createa
         - R (int): Number of proposal for each time step

        """
        T = len(pt)
        ut = self.u
        yt = self.y
        tt = self.t
        straj = np.empty((T,), dtype=object)

        # Initialise from end time estimates
        tmp = np.copy(pt[-1].pa.w)
        tmp -= np.max(tmp)
        tmp = np.exp(tmp)
        tmp = tmp / np.sum(tmp)
        cind = pf.sample(tmp, M)
        find = np.arange(M, dtype=int)
        #        anc = pt[-1].ancestors[cind]
        #        last_part = self.model.sample_smooth(part=pt[-1].pa.part[cind],
        #                                             ptraj=pt[:-1],
        #                                             anc=anc,
        #                                             future_trajs=None,
        #                                             find=find,
        #                                             ut=ut, yt=yt, tt=tt,
        #                                             cur_ind=T - 1)

        for t in reversed(range(T)):
            # Initialise from filtered estimate
            ft = straj[t + 1 :] if t < T - 1 else None

            # Initialize with filterted estimates
            pnew = pt[t].pa.part[cind]
            if t > 0:
                anc = pt[t].ancestors[cind]
                tmp = np.copy(pt[t - 1].pa.w)
                tmp -= np.max(tmp)
                tmp = np.exp(tmp)
                tmp = tmp / np.sum(tmp)
                ptraj = pt[:t]
            else:
                ptraj = None

            for _ in range(R):
                if t > 0:
                    # Propose new ancestors
                    panc = pf.sample(tmp, M)

                (pnew, acc) = mc_step(
                    model=self.model,
                    part=pnew,
                    ptraj=ptraj,
                    pind_prop=panc,
                    pind_curr=anc,
                    future_trajs=ft,
                    find=find,
                    ut=ut,
                    yt=yt,
                    tt=tt,
                    cur_ind=t,
                    reduced=reduced,
                )

                anc[acc] = panc[acc]

            fpart = self.model.sample_smooth(
                part=pnew,
                ptraj=ptraj,
                anc=anc,
                future_trajs=ft,
                find=find,
                ut=ut,
                yt=yt,
                tt=tt,
                cur_ind=t,
            )
            straj[t] = TrajectoryStep(ParticleApproximation(fpart))
            cind = anc

        self.traj = straj

        if hasattr(self.model, "post_smoothing"):
            # Do e.g. constrained smoothing for RBPS models
            self.traj = self.model.post_smoothing(self)

    def perform_mhips_pass(
        self,
        options: dict[str, Any] | None,
        reduced: bool = False,
    ) -> np.ndarray:
        """Runs MHIPS with the proposal density q as p(x_{t+1}|x_t)

        Args:
         - pt (ParticleTrajectory): Forward esimates
         - M (int): Number of backward trajectories
         - options (None): Unused

        """
        T = len(self.traj)
        # Handle last time-step seperately
        ut = self.u
        yt = self.y
        tt = self.t
        pind = np.arange(self.M, dtype=int)

        straj = np.empty((T,), dtype=object)
        pt = self.traj[: T - 1]
        (part, _acc) = mc_step(
            model=self.model,
            part=self.traj[-1].pa.part,
            ptraj=pt,
            pind_prop=pind,
            pind_curr=pind,
            future_trajs=None,
            find=pind,
            ut=ut,
            yt=yt,
            tt=tt,
            cur_ind=T - 1,
            reduced=reduced,
        )

        tmp = np.copy(
            self.model.sample_smooth(
                part=part,
                ptraj=pt,
                anc=pind,
                future_trajs=None,
                find=pind,
                ut=ut,
                yt=yt,
                tt=tt,
                cur_ind=T - 1,
            ),
        )
        straj[T - 1] = TrajectoryStep(ParticleApproximation(tmp), pind)

        for i in reversed(range(1, (T - 1))):
            ft = straj[(i + 1) :]
            pt = self.traj[:i]
            (part, _acc) = mc_step(
                model=self.model,
                part=self.traj[i].pa.part,
                ptraj=pt,
                pind_prop=pind,
                pind_curr=pind,
                future_trajs=ft,
                find=pind,
                ut=ut,
                yt=yt,
                tt=tt,
                cur_ind=i,
                reduced=reduced,
            )

            # The data dimension is not necessarily the same, since self.traj
            # contains data that has been processed by "post_smoothing".
            # This implementation assumes that the existing trajectory contains
            # enough space to hold the data, if that is not the case the
            # model class should extend "pre_mhips_pass" to allocate a larger
            # array
            # self.traj[i].pa.part[acc] = prop[acc]
            tmp = self.model.sample_smooth(
                part=part,
                ptraj=pt,
                anc=pind,
                future_trajs=ft,
                find=pind,
                ut=ut,
                yt=yt,
                tt=tt,
                cur_ind=i,
            )

            straj[i] = TrajectoryStep(ParticleApproximation(tmp), pind)

        ft = straj[1:]

        (part, _acc) = mc_step(
            model=self.model,
            part=self.traj[0].pa.part,
            ptraj=None,
            pind_prop=None,
            pind_curr=None,
            future_trajs=ft,
            find=pind,
            ut=ut,
            yt=yt,
            tt=tt,
            cur_ind=0,
            reduced=reduced,
        )

        tmp = self.model.sample_smooth(
            part,
            ptraj=None,
            anc=pind,
            future_trajs=ft,
            find=pind,
            ut=ut,
            yt=yt,
            tt=tt,
            cur_ind=0,
        )

        straj[0] = TrajectoryStep(ParticleApproximation(tmp), pind)

        return straj

    def get_smoothed_estimates(self) -> np.ndarray:
        """Return smoothed estimates (must first have called 'simulate')

        Returns:
         - (T, N, D) array

        T is the length of the dataset,
        N is the number of particles
        D is the dimension of each particle

        """
        T = len(self.traj)
        N = self.traj[0].pa.part.shape[0]
        D = self.traj[0].pa.part.shape[1]

        est = np.empty((T, N, D))

        for t in range(T):
            est[t] = self.traj[t].pa.part

        return est


def mc_step(
    model: Any,
    part: np.ndarray,
    ptraj: list[Any] | None,
    pind_prop: np.ndarray | None,
    pind_curr: np.ndarray | None,
    future_trajs: list[Any] | None,
    find: np.ndarray,
    ut: np.ndarray,
    yt: np.ndarray,
    tt: np.ndarray,
    cur_ind: int,
    reduced: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform a single iteration of the MCMC sampler used for MHIPS and MHBP

    Args:
     - model: model definition
     - partp_prop (array-like): proposed previous particle
     - partp_prop (array-like): current previous particle
     - up (array-like): input at time t-1
     - tp (array-like): timestamp at time t-1
     - curpart: current accepted paricle
     - yt (array-like): measurement at time t
     - ut (array-like): input at time t
     - tt (array-like): timestamp at time t
     - future_trajs (array-like): particle approximations of {x_{t+1:T|T}}

    """
    # The previously stored values for part already include the measurment from
    # cur_ind, we therefore need to recomputed the sufficient statistics
    # (for Rao-Blackwellized models)
    if ptraj is not None:
        oldpart = np.copy(ptraj[-1].pa.part[pind_curr])
        part = model.cond_predict_single_step(
            part=oldpart,
            past_trajs=ptraj[:-1],
            pind=ptraj[-1].ancestors[pind_curr],
            future_parts=part,
            find=np.arange(len(pind_curr)),
            ut=ut,
            yt=yt,
            tt=tt,
            cur_ind=cur_ind - 1,
        )
    else:
        part = model.cond_sampled_initial(part, tt[cur_ind])

    if reduced:
        if ptraj is not None:
            noise = model.sample_process_noise_full(
                ptraj=ptraj,
                ancestors=pind_prop,
                ut=ut[:cur_ind],
                tt=tt[:cur_ind],
            )

            xprop = np.copy(ptraj[-1].pa.part[pind_prop])

            model.update_full(
                particles=xprop,
                traj=ptraj,
                uvec=ut[:cur_ind],
                yvec=yt[:cur_ind],
                tvec=tt[:cur_ind],
                ancestors=pind_prop,
                noise=noise,
            )
        else:
            xprop = model.create_initial_estimate(len(future_trajs[0].pa.part))

        # Drawing from p(x_{t+1}|x_t), so these will be identical
        logp_prev_prop = 0.0
        logp_prev_curr = 0.0
        logp_q_prop = 0.0
        logp_q_curr = 0.0
    else:
        xprop = model.propose_smooth(
            ptraj=ptraj,
            anc=pind_prop,
            future_trajs=future_trajs,
            find=find,
            yt=yt,
            ut=ut,
            tt=tt,
            cur_ind=cur_ind,
        )

        # Accept/reject new sample
        logp_q_prop = model.logp_proposal(
            prop_part=xprop,
            ptraj=ptraj,
            anc=pind_prop,
            future_trajs=future_trajs,
            find=find,
            yt=yt,
            ut=ut,
            tt=tt,
            cur_ind=cur_ind,
        )
        logp_q_curr = model.logp_proposal(
            prop_part=part,
            ptraj=ptraj,
            anc=pind_curr,
            future_trajs=future_trajs,
            find=find,
            yt=yt,
            ut=ut,
            tt=tt,
            cur_ind=cur_ind,
        )

        if ptraj is not None:
            logp_prev_prop = model.logp_xnext_singlestep(
                part=ptraj[-1].pa.part[pind_prop],
                past_trajs=ptraj[:-1],
                pind=ptraj[-1].ancestors[pind_prop],
                future_parts=xprop,
                find=np.arange(len(xprop), dtype=int),
                ut=ut,
                yt=yt,
                tt=tt,
                cur_ind=cur_ind - 1,
            )
            logp_prev_curr = model.logp_xnext_singlestep(
                part=ptraj[-1].pa.part[pind_curr],
                past_trajs=ptraj[:-1],
                pind=ptraj[-1].ancestors[pind_curr],
                future_parts=part,
                find=np.arange(len(part), dtype=int),
                ut=ut,
                yt=yt,
                tt=tt,
                cur_ind=cur_ind - 1,
            )

        else:
            logp_prev_prop = model.eval_logp_x0(xprop, tt[0])
            logp_prev_curr = model.eval_logp_x0(part, tt[0])

    xpropy = np.copy(xprop)
    curparty = np.copy(part)
    if yt[cur_ind] is not None:
        logp_y_prop = model.measure_full(
            particles=xpropy,
            traj=ptraj,
            uvec=ut[: cur_ind + 1],
            yvec=yt[: (cur_ind + 1)],
            tvec=tt[: cur_ind + 1],
            ancestors=pind_prop,
        )

        logp_y_curr = model.measure_full(
            particles=curparty,
            traj=ptraj,
            uvec=ut[: cur_ind + 1],
            yvec=yt[: (cur_ind + 1)],
            tvec=tt[: cur_ind + 1],
            ancestors=pind_curr,
        )
    else:
        logp_y_prop = 0.0
        logp_y_curr = 0.0

    if future_trajs is not None:
        logp_next_prop = model.logp_xnext_full(
            part=xpropy,
            past_trajs=ptraj,
            pind=pind_prop,
            future_trajs=future_trajs,
            find=find,
            ut=ut,
            yt=yt,
            tt=tt,
            cur_ind=cur_ind,
        )
        logp_next_curr = model.logp_xnext_full(
            part=curparty,
            past_trajs=ptraj,
            pind=pind_curr,
            future_trajs=future_trajs,
            find=find,
            ut=ut,
            yt=yt,
            tt=tt,
            cur_ind=cur_ind,
        )
    else:
        logp_next_prop = 0.0
        logp_next_curr = 0.0

    # Calc ratio
    ratio = (
        (logp_prev_prop - logp_prev_curr)
        + (logp_y_prop - logp_y_curr)
        + (logp_next_prop - logp_next_curr)
        + (logp_q_curr - logp_q_prop)
    )

    rng = np.random.default_rng()
    test = np.log(rng.uniform(size=len(ratio)))
    acc = test < ratio
    curparty[acc] = xpropy[acc]
    return (curparty, acc)
