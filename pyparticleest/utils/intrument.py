"""
Created on Jun 25, 2014

@author: Jerker Nordh
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class OpCount:
    cnt_sample: int = 0
    cnt_update: int = 0
    cnt_measure: int = 0
    cnt_pdfxn: int = 0
    cnt_pdfxn_full: int = 0
    cnt_pdfxnmax: int = 0
    cnt_propsmooth: int = 0
    cnt_pdfsmooth: int = 0
    cnt_eval1st: int = 0
    cnt_eval_logp_x0: int = 0

    def __add__(self, other: "OpCount") -> "OpCount":
        return OpCount(
            **{
                field: getattr(self, field) + getattr(other, field)
                for field in self.__annotations__
            },
        )


class Instrumenter:
    """
    Count number of operations performed

    Wraps all calls and counts the number of calls * number of particles in
    each call.

    Results can be access through the member variables:

        - self.cnt_sample
        - self.cnt_update
        - self.cnt_measure
        - self.cnt_pdfxn
        - self.cnt_pdfxn_full
        - self.cnt_pdfxnmax
        - self.cnt_propsmooth
        - self.cnt_pdfsmooth
        - self.cnt_eval1st

    Args:
     - model: Object of encapsulated model class
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.oc = OpCount()

    def print_statistics(self) -> None:
        pass

    def print_total_ops(self) -> None:
        total = (
            self.oc.cnt_sample
            + self.oc.cnt_update
            + self.oc.cnt_measure
            + self.oc.cnt_pdfxn
            + self.oc.cnt_pdfxnmax
            + self.oc.cnt_propsmooth
            + self.oc.cnt_pdfsmooth
        )

    def create_initial_estimate(self, N: int) -> np.ndarray:
        """Sample N particle from initial distribution"""
        return self.model.create_initial_estimate(N)

    def sample_process_noise_full(
        self,
        ptraj: list[Any],
        ancestors: np.ndarray,
        ut: np.ndarray,
        tt: np.ndarray,
    ) -> np.ndarray:
        """Return process noise for input u"""
        self.oc.cnt_sample += len(ancestors)
        return self.model.sample_process_noise_full(ptraj, ancestors, ut, tt)

    def update_full(
        self,
        particles: np.ndarray,
        traj: list[Any],
        uvec: np.ndarray,
        yvec: np.ndarray,
        tvec: np.ndarray,
        ancestors: np.ndarray,
        noise: np.ndarray,
    ) -> Any:
        """Update estimate using 'data' as input"""
        self.oc.cnt_update += len(particles)
        return self.model.update_full(
            particles,
            traj,
            uvec,
            yvec,
            tvec,
            ancestors,
            noise,
        )

    def measure_full(
        self,
        particles: np.ndarray,
        traj: list[Any],
        uvec: np.ndarray,
        yvec: np.ndarray,
        tvec: np.ndarray,
        ancestors: np.ndarray,
    ) -> np.ndarray:
        """Return the log-pdf value of the measurement"""
        self.oc.cnt_measure += len(particles)
        return self.model.measure_full(particles, traj, uvec, yvec, tvec, ancestors)

    def copy_ind(
        self,
        particles: np.ndarray,
        new_ind: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.model.copy_ind(particles, new_ind)

    def logp_xnext(
        self,
        particles: np.ndarray,
        next_part: np.ndarray,
        u: Any,
        t: float,
    ) -> np.ndarray:
        """Return the log-pdf value for the possible future state 'next' given input u"""
        self.oc.cnt_pdfxn += max(len(particles), len(next_part))
        return self.model.logp_xnext(particles, next_part, u, t)

    def logp_xnext_max_full(
        self,
        part: np.ndarray,
        past_trajs: list[Any] | None,
        pind: np.ndarray,
        uvec: np.ndarray,
        yvec: np.ndarray,
        tvec: np.ndarray,
        cur_ind: int,
    ) -> np.ndarray:
        """Return the log-pdf value for the possible future state 'next' given input u"""
        self.oc.cnt_pdfxnmax += len(part)
        return self.model.logp_xnext_max_full(
            part,
            past_trajs,
            pind,
            uvec,
            yvec,
            tvec,
            cur_ind,
        )

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
        """Update ev. Rao-Blackwellized states conditioned on "next_part" """
        return self.model.sample_smooth(
            part,
            ptraj,
            anc,
            future_trajs,
            find,
            ut,
            yt,
            tt,
            cur_ind,
        )

    def propose_smooth(
        self,
        ptraj: list[Any] | None,
        anc: np.ndarray,
        future_trajs: list[Any] | None,
        find: np.ndarray,
        yt: np.ndarray,
        ut: np.ndarray,
        tt: np.ndarray,
        cur_ind: int,
    ) -> np.ndarray:
        """Sample from a distrubtion q(x_t | x_{t-1}, x_{t+1}, y_t)"""
        N = len(anc) if ptraj is not None else len(find)
        self.oc.cnt_propsmooth += N
        return self.model.propose_smooth(
            ptraj,
            anc,
            future_trajs,
            find,
            yt,
            ut,
            tt,
            cur_ind,
        )

    def logp_proposal(
        self,
        prop_part: np.ndarray,
        ptraj: list[Any] | None,
        anc: np.ndarray,
        future_trajs: list[Any] | None,
        find: np.ndarray,
        yt: np.ndarray,
        ut: np.ndarray,
        tt: np.ndarray,
        cur_ind: int,
    ) -> np.ndarray:
        """Eval log q(x_t | x_{t-1}, x_{t+1}, y_t)"""
        self.oc.cnt_pdfsmooth += len(prop_part)
        return self.model.logp_proposal(
            prop_part,
            ptraj,
            anc,
            future_trajs,
            find,
            yt,
            ut,
            tt,
            cur_ind,
        )

    def logp_xnext_full(
        self,
        part: np.ndarray,
        past_trajs: list[Any] | None,
        pind: np.ndarray,
        future_trajs: list[Any],
        find: np.ndarray,
        ut: np.ndarray,
        yt: np.ndarray,
        tt: np.ndarray,
        cur_ind: int,
    ) -> np.ndarray:
        self.oc.cnt_pdfxn += max(len(part), len(find))
        return self.model.logp_xnext_full(
            part,
            past_trajs,
            pind,
            future_trajs,
            find,
            ut,
            yt,
            tt,
            cur_ind,
        )

    def logp_xnext_singlestep(
        self,
        part: np.ndarray,
        past_trajs: list[Any] | None,
        pind: np.ndarray,
        future_parts: np.ndarray,
        find: np.ndarray,
        ut: np.ndarray,
        yt: np.ndarray,
        tt: np.ndarray,
        cur_ind: int,
    ) -> np.ndarray:
        self.oc.cnt_pdfxn += max(len(part), len(find))
        return self.model.logp_xnext_singlestep(
            part,
            past_trajs,
            pind,
            future_parts,
            find,
            ut,
            yt,
            tt,
            cur_ind,
        )

    def eval_1st_stage_weights(
        self,
        particles: np.ndarray,
        u: Any,
        y: Any,
        t: float,
    ) -> np.ndarray:
        self.oc.cnt_eval1st += len(particles)
        return self.model.eval_1st_stage_weights(particles, u, y, t)

    def pre_mhips_pass(self, st: Any) -> list[Any]:
        return self.model.pre_mhips_pass(st)

    def post_smoothing(self, st: Any) -> list[Any]:
        return self.model.post_smoothing(st)

    def eval_logp_x0(self, particles: np.ndarray, t: float) -> np.ndarray:
        self.oc.cnt_eval_logp_x0 += len(particles)
        return self.model.eval_logp_x0(particles, t)

    def cond_predict_single_step(
        self,
        part: np.ndarray,
        past_trajs: list[Any] | None,
        pind: np.ndarray,
        future_parts: np.ndarray,
        find: np.ndarray,
        ut: np.ndarray,
        yt: np.ndarray,
        tt: np.ndarray,
        cur_ind: int,
    ) -> np.ndarray:
        return self.model.cond_predict_single_step(
            part,
            past_trajs,
            pind,
            future_parts,
            find,
            ut,
            yt,
            tt,
            cur_ind,
        )

    def cond_sampled_initial(self, part: np.ndarray, t: float) -> np.ndarray:
        return self.model.cond_sampled_initial(part, t)
