"""
Utilities for evalutating probability density functions
"""

from collections.abc import Sequence

import numba as nb


@nb.njit(cache=True)
def _nb_unifsum_eval(
    p: float, c: float, w_min: float, w_diff: float, l: float, h: float, t: float,
) -> float:
    if p < l:
        return 0.0
    if p < (c - w_diff / 2.0):
        return t * (p - l) / w_min
    if p < (c + w_diff / 2.0):
        return t
    if p < h:
        return t * (c + w_diff / 2.0 - p + w_min) / w_min
    return 0.0


class unifsum:
    """
    pdf for sum of two uniform variables

    Args:
     a: lower limits (1st, 2nd)
     a: upper limits (1st, 2nd)
    """

    def __init__(self, a: Sequence[float], b: Sequence[float]) -> None:
        # a1 = numpy.min(a)
        # b1 = numpy.max(a)
        if a[0] < a[1]:
            a1 = a[0]
            b1 = a[1]
        else:
            a1 = a[1]
            b1 = a[0]

        # a2 = numpy.min(b)
        # b2 = numpy.max(b)
        if b[0] < b[1]:
            a2 = b[0]
            b2 = b[1]
        else:
            a2 = b[1]
            b2 = b[0]

        w1 = b1 - a1
        w2 = b2 - a2
        # w = numpy.mean([w1, w2])
        w = (w1 + w2) / 2.0
        # self.c = numpy.mean([a2, b2]) + numpy.mean([a1, b1])
        self.c = (a1 + a2 + b1 + b2) / 2.0
        # self.w_min = numpy.min([w1, w2])
        self.w_min = min(w2, w1)

        # self.w_diff = numpy.abs(numpy.diff([w1, w2]))
        # self.w_diff = numpy.abs(w1-w2)
        self.w_diff = w1 - w2 if w1 > w2 else w2 - w1
        self.l = self.c - w
        self.h = self.c + w
        self.t = 1.0 / (self.w_min + self.w_diff)

    def __call__(self, p: float) -> float:
        """
        Evaluate density a point p

        Args:
         - p (float): point at which to evaluate the pdf

        Returns:
         (float): the pdf value
        """
        return _nb_unifsum_eval(
            p, self.c, self.w_min, self.w_diff, self.l, self.h, self.t,
        )
