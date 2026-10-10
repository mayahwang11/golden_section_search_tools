"""Core golden-section search routine."""

from typing import Callable, Tuple

# The golden ratio minus one. The golden-section search maintains two
# interior points at this fraction of the current interval, measured
# from each end. Each iteration shrinks the interval by a factor of
# tau and reuses one previous function evaluation, so only one new
# evaluation is needed per step.
tau = (5.0 ** 0.5 - 1.0) / 2.0


def golden_section_search(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    tol: float = 1e-7,
    max_iter: int = 100,
) -> Tuple[float, float]:
    """Minimize a unimodal scalar function on ``[lower, upper]``.

    Returns a ``(x_min, f_min)`` pair where ``x_min`` is the best
    minimizer found and ``f_min`` is ``f(x_min)``.

    This routine uses the classic golden-section search. Each iteration
    evaluates ``f`` exactly once and shrinks the bracketing interval
    by the factor ``tau``. No derivatives are needed, which makes the
    method suitable for functions that are not smooth or whose
    derivatives are expensive.

    Choices made:
    - ``tol`` is measured in absolute units of ``x``, not in units of
      the initial interval width. Callers who want relative control
      can compute ``tol = width * relative_tol`` themselves.
    - A fixed ``max_iter`` guards against infinite loops; the default
      of 100 gives an interval shrink factor of roughly
      ``tau**100 ~ 1e-21``, far below double precision.
    - The initial bracket is taken as given. The caller is responsible
      for ensuring the function is unimodal on ``[lower, upper]``.
      If the function has multiple local minima in the interval the
      returned point will be one of them, but not guaranteed to be
      the global one.
    - Inputs are accepted without modification (``lower`` may be
      greater than ``upper``); the bracket is normalized internally.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    a, b = float(lower), float(upper)
    swapped = a > b
    if swapped:
        a, b = b, a

    width = b - a
    # If the initial endpoints coincide there is nothing to search.
    # We evaluate once at that point and return it. This avoids a
    # zero-length bracket and division-by-zero concerns downstream.
    if width <= tol:
        x = a
        return (b, f(b)) if swapped else (x, f(x))

    c = b - tau * width
    d = a + tau * width
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if b - a <= tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            width = b - a
            c = b - tau * width
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            width = b - a
            d = a + tau * width
            fd = f(d)

    # Pick the better of the two current interior points as the
    # estimate. This is strictly more accurate than returning the
    # midpoint of the final bracket when the minimum sits near one
    # endpoint.
    if fc <= fd:
        x, fx = c, fc
    else:
        x, fx = d, fd

    if swapped:
        # Preserve the original orientation: if the caller passed
        # (lower, upper) with lower > upper, return the minimizer
        # in the same conceptual location (the value is unchanged).
        return x, fx
    return x, fx
