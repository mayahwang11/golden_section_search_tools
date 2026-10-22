# golden_section_search

Minimizes a unimodal scalar function on a closed interval `[lower, upper]` without using derivatives, via the golden-section search.

```python
from golden_section_search import golden_section_search

x_min, f_min = golden_section_search(lambda x: (x - 2.0) ** 2, 0.0, 5.0)
print(x_min, f_min)  # ~2.0 ~0.0
```

## Why this exists

Derivative-free minimization is useful for functions that are noisy, non-smooth, or expensive to differentiate. The golden-section search is the simplest such method for the one-dimensional unimodal case: each iteration shrinks the bracketing interval by the constant factor `tau = (sqrt(5) - 1) / 2` and reuses one previous function evaluation, so only one new `f` call is needed per step. The trade-off versus Brent's method is linear (rather than superlinear) convergence; the payoff is a tiny, transparent implementation with no internal state to tune.

## API

`golden_section_search(f, lower, upper, tol=1e-7, max_iter=100) -> tuple[float, float]`

Returns `(x_min, f_min)` where `x_min` is the best minimizer found and `f_min` is `f(x_min)`. `tol` is an absolute tolerance on `x`. `max_iter` caps the number of iterations.

## Choices and edges

- `tol` is absolute, not relative to the initial interval width. If you want relative control, pass `tol = (upper - lower) * relative_tol`.
- The caller must guarantee unimodality on `[lower, upper]`. If the function has multiple local minima inside the bracket, the returned point will be one of them but is not guaranteed to be the global minimum.
- If `lower > upper` is passed, the bracket is normalized internally and the result is returned as if the bounds were given in ascending order; the minimizer's value is unaffected.
- If the initial interval is narrower than `tol`, the routine evaluates `f` once at `lower` and returns immediately.
- The returned `x_min` is one of the two final interior probe points (whichever has the smaller function value), not the midpoint of the final bracket. This is slightly more accurate when the minimum sits near an endpoint.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

