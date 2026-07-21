# Affine Arithmetic in Python

A from-scratch implementation of **affine arithmetic (AA)** for rigorous
uncertainty quantification, built on the de Figueiredo–Stolfi formulation
and extended with Rump & Kashiwagi–style improvements. Where plain interval
arithmetic silently loses correlation information between dependent
quantities, this library tracks it explicitly, so cascaded operations on
the same underlying variable don't blow up into needlessly wide bounds.

## Why affine arithmetic?

Interval arithmetic represents a quantity's uncertainty as a single
`[lo, hi]` box. That's simple, but it has a well-known failure mode: if a
variable `x` is used more than once in an expression, interval arithmetic
treats each occurrence as *independent*, even though they're the same
variable. A canonical example is `x - x`:

```python
x = X[0]              # x ranges over [-2, 2]
(x - x).interval       # affine arithmetic: exactly {0.0}
```

Plain interval subtraction on the same box gives `[-4, 4]` — a four-fold
overestimate of the true, zero-width answer. Affine arithmetic fixes this
by representing every quantity as a **first-order polynomial in a shared
set of noise symbols**:

```
x̂ = x₀ + Σ xᵢ · εᵢ,      εᵢ ∈ [-1, 1]
```

Two expressions that share a noise symbol `εᵢ` are provably correlated,
and that correlation is preserved through every subsequent affine
operation (addition, subtraction, and scalar multiplication are all
*exact* — no approximation, no accumulated error).

## What's in this repo

### `AffineScalar` / `AffineArray`
- `AffineArray` holds the shared state for a vector of correlated
  quantities (`x0`, the central values; `Xi`, the propagated noise
  coefficients; `Delta`, the independent remainder/rounding terms).
- `AffineScalar` is a lightweight view into one component of an
  `AffineArray` (or a fully standalone scalar), exposing the familiar
  `+`, `-`, `*`, `/`, and `.interval` interface.
- A **two-tier noise model** separates *propagated* symbols (`Xi` —
  symbols shared across variables, enabling correlation tracking) from
  *remainder* symbols (`Delta` — independent per-operation linearization
  and rounding error, following Rump & Kashiwagi's refinement of the
  original Stolfi/de Figueiredo model). This keeps the correlation
  structure clean while still accounting for every source of
  approximation error.
- **Outward rounding** (`_outer_bound`) at fixed significant-digit
  precision guarantees floating-point rounding never silently narrows a
  bound — every result is a *conservative* enclosure of the true range,
  not just a numerically convenient one.

### Exact affine operations
Addition, subtraction, and scalar multiplication of affine forms are
closed-form and error-free. Multiplication of two general affine forms
uses the improved quadratic error bound (rather than the older, looser
first-order bound), which tightens the enclosure for products of
already-uncertain quantities.

### Nonlinear function support
Every nonlinear elementary function is linearised using **two
interchangeable strategies**, selectable per call via `cheb=True/False`:

| Strategy | What it does |
|---|---|
| **Min-range** (default) | Picks the affine approximation with minimal enclosure *width*, cheap to compute, always valid. |
| **Chebyshev / minimax** (`cheb=True`) | Solves for the tangent-line(s) matching the secant slope exactly (Lemma 3 of the de Figueiredo–Stolfi framework), producing the *provably optimal* affine bound — typically tighter, at the cost of solving a small nonlinear equation for the tangent point(s). |

Supported functions, each with both strategies and correct handling of
domain edge cases (poles, cusps, straddling zero, symmetric/asymmetric
intervals):

- **Algebraic:** `inv`, `sqrt`, `abs`, and a fully general `pow(r)` for
  any real exponent — integer, rational (via exact `Fraction` reduction,
  correctly branching on odd/even numerator and denominator to
  distinguish domain-restricted "right-half" graphs, symmetric even
  graphs, and sign-preserving odd graphs), or irrational (routed through
  `exp(r · log(x))`).
- **Exponential/logarithmic:** `exp`, `log`.
- **Trigonometric:** `sin`, `cos`, `tan`, `cotan` — including correct
  period-counting logic to detect when an interval spans a full or
  half period, or straddles an extremum or a pole.
- **Hyperbolic:** `sinh`, `cosh`, `tanh`.
- **Inverse trigonometric:** `arcsin`, `arccos`, `arctan`, `arccot`.

### Domain-safety conventions
Functions with a restricted domain (`log`, `sqrt`, `inv`, `arcsin`,
`arccos`, `pow` with negative or fractional exponents) accept a
`warn=True/False` flag. When an input interval pokes outside the valid
domain, the function **clips the interval to the nearest valid range**
rather than raising — with `warn=True` surfacing a `RuntimeWarning`
describing exactly what was clipped, and `warn=False` (default) doing it
silently once a pipeline is trusted. Genuinely invalid inputs (e.g. an
interval straddling zero passed to a function with a true pole there,
like `pow` with a negative exponent) raise `ValueError` rather than
producing a nonsensical bound.

### Multidimensional support
`AffineArray.from_intervals` builds a full vector of correlated affine
variables from a box of intervals in one call, and `split_interval`
recursively subdivides a domain into a grid of sub-boxes — the
foundation for adaptive branch-and-bound refinement of affine bounds
(useful when a single linearization over a wide domain is too loose).

` examples_AA_classV3.txt'
It exercises basic arithmetic (including the x - x dependency-cancellation example), every elementary function (exp, log, sqrt, abs, trig, hyperbolic, inverse trig) in both cheb=True and cheb=False modes, the warn=True/False domain-clipping behavior for log/sqrt/arcsin/arccos, and — most thoroughly — every branch of pow() (all six sign/parity combinations, straddling-zero cases, the r=1 and irrational-exponent edge cases), cross-checking each affine bound against a dense numpy sweep of the true function to catch soundness violations.

` AA_scalar_test_functions_trial1.txt'
An applied benchmark script that runs the affine-arithmetic engine on three classic optimisation test functions (Branin, Ackley in both 2-D and 3-D, and Eggholder), each built compositionally from the class's primitives (pow, cos, sin, sqrt, abs, exp). For each function, it does adaptive domain refinement: repeatedly subdividing the unit hypercube via split_interval into 2^m pieces per dimension, re-evaluating both the Chebyshev and min-range bounds on every sub-box, and tracking how the aggregate enclosure tightens (and how the two strategies' ranges compare/intersect) as the mesh refines from m=0 to m=8.

` AA_WingWeight_model.txt'
The same adaptive-refinement methodology applied to a real 10-dimensional engineering model, the classic wing weight formula from aerospace design (a product of power-law terms in wing area, fuselage weight, taper ratio, sweep angle, dynamic pressure, thickness ratio, load factor, gross weight, and paint weight fraction). It's a good stress test for pow() specifically, since nearly every factor is raised to a small fractional exponent (0.758, 0.0035, 0.6, 0.006, 0.04, -0.3, 0.49), and for the class's composition machinery generally, since it chains multiplication, cos, and negative-exponent pow across ten correlated affine variables at once — at dim1=3 splits per dimension, that's already 8^10 sub-boxes being evaluated.

## Example

```python
from Affine_ArithmeticClassV3 import AffineArray
import numpy as np

X = AffineArray.from_intervals([(-1.0, 0.5), (0.4, 2.0)])
x, y = X[0], X[1]

# Correlated subtraction collapses exactly, unlike interval arithmetic:
print((x - x).interval)          # -> [0.0, 0.0]

# Nonlinear functions, either linearization strategy:
z = (x.cos(cheb=True) ** 2 if False else x.cos(cheb=True).pow(2, cheb=True))
print(z.interval)

# General rational power, correctly handling the pole at x=0:
w = y.pow(-1/3)
print(w.interval)
```

## Status

Core arithmetic, all listed elementary functions, and the general
rational/irrational `pow()` are implemented and under active test
(`test_affine_arithmetic.py` exercises every branch of `pow()` — all
sign/parity/cheb combinations — plus `warn=True/False` behaviour for
every domain-restricted function, cross-checked against dense pointwise
sampling of the true function). Ongoing work includes width-based sensitivity analysis for interval variables.

## References

- Stolfi, J., & de Figueiredo, L. H. (1997/2003). *Self-Validated
  Numerical Methods and Applications* — the foundational AA framework
  and Chebyshev/min-range linearization lemmas used throughout.
- Rump, S. M., & Kashiwagi, M. — improved multiplication error bounds
  and refinements to the original AA model, used in `Delta`/`Xi`
  separation and the multiplication routine here.
- Moore, R. E. — *Interval Analysis*, the classical foundation AA
  builds on and improves upon.
- Neumaier, A. — *Interval Methods for Systems of Equations*.
- Ferson, S. et al. — p-box and dependency-tracking motivation for
  uncertainty quantification.
