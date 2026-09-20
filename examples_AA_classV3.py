"""
Test / demonstration script for AffineScalar / AffineArray
(Affine_ArithmeticClassV3.py)

Structure
---------
  1. Construction & basic arithmetic (+, -, *, /)
  2. exp / log            (log demonstrates warn=True/False)
  3. sqrt                 (demonstrates warn=True/False)
  4. abs
  5. trig / hyperbolic     (sin, cos, tan, cotan, sinh, cosh, tanh)
  6. inverse trig          (arcsin/arccos demonstrate warn=True/False,
                            arctan, arccot)
  7. pow()                 <- the star of this file. Every branch of the
     rewritten pow() is exercised individually with cheb=False *and*
     cheb=True, so you can diff the min-range vs. Chebyshev bounds
     side by side, and see which corner case each test is targeting.

For every result we print both the affine object (x0/xi/delta) and its
outer .interval, plus a short "why this test exists" comment. Where it's
easy to eyeball, we also compare against the true pointwise range using
a dense numpy sweep, so you don't have to trust the bound blindly.

Run with:  python test_affine_arithmetic.py
"""

import numpy as np
import warnings

from Affine_ArithmeticClassV3 import AffineArray


def sweep_range(f, a, b, n=200000):
    """
    Ground-truth helper: evaluate the *real* function f pointwise across
    [a, b] on a dense grid and return (min, max). This is NOT a rigorous
    bound (it can miss a value between grid points) -- it's only here as
    a sanity check that the affine bound isn't obviously wrong or
    obviously too tight. Treat it as a smoke test, not a proof.
    """
    xs = np.linspace(a, b, n)
    ys = f(xs)
    return float(np.min(ys)), float(np.max(ys))


def check(label, affine_result, true_range=None):
    """Pretty-print one result, and flag if the bound fails to contain
    the sampled true range (which would indicate a real bug)."""
    lo, hi = affine_result.interval
    print(f"{label}")
    print(f"    x0={affine_result.x0:.6f}")
    print(f"    xi={np.array2string(affine_result.xi, precision=6)}")
    print(f"    delta={np.array2string(affine_result.delta, precision=6)}")
    print(f"    interval=[{lo:.6f}, {hi:.6f}]")
    if true_range is not None:
        tlo, thi = true_range
        ok = (lo <= tlo + 1e-9) and (hi >= thi - 1e-9)
        flag = "OK" if ok else "*** BOUND VIOLATION ***"
        print(f"    sampled true range=[{tlo:.6f}, {thi:.6f}]   {flag}")
    print()


# =====================================================================
# 1. CONSTRUCTION & BASIC ARITHMETIC
# =====================================================================
print("=" * 70)
print("1. CONSTRUCTION & BASIC ARITHMETIC")
print("=" * 70)

# Three independent affine variables, built from their defining intervals.
# x straddles zero, r straddles zero (smaller), s straddles zero (asymmetric).
X = AffineArray.from_intervals([
    (-2, 2),      # x: symmetric around 0
    (-1, 1),      # r: symmetric around 0, narrower
    (-0.5, 2.5),  # s: asymmetric, straddles 0
])

x = X[0]
r = X[1]
s = X[2]

check("x + 3            (affine + scalar)", x + 3)
check("x + r             (affine + affine, independent noise symbols)", x + r)
check("-5 + s            (scalar + affine, __radd__)", -5 + s)
check("s - r             (affine - affine)", s - r)

# Subtracting a variable from an expression that already contains it
# demonstrates that shared noise symbols correctly CANCEL (this is the
# whole point of affine arithmetic over plain interval arithmetic --
# dependency tracking). x - x should collapse the shared symbol away.
cancel_check = x - x
print("x - x  (dependency cancellation -- should be the exact point {0.0})")
print(f"    x0={cancel_check.x0}, interval={cancel_check.interval}")
print("    (plain interval arithmetic would instead give [-4, 4] here --")
print("     this comparison is the core motivating example for affine")
print("     arithmetic in the accompanying UQ presentation.)")
print()

check("s * r             (affine * affine, quadratic error term appended)", s * r)


# =====================================================================
# 2. exp / log   -- log demonstrates warn=True vs warn=False
# =====================================================================
print("=" * 70)
print("2. exp / log")
print("=" * 70)

Y = AffineArray.from_intervals([
    (1, 2),
    (1, 1.5),
    (-0.25, 1),   # y3: straddles zero -- log domain will need clipping
])

y1, y2, y3 = Y[0], Y[1], Y[2]

check("y3.exp(cheb=False)  [min-range]", y3.exp(cheb=False),
      sweep_range(np.exp, *y3.interval))
check("y3.exp(cheb=True)   [Chebyshev, tighter delta usually]", y3.exp(cheb=True),
      sweep_range(np.exp, *y3.interval))

# --- log(): warn=False (default) ------------------------------------
# y3's interval is (-0.25, 1), which pokes into x<=0 -- log is undefined
# there. With warn=False the function silently clips the lower bound to
# a small positive eps and proceeds (useful once you trust your pipeline
# and don't want console spam on every call).
print("log() with warn=False (default): silent clipping, no warning printed")
check("y3.log(warn=False)  [min-range]", y3.log(warn=False))

# --- log(): warn=True -------------------------------------------------
# Same call, but now we ask to be told about the clipping. Useful while
# debugging a pipeline, to catch places where an interval unexpectedly
# drifted into an invalid domain.
print("log() with warn=True: should print a RuntimeWarning below")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    lnVal = y3.log(warn=True)
    for w in caught:
        print(f"    >>> caught warning: {w.message}")
check("y3.log(warn=True)   [min-range]", lnVal)
check("y3.log(cheb=True, warn=True)  [Chebyshev]", y3.log(cheb=True, warn=True))


# =====================================================================
# 3. sqrt  -- also demonstrates warn=True/False, same rationale as log
# =====================================================================
print("=" * 70)
print("3. sqrt")
print("=" * 70)

print("sqrt() with warn=False: y3's interval dips below 0, clipped silently")
check("y3.sqrt(warn=False) [min-range]", y3.sqrt(warn=False))

print("sqrt() with warn=True: same call, but now warns about the clip")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sqrtVal = y3.sqrt(warn=True)
    for w in caught:
        print(f"    >>> caught warning: {w.message}")
check("y3.sqrt(warn=True)  [min-range]", sqrtVal)
check("y3.sqrt(cheb=True, warn=True) [Chebyshev]", y3.sqrt(cheb=True, warn=True))

# A fully positive interval needs no clipping at all -- warn has no effect.
check("y1.sqrt(warn=True)  [no clipping needed, y1=(1,2)]", y1.sqrt(warn=True),
      sweep_range(np.sqrt, *y1.interval))


# =====================================================================
# 4. abs
# =====================================================================
print("=" * 70)
print("4. abs")
print("=" * 70)

check("s.abs(cheb=False)  [straddles zero -> flat alpha=0 bound]", s.abs(cheb=False),
      sweep_range(np.abs, *s.interval))
check("s.abs(cheb=True)   [Chebyshev V-shape linearization]", s.abs(cheb=True),
      sweep_range(np.abs, *s.interval))


# =====================================================================
# 5. TRIG / HYPERBOLIC
# =====================================================================
print("=" * 70)
print("5. TRIG / HYPERBOLIC")
print("=" * 70)

Theta = AffineArray.from_intervals([
    (-np.pi / 2, np.pi),        # th1: crosses sin's max (pi/2)
    (0.25 * np.pi, 0.8 * np.pi),  # th2: interior, no extremum
    (0.4 * np.pi, 2.4 * np.pi),   # th3: spans more than a half period
])
th1, th2, th3 = Theta[0], Theta[1], Theta[2]

check("th2.sin(cheb=False)", th2.sin(cheb=False), sweep_range(np.sin, *th2.interval))
check("th2.sin(cheb=True)", th2.sin(cheb=True), sweep_range(np.sin, *th2.interval))
check("th1.cos(cheb=False) [interval crosses a sin/cos extremum]",
      th1.cos(cheb=False), sweep_range(np.cos, *th1.interval))
check("th1.cos(cheb=True)", th1.cos(cheb=True), sweep_range(np.cos, *th1.interval))

# Narrower theta range so tan()/cotan() don't hit a pole (tan has poles
# at odd multiples of pi/2; cotan has poles at multiples of pi).
Theta2 = AffineArray.from_intervals([
    (-np.pi / 2, np.pi / 2),
    (0.25 * np.pi, 0.8 * np.pi),
    (0.4 * np.pi, 2.4 * np.pi),
])
th1b, th2b = Theta2[0], Theta2[1]

check("th2b.cotan(cheb=False)", th2b.cotan(cheb=False))
check("th2b.cotan(cheb=True)", th2b.cotan(cheb=True))

check("th1b.sinh(cheb=False) [straddles zero]", th1b.sinh(cheb=False),
      sweep_range(np.sinh, *th1b.interval))
check("th1b.sinh(cheb=True)", th1b.sinh(cheb=True), sweep_range(np.sinh, *th1b.interval))

check("th2b.cosh(cheb=False)", th2b.cosh(cheb=False), sweep_range(np.cosh, *th2b.interval))
check("th2b.cosh(cheb=True)", th2b.cosh(cheb=True), sweep_range(np.cosh, *th2b.interval))

check("th1b.tanh(cheb=False) [straddles zero]", th1b.tanh(cheb=False),
      sweep_range(np.tanh, *th1b.interval))
check("th1b.tanh(cheb=True)", th1b.tanh(cheb=True), sweep_range(np.tanh, *th1b.interval))


# =====================================================================
# 6. INVERSE TRIG -- arcsin/arccos demonstrate warn=True/False
# =====================================================================
print("=" * 70)
print("6. INVERSE TRIG")
print("=" * 70)

# th1c's interval is (-1.5, 1.5), which pokes outside arcsin/arccos's
# valid domain [-1, 1] on both ends -- exactly the scenario warn=True
# is meant to surface.
Phi = AffineArray.from_intervals([
    (-1.5, 1.5),
    (0.0, 0.8),
    (-0.4, 0.25),
])
th1c, th2c, th3c = Phi[0], Phi[1], Phi[2]

print("arcsin() with warn=False: domain exceeded, clipped silently")
check("th1c.arcsin(warn=False)", th1c.arcsin(warn=False))

print("arcsin() with warn=True: same call, now warns about the clip")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    arcsinVal = th1c.arcsin(warn=True)
    for w in caught:
        print(f"    >>> caught warning: {w.message}")
check("th1c.arcsin(warn=True)", arcsinVal)
check("th1c.arcsin(cheb=True, warn=True)", th1c.arcsin(cheb=True, warn=True))

print("arccos() with warn=True: same domain-exceeded scenario")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    arccosVal = th1c.arccos(cheb=True, warn=True)
    for w in caught:
        print(f"    >>> caught warning: {w.message}")
check("th1c.arccos(cheb=True, warn=True)", arccosVal)

check("th1c.arctan(cheb=False)", th1c.arctan(cheb=False),
      sweep_range(np.arctan, *th1c.interval))
check("th1c.arctan(cheb=True)", th1c.arctan(cheb=True),
      sweep_range(np.arctan, *th1c.interval))
check("th1c.arccot(cheb=False)", th1c.arccot(cheb=False))
check("th1c.arccot(cheb=True)", th1c.arccot(cheb=True))


# =====================================================================
# 7. pow()  --  every branch, cheb=False AND cheb=True side by side
# =====================================================================
#
# pow() dispatches on the reduced fraction r = p/q (via Fraction):
#
#   q even            -> "right-half graph", domain x >= 0 only
#                         (e.g. r = 1/2, r = 3/2, r = -1/2)
#   q odd, p even     -> "even graph", f(x) = |x|^r, symmetric,
#                         defined for all real x (e.g. r = 2, r = 2/3)
#   q odd, p odd      -> "odd graph", f(x) = sign(x)*|x|^r, symmetric
#                         about the origin (e.g. r = 3, r = 1/3, r = -1/3)
#
# Within each of those, the code further branches on sign(r) and on
# whether the interval straddles zero. The test cases below are chosen
# specifically to hit each of those sub-branches at least once.
# =====================================================================
print("=" * 70)
print("7. pow()")
print("=" * 70)

W = AffineArray.from_intervals([
    (-1.0, 0.5),     # w1: straddles zero
    (-0.8, -0.2),    # w2: both negative
    (0.2, 0.4),      # w3: both positive
    (1.0, 4.0),      # w4: both positive, away from zero (clean pole tests)
])
w1, w2, w3, w4 = W[0], W[1], W[2], W[3]

# ---------------------------------------------------------------
# 7a. q even, r > 1  (e.g. r = 3/2): right-half graph, convex
# ---------------------------------------------------------------
print("--- q even, r=1.5 (needs x>=0; w1 straddles zero -> auto-clip) ---")
print("(w1 dips negative, so this exercises the a<=0 clipping branch)")
check("w1.pow(1.5, cheb=False)", w1.pow(1.5, cheb=False))
check("w1.pow(1.5, cheb=True)", w1.pow(1.5, cheb=True))

print("--- q even, r=0.5 (sqrt-equivalent via general pow path) ---")
check("w4.pow(0.5, cheb=False)", w4.pow(0.5, cheb=False),
      sweep_range(lambda t: t**0.5, *w4.interval))
check("w4.pow(0.5, cheb=True)", w4.pow(0.5, cheb=True),
      sweep_range(lambda t: t**0.5, *w4.interval))

# ---------------------------------------------------------------
# 7b. q even, r < 0  (e.g. r = -1/2): pole at x=0, needs eps-clip
# ---------------------------------------------------------------
print("--- q even, r=-0.5: pole at x=0 (not just a removable endpoint) ---")
check("w4.pow(-0.5, cheb=False)", w4.pow(-0.5, cheb=False),
      sweep_range(lambda t: t**-0.5, *w4.interval))
check("w4.pow(-0.5, cheb=True)", w4.pow(-0.5, cheb=True),
      sweep_range(lambda t: t**-0.5, *w4.interval))

# ---------------------------------------------------------------
# 7c. p even, q odd, r > 1  (e.g. r = 2): even graph, all three
#     sign sub-cases -- straddling zero, both negative, both positive
# ---------------------------------------------------------------
print("--- p even (r=2), interval STRADDLES zero (w1) ---")
print("(exercises the fmax/fmin, alpha=max(deriv,0) straddle branch)")
check("w1.pow(2, cheb=False)", w1.pow(2, cheb=False),
      sweep_range(lambda t: t**2, *w1.interval))
check("w1.pow(2, cheb=True)", w1.pow(2, cheb=True),
      sweep_range(lambda t: t**2, *w1.interval))

print("--- p even (r=2), both endpoints NEGATIVE (w2) ---")
check("w2.pow(2, cheb=False)", w2.pow(2, cheb=False),
      sweep_range(lambda t: t**2, *w2.interval))
check("w2.pow(2, cheb=True)", w2.pow(2, cheb=True),
      sweep_range(lambda t: t**2, *w2.interval))

print("--- p even (r=2), both endpoints POSITIVE (w3) ---")
check("w3.pow(2, cheb=False)", w3.pow(2, cheb=False),
      sweep_range(lambda t: t**2, *w3.interval))
check("w3.pow(2, cheb=True)", w3.pow(2, cheb=True),
      sweep_range(lambda t: t**2, *w3.interval))

# ---------------------------------------------------------------
# 7d. p even, q odd, 0 < r < 1  (e.g. r = 2/3): even graph, CUSP at
#     x=0 (derivative blows up there rather than vanishing). This is
#     exactly the sub-case that needed the "forced xs=0 hump" fallback
#     in the Chebyshev branch, and the flat alpha=0 bound elsewhere.
# ---------------------------------------------------------------
print("--- p even, 0<r<1 (r=2/3), STRADDLING zero: cusp/hump case ---")
print("(this is the subtle branch -- see the pow() derivation notes;")
print(" the Chebyshev tangent point is forced to x=0 here because the")
print(" un-forced tangent point falls outside [a,b] for a concave arm)")
check("w1.pow(2/3, cheb=False)", w1.pow(2 / 3, cheb=False),
      sweep_range(lambda t: np.abs(t)**(2 / 3), *w1.interval))
check("w1.pow(2/3, cheb=True)", w1.pow(2 / 3, cheb=True),
      sweep_range(lambda t: np.abs(t)**(2 / 3), *w1.interval))

# ---------------------------------------------------------------
# 7e. p even, q odd, r < 0  (e.g. r = -2/3): even graph, pole at x=0,
#     straddling zero is disallowed (raises ValueError)
# ---------------------------------------------------------------
print("--- p even, r=-2/3, both positive (w4) ---")
check("w4.pow(-2/3, cheb=False)", w4.pow(-2 / 3, cheb=False),
      sweep_range(lambda t: np.abs(t)**(-2 / 3), *w4.interval))
check("w4.pow(-2/3, cheb=True)", w4.pow(-2 / 3, cheb=True),
      sweep_range(lambda t: np.abs(t)**(-2 / 3), *w4.interval))

print("--- p even, r=-2/3, interval straddles zero: MUST raise ValueError ---")
try:
    w1.pow(-2 / 3)
    print("    *** BUG: expected ValueError, none was raised ***")
except ValueError as e:
    print(f"    OK, correctly raised: {e}")
print()

# ---------------------------------------------------------------
# 7f. p odd, q odd, r > 1  (e.g. r = 3): odd graph, monotonic cubic
#     shape. Straddling case needs the alpha=0 "flat at the s-curve's
#     center" fallback (r>=1 sub-branch); r=3 example below sits in a
#     range where the derivative's interior minimum at x=0 must be
#     found, not the endpoint derivatives.
# ---------------------------------------------------------------
print("--- p odd (r=3), STRADDLES zero: interior-minimum-derivative case ---")
check("w1.pow(3, cheb=False)", w1.pow(3, cheb=False),
      sweep_range(lambda t: t**3, *w1.interval))
check("w1.pow(3, cheb=True)", w1.pow(3, cheb=True),
      sweep_range(lambda t: t**3, *w1.interval))

print("--- p odd (r=3), both endpoints positive (w3) ---")
check("w3.pow(3, cheb=False)", w3.pow(3, cheb=False),
      sweep_range(lambda t: t**3, *w3.interval))
check("w3.pow(3, cheb=True)", w3.pow(3, cheb=True),
      sweep_range(lambda t: t**3, *w3.interval))

# ---------------------------------------------------------------
# 7g. p odd, q odd, 0 < r < 1  (e.g. r = 1/3): odd graph, cube-root
#     shape. Here the derivative has an interior MAXIMUM (blows up)
#     at x=0 rather than a minimum, so min(f'(a),f'(b)) already finds
#     the true minimum without any forcing -- contrast with 7f.
# ---------------------------------------------------------------
print("--- p odd, 0<r<1 (r=1/3), STRADDLES zero: cube-root shape ---")
check("w1.pow(1/3, cheb=False)", w1.pow(1 / 3, cheb=False),
      sweep_range(lambda t: np.sign(t) * np.abs(t)**(1 / 3), *w1.interval))
check("w1.pow(1/3, cheb=True)", w1.pow(1 / 3, cheb=True),
      sweep_range(lambda t: np.sign(t) * np.abs(t)**(1 / 3), *w1.interval))

# ---------------------------------------------------------------
# 7h. p odd, q odd, r < 0  (e.g. r = -1/3): odd graph, pole at x=0,
#     straddling disallowed. Both-negative and both-positive endpoint
#     selection tested separately since the code picks a or b directly
#     (relying on the sign split) rather than via min/max.
# ---------------------------------------------------------------
print("--- p odd, r=-1/3, both NEGATIVE (w2) ---")
check("w2.pow(-1/3, cheb=False)", w2.pow(-1 / 3, cheb=False),
      sweep_range(lambda t: np.sign(t) * np.abs(t)**(-1 / 3), *w2.interval))
check("w2.pow(-1/3, cheb=True)", w2.pow(-1 / 3, cheb=True),
      sweep_range(lambda t: np.sign(t) * np.abs(t)**(-1 / 3), *w2.interval))

print("--- p odd, r=-1/3, both POSITIVE (w3) ---")
check("w3.pow(-1/3, cheb=False)", w3.pow(-1 / 3, cheb=False),
      sweep_range(lambda t: np.sign(t) * np.abs(t)**(-1 / 3), *w3.interval))
check("w3.pow(-1/3, cheb=True)", w3.pow(-1 / 3, cheb=True),
      sweep_range(lambda t: np.sign(t) * np.abs(t)**(-1 / 3), *w3.interval))

print("--- p odd, r=-1/3, straddles zero: MUST raise ValueError ---")
try:
    w1.pow(-1 / 3)
    print("    *** BUG: expected ValueError, none was raised ***")
except ValueError as e:
    print(f"    OK, correctly raised: {e}")
print()

# ---------------------------------------------------------------
# 7i. Edge cases: r = 1 (identity shortcut) and irrational r
#     (routes through log/exp instead of the p/q dispatch)
# ---------------------------------------------------------------
print("--- r=1 exactly: should hit the identity shortcut, alpha=1,gamma=0 ---")
check("w3.pow(1)", w3.pow(1))

print("--- irrational r (1/sqrt(2)): routed via (log(x)*r).exp() ---")
check("w3.pow(1 / 2**0.5, cheb=False)", w3.pow(1 / 2**0.5, cheb=False),
      sweep_range(lambda t: t**(1 / 2**0.5), *w3.interval))
check("w3.pow(1 / 2**0.5, cheb=True)", w3.pow(1 / 2**0.5, cheb=True),
      sweep_range(lambda t: t**(1 / 2**0.5), *w3.interval))

# ---------------------------------------------------------------
# 7j. Composite check: cos(theta)^2 computed two ways, then compared
#     against the trig identity via 1/cos^2 -- a good end-to-end
#     regression test since it chains sin/cos, pow(2), and inv()/div.
# ---------------------------------------------------------------
print("--- composite: cos(theta)^2 and 1/cos(theta)^2, narrow theta ---")
Theta3 = AffineArray.from_intervals([(-10 * np.pi / 180, 10 * np.pi / 180)])
th4 = Theta3[0]

check("th4.cos(cheb=False)", th4.cos(cheb=False))
check("th4.cos(cheb=True)", th4.cos(cheb=True))

cosL2_mr = th4.cos().pow(2)
cosL2_cb = th4.cos(cheb=True).pow(2, cheb=True)
check("th4.cos().pow(2)                 [min-range chain]", cosL2_mr,
      sweep_range(lambda t: np.cos(t)**2, *th4.interval))
check("th4.cos(cheb=True).pow(2,cheb=True) [Chebyshev chain]", cosL2_cb,
      sweep_range(lambda t: np.cos(t)**2, *th4.interval))

inv_cosL2_mr = 1.0 / th4.cos().pow(2)
inv_cosL2_cb = 1.0 / th4.cos(cheb=True).pow(2, cheb=True)
check("1/(th4.cos().pow(2))                 [== sec^2, min-range]", inv_cosL2_mr,
      sweep_range(lambda t: 1.0 / np.cos(t)**2, *th4.interval))
check("1/(th4.cos(cheb=True).pow(2,cheb=True)) [== sec^2, Chebyshev]", inv_cosL2_cb,
      sweep_range(lambda t: 1.0 / np.cos(t)**2, *th4.interval))

print("=" * 70)
print("Done. Scan above for any '*** BOUND VIOLATION ***' or '*** BUG'")
print("markers -- those indicate an actual soundness failure, not just")
print("looseness (a loose-but-valid bound will never trigger the flag).")
print("=" * 70)