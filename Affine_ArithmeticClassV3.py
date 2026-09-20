import numpy as np
import math
import warnings

from itertools import product
# from Interval import Interval
from fractions import Fraction

sig_dig = 12  # global significant digits for outward rounding

class AffineScalar:

    # def __init__(self, parent=None, idx=None, x0=None, xi=None):

    #     # view mode
    #     if parent is not None:

    #         self.parent = parent
    #         self.idx = idx

    #         self._standalone = False

    #     # standalone mode
    #     else:

    #         self._x0 = float(x0)

    #         self._xi = np.asarray(xi, dtype=float)

    #         self._standalone = True

    def __init__(self, parent=None, idx=None, x0=None, xi=None, delta=None):

        # view mode
        if parent is not None:

            self.parent = parent
            self.idx = idx

            self._standalone = False

        # standalone mode
        else:

            self._x0 = float(x0)

            if xi is None:
                xi = []

            if delta is None:
                delta = []

            self._xi = np.asarray(xi, dtype=float)
            self._delta = np.asarray(delta, dtype=float)

            self._standalone = True

    def _oB(self, x, left):

        str_rep = f'{x:0.{sig_dig}g}'
        x_print = float(str_rep)

        prec = math.floor(
            math.log10(abs(x) + 1e-100) + 1.0
        ) - sig_dig

        least_sig = 10.0 ** prec * 0.5

        if left:
            if x_print > x and not (abs(x_print - x) < 1e-100):
                x -= least_sig
        else:
            if x_print < x and not (abs(x_print - x) < 1e-100):
                x += least_sig

        return round(x, sig_dig)

    def _outer_bound(self, lo, hi):

        return (
            self._oB(lo, left=True),
            self._oB(hi, left=False)
        )
    
    @property
    def x0(self):

        if self._standalone:

            return self._x0

        return self.parent.x0[self.idx]

    # @property
    # def xi(self):

    #     if self._standalone:

    #         return self._xi

    #     return self.parent.E[self.idx]

    @property
    def xi(self):

        if self._standalone:
            return self._xi

        return self.parent.Xi[self.idx]


    @property
    def delta(self):

        if self._standalone:
            return self._delta

        return self.parent.Delta[self.idx]

    # @property
    # def interval(self):

    #     r = np.sum(np.abs(self.xi))

    #     return (
    #         self.x0 - r,
    #         self.x0 + r
    #     )

    @property
    def interval(self):

        r = (
            np.sum(np.abs(self.xi))
            +
            np.sum(np.abs(self.delta))
        )

        return (
            self.x0 - r,
            self.x0 + r
        )
    
    @staticmethod
    def _align(xi1, xi2):

        n1 = len(xi1)

        n2 = len(xi2)

        if n1 < n2:

            xi1 = np.pad(
                xi1,
                (0, n2 - n1)
            )

        elif n2 < n1:

            xi2 = np.pad(
                xi2,
                (0, n1 - n2)
            )

        return xi1, xi2
    
    @staticmethod
    def _merge_delta(delta1, delta2):

        return np.concatenate((delta1, delta2))

    # def __add__(self, other):

    #     # affine + affine
    #     if isinstance(other, self.__class__):

    #         xi1, xi2 = self._align(
    #             self.xi,
    #             other.xi
    #         )

    #         return AffineScalar(
    #             x0 = self.x0 + other.x0,
    #             xi = xi1 + xi2
    #         )

    #     # affine + scalar
    #     elif isinstance(other, (int, float)):

    #         return AffineScalar(
    #             x0 = self.x0 + other,
    #             xi = self.xi.copy()
    #         )

    #     raise TypeError(
    #         "other must be AffineScalar, int, or float"
    #     )

    def __add__(self, other):

        # affine + affine
        if isinstance(other, self.__class__):

            xi1, xi2 = self._align(
                self.xi,
                other.xi
            )

            return AffineScalar(
                x0=self.x0 + other.x0,
                xi=xi1 + xi2,
                delta = self._merge_delta(
                    self.delta,
                    other.delta
                )
            )

        # affine + scalar
        elif isinstance(other, (int, float)):

            return AffineScalar(
                x0=self.x0 + other,
                xi=self.xi.copy(),
                delta=self.delta.copy()
            )

        raise TypeError(
            "other must be AffineScalar, int, or float"
        )


    # def __mul__(self, other):

    #     # affine * affine
    #     if isinstance(other, self.__class__):

    #         # ---------- align dimensions ----------

    #         xi1, xi2 = self._align(
    #             self.xi,
    #             other.xi
    #         )

    #         # ---------- affine part ----------

    #         x0_new = self.x0 * other.x0

    #         xi_aff = (
    #             self.x0 * xi2
    #             +
    #             other.x0 * xi1
    #         )

    #         # ---------- improved error term (26) ----------

    #         v = xi1 * xi2

    #         v_pos = np.maximum(v, 0.0)

    #         v_neg = np.maximum(-v, 0.0)

    #         diag_term = max(
    #             np.sum(v_pos),
    #             np.sum(v_neg)
    #         )

    #         offdiag = 0.0

    #         n = len(xi1)

    #         for i in range(n):

    #             for j in range(i + 1, n):

    #                 offdiag += abs(
    #                     xi1[i] * xi2[j]
    #                     +
    #                     xi1[j] * xi2[i]
    #                 )

    #         e0 = diag_term + offdiag

    #         _, e = self._outer_bound(0.0, float(e0))

    #         # ---------- append fresh noise symbol ----------

    #         xi_new = np.append(
    #             xi_aff,
    #             e
    #         )

    #         return AffineScalar(
    #             x0 = x0_new,
    #             xi = xi_new
    #         )

    #     # affine * scalar
    #     elif isinstance(other, (int, float)):

    #         return AffineScalar(
    #             x0 = self.x0 * other,
    #             xi = self.xi * other
    #         )

    #     raise TypeError(
    #         "other must be AffineScalar, int, or float"
    #     )
    def __mul__(self, other):

        if isinstance(other, self.__class__):

            xi1, xi2 = self._align(self.xi, other.xi)

            n1d, n2d = len(self.delta), len(other.delta)

            x0_new = self.x0 * other.x0

            xi_aff = self.x0 * xi2 + other.x0 * xi1

            # ---------- delta symbols are PRIVATE, never shared ----------
            # self's own delta symbols persist with coefficient other.x0;
            # other's own delta symbols persist with coefficient self.x0.
            # These must be concatenated into disjoint slots, not aligned
            # and summed positionally -- they are never the same symbol.

            delta_aff = np.concatenate((
                other.x0 * self.delta,
                self.x0 * other.delta
            ))

            # ---------- combined coefficient vectors over the FULL,
            # disjoint noise-symbol index set: [shared xi | self's own
            # delta | other's own delta], zero-padded where an operand
            # has no coefficient on a symbol it doesn't own ----------

            coeff1 = np.concatenate((xi1, self.delta, np.zeros(n2d)))
            coeff2 = np.concatenate((xi2, np.zeros(n1d), other.delta))

            v = coeff1 * coeff2

            v_pos = np.maximum(v, 0.0)
            v_neg = np.maximum(-v, 0.0)

            diag_term = max(np.sum(v_pos), np.sum(v_neg))

            offdiag = 0.0
            n = len(coeff1)

            for i in range(n):
                for j in range(i + 1, n):
                    offdiag += abs(coeff1[i]*coeff2[j] + coeff1[j]*coeff2[i])

            e0 = diag_term + offdiag

            _, e = self._outer_bound(0.0, float(e0))

            delta_new = np.append(delta_aff, e)

            return AffineScalar(x0=x0_new, xi=xi_aff, delta=delta_new)

        elif isinstance(other, (int, float)):

            return AffineScalar(
                x0=self.x0 * other,
                xi=self.xi * other,
                delta=self.delta * other
            )

        raise TypeError("other must be AffineScalar, int, or float")

    # def __mul__(self, other):

    #     # affine * affine
    #     if isinstance(other, self.__class__):

    #         # ---------- align propagated symbols ----------

    #         xi1, xi2 = self._align(
    #             self.xi,
    #             other.xi
    #         )

    #         # ---------- align remainder symbols ----------

    #         delta1, delta2 = self._align(
    #             self.delta,
    #             other.delta
    #         )

    #         # ---------- affine part ----------

    #         x0_new = self.x0 * other.x0

    #         xi_aff = (
    #             self.x0 * xi2
    #             +
    #             other.x0 * xi1
    #         )

    #         delta_aff = (
    #             self.x0 * delta2
    #             +
    #             other.x0 * delta1
    #         )

    #         # ---------- combine all independent symbols ----------

    #         coeff1 = np.concatenate(
    #             (xi1, delta1)
    #         )

    #         coeff2 = np.concatenate(
    #             (xi2, delta2)
    #         )

    #         coeff1, coeff2 = self._align(
    #             coeff1,
    #             coeff2
    #         )

    #         # ---------- improved error term (26) ----------

    #         v = coeff1 * coeff2

    #         v_pos = np.maximum(v, 0.0)

    #         v_neg = np.maximum(-v, 0.0)

    #         diag_term = max(
    #             np.sum(v_pos),
    #             np.sum(v_neg)
    #         )

    #         offdiag = 0.0

    #         n = len(coeff1)

    #         for i in range(n):

    #             for j in range(i + 1, n):

    #                 offdiag += abs(
    #                     coeff1[i] * coeff2[j]
    #                     +
    #                     coeff1[j] * coeff2[i]
    #                 )

    #         e0 = diag_term + offdiag

    #         _, e = self._outer_bound(
    #             0.0,
    #             float(e0)
    #         )

    #         # ---------- append fresh remainder symbol ----------

    #         delta_new = np.append(
    #             delta_aff,
    #             e
    #         )

    #         return AffineScalar(
    #             x0=x0_new,
    #             xi=xi_aff,
    #             delta=delta_new
    #         )

    #     # affine * scalar
    #     elif isinstance(other, (int, float)):

    #         return AffineScalar(
    #             x0=self.x0 * other,
    #             xi=self.xi * other,
    #             delta=self.delta * other
    #         )

    #     raise TypeError(
    #         "other must be AffineScalar, int, or float"
    #     )
    
    def __rmul__(self, other):

        return self * other
    
    def __neg__(self):

        return AffineScalar(
            x0 = -self.x0,
            xi = -self.xi,
            delta= -self.delta
        )
    
    def __sub__(self, other):

        return self + (-other)
    
    def __rsub__(self, other):

        return other + (-self)

    def __radd__(self, other):

        return self + other

    # def _affine_constructor(
    #     self,
    #     gamma,
    #     alpha,
    #     delta,
    #     interval = None
    #     ):
            
    #     if interval is None:

    #         x0_new = alpha * self.x0 + gamma

    #         xi_new = alpha * self.xi

    #         xi_new = np.append(
    #             xi_new,
    #             delta
    #         )

    #         y = AffineScalar(x0=x0_new, xi=xi_new)

    #     else:

    #         a , b = interval

    #         x0_new = alpha * 0.5 * (a + b) + gamma

    #         xi_new = alpha * 0.5 * (b - a)

    #         xi_new = np.append(
    #             xi_new,
    #             delta
    #         )

    #         y = AffineScalar(x0=x0_new, xi=xi_new)            

    #     return y

    def _affine_constructor(
        self,
        gamma,
        alpha,
        delta,
        center_shift=0.0,
        radius_scale=1.0,
        interval=None
        ):
            
        if interval is None:

            x0_new = alpha * (self.x0 + center_shift) + gamma

            xi_new = alpha * radius_scale * self.xi

            delta_new = alpha * radius_scale * self.delta

            delta_new = np.append(
                delta_new,
                delta
            )

            y = AffineScalar(
                x0=x0_new,
                xi=xi_new,
                delta=delta_new
            )

        else:

            a , b = interval

            x0_new = alpha * 0.5 * (a + b) + gamma

            xi_new = np.array([alpha * 0.5 * (b - a)])

            y = AffineScalar(x0=x0_new, xi=xi_new, delta=[delta])            

        return y
    
    def inv(self):

        lb, ub = self.interval

        a = lb
        b = ub

        if np.isclose(a, b):
            return AffineScalar(
                x0=1.0/a,
                xi=np.zeros_like(self.xi),
                delta=np.zeros(0)
            )

        if lb <= 0 <= ub:

            raise ValueError(
                "Interval contains zero"
            )

        else:

            a = min(abs(lb), abs(ub))
            b = max(abs(lb), abs(ub))

            alpha = -1.0 / b**2

            U = 1.0/a - alpha*a
            L = 2.0/b

            gamma = 0.5*(L + U)

            if lb < 0:
                gamma = -gamma

            delta0 = 0.5*(U - L)

        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,
            delta
        )

    # def inv(self, warn=False):

    #     lb, ub = self.interval

    #     if np.isclose(lb, ub):
    #         return AffineScalar(
    #             x0=1.0/lb,
    #             xi=np.zeros_like(self.xi),
    #             delta=np.zeros(0)
    #         )
        
    #     center_shift = 0.0
    #     radius_scale = 1.0


    #     if lb <= 0 <= ub:

    #         eps = 1e-12

    #         if warn:

    #             if warn:
    #                 warnings.warn(
    #                     f"inv(): lower bound {lb} is non-positive; "
    #                     f"clipping interval to [{eps}, {ub}] or [{lb}, {-eps}]",
    #                     RuntimeWarning,
    #                     stacklevel=2
    #                 )

    #         if np.abs(lb) > np.abs(ub):

    #             new_a = lb
    #             new_b = -eps

    #         else:

    #             new_a = eps
    #             new_b = ub

    #         old_center = self.x0
    #         old_radius = np.sum(np.abs(self.xi)) + np.sum(np.abs(self.delta))

    #         new_center = 0.5 * (new_a + new_b)
    #         new_radius = 0.5 * (new_b - new_a)

    #         center_shift = new_center - old_center
    #         radius_scale = new_radius / old_radius

    #         lb = new_a
    #         ub = new_b


    #     a = min(abs(lb), abs(ub))
    #     b = max(abs(lb), abs(ub))

    #     alpha = -1.0 / b**2

    #     U = 1.0/a - alpha*a
    #     L = 2.0/b

    #     gamma = 0.5*(L + U)

    #     if lb < 0:
    #         gamma = -gamma

    #     delta0 = 0.5*(U - L)

    #     _, delta = self._outer_bound(0.0, float(delta0))

    #     return self._affine_constructor(
    #         gamma,
    #         alpha,
    #         delta,
    #         center_shift=center_shift,
    #         radius_scale=radius_scale
    #     )
    
    def __truediv__(self, other):

        if isinstance(other, self.__class__):

            return self * other.inv()

        elif isinstance(other, (int, float)):

            if other == 0:
                raise ZeroDivisionError(
                    "division by zero"
                )

            return self * (1.0 / other)

        raise TypeError(
            "other must be AffineScalar, int, or float"
        )

    def __rtruediv__(self, other):

        if isinstance(other, (int, float)):

            return other * self.inv()

        raise TypeError(
            "other must be int or float"
        )

    # Nonlinear functions

    def exp(self, cheb=False):

        a, b = self.interval

        ea = np.exp(a)
        eb = np.exp(b)

        # ---------- Chebyshev approximation ----------
        if cheb:

            alpha = (eb - ea)/(b - a)

            xs = np.log(alpha)

            gamma = 0.5 * (
                ea
                + np.exp(xs)
                - alpha*(a + xs)
            )

            delta0 = 0.5 * np.abs(
                np.exp(xs)
                - ea
                - alpha*(xs - a)
            )

        # ---------- Min-range approximation ----------
        else:

            # if eb >=0:
            #  alpha = np.max(ea, 0)
            # else:
            #     alpha = eb

            alpha = np.max(ea, 0)

            gamma = 0.5 * (
                ea + eb
                - alpha*(a + b)
            )

            delta0 = 0.5 * np.abs(
                eb - ea
                - alpha*(b - a)
            )

        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,
            delta
        )
    
    def log(self, cheb=False, warn=False):

        a, b = self.interval

        if b<=0:
            raise ValueError(
                f"log undefined for interval [{a}, {b}] "   
            )
        
        if np.isclose(a, b):
            return AffineScalar(
                x0=np.log(a),
                xi=np.zeros_like(self.xi),
                delta=np.zeros(0)
            )

        center_shift = 0.0
        radius_scale = 1.0

        if a <= 0:

            new_a = 1e-12
            
            if warn:
                warnings.warn(
                    f"log(): lower bound {a} is non-positive; "
                    f"clipping interval to [{new_a}, {b}]",
                    RuntimeWarning,
                    stacklevel=2
                )

            old_center = self.x0
            old_radius = np.sum(np.abs(self.xi)) + np.sum(np.abs(self.delta))

            new_b = b

            new_center = 0.5 * (new_a + new_b)
            new_radius = 0.5 * (new_b - new_a)

            center_shift = new_center - old_center
            radius_scale = new_radius / old_radius

            a = new_a
            b = new_b

        fa = np.log(a)
        fb = np.log(b)

        # ---------- Chebyshev approximation ----------
        if cheb:

            alpha = (fb - fa)/(b - a)
            xs = 1/alpha

            gamma = 0.5 * (
                fa
                + np.log(xs)
                - alpha*(a + xs)
            )

            delta0 = 0.5 * np.abs(
                np.log(xs)
                - fa
                - alpha*(xs - a)
            )

        # ---------- Min-range approximation ----------
        else: 

            alpha = 1/b

            gamma = 0.5 * (
                fa + fb
                - alpha*(a + b)
            )

            delta0 = 0.5 * np.abs(
                fb - fa
                - alpha*(b - a)
            ) 

        _, delta = self._outer_bound(0.0, float(delta0))           

        return self._affine_constructor(
            gamma,
            alpha,
            delta,
            center_shift=center_shift,
            radius_scale=radius_scale
        )
    
    def abs(self, cheb=False):

        a, b = self.interval

        fa = np.abs(a)
        fb = np.abs(b)

        if a >= 0:

            alpha = 1
            gamma = 0
            delta0 = 0

        elif b <= 0:
            alpha = -1
            gamma = 0
            delta0 = 0

        else:

            # ---------- Chebyshev approximation ----------
            if cheb:

                alpha = (fb - fa)/(b - a)
                xs = 0

                gamma = 0.5 * (
                    fa
                    - alpha*(a + xs)
                )

                delta0 = 0.5 * np.abs(
                    - fa
                    - alpha*(xs - a)
                )

            else: 

                alpha = 0

                gamma = 0.5 * (
                    max(fa, fb)
                )

                delta0 = gamma 

        _, delta = self._outer_bound(0.0, float(delta0))           

        return self._affine_constructor(
            gamma,
            alpha,
            delta
        )

    # def abs(self, cheb=False):

    #     a, b = self.interval

    #     if np.isclose(a, b):
    #         return AffineScalar(
    #             x0=np.abs(a),
    #             xi=np.zeros_like(self.xi),
    #             delta=np.zeros(0)
    #         )
        
    #     fa = np.abs(a)
    #     fb = np.abs(b)

    #     center_shift = 0.0
    #     radius_scale = 1.0

    #     if a >= 0:

    #         alpha = 1
    #         gamma = 0
    #         delta0 = 0

    #     elif b <= 0:
    #         alpha = -1
    #         gamma = 0
    #         delta0 = 0

    #     else:

    #         # ---------- Chebyshev approximation ----------
    #         if cheb:

    #             alpha = (fb - fa)/(b - a)
    #             xs = 0

    #             gamma = 0.5 * (
    #                 fa
    #                 - alpha*(a + xs)
    #             )

    #             delta0 = 0.5 * np.abs(
    #                 - fa
    #                 - alpha*(xs - a)
    #             )

    #             # alpha = 0

    #             # gamma = 0.5 * (
    #             #     max(fa, fb)
    #             # )

    #             # delta0 = gamma 

    #         else: 

    #             alpha = 0

    #             gamma = 0.5 * (
    #                 max(fa, fb)
    #             )

    #             delta0 = gamma 

    #     _, delta = self._outer_bound(0.0, float(delta0))       

    #     new_f = self._affine_constructor(
    #         gamma,
    #         alpha,
    #         delta,
    #         center_shift=center_shift,
    #         radius_scale=radius_scale
    #     )

    #     fun_a , fun_b = new_f.interval

    #     if fun_a <= 0:

    #         old_center = new_f.x0
    #         old_radius = np.sum(np.abs(new_f.xi)) + np.sum(np.abs(new_f.delta))

    #         new_center = 0.5 * fun_b
    #         new_radius = 0.5 * fun_b

    #         center_shift = new_center - old_center

    #         if old_radius > 0:
    #             radius_scale = new_radius / old_radius
    #         else:
    #             radius_scale = 1.0

    #     return new_f._affine_constructor(
    #         gamma = 0,
    #         alpha = 1,
    #         delta = 0,
    #         center_shift=center_shift,
    #         radius_scale=radius_scale
    #     )
    
    def sqrt(self, cheb=False, warn=False):

        a, b = self.interval

        if b<=0:
            raise ValueError(
                f"sqrt undefined for interval [{a}, {b}] "   
            )
        
        if np.isclose(a, b):
            return AffineScalar(
                x0=np.sqrt(a),
                xi=np.zeros_like(self.xi),
                delta=np.zeros(0)
            )

        center_shift = 0.0
        radius_scale = 1.0

        if a <= 0:

            new_a = 0.0
            
            if warn:
                warnings.warn(
                    f"sqrt(): lower bound {a} is non-positive; "
                    f"clipping interval to [{new_a}, {b}]",
                    RuntimeWarning,
                    stacklevel=2
                )

            old_center = self.x0
            old_radius = np.sum(np.abs(self.xi)) + np.sum(np.abs(self.delta))

            new_b = b

            new_center = 0.5 * (new_a + new_b)
            new_radius = 0.5 * (new_b - new_a)

            center_shift = new_center - old_center
            radius_scale = new_radius / old_radius

            a = new_a
            b = new_b

        
        fa = np.sqrt(a)  
        fb = np.sqrt(b)

        # ---------- Chebyshev approximation ----------
        if cheb:

            alpha = alpha = (fb - fa)/(b - a)
            xs = 1/(4*alpha**2)
            gamma = 0.5*(fa + np.sqrt(xs) - alpha*(a + xs))
            delta0 = 0.5*np.abs(np.sqrt(xs) - fa - alpha*(xs - a))

        # ---------- Min-range approximation ----------
        else: 

            alpha = 1/(2*fb)
            gamma = 0.5*(fa + fb - alpha*(a + b))
            delta0 = 0.5*np.abs(fb - fa - alpha*(b - a))

        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,
            delta,
            center_shift=center_shift,
            radius_scale=radius_scale
        )
    
    # Trigonometric functions

    def sin(self, cheb=False):

        a, b = self.interval
        fa = np.sin(a)
        fb = np.sin(b)

        w = b - a

        fmax = max(fa, fb)
        fmin = min(fa, fb)

        # Does interval contain a point where sin = 1?
        kmax_lo = int(np.ceil((a - np.pi/2)/(2*np.pi)))
        kmax_hi = int(np.floor((b - np.pi/2)/(2*np.pi)))

        if kmax_lo <= kmax_hi:
            fmax = 1.0

        # Does interval contain a point where sin = -1?
        kmin_lo = int(np.ceil((a - 3*np.pi/2)/(2*np.pi)))
        kmin_hi = int(np.floor((b - 3*np.pi/2)/(2*np.pi)))

        if kmin_lo <= kmin_hi:
            fmin = -1.0

        # Interval spans at least a full period or half period

        if w >= np.pi:
            
            alpha = 0
            gamma = 0.5 * (fmax + fmin)
            delta0 = 0.5 * (fmax - fmin)
  
        # Chebyshev / Lemma 3

        else:
            if cheb:

                p = (fb - fa) / (b - a)

                # ---------- Case C ----------
                # odd function on symmetric interval

                if abs(a + b) < 1e-12:

                    theta = np.arccos(p)

                    alpha = p

                    gamma = 0.0

                    delta0 = abs(
                        np.sin(theta)
                        - p * theta
                    )
                
                else:
                    # ---------- Find roots of cos(x)=p ----------

                    theta = np.arccos(p)

                    roots = []

                    kmin = int(
                        np.floor((a - theta)/(2*np.pi))
                    ) - 1

                    kmax = int(
                        np.ceil((b + theta)/(2*np.pi))
                    ) + 1

                    for k in range(kmin, kmax + 1):

                        x1 = theta + 2*np.pi*k
                        x2 = -theta + 2*np.pi*k

                        if a <= x1 <= b:
                            roots.append(x1)

                        if a <= x2 <= b:
                            roots.append(x2)

                    roots = sorted(
                        set(np.round(roots, 12))
                    )

                    # ---------- Case A ----------

                    if len(roots) == 1:

                        xi = roots[0]

                        alpha = p

                        gamma = (
                            fa
                            + np.sin(xi)
                            - p*(a + xi)
                        ) / 2

                        delta0 = abs(
                            (
                                np.sin(xi)
                                - fa
                                - p*(xi - a)
                            ) / 2
                        )
                    # ---------- Case B ----------

                    elif len(roots) >= 2:

                        xi1 = roots[0]
                        xi2 = roots[-1]

                        alpha = p

                        gamma = (
                            np.sin(xi1)
                            + np.sin(xi2)
                            - p*(xi1 + xi2)
                        ) / 2

                        delta0 = abs(
                            (
                                np.sin(xi2)
                                - np.sin(xi1)
                                - p*(xi2 - xi1)
                            ) / 2
                        )

                    else:

                        raise RuntimeError(
                            "No solution of cos(x)=p found."
                        )       

            # Minimum range 
            else:

                if fmax == 1 or fmin == -1:
                    alpha = 0
                    gamma = 0.5*(fmin + fmax)
                    delta0 = 0.5*(fmax - fmin)

                else:

                    ca = np.cos(a)
                    cb = np.cos(b)

                    if np.abs(ca) <= np.abs(cb):
                        alpha = ca
                    else:
                        alpha = cb

                    gamma = 0.5*(fa + fb - alpha*(a + b))
                    delta0 = 0.5*np.abs(fb - fa - alpha*(b - a))

        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,
            delta
        )
    
    def cos(self, cheb=False):
        return (self + np.pi/2).sin(cheb=cheb)
    
    def tan(self, cheb=False):

        a, b = self.interval

        k_lo = int(np.ceil((a - np.pi/2)/(np.pi)))
        k_hi = int(np.floor((b - np.pi/2)/(np.pi)))

        if k_lo <= k_hi:
            raise ValueError("interval contains an odd multiple of pi/2")
        
        
        # Count multiples of pi in [a,b]
        k1 = int(np.ceil(a/np.pi))
        k2 = int(np.floor(b/np.pi))

        nroots = max(0, k2 - k1 + 1)
        
        if abs(b-a) < 1e-12:
                return self._affine_constructor(
                    np.tan(a),
                    1/np.cos(a)**2,
                    0.0
                )


        else:

            fa = np.tan(a)
            fb = np.tan(b)


            # Chebyshev / Lemma 3

            if cheb:
                
                p = (fb - fa)/(b - a)

                if p < 1.0 - 1e-12:
                    raise RuntimeError(
                    "Invalid chord slope for tan."
                    )

                
                # Case A:
                # No multiple of pi in interval
                # odd function on symmetric interval
                #--------------------------------------------
                if nroots == 0:

                    theta = np.arccos(1.0/np.sqrt(p))

                    roots = []

                    kmin = int(
                        np.floor((a - np.pi - theta)/(np.pi))
                    ) - 1

                    kmax = int(
                        np.ceil((b + np.pi + theta)/(np.pi))
                    ) + 1

                    for k in range(kmin, kmax + 1):

                        x1 = theta + k*np.pi
                        x2 = -theta + k*np.pi

                        if a <= x1 <= b:
                            roots.append(x1)

                        if a <= x2 <= b:
                            roots.append(x2)

                    roots = sorted(
                        set(np.round(roots, 12))
                    )

                    if len(roots) == 0:

                        raise RuntimeError(
                            "Expected one solution of sec^2(x)=p."
                        )

                    xi = roots[0]

                    alpha = p

                    gamma = (
                        fa
                        + np.tan(xi)
                        - p*(a + xi)
                    )/2

                    delta0 = abs(
                        (
                            np.tan(xi)
                            - fa
                            - p*(xi - a)
                        )/2
                    )

                
                # Case B/C:
                # Exactly one multiple of pi
                # ----------------------------------

                elif nroots == 1:

                    center = k1*np.pi

                    theta = np.arccos(
                        1.0/np.sqrt(p)
                    )

                    xi1 = center - theta
                    xi2 = center + theta

                    # if not (a <= xi1 <= b and a <= xi2 <= b):
                    #     raise RuntimeError(
                    #         "Case B roots not inside interval."
                    #     )

                    alpha = p

                    if not (a <= xi1 <= b and a <= xi2 <= b):

                        gamma = (
                            fa
                            + np.tan(xi1)
                            - p*(a + xi1)
                        )/2

                        delta0 = abs(
                            (
                                np.tan(xi1)
                                - fa
                                - p*(xi1 - a)
                            )/2
                        )

                    # ---------- Case C ----------
                    # symmetric interval about k*pi

                    elif abs((a + b)/2 - center) < 1e-12:

                        gamma = 0.0

                        delta0 = abs(
                            np.tan(xi2)
                            - p*(xi2 - center)
                        )

                        

                    # ---------- Case B ----------

                    else:

                        gamma = (
                            np.tan(xi1)
                            + np.tan(xi2)
                            - p*(xi1 + xi2)
                        )/2

                        delta0 = abs(
                            (
                                np.tan(xi2)
                                - np.tan(xi1)
                                - p*(xi2 - xi1)
                            )/2
                        )
                else:

                    raise RuntimeError(
                        "Unexpected number of multiples of pi."
                    )


            # Minimum range 
            else: 

                if nroots >= 1:
                    alpha = 1.0

                else:
                    alpha = min(
                        1/np.cos(a)**2,
                        1/np.cos(b)**2
                    )

                gamma = 0.5*(fa + fb - alpha*(a + b))
                delta0 = 0.5*np.abs(fb - fa - alpha*(b - a))



        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,                                                                                                                                                                                                                       
            delta
        )
    
    def cotan(self, cheb=False):

        a, b = self.interval

        # singularities: multiples of pi
        k_lo = int(np.ceil(a/np.pi))
        k_hi = int(np.floor(b/np.pi))

        if k_lo <= k_hi:
            raise ValueError(
                "interval contains a multiple of pi"
            )
        
        # count odd multiples of pi/2
        k1 = int(np.ceil((a - np.pi/2)/np.pi))
        k2 = int(np.floor((b - np.pi/2)/np.pi))

        nroots = max(0, k2 - k1 + 1)

        if abs(b - a) < 1e-12:

            return self._affine_constructor(
                np.cot(a),              
                -1.0/(np.sin(a)**2),
                0.0
            )

        fa = 1.0/np.tan(a)
        fb = 1.0/np.tan(b)

        # --------------------------------------------------
        # Chebyshev / Lemma 3
        # --------------------------------------------------

        if cheb:

            p = (fb - fa)/(b - a)

            if p > -1.0 + 1e-12:
                raise RuntimeError(
                    "Invalid chord slope for cot."
                )
            
            # -------------------------
            # Case A
            # -------------------------

            if nroots == 0:

                theta = np.arcsin(
                    1.0/np.sqrt(-p)
                )

                roots = []

                kmin = int(
                    np.floor((a - np.pi - theta)/np.pi)
                ) - 1

                kmax = int(
                    np.ceil((b + np.pi + theta)/np.pi)
                ) + 1

                for k in range(kmin, kmax + 1):

                    x1 = theta + k*np.pi
                    x2 = np.pi - theta + k*np.pi

                    if a <= x1 <= b:
                        roots.append(x1)

                    if a <= x2 <= b:
                        roots.append(x2)

                roots = sorted(
                    set(np.round(roots, 12))
                )

                if len(roots) == 0:
                    raise RuntimeError(
                        "Expected one solution."
                    )

                xi = roots[0]

                alpha = p

                gamma = (
                    fa
                    + 1.0/np.tan(xi)
                    - p*(a + xi)
                )/2

                delta0 = abs(
                    (
                        1.0/np.tan(xi)
                        - fa
                        - p*(xi - a)
                    )/2
                )
            
            # -------------------------
            # Case B/C
            # -------------------------

            elif nroots == 1:

                center = (
                    (k1 + 0.5)*np.pi
                )

                theta = np.arcsin(
                    1.0/np.sqrt(-p)
                )

                xi1 = center - theta
                xi2 = center + theta

                alpha = p

                if not (a <= xi1 <= b and a <= xi2 <= b):

                        gamma = (
                            fa
                            + 1.0/np.tan(xi1)
                            - p*(a + xi1)
                        )/2

                        delta0 = abs(
                            (
                                1.0/np.tan(xi1)
                                - fa
                                - p*(xi1 - a)
                            )/2
                        )

                # symmetric interval
                elif abs((a + b)/2 - center) < 1e-12:

                    gamma = 0.0

                    delta0 = abs(
                        1.0/np.tan(xi2)
                        - p*(xi2 - center)
                    )

                else:

                    gamma = (
                        1.0/np.tan(xi1)
                        + 1.0/np.tan(xi2)
                        - p*(xi1 + xi2)
                    )/2

                    delta0 = abs(
                        (
                            1.0/np.tan(xi2)
                            - 1.0/np.tan(xi1)
                            - p*(xi2 - xi1)
                        )/2
                    )

            else:

                raise RuntimeError(
                    "Unexpected number of odd multiples of pi/2."
                )
            
        # --------------------------------------------------
        # Minimum-range version
        # --------------------------------------------------

        else:

            if nroots >= 1:

                alpha = -1.0

            else:

                alpha = max(
                    -1.0/(np.sin(a)**2),
                    -1.0/(np.sin(b)**2)
                )

            gamma = 0.5*(
                fa + fb
                - alpha*(a + b)
            )

            delta0 = 0.5*np.abs(
                fb - fa
                - alpha*(b - a)
            )
        
        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,                                                                                                                                                                                                                       
            delta
        )
    
    # def sinh(self, cheb=False):

    #     return (self.exp(cheb=cheb) - (-self).exp(cheb=cheb)) * 0.5

    def sinh(self, cheb=False):

        a, b = self.interval

        if abs(b-a) < 1e-12:

            return self._affine_constructor(
                np.sinh(a),
                np.cosh(a),
                0.0
            )

        fa = np.sinh(a)
        fb = np.sinh(b)

        # ---------- Chebyshev approximation ----------
        if cheb:

            p = (fb - fa)/(b - a)

            if p < 1.0 - 1e-12:
                raise RuntimeError(
                    "Invalid chord slope for sinh."
                )
            
            p = max(p, 1.0)
            
            alpha = p

            xs = np.arccosh(alpha)

            # Case A
            if a >= 0.0:

                xi = xs

            elif b <= 0.0:

                xi = -xs

            if a >= 0.0 or b <= 0.0:

                if not (a <= xi <= b):
                    raise RuntimeError(
                        "Unexpected stationary point."
                    )
                
                gamma = (
                    fa
                    + np.sinh(xi)
                    - alpha*(a + xi)
                )/2

                delta0 = abs(
                    (
                        np.sinh(xi)
                        - fa
                        - alpha*(xi - a)
                    )/2
                )

            # Case C
            elif abs(a + b) < 1e-12:

                gamma = 0.0

                delta0 = abs(
                    np.sinh(xs)
                    - alpha*xs
                )

            # Case B
            else:

                xi1 = -xs
                xi2 = xs

                inside1 = (a <= xi1 <= b)
                inside2 = (a <= xi2 <= b)

                if inside1 and inside2:

                    gamma = (
                        np.sinh(xi1)
                        + np.sinh(xi2)
                        - alpha*(xi1 + xi2)
                    )/2

                    delta0 = abs(
                        (
                            np.sinh(xi2)
                            - np.sinh(xi1)
                            - alpha*(xi2 - xi1)
                        )/2
                    )

                elif inside1 or inside2:
                    # Fall back to Case A using whichever root exists
                    xi = xi1 if inside1 else xi2

                    gamma = (
                        fa
                        + np.sinh(xi)
                        - alpha*(a + xi)
                    )/2

                    delta0 = abs(
                        (
                            np.sinh(xi)
                            - fa
                            - alpha*(xi - a)
                        )/2
                    )

                else:
                    raise RuntimeError(
                        "No stationary point found."
                    )

        # ---------- Min-range approximation ----------
        else:

            if a <= 0 <= b:

                alpha = 1
                
            else:
                alpha = min(np.cosh(a), np.cosh(b))

            gamma = 0.5*(
                fa + fb
                - alpha*(a + b)
            )

            delta0 = 0.5*np.abs(
                fb - fa
                - alpha*(b - a)
            )

        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,                                                                                                                                                                                                                       
            delta
        )


    def cosh(self, cheb=False):

        a, b = self.interval

        if abs(b-a) < 1e-12:

            return self._affine_constructor(
                np.cosh(a),
                np.sinh(a),
                0.0
            )

        fa = np.cosh(a)
        fb = np.cosh(b)

        # ---------- Chebyshev approximation ----------
        if cheb:

            alpha = (fb - fa)/(b - a)

            xs = np.arcsinh(alpha)

            gamma = 0.5 * (
                fa
                + np.cosh(xs)
                - alpha*(a + xs)
            )

            delta0 = 0.5 * np.abs(
                np.cosh(xs)
                - fa
                - alpha*(xs - a)
            )

        # ---------- Min-range approximation ----------
        else:

            fmax = max(fa, fb)
            fmin = min(fa, fb)

            if a <= 0 <= b:

                alpha = 0.0
                fmin = 1.0

            elif b < 0:

                alpha = np.sinh(b)

            else:

                alpha = np.sinh(a)

           
            gamma = 0.5 * (
                fmax + fmin
                - alpha*(a + b)
            )

            delta0 = 0.5 * np.abs(
                fmax - fmin
                - alpha*(b - a)
            )

        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,                                                                                                                                                                                                                       
            delta
        )

    def tanh(self, cheb=False):

        a, b = self.interval

        fa = np.tanh(a)
        fb = np.tanh(b)

        if abs(b - a) < 1e-12:

            return self._affine_constructor(
                np.tanh(a),              
                1.0/((np.cosh(a))**2),
                0.0
            )
        
        # --------------------------------------------------
        # Chebyshev / Lemma 3
        # --------------------------------------------------
        
        if cheb:

            p = (fb - fa)/(b - a)

            if not (0.0 <= p <= 1.0 + 1e-12):
                raise RuntimeError(
                    "Invalid chord slope for tanh."
                )

            p = np.clip(p, 0.0, 1.0)

            alpha = p

            xs = np.arctanh(
                np.sqrt(1.0 - p)
            )

            # Case A
            if a >= 0.0:

                xi = xs

            elif b <= 0.0:

                xi = -xs

            if a >= 0.0 or b <= 0.0:

                if not (a <= xi <= b):
                    raise RuntimeError(
                        "Unexpected stationary point."
                    )
                
                gamma = (
                    fa
                    + np.tanh(xi)
                    - alpha*(a + xi)
                )/2

                delta0 = abs(
                    (
                        np.tanh(xi)
                        - fa
                        - alpha*(xi - a)
                    )/2
                )

            # Case C
            elif abs(a + b) < 1e-12:

                gamma = 0.0

                delta0 = abs(
                    np.tanh(xs)
                    - alpha*xs
                )

            # Case B
            else:

                xi1 = -xs
                xi2 = xs

                inside1 = (a <= xi1 <= b)
                inside2 = (a <= xi2 <= b)

                if inside1 and inside2:

                    gamma = (
                        np.tanh(xi1)
                        + np.tanh(xi2)
                        - alpha*(xi1 + xi2)
                    )/2

                    delta0 = abs(
                        (
                            np.tanh(xi2)
                            - np.tanh(xi1)
                            - alpha*(xi2 - xi1)
                        )/2
                    )

                elif inside1 or inside2:
                    # Fall back to Case A using whichever root exists
                    xi = xi1 if inside1 else xi2

                    gamma = (
                        fa
                        + np.tanh(xi)
                        - alpha*(a + xi)
                    )/2

                    delta0 = abs(
                        (
                            np.tanh(xi)
                            - fa
                            - alpha*(xi - a)
                        )/2
                    )

                else:
                    raise RuntimeError(
                        "No stationary point found."
                    )


        # --------------------------------------------------
        # Minimum-range version
        # --------------------------------------------------     
        else:
            
            alpha = min(1-(np.tanh(a))**2, 1-(np.tanh(b))**2)

            gamma = 0.5*(
                fa + fb
                - alpha*(a + b)
            )

            delta0 = 0.5*np.abs(
                fb - fa
                - alpha*(b - a)
            )

        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,                                                                                                                                                                                                                       
            delta
        )
    
    def arcsin(self, cheb=False, warn=False):

        a, b = self.interval

        tol = 1e-12

        # if a < -1.0 - tol or b > 1.0 + tol:
        #     raise RuntimeError(
        #         "arcsin domain exceeded."
        #     )

        # a = np.clip(a, -1.0, 1.0)
        # b = np.clip(b, -1.0, 1.0)

        center_shift = 0.0
        radius_scale = 1.0

        if a < -1.0 - tol or b > 1.0 + tol:

            new_a = np.clip(a, -1.0, 1.0)
            new_b = np.clip(b, -1.0, 1.0)
            
            if warn:
                warnings.warn(
                    f"arcsin(): lower bound {a} and upper bound {b}; "
                    f"clipping interval to [{new_a}, {new_b}]",
                    RuntimeWarning,
                    stacklevel=2
                )

            old_center = self.x0
            old_radius = np.sum(np.abs(self.xi)) + np.sum(np.abs(self.delta))

            new_center = 0.5 * (new_a + new_b)
            new_radius = 0.5 * (new_b - new_a)

            center_shift = new_center - old_center
            radius_scale = new_radius / old_radius

            a = new_a
            b = new_b        

        fa = np.arcsin(a)
        fb = np.arcsin(b)

        if abs(b - a) < 1e-12:

            if abs(abs(a)-1.0) < 1e-12:
                raise RuntimeError(
                    "Derivative of arcsin undefined at ±1."
                )

            return self._affine_constructor(
                np.arcsin(a),              
                1.0/(np.sqrt(1 - a**2)),
                0.0
            )    

        # --------------------------------------------------
        # Chebyshev / Lemma 3
        # --------------------------------------------------
        if cheb:

            p = (fb - fa)/(b - a)

            if p < 1.0 - tol:
                raise RuntimeError(
                    "Invalid chord slope for arcsin."
                )

            xs = np.sqrt(1 - 1/p**2)

            alpha = p

            # Case A
            if a >= 0.0:

                xi = xs

            elif b <= 0.0:

                xi = -xs

            if a >= 0.0 or b <= 0.0:

                if not (a <= xi <= b):
                    raise RuntimeError(
                        "Unexpected stationary point."
                    )
                
                gamma = (
                    fa
                    + np.arcsin(xi)
                    - alpha*(a + xi)
                )/2

                delta0 = abs(
                    (
                        np.arcsin(xi)
                        - fa
                        - alpha*(xi - a)
                    )/2
                )
            
            # Case C
            elif abs(a + b) < tol:

                gamma = 0.0

                delta0 = abs(
                    np.arcsin(xs)
                    - alpha*xs
                )

            # Case B
            else:

                xi1 = -xs
                xi2 = xs

                inside1 = (a <= xi1 <= b)
                inside2 = (a <= xi2 <= b)

                if inside1 and inside2:

                    gamma = (
                        np.arcsin(xi1)
                        + np.arcsin(xi2)
                        - alpha*(xi1 + xi2)
                    )/2

                    delta0 = abs(
                        (
                            np.arcsin(xi2)
                            - np.arcsin(xi1)
                            - alpha*(xi2 - xi1)
                        )/2
                    )

                elif inside1 or inside2:
                    # Fall back to Case A using whichever root exists
                    xi = xi1 if inside1 else xi2

                    gamma = (
                        fa
                        + np.arcsin(xi)
                        - alpha*(a + xi)
                    )/2

                    delta0 = abs(
                        (
                            np.arcsin(xi)
                            - fa
                            - alpha*(xi - a)
                        )/2
                    )

                else:
                    raise RuntimeError(
                        "No stationary point found."
                    )
            

        # --------------------------------------------------
        # Minimum-range version
        # -------------------------------------------------- 
        else:

            if a <= 0 <= b:

                alpha = 1
                
            else:
                alpha = min(1/np.sqrt(1 - a**2), 1/np.sqrt(1 - b**2))

            gamma = 0.5*(
                fa + fb
                - alpha*(a + b)
            )

            delta0 = 0.5*np.abs(
                fb - fa
                - alpha*(b - a)
            )

        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,                                                                                                                                                                                                                       
            delta,
            center_shift=center_shift,
            radius_scale=radius_scale
        )
    
    def arccos(self, cheb=False, warn=False):

        a, b = self.interval

        tol = 1e-12        

        if a < -1.0 - tol or b > 1.0 + tol:

            new_a = np.clip(a, -1.0, 1.0)
            new_b = np.clip(b, -1.0, 1.0)
            
            if warn:
                warnings.warn(
                    f"arccos(): lower bound {a} and upper bound {b}; "
                    f"clipping interval to [{new_a}, {new_b}]",
                    RuntimeWarning,
                    stacklevel=2
                )    

        return (-(self.arcsin(cheb=cheb)) + np.pi/2) 

    def arctan(self, cheb=False):
        
        a, b = self.interval

        fa = np.arctan(a)
        fb = np.arctan(b)

        tol = 1e-12

        if abs(b - a) < 1e-12:

            return self._affine_constructor(
                np.arctan(a),              
                1.0/(1 + a**2),
                0.0
            )
        
        
        # --------------------------------------------------
        # Chebyshev / Lemma 3
        # --------------------------------------------------
        if cheb:

            p = (fb - fa)/(b - a)

            if not (0.0 <= p <= 1.0 + tol):
                raise RuntimeError(
                    "Invalid chord slope for arctan."
                )

            p = np.clip(p, 0.0, 1.0)

            alpha = p

            xs = np.sqrt(1/p - 1)


            # Case A
            if a >= 0.0:

                xi = xs

            elif b <= 0.0:

                xi = -xs

            if a >= 0.0 or b <= 0.0:

                if not (a <= xi <= b):
                    raise RuntimeError(
                        "Unexpected stationary point."
                    )
                
                gamma = (
                    fa
                    + np.arctan(xi)
                    - alpha*(a + xi)
                )/2

                delta0 = abs(
                    (
                        np.arctan(xi)
                        - fa
                        - alpha*(xi - a)
                    )/2
                )

            # Case C
            elif abs(a + b) < tol:

                gamma = 0.0

                delta0 = abs(
                    np.arctan(xs)
                    - alpha*xs
                )

            # Case B
            else:

                xi1 = -xs
                xi2 = xs

                inside1 = (a <= xi1 <= b)
                inside2 = (a <= xi2 <= b)

                if inside1 and inside2:

                    gamma = (
                        np.arctan(xi1)
                        + np.arctan(xi2)
                        - alpha*(xi1 + xi2)
                    )/2

                    delta0 = abs(
                        (
                            np.arctan(xi2)
                            - np.arctan(xi1)
                            - alpha*(xi2 - xi1)
                        )/2
                    )

                elif inside1 or inside2:
                    # Fall back to Case A using whichever root exists
                    xi = xi1 if inside1 else xi2

                    gamma = (
                        fa
                        + np.arctan(xi)
                        - alpha*(a + xi)
                    )/2

                    delta0 = abs(
                        (
                            np.arctan(xi)
                            - fa
                            - alpha*(xi - a)
                        )/2
                    )

                else:
                    raise RuntimeError(
                        "No stationary point found."
                    )


        # --------------------------------------------------
        # Minimum-range version
        # --------------------------------------------------     
        else:
            
            alpha = min(1/(1 + a**2), 1/(1 + b**2))

            gamma = 0.5*(
                fa + fb
                - alpha*(a + b)
            )

            delta0 = 0.5*np.abs(
                fb - fa
                - alpha*(b - a)
            )

        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,                                                                                                                                                                                                                       
            delta
        )

    def arccot(self, cheb=False):

        return ((-self).arctan(cheb=cheb) + np.pi/2)
    
    
    # def powerI(self, r):
    #     a, b = self.interval
    #     I = Interval(a, b)
    #     return I ** Interval(r)
    

    @staticmethod
    def _deriv_abs_pow(x, r, tol=1e-12):

        """Derivative of f(x) = |x|^r, i.e. r * sign(x) * |x|^(r-1)."""

        if abs(x) < tol:

            return 0.0
        
        return r * np.sign(x) * abs(x)**(r - 1)
    
    @staticmethod
    def _deriv_odd_pow(x, r):
        """Derivative of f(x) = sign(x)*|x|^r, i.e. r * |x|^(r-1)."""
        return r * abs(x)**(r - 1)
    
    def pow(self, r: int | float | Fraction, cheb=False, warn=False):

        a, b = self.interval

        if np.isclose(a, b):

            return AffineScalar(
                x0=a**r,
                xi=np.zeros_like(self.xi),
                delta=np.zeros(0)
            )

        if math.isclose(r, 1.0, abs_tol=1e-12):

            return self._affine_constructor(
                0.0,
                1.0,
                0.0
            )
        
        frac = Fraction(r).limit_denominator(10000)

        # Irrational exponent
        if not math.isclose(
            float(frac),
            r,
            rel_tol=1e-12,
            abs_tol=1e-15
        ):
            return ((self.log(cheb=cheb, warn=warn)) * r).exp(cheb=cheb)
        
        p = frac.numerator
        q = frac.denominator

        center_shift = 0.0
        radius_scale = 1.0

        eps = 1e-12

        if q % 2 == 0:
            # right-half graph (x >= 0)

            if a <= 0:

                new_a = 0.0 if r > 0 else eps
            
                if warn:
                    warnings.warn(
                        f"pow(): lower bound {a} is non-positive; "
                        f"clipping interval to [{new_a}, {b}]",
                        RuntimeWarning,
                        stacklevel=2
                    )

                old_center = self.x0
                old_radius = np.sum(np.abs(self.xi)) + np.sum(np.abs(self.delta))

                new_b = b

                new_center = 0.5 * (new_a + new_b)
                new_radius = 0.5 * (new_b - new_a)

                center_shift = new_center - old_center
                radius_scale = new_radius / old_radius

                a = new_a
                b = new_b

            fa = a**r
            fb = b**r

            if cheb:

                alpha = (fb - fa)/(b - a)

                xs = (alpha/r)**(1.0/(r-1))
                fxs = xs**r

                gamma = 0.5*( fxs + fa - alpha*(a + xs) )
                delta0 = 0.5*abs(fxs - fa -alpha*(xs - a))

            else:

                if r < 0:
                    alpha = max(r*a**(r - 1), r*b**(r - 1))

                else:
                    alpha = min(r*a**(r - 1), r*b**(r - 1))
            
                gamma = 0.5*( fa + fb - alpha*(a + b) )
                delta0 = 0.5*abs(fb - fa - alpha*(b - a))


        elif p % 2 == 0:
            # even graph

            if r > 0:

                fa = abs(a)**r
                fb = abs(b)**r

                # --------------------------------------------------
                # Chebyshev / Lemma 3
                # --------------------------------------------------

                if cheb:

                    alpha = (fb - fa) / (b - a)

                    xs = np.sign(alpha/r) * abs(alpha/r)**(1/(r - 1))

                    if not (a <= xs <= b):

                        xs = 0.0
       
                    fxs = abs(xs)**r

                    gamma = 0.5 * (fxs + fa - alpha * (a + xs))
                    delta0 = 0.5 *abs(fxs - fa - alpha * (xs - a))

                else:

                    if a <= 0 <= b:
                        # straddles zero: minimum sits at x = 0
                        alpha = max(self._deriv_abs_pow(a, r), 0.0)
                        fmin = 0.0

                        fmax = max(fa , fb)
                        gamma = 0.5 * (fmax + fmin - alpha * (a + b))
                        delta0 = 0.5 *abs(fmax - fmin - alpha * (b - a))  

                    elif b <= 0:
                        # both endpoints negative, f decreasing
                        alpha = max(self._deriv_abs_pow(a, r), self._deriv_abs_pow(b, r))

                        gamma = 0.5 * (fa + fb - alpha * (a + b))
                        delta0 = 0.5 * abs(fb - fa - alpha * (b - a)) 

                    else:
                        # both endpoints positive, f increasing
                        alpha = min(self._deriv_abs_pow(a, r), self._deriv_abs_pow(b, r))

                        gamma = 0.5 * (fa + fb - alpha * (a + b))
                        delta0 = 0.5 * abs(fb - fa - alpha * (b - a)) 

            else:

                if a <= 0 <= b:

                    raise ValueError(
                      "Interval contains zero"
                    )
                
                fa = abs(a)**r
                fb = abs(b)**r

                if cheb:

                    alpha = (fb - fa)/(b - a)

                    xs = abs(alpha / r)**(1.0 / (r - 1))

                    xi = xs if a > 0 else -xs

                    fxs = abs(xi)**r
                    
                    gamma = 0.5 * (fxs + fa - alpha * (a + xi))
                    delta0 = 0.5 * abs(fxs - fa - alpha * (xi - a))

                else:

                    if b <= 0:

                        alpha = self._deriv_abs_pow(a, r)

                    else:

                        alpha = self._deriv_abs_pow(b, r)

                    gamma = 0.5 * (fa + fb - alpha * (a + b))
                    delta0 = 0.5 * abs(fb - fa - alpha * (b - a))

        else:
            # odd graph

            if r > 0:

                fa = np.sign(a)*abs(a)**r
                fb = np.sign(b)*abs(b)**r

                # --------------------------------------------------
                # Chebyshev / Lemma 3
                # --------------------------------------------------

                if cheb:

                    alpha = (fb - fa) / (b - a)

                    xs = abs(alpha/r)**(1.0/(r-1))
                
                    # Case A
                    if a >= 0.0:

                        xi = xs

                    elif b <= 0.0:

                        xi = -xs

                    if a >= 0.0 or b <= 0.0:
                
                        gamma = (
                            fa
                            + np.sign(xi) * abs(xi)**r
                            - alpha*(a + xi)
                        )/2

                        delta0 = abs(
                            (
                                np.sign(xi) * abs(xi)**r
                                - fa
                                - alpha*(xi - a)
                            )/2
                        )

                    # Case C
                    elif abs(a + b) < eps:

                        gamma = 0.0

                        delta0 = abs(
                            np.sign(xs) * abs(xs)**r
                            - alpha*xs
                        )

                    # Case B
                    else:

                        xi1 = -xs
                        xi2 = xs

                        inside1 = (a <= xi1 <= b)
                        inside2 = (a <= xi2 <= b)

                        if inside1 and inside2:

                            gamma = (
                                np.sign(xi1) * abs(xi1)**r
                                + np.sign(xi2) * abs(xi2)**r
                                - alpha*(xi1 + xi2)
                            )/2

                            delta0 = abs(
                                (
                                    np.sign(xi2) * abs(xi2)**r
                                    - np.sign(xi1) * abs(xi1)**r
                                    - alpha*(xi2 - xi1)
                                )/2
                            )

                        elif inside1 or inside2:
                        # Fall back to Case A using whichever root exists
                            xi = xi1 if inside1 else xi2

                            gamma = (
                                fa
                                + np.sign(xi) * abs(xi)**r
                                - alpha*(a + xi)
                            )/2

                            delta0 = abs(
                                (
                                    np.sign(xi) * abs(xi)**r
                                    - fa
                                    - alpha*(xi - a)
                                )/2
                            )

                        else:
                            raise RuntimeError(
                                "No stationary point found."
                            )
                        
                else:

                    if a <= 0 <= b and r >= 1:

                        alpha = 0.0
                        gamma = 0.5 * (fa + fb)
                        delta0 = 0.5 * abs(fb - fa) 

                    else:

                        alpha = min(
                            r/abs(a)**(1-r),
                            r/abs(b)**(1-r)
                        )

                        gamma = 0.5*(
                            fa + fb
                            - alpha*(a+b)
                        )

                        delta0 = 0.5*abs(
                            fb-fa
                            - alpha*(b-a)
                        )   

            else:

                if a <= 0 <= b:

                    raise ValueError(
                      "Interval contains zero"
                    )
                
                fa = np.sign(a) * abs(a)**r
                fb = np.sign(b) * abs(b)**r

                if cheb:

                    alpha = (fb - fa)/(b -a)

                    xs_mag = abs(alpha / r)**(1.0 / (r - 1))
                    xs = xs_mag if a > 0 else -xs_mag

                    fxs = np.sign(xs) * abs(xs)**r

                    gamma = 0.5 * (fxs + fa - alpha * (a + xs))
                    delta0 = 0.5 * abs(fxs - fa - alpha * (xs - a))

                else:

                    if b <= 0:

                        alpha = self._deriv_odd_pow(a, r)
                    else:

                        alpha = self._deriv_odd_pow(b, r)

                    gamma = 0.5 * (fa + fb - alpha * (a + b))
                    delta0 = 0.5 * abs(fb - fa - alpha * (b - a))


        _, delta = self._outer_bound(0.0, float(delta0))

        return self._affine_constructor(
            gamma,
            alpha,                                                                                                                                                                                                                       
            delta,
            center_shift=center_shift,
            radius_scale=radius_scale
        )

    def __repr__(self):

        return (
            f"AffineScalar("
            f"x0={self.x0}, "
            f"xi={self.xi}), "
            f"delta ={self.delta}"
        )
    

class AffineArray:

    # def __init__(self, x0, E):

    #     self.x0 = np.asarray(x0, dtype=float)

    #     self.E = np.asarray(E, dtype=float)

    def __init__(self, x0, Xi=None, Delta=None):

        self.x0 = np.asarray(x0, dtype=float)

        if Xi is None:
            Xi = np.zeros((len(x0), 0))

        if Delta is None:
            Delta = np.zeros((len(x0), 0))

        self.Xi = np.asarray(Xi, dtype=float)
        self.Delta = np.asarray(Delta, dtype=float)

    # @classmethod
    # def from_intervals(cls, intervals):

    #     intervals = np.asarray(intervals, dtype=float)

    #     lb = intervals[:, 0]
    #     ub = intervals[:, 1]

    #     x0 = 0.5 * (lb + ub)

    #     r = 0.5 * (ub - lb)

    #     E = np.diag(r)

    #     return cls(x0, E)

    @classmethod
    def from_intervals(cls, intervals):

        intervals = np.asarray(intervals, dtype=float)

        lb = intervals[:, 0]
        ub = intervals[:, 1]

        x0 = 0.5 * (lb + ub)

        r = 0.5 * (ub - lb)

        Xi = np.diag(r)

        Delta = np.zeros((len(x0), 0))

        return cls(x0, Xi, Delta)

    # @property
    # def interval(self):

    #     rad = np.sum(np.abs(self.E), axis=1)

    #     lb = self.x0 - rad
    #     ub = self.x0 + rad

    #     # return np.column_stack((lb, ub))
    #     return [
    #         float(lb),
    #         float(ub)
    #     ]

    @property
    def interval(self):

        rad = (
            np.sum(np.abs(self.Xi), axis=1)
            + np.sum(np.abs(self.Delta), axis=1)
        )

        lb = self.x0 - rad
        ub = self.x0 + rad

        return np.column_stack((lb, ub))

    def split_interval(self, splits_per_dim):

        lower = []
        upper = []

        for x in self:
            a, b = x.interval
            lower.append(a)
            upper.append(b)
        
        dim_intervals = []

        for d in range(len(lower)):

            grid = np.linspace(
                lower[d],
                upper[d],
                splits_per_dim[d] + 1
            )

            dim_intervals.append([
                (grid[i], grid[i+1])
                for i in range(splits_per_dim[d])
            ])

        sub_arrays = []

        for combo in product(*dim_intervals):

            sub_arrays.append(
                AffineArray.from_intervals(combo)
            )

        return sub_arrays

    # def __getitem__(self, idx):

    #     # print("getitem:", idx)

    #     if idx < 0 or idx >= len(self.x0):
    #         raise IndexError

    #     return AffineScalar(self, idx)

    # def __repr__(self):

    #     return (
    #         f"AffineArray(\n"
    #         f"x0={self.x0},\n"
    #         f"E=\n{self.E}\n)"
    #     )

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.x0):
            raise IndexError

        return AffineScalar(self, idx)


    def __repr__(self):

        return (
            f"AffineArray(\n"
            f"x0={self.x0},\n"
            f"Xi=\n{self.Xi},\n"
            f"Delta=\n{self.Delta}\n)"
        )