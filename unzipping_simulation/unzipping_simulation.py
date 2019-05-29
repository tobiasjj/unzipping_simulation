#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# unzipping_simulation, simulate the force extension of unzipipng DNA
# Copyright 2018 Tobias Jachowski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO: Distributed calculation over the network, see
# https://eli.thegreenplace.net/2012/01/24/distributed-computing-in-python-with-multiprocessing/

import cloudpickle
import copy
import hashlib
import math
import mpmath
import numpy as np
import pickle
import time
import warnings
from multiprocessing import Pool
from scipy import constants
from scipy.integrate import quad
from scipy.optimize import fminbound, minimize_scalar
# from scipy.optimize import fmin_l_bfgs_b, minimize, brent
# minimize_scalar alone causes imprecise results
from scipy.optimize._minimize import _minimize_lbfgsb

from matplotlib import pyplot as plt
# import cn_plot_style as cnps

# Boltzmann constant
kB = constants.value('Boltzmann constant')
cal = constants.calorie
Na = constants.Avogadro
kcal = 1e+3*cal

# Set digits of precision
mpmath.mp.dps = 30


def coth(x):
    """
    Cotangens hyperbolicus
    """
    typ = type(x)
    if typ is mpmath.mpf or typ is mpmath.mpc:
        return mpmath.coth(x)
    else:
        # return np.cosh(x) / np.sinh(x)  # can cause overflows
        return (1 + np.exp(-2*x)) / (1 - np.exp(-2*x))


def ext_ssDNA(F, nbs=0, S=None, L_p=None, z=None, T=298.2, avoid_neg_ext=True):
    """
    Freely jointed chain (FJC) model, relating the total polymer length
    ext_ssDNA to an applied force F.

    Bockelmann, U., Essevaz-Roulet, B., Heslot, F. (1998). "DNA strand
    separation studied by single molecule force measurements". Physical Review
    E, 58(2), 2386-94.

    Steven B. Smith, Yujia Cui, Carlos Bustamante (1996). "Overstretching
    B-DNA: The Elastic Response of Individual Double-Stranded and
    Single-Stranded DNA Molecules". Science Reports, 271, 795-799

    Contour length of ssDNA: L_0 = nbs*z

    Kuhn length b (in FJC b = 2 * persistence length), in paper: b = 1.5 nm

    Parameters
    ----------
    nbs : int
        Number of bases of ssDNA
    F : float
        Force in N
    S : float
        Stretch modulus in N
    L_p : float
        Persistence length in m
    z : float
        Length of a single base in m
    T : float
        Temperature in K

    Returns
    -------
    float
        Extension in m
    """
    S = S or 800e-12
    L_p = L_p or 7.97e-10
    z = z or 0.537e-9

    if F == 0:
        return 0

    if nbs <= 0:
        return 0

    sign = 1
    if F < 0:
        F = -F
        sign = -1

    # Prevent float overflow in downstream calculation leading to
    # too high x value. The exact value seems to depend on the system.
    # On a laptop with Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz a
    # value of F < 1.28323360182078e-26 was sufficient. However, on
    # another Intel system it was not. Therefore, to be one the save
    # side, choose a value as a break criterion of F < 1e-17 N
    # (i.e. 10 aN), which is more than sufficiently small to not
    # affect a precise determination of x and still should work on
    # most systems.
    # Alternatively, one could use mpmath.coth, but would have to live
    # with a 10 fold increase in execution time.
    if F < 1e-17:
        return 0

    b = 2*L_p
    # modified FJC model incorporating Kuhn segments that can stretch
    # entropic (coth term) and stretching (F/S term) contribution
    x = nbs*z * (coth(F*b / (kB*T)) - kB*T / (F*b)) * (1 + F/S)

    if avoid_neg_ext:
        x = max(x, 0)

    return sign * x


def _F_ssDNA(x, nbs=0, S=None, L_p=None, z=None, T=298.2, f_min=None,
             f_max=None, xtol=None, avoid_neg_F=True):
    """
    Freely jointed chain (FJC) model, relating the applied force F to a
    total polymer length ext_ssDNA.

    Contour length of ssDNA: L_SS = j*z

    Kuhn length b (in FJC b = 2 * persistence length), in paper: b=1.5e-9

    Parameters
    ----------
    nbs : int
        Number of bases of ssDNA
    x : float
        Extension in m
    S : float
        Stretch modulus in N
    L_p : float
        Persistence length in m
    z : float
        Length of a single base in m
    T : float
        Temperature in K
    """
    f_min = f_min or 0e-12
    f_max = f_max or 200e-12
    xtol = xtol or 1e-18

    if x == 0:
        return 0

    if nbs <= 0:
        return 0

    sign = 1
    if x < 0:
        x = -x
        sign = -1

    # Numerical invert ext_ssDNA function
    def f_ssDNA_cost(f):
        return (ext_ssDNA(f, nbs=nbs, S=S, L_p=L_p, z=z, T=T,
                          avoid_neg_ext=False) - x)**2

    # Find the force, which will result in input extension
    # To speed up the minimization, first find an unprecise
    # answer with a quick algorithm and than make it precise
    # f_fit = minimize(f_ssDNA_cost,
    #                  x0=12e-9,
    #                  bounds=((f_min, f_max), ),
    #                  tol=xtol).x
    # f_fit = brent(f_ssDNA_cost,
    #               brack=(f_min, f_max),
    #               tol=xtol)
    f_fit = minimize_scalar(f_ssDNA_cost,
                            bounds=(f_min, f_max),
                            options={'xtol': xtol}).x
    f_min = f_fit - 1e-10
    f_max = f_fit + 1e-10
    f_fit = fminbound(f_ssDNA_cost,
                      f_min,
                      f_max,
                      xtol=xtol)

    if avoid_neg_F:
        f_fit = max(f_fit, 0)

    return sign * f_fit


def _ext_dsDNA_wlc(F, nbp=0, pitch=None, L_p=None, T=298.2, x_min=0e-12,
                   xtol=1e-15):
    # See also default value in function in `F_dsDNA_wlc`
    pitch = pitch or 0.338e-9
    if F == 0:
        return 0

    if nbp <= 0:
        return 0

    sign = 1
    if F < 0:
        F = -F
        sign = -1

    # WLC only valid in the interval x = (-L_0, L_0)
    # - 1e-25  # corresponds to force of inf # >> 2e+9 N
    x_max = nbp * pitch

    # Numerical invert ext_dsDNA function
    def ext_dsDNA_cost(x):
        return (F_dsDNA_wlc(x, nbp=nbp, pitch=pitch, L_p=L_p, T=T) - F)**2

    # Find the force, which will result in input extension
    # To speed up the minimization, first find an unprecise
    # answer with a quick algorithm and than make it precise
    # x_fit = minimize_scalar(ext_dsDNA_cost,
    #                         bounds=(x_min, x_max),
    #                         options={'xtol': xtol}).x
    # x_min = x_fit - 1e-10
    # x_max = x_fit + 1e-10
    x_fit = fminbound(ext_dsDNA_cost,
                      x_min,
                      x_max,
                      xtol=xtol)
    return sign * x_fit


def F_dsDNA_wlc(x, nbp=0, pitch=None, L_p=None, T=298.2):
    """
    A worm-like chain model.

    Parameters
    ----------
    x : float
        Extension (m)
    pitch : float
        Contour length (m). Also denoted as 'L_0'.
    L_p : float
        Persistence length (m)
    T : float
        Temperature (K)

    Returns
    -------
    1D numpy.ndarray of type float
        Force (N).
    """
    pitch = pitch or 0.338e-9
    L_p = L_p or 50e-9

    if x == 0:
        return 0

    if nbp <= 0:
        return 0

    sign = 1
    if x < 0:
        x = -x
        sign = -1

    # Contour length
    L_0 = nbp*pitch

    if x >= L_0:
        return float('inf') * sign

    # Marko, J.F.; Eric D. Siggia. "Stretching DNA". Macromolecules. 1995. 28:
    # 8759–8770. doi:10.1021/ma00130a008
    # F = kB * T / L_p * (1 / (4 * (1 - x / L_0)**2) - 1/4 + x / L_0)

    # Relative extension
    x = x / L_0
    
    # Petrosyan, R. "Improved approximations for some polymer extension
    # models". Rehol Acta. 2016. doi:10.1007/s00397-016-0977-9
    F = kB * T / L_p * (1 / (4 * (1 - x)**2) - 1/4 + x - 0.8 * x**2.15)

    return F * sign


def poly10(c, x):
    return (c[0]*x**10
            + c[1]*x**9
            + c[2]*x**8
            + c[3]*x**7
            + c[4]*x**6
            + c[5]*x**5
            + c[6]*x**4
            + c[7]*x**3
            + c[8]*x**2
            + c[9]*x
            + c[10])


def poly11(c, x):
    return (c[0]*x**11
            + c[1]*x**10
            + c[2]*x**9
            + c[3]*x**8
            + c[4]*x**7
            + c[5]*x**6
            + c[6]*x**5
            + c[7]*x**4
            + c[8]*x**3
            + c[9]*x**2
            + c[10]*x
            + c[11])


power = np.arange(20)[::-1]
def polyn(c, x):
    return np.dot(c, x ** power[-c.size:])


def polynom(c, x, n):
    if n == 10:
        return poly10(c, x)
    if n == 11:
        return poly11(c, x)
    return polyn(c, x)


def polynomial(f, x_min, x_max, num_x, p_deg, verbose=False,
               factors_only=False, key=0, **kwargs):
    # Create test data to be fitted, according to the actual
    # fjc model for a Y range which is expected to
    # happen during unzipping
    X = np.linspace(x_min, x_max, num_x)
    Y = np.array([f(x, key, **kwargs) for x in X])

    # Create polynomial for the f(x) curve
    c = np.polyfit(X, Y, p_deg)
    fp = lambda x: polyn(c, x)

    if verbose:
        # Calculate STD (STE) of polynomial
        X = np.linspace(x_min, x_max, num_x)
        Y = np.array([f(x, key, **kwargs) for x in X])
        Yp = np.array([fp(x) for x in X])
        std = np.sqrt(((Y - Yp)**2).sum() / (len(Y) - 1))
        # ste = std / np.sqrt(len(Y))
        print('The STE of the polynomial is: ', std)

    if factors_only:
        return c

    return fp


class unboundfunction(object):
    """
    Class to hold references to functions and still be able to pickle them.
    To reference the function you want to bind:
    function_reference = boundmethod(function)
    """
    def __init__(self, ft, **kwargs):
        self.ft = ft

    def __getstate__(self):
        return cloudpickle.dumps(self.ft)

    def __setstate__(self, cloudepickled_ft):
        self.ft = cloudpickle.loads(cloudepickled_ft)

    def __call__(self, *a, **kw):
        return self.ft(*a, **kw)


class DNA_MODEL_APPROX(object):
    def __init__(self, f, yx_f, key,
                 x_min=0e-12, x_mid=25e-12, x_max=200e-12,
                 num_f_low=501, num_f_high=501, p_deg_low=10, p_deg_high=10,
                 **kwargs):
        self._f = f
        self._yx_f = unboundfunction(yx_f)
        self._p_min = x_min
        self._p_mid = x_mid
        self._p_max = x_max
        self._num_f_low = num_f_low
        self._p_deg_low = p_deg_low
        self._num_f_high = num_f_high
        self._p_deg_high = p_deg_high
        self._kwargs = {}
        self._key = key

        # Set the default values for the model
        self._kwargs = {}
        self._kwargs.update(kwargs)

        # Set the approximation
        self.reset()

    def reset(self):
        self.Fp_low = {}
        self.Fp_high = {}
        self.x_min = {}
        self.x_mid = {}
        self.x_max = {}
        self.keys = set([])

    def calculate_approximation(self, key, verbose=False):
        # if verbose:
        #     print('\rCalculating approximation model for {:04d} ...'
        #           ''.format(key), end='', flush=True)
        # Set default values for model
        kwargs = self._kwargs

        # Get intervals for low/high approximation
        x_min = self.yx_f(self._p_min, key, **kwargs)
        x_mid = self.yx_f(self._p_mid, key, **kwargs)
        x_max = self.yx_f(self._p_max, key, **kwargs)

        # Calculate factors for low approximation
        num_f = self.num_f_low
        p_deg = self.p_deg_low
        cp_low = polynomial(self.f, x_min, x_mid, num_f, p_deg,
                            verbose=verbose, factors_only=True, key=key,
                            **kwargs)

        # Calculate high approximation
        num_f = self.num_f_high
        p_deg = self.p_deg_high
        cp_high = polynomial(self.f, x_mid, x_max, num_f, p_deg,
                             verbose=verbose, factors_only=True, key=key,
                             **kwargs)

        return key, cp_low, cp_high, x_min, x_mid, x_max

    def set_approximation(self, ap, key=None):
        key = key or ap[0]
        fp_l = lambda x, c=ap[1]: polyn(c, x)
        fp_h = lambda x, c=ap[2]: polyn(c, x)
        # def fp_l(x, c=ap[1]):
        #     return polyn(c, x)
        # def fp_h(x, c=ap[2]):
        #     return polyn(c,x)
        x_min = ap[3]
        x_mid = ap[4]
        x_max = ap[5]

        self.Fp_low[key] = unboundfunction(fp_l)
        self.Fp_high[key] = unboundfunction(fp_h)
        self.x_min[key] = x_min
        self.x_mid[key] = x_mid
        self.x_max[key] = x_max
        self.keys.add(key)

    def __call__(self, x=0.0, key=0, avoid_neg_Y=True, verbose=False):
        # Automatically create Fp(x) polynomial and cache it for future calls
        if key not in self.keys:
            ap = self.calculate_approximation(key, verbose=verbose)
            self.set_approximation(ap, key)

        # Fallback, if x out of fitted range from polynomial
        if x < self.x_min[key] or self.x_max[key] < x:
            if verbose:
                print('Calculation in {} for {:.3e} !< {:.3e} !< {:.3e} is out'
                      'of range!'.format(
                            self.__class__.__name__,
                            self.x_min[key], x, self.x_max[key]),
                      'Adjust x_min or x_max for polynomial approximation.')
            return self.f(x, **self._kwargs)

        # Return approximation according to the low/high range
        if self.x_min[key] <= x and x <= self.x_mid[key]:
            y = self.Fp_low[key](x)
        elif self.x_mid[key] < x and x <= self.x_max[key]:
            y = self.Fp_high[key](x)

        if avoid_neg_Y:
            y = max(y, 0)

        return y

    @property
    def f(self):
        return self._f

    @property
    def yx_f(self):
        return self._yx_f

    @property
    def key(self):
        return self._key

    @property
    def num_f_low(self):
        return self._num_f_low

    @property
    def p_deg_low(self):
        return self._p_deg_low

    @property
    def num_f_high(self):
        return self._num_f_high

    @property
    def p_deg_high(self):
        return self._p_deg_high

    # does not work well with multiprocessing
    # def __getattr__(self, name):
    #     """
    #     Allow attributes to be used as kwargs keys
    #     """
    #     if name in self._kwargs:
    #         return self._kwargs[name]
    #     else:
    #         raise AttributeError(name)


class F_SSDNA(DNA_MODEL_APPROX):
    def __init__(self, S=None, L_p=None, z=None, T=298.2,
                 f_min=0e-12, f_mid=25e-12, f_max=200e-12,
                 num_f_low=501, num_f_high=501, p_deg_low=10, p_deg_high=8):
        f = _F_ssDNA
        yx_f = ext_ssDNA
        super().__init__(f, yx_f, key='nbs', S=S, L_p=L_p, z=z, T=T,
                         x_min=f_min, x_mid=f_mid, x_max=f_max,
                         num_f_low=num_f_low, num_f_high=num_f_high,
                         p_deg_low=p_deg_low, p_deg_high=p_deg_high)

    def __call__(self, x=0.0, nbs=0, avoid_neg_F=True, verbose=False,
                 **ignored):
        if nbs <= 0:
            return 0

        if x == 0.0:
            return 0.0
        sign = 1
        if x < 0:
            x = -x
            sign = -1
        f = super().__call__(x=x, key=nbs, avoid_neg_Y=avoid_neg_F,
                             verbose=verbose)

        return sign * f


class E_EXT_SSDNA(DNA_MODEL_APPROX):
    def __init__(self, S=None, L_p=None, z=None, T=298.2,
                 f_min=0e-12, f_mid=25e-12, f_max=200e-12,
                 num_f_low=501, num_f_high=501,
                 p_deg_low=11, p_deg_high=8):
        f = _E_ext_ssDNA
        yx_f = ext_ssDNA
        super().__init__(f, yx_f, key='nbs', S=S, L_p=L_p, z=z, T=T,
                         x_min=f_min, x_mid=f_mid, x_max=f_max,
                         num_f_low=num_f_low, num_f_high=num_f_high,
                         p_deg_low=p_deg_low, p_deg_high=p_deg_high)

    def __call__(self, x=0.0, nbs=0, avoid_neg_E=True, verbose=False,
                 **ignored):
        if nbs <= 0:
            return 0.0

        if x == 0.0:
            return 0.0

        # There are no negative energies, even for negative extensions
        if x < 0:
            x = -x

        return super().__call__(x=x, key=nbs, avoid_neg_Y=avoid_neg_E,
                                verbose=verbose)


class EXT_DSDNA(DNA_MODEL_APPROX):
    def __init__(self, pitch=None, L_p=None, T=298.2,
                 f_min=0e-12, f_mid=4e-12, f_max=200e-12,
                 num_f_low=501, num_f_high=501, p_deg_low=14, p_deg_high=16):
        f = _ext_dsDNA_wlc

        def yx_f(x, *args, **kwargs):
            return x
        super().__init__(f, yx_f, key='nbp', pitch=pitch, L_p=L_p, T=T,
                         x_min=f_min, x_mid=f_mid, x_max=f_max,
                         num_f_low=num_f_low, num_f_high=num_f_high,
                         p_deg_low=p_deg_low, p_deg_high=p_deg_high)

    def __call__(self, F=0.0, nbp=0, avoid_neg_e=True, verbose=False,
                 **ignored):
        if nbp <= 0:
            return 0

        if F == 0.0:
            return 0.0

        sign = 1
        if F < 0:
            F = -F
            sign = -1

        if F > self._p_max:
            return self._kwargs['pitch'] + nbp
            # return float('inf')

        x = super().__call__(x=F, key=nbp, avoid_neg_Y=avoid_neg_e,
                             verbose=verbose)

        return sign * x


def init_buf_ext_dsDNA_wlc(nbp=0, pitch=None, L_p=None, T=298.2,
                           f_min=0e-12, f_mid=4e-12, f_max=200e-12,
                           num_f_low=501, num_f_high=501,
                           p_deg_low=14, p_deg_high=16):
    # Assign DNA model function to the variable of the global (module) scope,
    # such that `multiprocessing.Pool` will see these variables.
    global ext_dsDNA_wlc
    # Initialize the approximations of the dsDNA model function
    # with fixed model function parameters and substitute the original
    # DNA model functons
    ext_dsDNA_wlc = EXT_DSDNA(pitch=pitch, L_p=L_p, T=T,
                              f_min=f_min, f_mid=f_mid, f_max=f_max,
                              num_f_low=num_f_low, num_f_high=num_f_high,
                              p_deg_low=p_deg_low, p_deg_high=p_deg_high)

    # Initialize the ext_dsDNA_wlc object approximation with the needed nbp
    if nbp > 0:
        ap_ext_dsDNA = ext_dsDNA_wlc.calculate_approximation(nbp)
        ext_dsDNA_wlc.set_approximation(ap_ext_dsDNA, nbp)

    return ext_dsDNA_wlc


def init_buf_E_ext_ssDNA(read=True, write=False, filename='E_ext_ssDNA',
                         processes=8, bases='', nbs=0, nbs_loop=0,
                         S=None, L_p=None, z=None,
                         T=298.2, f_min=0e-12, f_mid=25e-12, f_max=200e-12,
                         num_f_low=501, num_f_high=501,
                         p_deg_low=11, p_deg_high=8):
    # Assign DNA model function to the variable of the global (module) scope,
    # such that `multiprocessing.Pool` will see these variables.
    global E_ext_ssDNA

    # Initialize the E_ext_ssDNA objects approximation with all needed nbs,
    # either read from file or calculate
    if read:
        # read aps_E_ext_ssDNA from file
        with open(filename, 'rb') as f:
            E_ext_ssDNA = pickle.load(f)

        model_kwargs = {
            'S': S,
            'L_p': L_p,
            'z': z,
            'T': T
        }

        for k in model_kwargs.keys():
            if model_kwargs[k] != E_ext_ssDNA._kwargs[k]:
                warnings.warn('{} in E_ext_ssDNA was set from read model to:'
                              '{}'.format(k, E_ext_ssDNA._kwargs[k]))

    else:
        # Initialize the approximations of the ssDNA model function
        # with fixed model function parameters and substitute the original
        # DNA model functons
        E_ext_ssDNA = E_EXT_SSDNA(S=S, L_p=L_p, z=z, T=T,
                                  f_min=f_min, f_mid=f_mid, f_max=f_max,
                                  num_f_low=num_f_low, num_f_high=num_f_high,
                                  p_deg_low=p_deg_low, p_deg_high=p_deg_high)

        # Define a closure to be executed by the pool
        def f(nbs):
            # E_ext_ssDNA, nbs = args
            print('\rCalculating approximation model for nbs = {:04d} ...'
                  ''.format(nbs), end='', flush=True)
            return E_ext_ssDNA.calculate_approximation(nbs)
        f = unboundfunction(f)

        # Use all available CPUs for the calculation to speed up calculations
        with Pool(processes=processes) as pool:
            start = time.time()
            # Calculate all possible bases of ssDNA, i.e. 10 bases of 5*pT
            # spacer, + all possibilities of unzipped basepairs (1 to 811
            # basepairs * 2), + 10 bases hairpin, if last basepair is unzipped
            nuz_max = len(bases)
            nob = list(range(nbs, nuz_max*2+nbs+1, 2))
            nob.append(nob[-1] + nbs_loop)
            # args = [(E_ext_ssDNA, nbs) for nbs in nob]
            aps_E_ext_ssDNA = pool.map(f, nob)
            stop = time.time()
            print('\nDone, elapsed time: {:.1f} s'.format(stop - start))

        for ap in aps_E_ext_ssDNA:
            E_ext_ssDNA.set_approximation(ap)

        if write:
            # save E_ext_ssDNA to file
            with open(filename, 'wb') as f:
                pickle.dump(E_ext_ssDNA, f)

    return E_ext_ssDNA


def F_lev(x0, x_ss=0.0, x_ds=0.0, kappa=0.0):
    """
    Parameters
    ----------
    x0 : float
        Total displacement (m)
    x_ss : float
        Extension of ssDNA (m)
    x_ds : float
        Extension of dsDNA (m)
    kappa : float
        Spring constant (N/m)
    """
    return kappa * (x0 - x_ss - x_ds)


def F_construct_2D(x0, x_ss=0.0, x_ds=0.0, r=0.0, z0=0.0, kappa=0.0,
                   xtol=1e-18):
    """
    Parameters
    ----------
    x0 : float
        Total displacement (m)
    x_ss : float
        Extension of ssDNA (m)
    x_ds : float
        Extension of dsDNA (m)
    r : float
        Radius of the bead/handle (m)
    z0 : float
        Distance of the bead surface to the glass surface, if
        the bead is in its resting position, i.e. no force in
        the vertical direction is applied (m).
    kappa : float or np.ndarray of type float
        Spring constant (N/m). If `kappa` of type float, only
        one axis (i.e. X or Y) is considered. If `kappa` of
        type np.ndarray, the first number is X (or Y) axis and
        the second number is Z.
    """
    # Pythagoras:
    #   a is horizontal distance of attachment point to the center of the bead
    #   b is vertical distance of the surface to the center of the bead
    #   c is extension of the construct (x_ss + x_ds) plus the bead radius (r)
    #   dx is the horizontal displacement of the bead (x or y)
    #   dz is the vertical displacement of the bead (z)
    # a = x0 - dx
    # b = z0 + r - dz
    c = x_ss + x_ds + r
    # a**2 + b**2 = c**2
    # ->
    # (x0 - dx)**2 + (z0 + r - dz)**2 = c**2
    # dz = z0 + r - math.sqrt(c**2 - (x0 - dx)**2)
    # dx = x0 - math.sqrt(c**2 - (z0 + r - dz)**2)

    # construct is longer than possible stretching with dx/dz >= 0.
    # -> bead will equilibrate in the middle of the trap with zero force
    if c**2 >= x0**2 + (z0 + r)**2:
        return 0.0, 0.0, 0.0

    # If z0 is 0 or the stiffness of z is 0 bead will always
    # touch the surface and dx only depends on x0, x_ss, x_ds, and r.
    if z0 == 0 or isinstance(kappa, float):
        if not isinstance(kappa, float):
            kappa = kappa[0]
        dx = x0 - math.sqrt(c**2 - r**2)
        dz = z0
        # force that need to be acting on the construct to
        # result in a corresponding horizontal force (in x/y)
        fx = dx * kappa  # * (x0 - dx) / c
        cos_alpha = (x0 - dx) / c
        fxz = fx / cos_alpha

        return fxz, dx, dz

    # displacement dz dependent upon dx
    def _dz(dx):
        # print('z0 {:.1e}, c {:.1e}, x0 {:.1e}, dx {:.1e}'
        #       ''.format(z0, c, x0, dx))
        return z0 + r - math.sqrt(c**2 - (x0 - dx)**2)

    # displacement dx dependent upon dz
    def _dx(dz):
        # x0 8.0e-07, c 1.3e-07, z0 2.0e-07, r 0.0e+00, dz 0.0e+00
        # print('x0 {:.1e}, c {:.1e}, z0 {:.1e}, r {:.1e}, dz {:.1e}'
        #       ''.format(x0, c, z0, r, dz))
        return x0 - math.sqrt(c**2 - (z0 + r - dz)**2)

    # difference of the ratio of the force in x/z to the ratio of a/b
    # the construct with the handle equilibrates where diff == 0
    def diff_tan_fxz_ab(dx):
        a = x0 - dx
        b = z0 + r - _dz(dx)
        fx = dx * kappa[0]
        fz = _dz(dx) * kappa[1]
        diff = b/a - fz/fx
        # diff = math.sqrt(c**2 - (x0 - dx)**2)
        #                 / (x0 - dx)
        #                 - (_dz(dx) * kappa[1])
        #                 / (dx * kappa[0])
        return diff**2

    # if construct is shorter than z0 + r, dz has to be at least the difference
    dz_min = max(0, z0 + r - c)
    # dz can be at max as large as z0, then the bead touches the surface
    dz_max = z0
    # dx has to be at least x0 - c
    dx_min = max(0, x0 - c, _dx(dz_max))
    dx_max = max(0, _dx(dz_min))
    # print('dx_min {:.1e}, dx_max {:.1e}'.format(dx_min, dx_max))

    # Calculate the displacement of x (and z), where the angle between the
    # force vector of the construct and the force vector of the bead
    # displacement is 0° (180°)
    # Unfortunately, there is no analytical solution to this ...
    dx = fminbound(diff_tan_fxz_ab, dx_min, dx_max, xtol=xtol)

    # the force needed to be acting on the construct to result in a
    # corresponding force acting on the handle
    # the resulting force is the combination of the horizontal force acting on
    # the handle and  the normal force of the bead touching the surface and/or
    # the vertical trapping force acting on the handle
    fx = dx * kappa[0]
    cos_alpha = (x0 - dx) / c
    # print(fx / f(dx) - cos_alpha)
    fxz = fx / cos_alpha

    # print(dx, dz_min, dz_max, dx_min, dx_max)
    # dz = z0 + r - math.sqrt(c**2 - (x0 - dx)**2)
    # print('dx {:.1e}, dz {:.1e}'
    #       ''.format(dx, z0 + r - math.sqrt(c**2 - (x0 - dx)**2)))
    # a = x0 - dx
    # b = z0 + r - _dz(dx)
    # print('x0 {:.3e}, a {:.3e}, b {:.3e}, c {:.3e}'.format(x0, a, b, c))
    # #print('dx {:.3e}, dz {:.3e}, fx{:.1e}, fz {:.1e}'
    # #      ''.format(dx, _dz(dx), dx*kappa[0], _dz(dx)*kappa[1]))
    # #print('dzmin {:.1e}, dzmax {:.1e}, dxmin {:.1e}, dxmax {:.1e}, f {:.1e}'
    # #      ''.format(dz_min, dz_max, dx_min, dx_max, f(dx)))
    return fxz, dx, _dz(dx)


def F_construct_3D(A0, x_ss=0.0, x_ds=0.0, f_dna=0.0, r0_sph=None,
                   kappa=None, k_rot=None,
                   factr=1e5, gtol=1e-5, eps_angle=1e-8,
                   high_precision_eps_angle=1e-12, eps_rot=1e-8,
                   verbose=False, deep_verbose=False, print_result=False,
                   return_plus=False):
    """
    Origin of the coordinate system is the center of the trap [0, 0, 0].
    The coordinates are given for a right handed cartesian coordinate system.

    Parameters
    ----------
    A0 : np.ndarray of type float
        Position (m) of the DNA attachment point on the glass surface relative
        to the trap center: [x, y, z].
    x_ss : float
        Extension of ssDNA (m).
    x_ds : float
        Extension of dsDNA (m).
    f_dna : float
        Force (N) acting on the DNA construct that corresponds to the
        extensions `x_ss` and `x_ds`.
    r0_sph : np.ndarray of type float, optional
        Radius vector of the bead at zero force, i.e. where the attachment
        point of the DNA is located on the bead, when the bead is able to
        freely orient itself in the trap and no external force is acting on it.
        Vector is given in spherical units: [radius, θ, φ], where radius is the
        radius of the bead in m, θ is the inclination, and φ is the azimuth.
        Defaults to [0, 0, 0].
    kappa : np.ndarray of type float
        Spring constant (N/m) of the trap: [x, y, z].
    k_rot : np.ndarray of type float, optional
        Rotational stiffness (N/rad) of the bead relative to `r0_sph`, with two
        components pointing in the direction of θ (latitude) and φ (longitude):
        [θ, φ]. Defaults to [0, 0].
    factr : float, optional
        The iteration stops when
        ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``,
        where ``eps`` is the machine precision, which is automatically
        generated by the code. Typical values for `factr` are: 1e12 for
        low accuracy; 1e7 for moderate accuracy; 10.0 for extremely
        high accuracy. See Notes for relationship to `ftol`, which is exposed
        (instead of `factr`) by the `scipy.optimize.minimize` interface to
        L-BFGS-B.
    gtol : float
        The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
        <= gtol`` where ``pg_i`` is the i-th component of the
        projected gradient.
    eps_angle : float
        Step size used for numerical approximation of the jacobian for the
        fitting of the angle of the DNA/bead construct.
    eps_rot : float
        Step size used for numerical approximation of the jacobian for the
        fitting of the rotation of the radius vector of the bead.

    Notes
    -----
    The option `ftol` is exposed via the `scipy.optimize.minimize` interface,
    but calling `scipy.optimize.fmin_l_bfgs_b` directly exposes `factr`. The
    relationship between the two is ``ftol = factr * numpy.finfo(float).eps``.
    I.e., `factr` multiplies the default machine floating-point precision to
    arrive at `ftol`
    """
    # Radius vector of the bead and radius
    if r0_sph is None:
        # [radius, theta, phi]
        r0_sph = np.array([0, 0, 0])
    r0_sph = normalize_phi(r0_sph)
    radius = r0_sph[0]
    angles_r0 = r0_sph[1:]

    if kappa is None:
        kappa = np.array([0, 0, 0])

    # Rotational stiffness in (N/rad) of r[1:] - r0[1:] for [theta (latitute),
    # phi (longitude)]
    if k_rot is None:
        k_rot = np.array([0, 0])

    # Length of the DNA construct
    l_dna = x_ss + x_ds

    # Initial distance of DNA attachment point on the glass surface to the bead
    # center, i.e. the length of the DNA/bead construct
    l_c = l_dna + radius
    # Apparent measured extension of dna in the experiment, if rotation is
    # neglected
    l_dna_app = l_c - radius

    # Distance of the DNA glass attachment point to the trap center
    l_A0 = np.linalg.norm(A0)

    # Check, if the smallest possible DNA/bead construct length is longer than
    # or equal to the distance of the attachment point to the trap center. If
    # so, the bead will equilibrate in the middle of the trap with zero force.
    if np.all(k_rot == 0):
        l_c_min = l_dna
    else:
        l_c_min = l_dna - radius
    if l_c_min >= l_A0:
        if return_plus:
            return (np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(2),
                    np.zeros(2))
        return 0, np.zeros(3), np.zeros(2), l_dna_app

    # Iteratively fit the direction of the DNA/bead construct and the
    # bead center -> DNA bead attachment point

    # index and maximum number of iterations
    i = 0
    i_max = 50

    # few initializations of needed variables
    angles_dna = None
    angles_r = None
    angles_c = None
    d_angles_c_dna = None
    d_angles_c_r_opp = np.zeros(2)
    # f_bead_mag = 0
    f_dna_total_mag = 0

    # terminating condition thresholds
    delta_l_c = 1e-9
    # delta_f_bead = 1e-12
    delta_f_dna_total = 1e-12
    threshold_delta_l_c = 0.001e-9  # m
    # threshold_delta_f_bead = 0.001e-12  # N
    threshold_delta_f_dna_total = 0.001e-12  # N

    # high precision switch for near zero force calculations
    proximity_threshold = 0.01e-9  # m
    precision = 1

    while (i < i_max
           and ((l_c >= l_A0 and delta_l_c > threshold_delta_l_c / precision)
                # or delta_l_c > threshold_delta_l_c / precision
                # or delta_f_bead > threshold_delta_f_bead / precision
                or delta_f_dna_total > threshold_delta_f_dna_total / precision
                )):
        i += 1
        if verbose:
            print('Round {}'.format(i))

        # Fit the angles of the DNA/bead construct
        if l_c >= l_A0:
            if verbose:
                print('DNA/bead construct extension too long, assume construct'
                      'pointing through the trap center')
            init_c = np.array(cart2sph(*-A0))[1:]
            angles_c = init_c
            success_c = True
        else:
            # If the DNA/bead construct length is close to the DNA glass
            # attachment point -> trap center distance, tiny changes of an
            # angle of the DNA/bead construct lead to huge changes in the
            # function `cost_angle_opp_force_bead_attachment()`.
            # Make sure to be able to still properly fit.
            if l_A0 - l_c < proximity_threshold and np.any(k_rot != 0):
                if verbose:
                    print('DNA/bead construct extension close to the trap'
                          'center -> high precision mode')
                precision = 100
                _eps_angle = high_precision_eps_angle
            else:
                precision = 1
                _eps_angle = eps_angle
            # fit angles of DNA/bead construct
            init_c, angles_c, success_c \
                = minimize_angle_cost(A0, l_c, radius, kappa, d_angles_c_r_opp,
                                      init_c=angles_c, factr=factr, gtol=gtol,
                                      eps=_eps_angle, verbose=deep_verbose,
                                      print_result=print_result)

        # calculate angles of dna based on the rotation of the DNA/bead
        # construct c
        if angles_dna is None:
            # First iteration, angles of dna point in the direction of the
            # DNA/bead construct
            angles_dna = angles_c
        else:
            # correct angles of dna by the change of angles of the DNA/bead
            # construct c
            angles_dna = angles_c + d_angles_c_dna
            angles_dna = normalize_phi(angles_dna)
        # calculate the true extension of the DNA
        dna = np.array(sph2cart(l_dna, *angles_dna))

        # Fit the angles of the DNA bead attachment vector r
        if np.all(k_rot == 0):
            if verbose:
                print('No bead rotation.')
            # set angles of r such a way, that r is pointing in the opposite
            # direction of the dna
            init_r = np.array(cart2sph(*-dna))[1:]
            angles_r = init_r
            success_r = True
            i_max = 1
        else:
            # fit angles of r
            init_r, angles_r, success_r \
                = minimize_rot_cost(angles_dna, f_dna, radius, angles_r0,
                                    k_rot, A0, init_r=angles_r, factr=factr,
                                    gtol=gtol, eps=eps_rot,
                                    verbose=deep_verbose,
                                    print_result=print_result)

        # calculate angles and length of DNA/bead construct
        # based on the rotated DNA bead attachment vector r
        r = np.array(sph2cart(radius, *angles_r))
        c = dna - r
        angles_c = np.array(cart2sph(*c)[1:])
        l_c, l_c_old = np.linalg.norm(c), l_c
        l_dna_app = l_c - radius
        # calculate the difference of the angles of the DNA/bead construct c
        # to the dna
        # and to the vector opposing the bead center/DNA attachment point
        # vector r for later recalculation after having changed the angles of c
        # after the next iteration
        d_angles_c_dna = angles_dna - angles_c
        d_angles_c_r_opp[0] = (math.pi - angles_r[0]) - angles_c[0]  # theta
        d_angles_c_r_opp[1] = (math.pi + angles_r[1]) - angles_c[1]  # phi
        # d_angles_c_r_opp = normalize_phi(d_angles_c_r_opp)

        # calculate optimized displacement, angles differences and
        # corresponding forces
        d_angles = angles_r0 - angles_r
        d, f_bead = cost_angle_opp_force_bead_attachment(angles_c, l_c, A0,
                                                         kappa, cost=False)
        # f_bead_mag, f_bead_old = np.linalg.norm(f_bead), f_bead_mag
        f_cnstr = np.array(sph2cart(-f_dna, *angles_dna))
        f_bead_rot, f_cnstr_rot \
            = cost_rot_force_bead_construct(angles_r, radius, angles_r0, k_rot,
                                            f_cnstr, cost=False)
        f_dna_total = - (f_bead + f_bead_rot)
        f_dna_total_mag, f_dna_total_old = \
            np.linalg.norm(f_dna_total), f_dna_total_mag

        # calculate terminating conditions
        delta_l_c = abs(l_c_old - l_c)
        # delta_f_bead = abs(f_bead_old - f_bead_mag)
        delta_f_dna_total = abs(f_dna_total_old - f_dna_total_mag)

        if verbose:
            print('dna: {:.3f} nm, r: {:.3f} nm, c: {:.3f} nm'.format(
                np.linalg.norm(dna)*1e9,
                np.linalg.norm(r)*1e9,
                np.linalg.norm(c)*1e9))
            # print('a_dna: {}°, a_r: {}°, a_c_rot: {}°, a_c: {}°, a_dna-r:'
            #       '{:.2f}°'.format(
            print('a_dna: {}°, a_r: {}°, a_c: {}°, d_a_c_r_opp: {}°, a_dna-r:'
                  '{:.2f}°'.format(
                          angles_dna*180/math.pi,
                          angles_r*180/math.pi,
                          angles_c*180/math.pi,
                          d_angles_c_r_opp*180/math.pi,
                          angle(dna, r)*180/math.pi))
            print('f_bead: {:.3f} pN, f_bead_rot: {:.3f} pN, f_cnstr_rot:'
                  '{:.3f} pN, f_dna_total: {:.3f} pN'.format(
                          np.linalg.norm(f_bead)*1e12,
                          np.linalg.norm(f_bead_rot)*1e12,
                          np.linalg.norm(f_cnstr_rot)*1e12,
                          np.linalg.norm(f_dna_total)*1e12))
            print()

    if verbose and angle(dna, r) < (math.pi / 2):
        print('DNA attachment point tangent angle: {:.2f} < 90°!'.format(
                angle(dna, r)*180/math.pi))
        print('-> DNA would wind around the bead.')

    # Check, if the DNA/bead construct is longer than the distance of the
    # attachment point to the trap center. If the DNA/bead construct is longer
    # than possible stretching with displacement ||d|| >= 0, the bead will
    # equilibrate in the middle of the trap with zero force.
    if l_c >= l_A0:
        if verbose:
            print('DNA/bead construct extension too long, return zero foce.')
        if return_plus:
            return (np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(2),
                    np.zeros(2))
        return 0, np.zeros(3), np.zeros(2), l_dna_app

    if return_plus:
        return f_dna_total, f_bead, d, angles_c, angles_r
    return f_dna_total_mag, d, d_angles, l_dna_app


def minimize_angle_cost(A0, l_c, radius, kappa, d_angles_c_r_opp=None,
                        init_c=None, copy_init=False, factr=1e5, gtol=1e-5,
                        eps=1e-8, verbose=False, print_result=False):
    ftol = factr * np.finfo(float).eps

    # Set offset_phi_c to the direction of attachment point -> trap center
    _, theta_c, offset_phi_c = cart2sph(*-A0)

    # Boundaries of theta and phi for the DNA attachment point -> bead center
    # vector.
    # The construct can point straight upwards to sidewards, where the bead
    # would touch the glass surface
    theta_c_min = 0
    sin_theta = min(1, radius / l_c)
    theta_c_max = math.pi / 2 - math.asin(sin_theta)
    # The construct can point towards the hemisphere of the trap center
    # (i.e. +/- 90°)
    phi_c_min = - math.pi / 2
    phi_c_max = math.pi / 2
    bounds = ((theta_c_min, theta_c_max), (phi_c_min, phi_c_max))

    if d_angles_c_r_opp is None:
        d_angles_c_r_opp = np.zeros(2)

    if init_c is None:
        # Find proper start values for theta_c and phi_c
        # Geometrically assume the DNA attachment point -> bead center vector
        # pointing towards the center of the trap.
        init_c = np.array([theta_c, 0])
    else:
        # correct phi of init_c by the offset of phi
        if copy_init:
            init_c = init_c.copy()
        init_c[1] -= offset_phi_c
        init_c = normalize_phi(init_c)

    if verbose:
        print('## ANGLE CONSTRUCT MINIMIZATION ##')
        print('bounds theta: {:.2f}° -> {:.2f}°, phi: {:.2f}° -> {:.2f}°'
              ''.format(theta_c_min*180/math.pi,
                        theta_c_max*180/math.pi,
                        normalize(phi_c_min + offset_phi_c)*180/math.pi,
                        normalize(phi_c_max + offset_phi_c)*180/math.pi))
        print('offset phi: {:.2f}°'.format(offset_phi_c*180/math.pi))

    # Iteratively change theta and phi of the DNA attachment point -> bead
    # center vector such a way that the angle of the attachment point -> bead
    # center vector c and the force vector f_bead are pointing in the exact
    # opposite direction (method 'L-BFGS-B').
    res = _minimize_lbfgsb(cost_angle_opp_force_bead_attachment,
                           x0=init_c,
                           bounds=bounds,
                           args=(l_c, A0, kappa, d_angles_c_r_opp,
                                 offset_phi_c, True, verbose),
                           ftol=ftol,
                           gtol=gtol,
                           eps=eps)
    angles_c = res.x
    success_c = res.success

    # correct for the offset phi
    angles_c[1] += offset_phi_c
    angles_c = normalize_phi(angles_c)
    init_c[1] += offset_phi_c

    if verbose:
        d, f_bead = cost_angle_opp_force_bead_attachment(angles_c, l_c, A0,
                                                         kappa, cost=False,
                                                         verbose=False)
        print('----------')
        print('## ANGLE CONSTRUCT RESULT ##')
        if print_result:
            print(res)
            print('----------')
        print('ANGLE:       θ (deg)   φ (deg)')
        print('f_bead:    {:8.3f}  {:8.3f}'.format(
                *np.array(cart2sph(*f_bead))[1:]*180/math.pi))
        print('construct: {:8.3f}  {:8.3f}'.format(*angles_c*180/math.pi))
        print('-> force: {:.3f} pN'.format(np.sqrt(np.sum(f_bead**2))*1e12))
        print()
    return init_c, angles_c, success_c


def cost_angle_opp_force_bead_attachment(angles_c, l_c, A0, kappa,
                                         d_angles_c_r_opp=None, offset_phi_c=0,
                                         cost=True, verbose=False):
    """
    l_c : float
          length of attachment point to bead center
    """
    if verbose:
        print('  # CALCULATE ANGLE CONSTRUCT COST ...')
        print('    theta_c: {:.6f}, delta_phi_c: {:.6f}'.format(
                *angles_c*180/math.pi))

    # 1. calculate attachment point -> bead center vector of the construct for
    # given theta and phi
    c = np.array(sph2cart(l_c, *angles_c, offset_phi=offset_phi_c))

    # 2. calculate position of the center of the bead (i.e. displacement
    # vector) for a given attachment point -> bead center vector c
    d = A0 + c

    # 3. calculate force vector of bead due to displacement
    f_bead = - d * kappa

    if cost:
        # 4. calculate the angle between f_bead and the vector opposing the r
        # vector, which is connected to the attachment point -> bead center
        # vector c (i.e. the vector opposing the force vector along the bead
        # center / DNA attachment point axis). If they are pointing in the same
        # direction, angle_opp is 0.
        c_r_opp = np.array(sph2cart(l_c, *(angles_c + d_angles_c_r_opp),
                                    offset_phi=offset_phi_c))
        angle_opp = angle(f_bead, c_r_opp)
        # print(angle_opp*180/math.pi)

    if verbose:
        print('    f_bead_theta_phi: {}°'.format(
                np.array(cart2sph(*f_bead))[1:]*180/math.pi))
        print('    c_theta_phi: {}°'.format(angles_c*180/math.pi))
        print('    c_r_opp_theta_phi: {}°'.format(
                (angles_c + d_angles_c_r_opp)*180/math.pi))
        print('    angle_opp: {:.3f}°'.format(angle_opp*180/math.pi))
    if cost:
        return angle_opp**2
    else:
        return d, f_bead


def minimize_rot_cost(angles_dna, f_dna, radius, angles_r0, k_rot, A0,
                      init_r=None, copy_init=False, factr=1e5, gtol=1e-5,
                      eps=1e-8, verbose=False, print_result=False):
    ftol = factr * np.finfo(float).eps

    # set offset_phi_r to the angle of phi_r0
    offset_phi_r = angles_r0[1]

    # correct dna vector by offset_phi_r
    angles_dna = angles_dna.copy()
    angles_dna[1] -= offset_phi_r
    angles_dna = normalize_phi(angles_dna)

    # Force vector of the construct, magnitude defined by f_dna
    f_cnstr = np.array(sph2cart(- f_dna, *angles_dna, offset_phi=offset_phi_r))

    # determine on which hemisphere of the bead relative to r0 the DNA is
    # attached to
    phi_A = math.atan2(A0[1], A0[0])
    delta_phi = normalize(phi_A - angles_r0[1])
    hemisphere = math.copysign(1, delta_phi)

    # correct r0 vector by offset_phi_r
    angles_r0 = angles_r0.copy()
    angles_r0[1] -= offset_phi_r
    angles_r0 = normalize_phi(angles_r0)

    # Set theta and phi of r assuming no rotational force
    # theta_dna can be between 0° and 90°, theta_r can be between 0° and 180°
    #   -> 180° - theta_c
    # phi_dna can be between -180° and 180°, phi_r can be between -180° and
    # 180°
    # r is pointing to the hemisphere on which the DNA is attached to the glass
    #   -> 180° - |phi_dna| * sign(hemisphere)
    theta_r = math.pi - angles_dna[0]
    phi_r = (math.pi - abs(angles_dna[1])) * hemisphere

    # Boundaries of theta and phi for the bead attachment point vector
    theta_r_min = min(theta_r, angles_r0[0])
    theta_r_max = max(theta_r, angles_r0[0])
    phi_r_min = min(phi_r, angles_r0[1])
    phi_r_max = max(phi_r, angles_r0[1])
    bounds = ((theta_r_min, theta_r_max), (phi_r_min, phi_r_max))

    if init_r is None:
        # Set the start values of theta and phi to the midpoints of r and r0
        init_theta_r = (theta_r + angles_r0[0]) / 2
        init_phi_r = (phi_r + angles_r0[1]) / 2
        init_r = np.array([init_theta_r, init_phi_r])
    else:
        # correct phi of init_r by the offset of phi
        if copy_init:
            init_r = init_r.copy()
        init_r[1] -= offset_phi_r
        init_r = normalize_phi(init_r)

    if verbose:
        print('## ROTATION BEAD MIMIMIZATION ##')
        print('bounds theta: {:.2f}° -> {:.2f}°, phi: {:.2f}° -> {:.2f}°'
              ''.format(theta_r_min*180/math.pi,
                        theta_r_max*180/math.pi,
                        normalize(phi_r_min + offset_phi_r)*180/math.pi,
                        normalize(phi_r_max + offset_phi_r)*180/math.pi))
        print('offset phi: {:.2f}°'.format(offset_phi_r*180/math.pi))

    res = _minimize_lbfgsb(cost_rot_force_bead_construct,
                           x0=init_r,
                           bounds=bounds,
                           args=(radius, angles_r0, k_rot, f_cnstr,
                                 offset_phi_r, True, verbose),
                           ftol=ftol,
                           gtol=gtol,
                           eps=eps)
    angles_r = res.x
    success_r = res.success

    # correct for the offset phi
    angles_r[1] += offset_phi_r
    angles_r = normalize_phi(angles_r)
    init_r[1] += offset_phi_r

    if verbose:
        f_bead_rot, f_cnstr_rot \
            = cost_rot_force_bead_construct(angles_r, radius, angles_r0, k_rot,
                                            f_cnstr, cost=False, verbose=False)
        print('----------')
        print('## ROTATION BEAD RESULT ##')
        if print_result:
            print(res)
            print('----------')
        print('f_bead_rot:  {} pN'.format(f_bead_rot*1e12))
        print('f_cnstr_rot: {} pN'.format(f_cnstr_rot*1e12))
        print('f_total_rot: {:.3f} pN'.format(
                                np.linalg.norm(f_bead_rot + f_cnstr_rot)*1e12))
        print('ANGLE:       θ (deg)   φ (deg)')
        print('angles_r:  {:8.3f}  {:8.3f}'.format(*angles_r*180/math.pi))
        print()
    return init_r, angles_r, success_r


def F_rot(d_angles, k_rot):
    # 3. calculate rotational force component magnitudes in the direction of
    # theta and phi (see Pedaci et.al., "Calibration of the optical torque
    # wrench", 2012, Optics Express)
    # f_bead_rot_mag = k_rot * angles_d
    # For phi the force is proportional to 1/cos(theta - 90°). It is lowest
    # where theta is 90° and increases towards the poles at theta equal to 0°
    # and 180°.
    # At the poles (0° and 180°) the force due to phi would be infinite, but
    # does not play any role and only the force along theta should be
    # considered.
    if len(d_angles) != 2:
        return k_rot * np.sin(2 * d_angles)
    cos_theta_shifted = abs(math.cos(d_angles[0] - math.pi/2))
    if cos_theta_shifted != 0:
        factor_phi = 1 / cos_theta_shifted
    else:
        factor_phi = 0
    return k_rot * np.sin(2 * d_angles) * np.array([1, factor_phi])


def cost_rot_force_bead_construct(angles_r, radius, angles_r0, k_rot, f_cnstr,
                                  offset_phi_r=0, cost=True, verbose=False):
    if verbose:
        print('  # CALCULATE ROTATION BEAD COST ...')
        print('    theta_r: {:07.6f}, delta_phi_r: {:07.6f}'
              ''.format(*angles_r*180/math.pi))
    # 1. rotate attachment point vector of the bead.
    r = np.array(sph2cart(radius, *angles_r, offset_phi=offset_phi_r))

    # 2. calculate difference of angles of r and r0
    d_angles = angles_r0 - angles_r

    # 3. calculate rotational force component magnitudes in the direction of
    # theta and phi
    # f_bead_rot_mag = k_rot * angles_d
    f_bead_rot_mag = F_rot(d_angles, k_rot)

    # 4. calculate rotational force components in the direction of theta and
    # phi
    f_bead_rot_cmp = np.r_[0, f_bead_rot_mag]

    # 5. calculate rotational force vector of bead acting on attachment point,
    # i.e. rotate/transform coordinates of f_bead_rot_cmp into coordinates with
    # the tangent of the attachment point of the DNA on the bead as origin.
    f_bead_rot = coord_sph2cart(angles_r[0], angles_r[1], f_bead_rot_cmp,
                                offset_phi=offset_phi_r)

    # 6. Calculate rotational force of construct f_cnstr_rot acting on
    # attachment point, i.e. the component of f_cnstr antiparallel to
    # f_bead_rot.
    # f_bead_rot != - f_cnstr_rot
    # f_bead_rot = - parallel_cmp(f_cnstr, f_bead_rot)
    # f_bead_rot = (- (f_cnstr.dot(f_bead_rot))
    #               / np.linalg.norm(f_bead_rot)**2
    #               * f_bead_rot)
    # 0 = (||f_cnstr|| / ||f_bead_rot||
    #      * (sin(phi_r) * sin(phi_c)
    #      * cos(theta_c - theta_r)
    #      + cos(phi_r) * cos(phi_c))
    #      + 1)
    # calculate force vector of the construct for given theta and phi
    f_cnstr_rot = orthogonal_cmp(f_cnstr, r)

    # 6. Calculate the total rotational force of bead and rotational force of
    # construct.
    f_total_rot = np.linalg.norm(f_bead_rot + f_cnstr_rot)

    if verbose:
        print('    f_bead_rot: {} pN, {:.3f} pN'
              ''.format(f_bead_rot*1e12, np.linalg.norm(f_bead_rot)*1e12))
        print('    f_cnstr_rot: {} pN, {:.3f} pN'
              ''.format(f_cnstr_rot*1e12, np.linalg.norm(f_cnstr_rot)*1e12))
        print('    f_total_rot: {:.3f} pN'.format(f_total_rot*1e12))
    if cost:
        return (f_total_rot*1e12)**2
    else:
        return f_bead_rot, f_cnstr_rot


def attachment_point(x0, y0=0.0, h0=0.0, radius=0.0):
    """
    x0 : float
        Position of the stage x (m) relative to the trap center.
    y0 : float
        Position of the stage y (m) relative to the trap center.
    h0 : float
        Distance (m) of the bead surface to the glass surface, if
        the bead is in its resting position, i.e. no force in the
        vertical direction is applied.
    radius : float
        Radius of the bead (m).
    """
    return np.array([x0, y0, - (h0 + radius)])


def normalize_phi(angles):
    angles[1] = normalize(angles[1])
    return angles


def normalize(phi):
    if phi > math.pi:
        phi -= 2*math.pi
    if phi <= -math.pi:
        phi += 2*math.pi
    return phi


def cart2sph(x, y, z, offset_phi=0, positive_phi=False):
    """
    offset_phi : float
        angle in Euclidian plane that should point in the direction of positive
        x
    """
    # cart2sph -- Transform Cartesian to spherical coordinates
    # Spherical coordinates (r, θ, φ) as commonly used in physics (ISO
    # convention): radial distance r, inclination θ (theta), and azimuth φ
    # (phi).
    hxy = math.hypot(x, y)
    r = math.hypot(hxy, z)
    theta = math.atan2(hxy, z)
    phi = math.atan2(y, x) - offset_phi
    if positive_phi and phi < 0:
        phi += 2 * math.pi
    return r, theta, phi


def sph2cart(r, theta, phi, offset_phi=0):
    """
    offset_phi : float
        angle in Euclidian plane that points in the directon of positive x
    """
    # sph2cart -- Transform spherical to Cartesian coordinates
    # Spherical coordinates (r, θ, φ) as commonly used in physics (ISO
    # convention): radial distance r, inclination θ (theta), and azimuth φ
    # (phi).
    phi += offset_phi
    rsin_theta = r * math.sin(theta)
    x = rsin_theta * math.cos(phi)
    y = rsin_theta * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z


def coord_sph2cart(theta, phi, v, offset_phi=0):
    # v is vector with components pointing in the direction of the
    # v[0] radius vector
    # v[1] circle formed by changing theta (inclination)
    # v[2] circle formed by changin phi (azimut)
    # returns a vector rotated according to the local orthogonal unit vectors
    # of the spherical coordinate system
    phi += offset_phi
    sint = math.sin(theta)
    cost = math.cos(theta)
    sinp = math.sin(phi)
    cosp = math.cos(phi)
    return np.array([
        [sint*cosp, cost*cosp, -sinp],
        [sint*sinp, cost*sinp, cosp],
        [cost,     -sint,      0]
    ]).dot(v)


def angle(v1, v2):
    # angle between two vectors
    # return math.atan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))
    # does not work as well for small angles, but is faster:
    cos_theta = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = max(-1, cos_theta)
    cos_theta = min(1, cos_theta)
    return math.acos(cos_theta)


def parallel_cmp(v1, v2):
    # project v1 onto v2
    # component of v1 parallel to v2
    amp_v2 = np.linalg.norm(v2)
    if amp_v2 == 0:
        return(v2)
    return (v1.dot(v2) / amp_v2**2) * v2


def orthogonal_cmp(v1, v2):
    # component of v1 orthogonal to v2
    return v1 - parallel_cmp(v1, v2)


def F_0(x0, nbs=0, S=None, L_p_ssDNA=None, z=None, T=298.2, nbp=0, pitch=None,
        L_p_dsDNA=None, f_min=0e-12, f_max=200e-12, xtol=1e-18, kappa=0.0):
    """
    Find most probable force for a given sample displacement `x0`, a number of
    separated basepairs `j`, and number of basepairs `nbp` of dsDNA spacers.

    See equation (10) in Bockelmann 1998

    Parameters
    ----------
    x0 : float
        Displacement of the sample (surface) relative to the
        lever/handle (attachment of DNA), i.e. the end to end distance
        of ssDNA termini which are moved apart plus dsDNA spacer length
        plus displacement of lever in m.
    nbs : int
        Number of bases of ssDNA strand
    nbp : int
        Number of basepairs of dsDNA spacers
    kappa : float
        Stiffness of lever (handle) attached to DNA in N/m
    """
    # Find the equilibrium force F0 at given j and x0
    def f_lev_cost(f):
        x_ss = ext_ssDNA(f, nbs=nbs, S=S, L_p=L_p_ssDNA, z=z, T=T)
        x_ds = ext_dsDNA_wlc(f, nbp=nbp, pitch=pitch, L_p=L_p_dsDNA, T=T)
        return (f - F_lev(x0, x_ss=x_ss, x_ds=x_ds, kappa=kappa))**2

    # Find the force, which will result in input extension
    f0 = fminbound(f_lev_cost,
                   f_min,
                   f_max,
                   xtol=xtol)
    return f0


def F_0_2D(x0, nbs=0, S=None, L_p_ssDNA=None, z=None, T=298.2,
           nbp=0, pitch=None, L_p_dsDNA=None,
           f_min=0e-12, f_max=200e-12, xtol=1e-18,
           r=0.0, z0=0.0, kappa=0.0):
    """
    Find most probable force for a given sample displacement `x0`, a number of
    separated basepairs `j`, and number of basepairs `nbp` of dsDNA spacers.

    See equation (10) in Bockelmann 1998

    Parameters
    ----------
    x0 : float
        Displacement of the sample (surface) relative to the
        lever/handle (attachment of DNA), i.e. the end to end distance
        of ssDNA termini which are moved apart plus dsDNA spacer length
        plus displacement of lever in m.
    nbs : int
        Number of bases of ssDNA strand
    nbp : int
        Number of basepairs of dsDNA spacers
    kappa : float
        Stiffness of lever (handle) attached to DNA in N/m
    """
    # Find the equilibrium force F0 at given j and x0
    def f_lev_cost(f, d2D=False):
        x_ss = ext_ssDNA(f, nbs=nbs, S=S, L_p=L_p_ssDNA, z=z, T=T)
        x_ds = ext_dsDNA_wlc(f, nbp=nbp, pitch=pitch, L_p=L_p_dsDNA, T=T)
        f_construct, dx, dz = F_construct_2D(x0, x_ss=x_ss, x_ds=x_ds,
                                             r=r, z0=z0, kappa=kappa)
        if d2D:
            return dx, dz
        return (f - f_construct)**2

    # Find the force, which will result in input extension
    f0 = fminbound(f_lev_cost,
                   f_min,
                   f_max,
                   xtol=xtol)

    # calculate bead displacements in the trap for found force
    d2D = f_lev_cost(f0, d2D=True)
    return f0, d2D


def F_0_3D(A0, nbs=0, S=None, L_p_ssDNA=None, z=None, T=298.2,
           nbp=0, pitch=None, L_p_dsDNA=None,
           f_min=0e-12, f_max=200e-12, xtol=1e-18,
           r0_sph=None, kappa=None, k_rot=None,
           verbose=False):
    """
    Find most probable force for a given sample displacement `x0`, a number of
    separated basepairs `j`, and number of basepairs `nbp` of dsDNA spacers.

    Extended 3D version of the equation (10) in Bockelmann 1998, including
    rotation of the bead

    Parameters
    ----------
    A0 : np.ndarray of type float
        Position (m) of the DNA attachment point on the glass surface relative
        to the trap center: [x, y, z].
    nbs : int
        Number of bases of ssDNA strand
    nbp : int
        Number of basepairs of dsDNA spacers
    r0_sph : np.ndarray of type floatt
        Radiust
        Radiung_aasds
        Radius vector of the bead [radius, theta, phi] ([m, rad, rad]).
    kappa : np.ndarray of type float
        Stiffness for [x, y, z] of lever (handle) attached to DNA in N/m.
    k_rot : np.ndarray of type float, optional
        Rotational stiffness (N/rad) of the bead relative to `r0_sph`, with two
        components pointing in the direction of θ (latitude) and φ (longitude):
        [θ, φ]. Defaults to [0, 0].
    """
    # Find the equilibrium force F0 at given j and x0
    def f_lev_cost(f, return_d=False):
        x_ss = ext_ssDNA(f, nbs=nbs, S=S, L_p=L_p_ssDNA, z=z, T=T)
        x_ds = ext_dsDNA_wlc(f, nbp=nbp, pitch=pitch, L_p=L_p_dsDNA, T=T)
        f_construct, d, d_angles, ext_app = \
            F_construct_3D(A0, x_ss=x_ss, x_ds=x_ds, f_dna=f,
                           r0_sph=r0_sph, kappa=kappa, k_rot=k_rot,
                           verbose=verbose)
        if return_d:
            return d, d_angles, ext_app
        return (f - f_construct)**2

    # Find the force, which will result in input extension
    f0 = fminbound(f_lev_cost,
                   f_min,
                   f_max,
                   xtol=xtol)

    # calculate bead displacements in the trap for found force
    d, d_angles, ext_app = f_lev_cost(f0, return_d=True)
    return f0, d, d_angles, ext_app


_E_pair = {
    # Huguet et. al. 2010, table 1, 1M NaCl
    # Energies from Huguet 2010 are given for 298 K

    # one Purine and one Pyrimidine or
    # two succesive Purines/Pyrimidines with same bases
    'AA': 1.23*kcal/Na,
    'TT': 1.23*kcal/Na,  #
    'AC': 1.49*kcal/Na,
    'TG': 1.49*kcal/Na,  #
    'AG': 1.36*kcal/Na,
    'TC': 1.36*kcal/Na,  #
    'CA': 1.66*kcal/Na,
    'GT': 1.66*kcal/Na,  #
    'CC': 1.93*kcal/Na,
    'GG': 1.93*kcal/Na,  #
    'GA': 1.47*kcal/Na,
    'CT': 1.47*kcal/Na,  #

    # two succesive Purines/Pyrimidines with different bases
    'AT': 1.17*kcal/Na,
    'CG': 2.37*kcal/Na,
    'GC': 2.36*kcal/Na,
    'TA': 0.84*kcal/Na,

    # TODO: include proper energy term for the first and last bp

    # Ander PhD thesis 2011
    # kB*T = 4.1 pN*nm -> T ~ 298 K
    'A': 1.2*kB*298,
    'T': 1.2*kB*298,
    'G': 3.4*kB*298,
    'C': 3.4*kB*298

    # energies Bockelmann et. al. 1997
    # for S=800pN, L_p_ssDNA=0.75nm, z=0.61nm/bp
    #   'AT': 1.3*kB*298
    #   'GC': 2.9*kB*298
    # for S=800pN, L_p_ssDNA=1.35nm, z=0.56nm/bp
    #   'AT': 1.6*kB*298
    #   'GC': 3.2*kB*298
}

_M_pair = {
    # Huguet et. al. 2010, table 1, NaCl concentration correction factor
    # Energies from Huguet et. al. 2010 are given for 298 K
    'AA': 0.145*kcal/Na,
    'TT': 0.145*kcal/Na,  #
    'AC': 0.099*kcal/Na,
    'TG': 0.099*kcal/Na,  #
    'AG': 0.070*kcal/Na,
    'TC': 0.070*kcal/Na,  #
    'CA': 0.091*kcal/Na,
    'GT': 0.091*kcal/Na,  #
    'CC': 0.063*kcal/Na,
    'GG': 0.063*kcal/Na,  #
    'GA': 0.155*kcal/Na,
    'CT': 0.155*kcal/Na,  #

    'AT': 0.117*kcal/Na,
    'CG': 0.132*kcal/Na,
    'GC': 0.079*kcal/Na,
    'TA': 0.091*kcal/Na,
}

_DH_pair = {
    # Huguet et. al. 2010, table 2, enthalpy (kcal/mol)
    'AA': 7.28*kcal/Na,
    'TT': 7.28*kcal/Na,  #
    'AC': 5.80*kcal/Na,
    'TG': 5.80*kcal/Na,  #
    'AG': 5.21*kcal/Na,
    'TC': 5.21*kcal/Na,  #
    'CA': 8.96*kcal/Na,
    'GT': 8.96*kcal/Na,  #
    'CC': 8.57*kcal/Na,
    'GG': 8.57*kcal/Na,  #
    'GA': 8.16*kcal/Na,
    'CT': 8.16*kcal/Na,  #

    'AT': 4.63*kcal/Na,
    'CG': 9.66*kcal/Na,
    'GC': 10.10*kcal/Na,
    'TA': 8.31*kcal/Na
}

_DS_pair = {
    # Huguet et. al. 2010, table 2, entropy (cal/mol)
    'AA': 20.28*cal/Na,
    'TT': 20.28*cal/Na,  #
    'AC': 14.46*cal/Na,
    'TG': 14.46*cal/Na,  #
    'AG': 12.89*cal/Na,
    'TC': 12.89*cal/Na,  #
    'CA': 24.48*cal/Na,
    'GT': 24.48*cal/Na,  #
    'CC': 22.30*cal/Na,
    'GG': 22.30*cal/Na,  #
    'GA': 22.46*cal/Na,
    'CT': 22.46*cal/Na,  #

    'AT': 11.62*cal/Na,
    'CG': 24.43*cal/Na,
    'GC': 25.96*cal/Na,
    'TA': 25.06*cal/Na
}


def E_pair(bases, NNBP=False, c=None, T=None):
    """
    Work necessary to separate base pairs A-T and G-C of a DNA double helix.

    Includes the contributions of unpairing, unstacking, and rearrangement of
    bases.

    Parmeters
    ---------
    bases : str
        Sequence of bases 'A', 'T', 'C', and 'G'.
    NNBP : bool
        Nearest-neighbour base-pair determination of the base-pair energies
    c : float
        Concentration of monovalent cations in mol, defaults to 1 M.
    T : float
        T is not considered
    """
    c = 1 if c is None else c
    bases = bases.upper()
    if NNBP:
        # TODO: include proper energy term for the first and last bp
        e_pair = [_E_pair[''.join((a, b))]
                  for a, b
                  in zip(bases[:-1], bases[1:])]
        m_pair = [_M_pair[''.join((a, b))]
                  for a, b
                  in zip(bases[:-1], bases[1:])]
        e_pair = np.array(e_pair)
        m_pair = np.array(m_pair)
        e = e_pair + m_pair * np.log(c)
    else:
        e = np.array([_E_pair[base] for base in bases])
    return e


def E_pair_T(bases, NNBP=False, c=None, T=298.2):
    """
    Work necessary to separate base pairs A-T and G-C of a DNA double helix.

    Includes the contributions of unpairing, unstacking, and rearrangement of
    bases.

    Parmeters
    ---------
    bases : str
        Sequence of bases 'A', 'T', 'C', and 'G'.
    NNBP : bool
        Nearest-neighbour base-pair determination of the base-pair energies
    c : float
        Concentration of monovalent cations in mol
    T : float
        Temperature in K
    """
    c = 1 if c is None else c
    bases = bases.upper()
    if NNBP:
        dh_pair = [_DH_pair[''.join((a, b))]
                   for a, b
                   in zip(bases[:-1], bases[1:])]
        ds_pair = [_DS_pair[''.join((a, b))]
                   for a, b
                   in zip(bases[:-1], bases[1:])]
        m_pair = [_M_pair[''.join((a, b))]
                  for a, b
                  in zip(bases[:-1], bases[1:])]
        dh_pair = np.array(dh_pair)
        ds_pair = np.array(ds_pair)
        m_pair = np.array(m_pair)

        # salt dependent entropy
        # only entropy depends on salt concentration
        ds_pair_salt = ds_pair - m_pair/298 * np.log(c)

        # temperature dependent energy
        e_pair = dh_pair - T*ds_pair_salt

        e = e_pair  # + m_pair * np.log(c)
    else:
        e = np.array([_E_pair[base] for base in bases])
    return e


def E_unzip_DNA(bases, nuz=0, NNBP=False, c=None, T=298.2):
    """
    Work necessary to separate two single strands of DNA double helix of `nuz`
    base pairs.

    Includes the contributions of unpairing, unstacking, and rearrangement of
    bases.

    Parameters
    ----------
    bases : str
        Sequence of bases 'A', 'T', 'C', and 'G'.
    nuz : int
        Number of base(pair)s up to where the unpairing energy should be
        calculated ([1,`nuz`]). If `nuz` is 1, calculate energy for first
        basepair.
    T : float
        Temperature in K
    """
    if nuz <= 0:
        return 0

    # if NNBP:
        # TODO: include proper energy term for the first and last bp

    return np.sum(E_pair(bases[:nuz], NNBP=NNBP, c=c, T=T))


def _E_ext_ssDNA(x, nbs=0, S=None, L_p=None, z=None, T=298.2):
    """
    Elastic energy stored in a single strand of j bases
    extended by force F to length x.

    Parameters
    ----------
    nbs : intf
        Number of bases of ssDNA
    x : float
        Extension of ssDNA in m
    z : float
        Length of a single base in m
    """
    if nbs <= 0:
        return 0

    if x < 0:
        x = -x

    # Slow variant of numerical integration
    # E _fjc = quad(F_ssDNA, 0, x, (j, S, L_p, z, T))[0]

    f = F_ssDNA(x, nbs=nbs, S=S, L_p=L_p, z=z, T=T)
    # integral_ext_dF = ext_ssDNA_int(f, j, S=S, L_p=L_p, z=z, T=T)
    # The ext_ssDNA_int seems to be not correct -> numerical integration
    integral_ext_dF = quad(ext_ssDNA, 0, f, (nbs, S, L_p, z, T))[0]

    E_fjc = f * x - integral_ext_dF

    # There is no negative energy.
    E_fjc = max(E_fjc, 0)

    return E_fjc


def E_ext_dsDNA_wlc(x, nbp=0, pitch=None, L_p=None, T=298.2):
    """
    Elastic energy stored in a double strand of nbp basepairs
    extended by force F to length x.

    Integral of the worm-like chain model [1].

    [1] Marko, J.F.; Eric D. Siggia. "Stretching DNA". Macromolecules. 1995.
    28: 8759–8770. doi:10.1021/ma00130a008

    Parameters
    ----------
    x : float
        Extension (m)
    L_0 : float
        Contour length (m)
    L_p : float
        Persistence length (m)
    T : float
        Temperature (K)
    """
    pitch = pitch or 0.338e-9
    L_p = L_p or 50e-9

    if nbp <= 0:
        return 0

    if x < 0:
        x = -x

    L_0 = nbp*pitch

    # WLC only valid in the interval x = (-L_0, L_0)
    # Higher x would lead to wrongly calculated energies.
    # if x > L_0, even negative energies are possible, which
    # would lead to exceptionally high valies in the partition
    # function.
    if x >= L_0:
        return float('inf')

    def integral(x):
        # from wolfram alpha
        # return (kB * T * (L_0**2 / (L_0 - x) + (2 * x**2) / L_0 - x)) / (4 * L_p)
        # (k T (L^2/(L - x) + (2 x^2)/L - x))/(4 P)

        # Petrosyan, R. "Improved approximations for some polymer extension
        # models". Rehol Acta. 2016. doi:10.1007/s00397-016-0977-9
        return (kB * T * (L_0**2 / (L_0 - x) + (2 * x**2) / L_0 - 1.01587 * x *
                          (x/L_0)**2.15 - x)) / (4 * L_p)
        # (k T (L^2/(L - x) + (2 x^2)/L - 1.01587 x^1 (x/L)^2.15 - x))/(4 P)
    return integral(x) - integral(0)


def E_lev(d, kappa):
    """
    The elastic energy of the lever/handle.

    Parameters
    ----------
    kappa : float
        Stiffness of lever in N/m
    d : float
        Displacement of lever in m
    """
    return 1/2 * kappa * d*d


def E_rot(d_angles, k_rot, radius, shifted=True):
    """
    The rotational energy of the lever/handle.

    Parameters
    ----------
    d_angles : np.ndarray of type float
        Angles of rotation of the bead in rad [theta, phi].
    k_rot : np.ndarray of type float
        Rotational stiffness (N/rad) of the bead relative to `r0_sph`, with two
        components pointing in the direction of θ (latitude) and φ (longitude):
        [θ, φ]. Defaults to [0, 0].
    radius : float
        Radius of the rotated bead in m.
    """
    # return 1/2 * k_rot * radius * d_angles**2
    # (see Pedaci et.al., "Calibration of the optical torque wrench", 2012,
    # Optics Express)
    if shifted:
        return (1 - np.cos(2 * d_angles)) * (1/2 * k_rot * radius)
    return - 1/2 * k_rot * radius * np.cos(2 * d_angles)


def E_tot(bases='', nuz=0, nbs=0, x_ss=0.0, nbp=0, x_ds=0.0,
          d=0.0, kappa=0.0, d_angles=0.0, k_rot=0.0, radius=0.0,
          S=None, L_p_ssDNA=None, z=None,
          pitch=None, L_p_dsDNA=None,
          NNBP=False, c=None, e_loop=0.0, T=298.2, verbose=False):
    """
    Parameters
    ----------
    bases : str
        Sequence of sense strand of dsDNA which is (will be) unzipped
    nuz : int
        Number of unzipped basepairs to calculate the unzip energy.
    nbs : int
        Number of ssDNA bases
    x_ss : float
        Extension of an ssDNA strand
    nbp : int
        Number of basepairs of the spacer dsDNA
    x_ds : float
        Extension of the spacer dsDNA
    kappa : float
        Stiffness of lever (handle) attached to DNA in N/m
    e_loop : float
        Free energy for opening the last bp and terminal hairpin (kcal/mol).
    """

    e_ext_ssDNA = E_ext_ssDNA(x_ss, nbs=nbs, z=z, L_p=L_p_ssDNA, S=S, T=T)
    e_ext_dsDNA = E_ext_dsDNA_wlc(x_ds, nbp=nbp, pitch=pitch, L_p=L_p_dsDNA,
                                  T=T)
    e_unzip_DNA = E_unzip_DNA(bases, nuz=nuz, NNBP=NNBP, c=c, T=T)
    e_lev = np.sum(E_lev(d, kappa))
    e_rot = np.sum(E_rot(d_angles, k_rot, radius))

    # Include proper energy term for opening the terminal hairpin, only if all
    # bps are already unzipped and hairpin is to be opened
    if nuz >= len(bases) + 1:
        e_loop = e_loop*kcal/Na
    else:
        e_loop = 0.0

    e_total = (
        e_ext_ssDNA
        + e_ext_dsDNA
        + e_unzip_DNA
        + e_lev
        + e_rot
        + e_loop
    )

    if verbose:
        print('E_ext_ssDNA: ' + str(e_ext_ssDNA/(kB*T)))
        print('E_ext_dsDNA: ' + str(e_ext_dsDNA/(kB*T)))
        print('E_unzip_DNA: ' + str(e_unzip_DNA/(kB*T)))
        print('E_lev: ' + str(e_lev/(kB*T)))
        print('E_rot: ' + str(e_rot/(kB*T)))

    return e_total


def equilibrium_xfe0(x0, bases='', nuz=0, nbs=0, nbp=0, nbs_loop=0,
                     r=0.0, z0=0.0, kappa=0.0,
                     S=None, L_p_ssDNA=None, z=None,
                     pitch=None, L_p_dsDNA=None,
                     NNBP=False, c=None, e_loop=0.0, T=298.2, verbose=False):
    """
    Calculate the equilibrium extension, force, and energy for a given stage
    displacement `x0` and a fixed set of the following parameters.

    Parameters
    ----------
    x0 : float
        Total displacement (m)
    nuz : int
        Number of unzipped basepairs
    nbs : int
        Number of extra ssDNA bases in the construct
    nbp : int
        Number of basepairs of dsDNA spacer
    nbs_loop : int
        Number of extra ssDNA bases in the hairpin
    kappa : float
        Stiffness of lever (handle) attached to DNA in N/m
    """
    # One unzipped basepair leads to 2 free ssDNA bases
    nbs = 2*nuz + nbs

    # If unzipping fork has reached the last basepair and end loop of unzipping
    # construct should be unzipped, elongates the ssDNA by nbs_loop bases
    if nbs_loop > 0 and nuz >= len(bases) + 1:
        nbs += nbs_loop

    # Calculate most probable force for
    #   number of unzipped bases nbs and
    #   number of basepairs nbp and
    #   stage displacement x0
    f0, d2D = F_0_2D(x0, nbs=nbs, S=S, L_p_ssDNA=L_p_ssDNA, z=z, T=T,
                     nbp=nbp, pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                     r=r, z0=z0, kappa=kappa)

    # Calculate most probable extension for most probable force for
    #   both of the two ssDNA strands and
    #   one dsDNA strand for
    #   number of unzipped base pairs j
    x0_ss = ext_ssDNA(f0, nbs=nbs, S=S, L_p=L_p_ssDNA, z=z, T=T)
    x0_ds = ext_dsDNA_wlc(f0, nbp=nbp, pitch=pitch, L_p=L_p_dsDNA, T=T)
    e0 = E_tot(bases=bases, nuz=nuz, nbs=nbs, x_ss=x0_ss, nbp=nbp, x_ds=x0_ds,
               d=d2D, kappa=kappa,
               S=S, L_p_ssDNA=L_p_ssDNA, z=z,
               pitch=pitch, L_p_dsDNA=L_p_dsDNA,
               NNBP=NNBP, c=c, e_loop=e_loop, T=T, verbose=verbose)

    if verbose:
        template = "nuz: {:03d}, f0: {:.3e}, e0: {:.3e}"
        print(template.format(nuz, f0, e0))

    return x0_ss, x0_ds, d2D, f0, e0


def xfe0_all_nuz(x0, h0=0.0, bases='', nbs=0, nbp=0, nbs_loop=0,
                 radius=0.0, kappa=0.0,
                 S=None, L_p_ssDNA=None, z=None,
                 pitch=None, L_p_dsDNA=None,
                 NNBP=False, c=0, e_loop=0.0, T=298.2,
                 boltzmann_factor=1e-9, verbose=False):
    """
    Calculate the equilibrium extensions, forces and energies for a given stage
    displacement `x0` for all possible numbers of unzipped basepairs and find
    the number of unzipped bases, at which the unzipping fork will most likely
    fluctuate.

    Parameters
    ----------
    bases : str
        Sequence of sense strand of dsDNA which is (will be) unzipped
    nbs : int
        Number of extra ssDNA bases in the construct
    nbp : int
        Number of basepairs of dsDNA spacer
    nbs_loop : int
        Number of extra ssDNA bases in the hairpin
    kappa : float
        Stiffness of lever (handle) attached to DNA in N/m
    """
    # Maximum number of unzippabed bps
    nuz_max = len(bases)
    
    # If hairpin exists, add one possible unzipping event representative for
    # opening the hairpin
    if nbs_loop > 0:
        nuz_max += 1

    # Create a list of all possible numbers of unzipped basepairs
    NUZ0 = np.arange(0, nuz_max+1)

    # Go through all possible numbers of unzipped basepairs
    # and calculate the equilibrium forces and energies, while
    # ignoring the fluctuations of the extensions of the DNA
    # and the bead in the trap
    X0_ss = []
    X0_ds = []
    D0 = []
    F0 = []
    E0 = []
    W0 = []

    for nuz in NUZ0:
        if nuz % 25 == 0:
            verbose = verbose
        else:
            verbose = False
        x0_ss, x0_ds, d2D, f0, e0 = \
            equilibrium_xfe0(x0, bases=bases, nuz=nuz, nbs=nbs, nbp=nbp,
                             nbs_loop=nbs_loop,
                             r=radius, z0=h0, kappa=kappa,
                             S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                             pitch=pitch, L_p_dsDNA=L_p_dsDNA, NNBP=NNBP,
                             c=c, e_loop=e_loop, T=T, verbose=verbose)
        X0_ss.append(x0_ss)
        X0_ds.append(x0_ds)
        D0.append(d2D)
        F0.append(f0)
        E0.append(e0)
        W0.append(mpmath.exp(- e0 / (kB*T)))

    X0_ss = np.array(X0_ss)
    X0_ds = np.array(X0_ds)
    D0 = np.array(D0)
    F0 = np.array(F0)
    E0 = np.array(E0)
    W0 = np.array(W0)

    # Calculate weighted averages of:
    W0_sum = W0.sum()
    P0 = W0 / W0_sum
    #   unzipped basepairs
    NUZ0_avg = (NUZ0 * W0).sum() / W0_sum
    #   bead displacements
    D0_avg = (D0 * W0[np.newaxis].T).sum(axis=0) / W0_sum
    #   force
    F0_avg = (F0 * W0).sum() / W0_sum
    #   extension of the construct
    x = ((X0_ss + X0_ds) * W0).sum() / W0_sum

    # Automatically detect the number of unzipped bases, at which
    # the unzipping fork will fluctuate, i.e. select the
    # number of basepairs which have significant weights (weights
    # with a minimum probability compared to the largest weight).
    # Calculate later used variables.
    idx_max = W0.argmax()
    NUZ0_max_W0 = NUZ0[idx_max]
    F0_max_W0 = F0[idx_max]
    W0_max = W0[idx_max]

    # boltzmann_factor = 1e-9
    # mpmath.exp(-20) > 1e-9 -> corresponds to more than 20 kT difference
    NUZ0_min = NUZ0[W0 / W0_max >= boltzmann_factor].min()
    NUZ0_max = NUZ0[W0 / W0_max >= boltzmann_factor].max()

    r = {
        'x': x,
        'x0': x0,
        'NUZ0': NUZ0,
        'D0': D0,
        'F0': F0,
        'E0': E0,
        'W0': W0,
        'P0': P0,
        'NUZ0_avg': NUZ0_avg,
        'D0_avg': D0_avg,
        'F0_avg': F0_avg,
        'W0_max': W0_max,
        'NUZ0_max_W0': NUZ0_max_W0,
        'F0_max_W0': F0_max_W0,
        'NUZ0_min': NUZ0_min,
        'NUZ0_max': NUZ0_max,
        'settings': {
            'h0': h0,
            'bases': bases,
            'nbs': nbs,
            'nbp': nbp,
            'nbs_loop': nbs_loop,
            'radius': radius,
            'kappa': kappa,
            'S': S,
            'L_p_ssDNA': L_p_ssDNA,
            'z': z,
            'pitch': pitch,
            'L_p_dsDNA': L_p_dsDNA,
            'NNBP': NNBP,
            'c': c,
            'e_loop': e_loop,
            'T': T,
            'boltzmann_factor': boltzmann_factor
        }
    }
    return r


def approx_eq_nuz(x0, bases='', nbs=0, nbp=0,
                  r=0.0, z0=0.0, kappa=0.0,
                  S=None, L_p_ssDNA=None, z=None,
                  pitch=None, L_p_dsDNA=None,
                  NNBP=False, c=None, T=298.2,
                  spacing=5, min_stepsize=10, verbose=False):
    """
    Find the approximate number of unzipped basepairs the unzipping construct
    automatically adjust itself when in equilibrium.
    The search is performed in a binary mode, i.e. the number of calculations
    to find the number of unzipped basepairs is of class O(log(n)), where n is
    the number of basepairs in the unzipping seagment.
    """
    # maximal number of unzipped basepairs
    nuz_max = len(bases)

    # verify sppacing and set limits for nuz
    spacing = min(spacing, nuz_max)
    minimum = 0
    maximum = nuz_max - spacing

    # initialize step size and starting nuz
    step = int(round((maximum - minimum) / 2))
    nuz = int(round((maximum - minimum) / 2))

    def unzip_for_eq(nuz=0):
        """
        Calculate the gradient of the energy.
        Return True, if unzipping construct has to be further unzipped, to
        reach equilibrium. Return False, if unziping construct has to be
        further annealed, to reach equilibrium. Ignore the opening of the
        endloop (nbs_loop=0, e_loop=0.0) for finding the minimum of the total
        energy, to avoid falsly high numbers of unzipped basepairs, due to
        energy jump upon opening of the end loop.
        """
        nuzl = nuz
        nuzr = nuz + spacing

        _, _, _, f0l, e0l = \
            equilibrium_xfe0(x0, bases=bases, nuz=nuzl, nbs=nbs, nbp=nbp,
                             nbs_loop=0, r=r, z0=z0, kappa=kappa,
                             S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                             pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                             NNBP=NNBP, c=c, e_loop=0.0, T=T, verbose=verbose)
        _, _, _, f0r, e0r = \
            equilibrium_xfe0(x0, bases=bases, nuz=nuzr, nbs=nbs, nbp=nbp,
                             nbs_loop=0, r=r, z0=z0, kappa=kappa,
                             S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                             pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                             NNBP=NNBP, c=c, e_loop=0.0, T=T, verbose=verbose)
        return e0l > e0r

    # Search for the approximate number of unzipped basepairs, to be in
    # equilibrium
    i = 0
    while step > min_stepsize:
        i += 1
        if unzip_for_eq(nuz=nuz):
            if verbose:
                print('nuz + step -> new: {} + {}'.format(nuz, step),
                      end=' -> ')
            nuz += step
        else:
            if verbose:
                print('nuz - step -> new: {} - {}'.format(nuz, step),
                      end=' -> ')
            nuz -= step
        if verbose:
            print(nuz)
        if nuz < minimum or nuz > maximum:
            # unzipping construct has to be either fully closed or fully opened
            # to be in equilibrium -> stop the loop and return either 0 or
            # nuz_max
            step = 0
            nuz = max(0, nuz)
            nuz = min(nuz, nuz_max)
        # half the stepsize
        step = int(round(step / 2))
    if verbose:
        print('Number of iterations to find approximation of eq nuz: {}'
              ''.format(i))
    return nuz


def xfe0_fast_nuz(x0, h0=0.0, bases='', nuz_est=-1, nbs=0, nbp=0, nbs_loop=0,
                  radius=0.0, kappa=0.0,
                  S=None, L_p_ssDNA=None, z=None,
                  pitch=None, L_p_dsDNA=None,
                  NNBP=False, c=0, e_loop=0.0, T=298.2,
                  spacing=5, min_stepsize=10,
                  boltzmann_factor=1e-9, verbose=False):
    """
    Calculate the equilibrium extensions, forces and energies for a given stage
    displacement `x0` for most probable numbers of unzipped basepairs and find
    the number of unzipped bases, at which the unzipping fork will most likely
    fluctuate.

    Parameters
    ----------
    bases : str
        Sequence of sense strand of dsDNA which is (will be) unzipped
    nuz_est : int
        Estimate number of unzipped basepairs. 0 <= `nuz_est` <= `nuz_max`.
        If `nuz_est` < 0, the number is approximated automatically with a
        binary search using the function `approx_eq_nuz`.
    nbs : int
        Number of extra ssDNA bases in the construct
    nbp : int
        Number of basepairs of dsDNA spacer
    nbs_loop : int
        Number of extra ssDNA bases in the hairpin
    kappa : float
        Stiffness of lever (handle) attached to DNA in N/m
    boltzmann_factor : float
        The minimum probability each number of unzipped basepairs (nuz) state
        has to have relative to the most probable one to be considered in the
        calculation. The smaller the boltzmann_factor, the more exact the
        result is. The larger the boltzmann factor is, the faster the
        calculation.
    """
    # Maximum number of unzippabed bps
    nuz_max = len(bases)
    
    # If hairpin exists, add one possible unzipping event representative for
    # opening the hairpin
    if nbs_loop > 0:
        nuz_max += 1

    if boltzmann_factor <= 0:
        # All nuz will be calculated, start in the middle
        nuz_est = int(round(nuz_max / 2))
    elif nuz_est < 0:
        # Autodetermine the approximate nuz which will be in equilibrium
        nuz_est = approx_eq_nuz(x0, bases=bases, nbs=nbs, nbp=nbp,
                                r=radius, z0=h0, kappa=kappa,
                                S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                                pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                                NNBP=NNBP, c=c, T=T,
                                spacing=spacing, min_stepsize=min_stepsize,
                                verbose=verbose)
    else:
        # Set nuz_est to valid value
        nuz_est = max(0, nuz_est)
        nuz_est = min(nuz_est, nuz_max)

    # Go through all possible numbers of unzipped basepairs
    # and calculate the equilibrium forces and energies, while
    # ignoring the fluctuations of the extensions of the DNA
    # and the bead in the trap
    #
    # Speed up calculation, i.e. calculate only around the most likely nuz:
    # 1. If no nuz_est given, perform binary search to find most likely nuz
    NUZ0 = []
    X0_ss = []
    X0_ds = []
    D0 = []
    F0 = []
    E0 = []
    W0 = []

    # Calculate force, extension, and weight for given number of unzipped
    # basepairs
    def eq_few0(nuz, w0_likely):
        x0_ss, x0_ds, d2D, f0, e0 = \
            equilibrium_xfe0(x0, bases=bases, nuz=nuz, nbs=nbs, nbp=nbp,
                             nbs_loop=nbs_loop,
                             r=radius, z0=h0, kappa=kappa,
                             S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                             pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                             NNBP=NNBP, c=c, e_loop=e_loop, T=T,
                             verbose=verbose)
        NUZ0.append(nuz)
        X0_ss.append(x0_ss)
        X0_ds.append(x0_ds)
        D0.append(d2D)
        F0.append(f0)
        E0.append(e0)
        w0 = mpmath.exp(- e0 / (kB*T))
        W0.append(w0)

        # 3a. Set new minimum energy, if energy is smaller than all previous
        # calculated energies, and if energy is not calculated from max nuz
        # opened, which could result in a huge drop of the energy, due to the
        # loop opening and therefore in too many falsly neglected calculated
        # energies, due to boltzmann_factor selection
        if w0 > w0_likely and nuz < nuz_max:
            w0_likely = w0
        stop = False
        # 3b. if energy difference > 20 kT stop calculation
        if w0_likely != 0 and w0 / w0_likely < boltzmann_factor:
            stop = True
        return w0_likely, stop

    # 2. Calculate energy for most likely nuz
    w0_likely, _ = eq_few0(nuz_est, 0)

    # 3. Calculate neighbouring nuzes (nuz_left / nuz_right)
    nuz_left = nuz_est - 1
    nuz_right = nuz_est + 1
    stop_left = nuz_left < 0
    stop_right = nuz_right > nuz_max
    while not (stop_left and stop_right):
        if not stop_left:
            w0_likely, stop_left = eq_few0(nuz_left, w0_likely)
            nuz_left -= 1
            # stop, if nuz_left is negative
            stop_left = stop_left or nuz_left < 0
        if not stop_right:
            w0_likely, stop_right = eq_few0(nuz_right, w0_likely)
            nuz_right += 1
            # stop, if nuz_right is larger than number of unzippable basepairs
            stop_right = stop_right or nuz_right > nuz_max

    # Select nuz datapoints that are at least equally likely as
    # `boltzmann_factor`
    # boltzmann_factor = 1e-9
    # corresponds to - log(1e-9) ~= 20.03 kT difference
    W0 = np.array(W0)
    idx_vld = W0 / w0_likely >= boltzmann_factor

    # Sort nuz in ascending order
    NUZ0 = np.array(NUZ0)[idx_vld]
    idx_srt = np.argsort(NUZ0)
    NUZ0 = NUZ0[idx_srt]

    X0_ss = np.array(X0_ss)[idx_vld][idx_srt]
    X0_ds = np.array(X0_ds)[idx_vld][idx_srt]
    D0 = np.array(D0)[idx_vld][idx_srt]
    F0 = np.array(F0)[idx_vld][idx_srt]
    E0 = np.array(E0)[idx_vld][idx_srt]
    W0 = W0[idx_vld][idx_srt]

    # Calculate weighted averages of:
    W0_sum = W0.sum()
    P0 = W0 / W0_sum
    #   unzipped basepairs
    NUZ0_avg = (NUZ0 * W0).sum() / W0_sum
    #   bead displacements
    D0_avg = (D0 * W0[np.newaxis].T).sum(axis=0) / W0_sum
    #   force
    F0_avg = (F0 * W0).sum() / W0_sum
    #   extension of the construct
    x = ((X0_ss + X0_ds) * W0).sum() / W0_sum

    # Automatically detect the number of unzipped bases, at which
    # the unzipping fork will fluctuate, i.e. select the
    # number of basepairs which have significant weights (weights
    # with a minimum probability compared to the largest weight).
    # Calculate later used variables.
    idx_max = W0.argmax()
    NUZ0_max_W0 = NUZ0[idx_max]
    F0_max_W0 = F0[idx_max]
    W0_max = W0[idx_max]

    # boltzmann_factor = 1e-9
    # mpmath.exp(-20) > 1e-9 -> corresponds to more than 20 kT difference
    NUZ0_min = NUZ0.min()
    NUZ0_max = NUZ0.max()

    r = {
        'x': x,
        'x0': x0,
        'NUZ0': NUZ0,
        'D0': D0,
        'F0': F0,
        'E0': E0,
        'W0': W0,
        'P0': P0,
        'NUZ0_avg': NUZ0_avg,
        'D0_avg': D0_avg,
        'F0_avg': F0_avg,
        'W0_max': W0_max,
        'NUZ0_max_W0': NUZ0_max_W0,
        'F0_max_W0': F0_max_W0,
        'NUZ0_min': NUZ0_min,
        'NUZ0_max': NUZ0_max,
        'settings': {
            'h0': h0,
            'bases': bases,
            'nbs': nbs,
            'nbp': nbp,
            'nbs_loop': nbs_loop,
            'radius': radius,
            'kappa': kappa,
            'S': S,
            'L_p_ssDNA': L_p_ssDNA,
            'z': z,
            'pitch': pitch,
            'L_p_dsDNA': L_p_dsDNA,
            'NNBP': NNBP,
            'c': c,
            'e_loop': e_loop,
            'T': T,
            'spacing': spacing,
            'min_stepsize': min_stepsize,
            'boltzmann_factor': boltzmann_factor
        }
    }
    return r


class _xfe0_fast_nuz_chained(object):
    """Speed up calculation of xfe0_fast_nuz by taking the nuz_est from
    previous calculation for next calculation

    The object of this class is a drop in replacement for the original
    `xfe0_fast_nuz` function, if using the the multiprocessing package.

    Each process gets its own copy of the a _xfe0_fast_nuz_chained object,
    which is initialized with nuz_est = -1. Upon each call nuz_est is set to
    the previous outcome of the calculated NUZ0_avg.
    """
    def __init__(self):
        self.nuz_est = -1

    def __call__(self, x0, h0=0.0, bases='', nuz_est=-1, nbs=0, nbp=0, nbs_loop=0,
                 radius=0.0, kappa=0.0,
                 S=None, L_p_ssDNA=None, z=None,
                 pitch=None, L_p_dsDNA=None,
                 NNBP=False, c=0, e_loop=0.0, T=298.2,
                 spacing=5, min_stepsize=10,
                 boltzmann_factor=1e-9, verbose=False):
        if nuz_est == -1:
            nuz_est = self.nuz_est
        # print('x0 {}, nuz_est {}'.format(x0, nuz_est))
        r = xfe0_fast_nuz(x0, h0=h0, bases=bases, nuz_est=nuz_est, nbs=nbs, nbp=nbp,
                          nbs_loop=nbs_loop,
                          radius=radius, kappa=kappa,
                          S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                          pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                          NNBP=NNBP, c=c, e_loop=e_loop, T=T,
                          boltzmann_factor=boltzmann_factor, verbose=verbose)
        self.nuz_est = int(round(r['NUZ0_avg']))
        return r


def unzipping_force_energy(x0_min, x0_max, h0=0.0, resolution=1e-9, processes=8,
                           bases='', nbs=0, nbp=0, nbs_loop=0,
                           radius=0.0, kappa=0.0,
                           S=None, L_p_ssDNA=None, z=None,
                           pitch=None, L_p_dsDNA=None,
                           NNBP=False, c=0, e_loop=0.0, T=298.2,
                           spacing=5, min_stepsize=10,
                           boltzmann_factor=1e-9,
                           individual_points=False, verbose=False,
                           F_ssDNA_mod=None, E_ext_ssDNA_mod=None,
                           ext_dsDNA_wlc_mod=None):
    # Assign DNA model functions to the variables of the global (module) scope,
    # such that `multiprocessing.Pool` will see these variables.
    global F_ssDNA
    global E_ext_ssDNA
    global ext_dsDNA_wlc

    # Set DNA model functions to the unbuffered default functions
    F_ssDNA = F_ssDNA_mod or _F_ssDNA
    E_ext_ssDNA = E_ext_ssDNA_mod or _E_ext_ssDNA
    # Initialize the approximations of the ssDNA/dsDNA model functions with
    # fixed model function parameters and substitute the original DNA model
    # functions
    # F_ssDNA is implicitly buffered with `ext_dsDNA_wlc`.
    # Buffered `E_ext_ssDNA` does not speed up calculation.
    # E_ext_ssDNA = \
    #     init_buf_E_ext_ssDNA(read=False, write=False, filename='E_ext_ssDNA',
    #                          processes=processes,
    #                          bases=bases, nbs=nbs, nbs_loop=nbs_loop,
    #                          S=S, L_p=L_p_ssDNA, z=z, T=T)
    ext_dsDNA_wlc = ext_dsDNA_wlc_mod or \
        init_buf_ext_dsDNA_wlc(nbp=nbp, pitch=pitch, L_p=L_p_dsDNA, T=T)

    resolution = int(np.round((x0_max - x0_min) / resolution + 1))
    X0 = np.linspace(x0_min, x0_max, resolution)

    # Speed up calculation with the multiprocessing package,
    # by taking the nuz_est from previous calculation for
    # each subsequent calculation
    xfe0_fast_nuz = _xfe0_fast_nuz_chained()

    # Define a closure to be executed by the pool
    def f(x0):
        print('\rCalculating equilibrium for stage displacement x0 = {:.3e}'
              '...'.format(x0), end='', flush=True)
        return xfe0_fast_nuz(x0=x0, h0=h0, bases=bases, nbs=nbs, nbp=nbp,
                             nbs_loop=nbs_loop,
                             radius=radius, kappa=kappa,
                             S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                             pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                             NNBP=NNBP, c=c, e_loop=e_loop, T=T,
                             spacing=spacing, min_stepsize=min_stepsize,
                             boltzmann_factor=boltzmann_factor)
        # nuz_est = int(round(r['NUZ0_avg']))
    f = unboundfunction(f)

    # Process function in pool with 8 parallelly executed processes
    with Pool(processes=processes) as pool:
        start = time.time()
        XFE0 = pool.map(f, X0)
        stop = time.time()
        print('\nDone, elapsed time: {:.1f} s'.format(stop - start))

    # combine all individually simulated points into one array
    XFE = {
        'X': np.array([xfe0['x'] for xfe0 in XFE0]),
        'X0': np.array([xfe0['x0'] for xfe0 in XFE0]),
        'NUZ0_avg': np.array([xfe0['NUZ0_avg'] for xfe0 in XFE0]),
        'NUZ0_max_W0': np.array([xfe0['NUZ0_max_W0'] for xfe0 in XFE0]),
        'F0_avg': np.array([xfe0['F0_avg'] for xfe0 in XFE0]),
        'F0_max_W0': np.array([xfe0['F0_max_W0'] for xfe0 in XFE0]),
        'settings': {
            'x0_min': x0_min,
            'x0_max': x0_max,
            'h0': h0,
            'resolution': resolution,
            'bases': bases,
            'nbs': nbs,
            'nbp': nbp,
            'nbs_loop': nbs_loop,
            'radius': radius,
            'kappa': kappa,
            'S': S,
            'L_p_ssDNA': L_p_ssDNA,
            'z': z,
            'pitch': pitch,
            'L_p_dsDNA': L_p_dsDNA,
            'NNBP': NNBP,
            'c': c,
            'e_loop': e_loop,
            'T': T,
            'spacing': spacing,
            'min_stepsize': min_stepsize,
            'boltzmann_factor': boltzmann_factor
        }
    }

    if individual_points:
        return XFE, XFE0
    return XFE


def plot_unzip_energy(x0, h0=0.0, bases='', nuz_est=-1, nbs=0, nbp=0, nbs_loop=0,
                      radius=0.0, kappa=0.0,
                      S=None, L_p_ssDNA=None, z=None,
                      pitch=None, L_p_dsDNA=None,
                      NNBP=False, c=0, e_loop=0.0, T=298.2,
                      spacing=5, min_stepsize=10,
                      boltzmann_factor=1e-9,
                      verbose=False, compatibility=False,
                      axes=None):

    if compatibility:
        xfe0 = xfe0_all_nuz(x0, h0=h0, bases=bases, nbs=nbs, nbp=nbp,
                            nbs_loop=nbs_loop,
                            radius=radius, kappa=kappa,
                            S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                            pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                            NNBP=NNBP, c=c, e_loop=e_loop, T=T,
                            boltzmann_factor=boltzmann_factor,
                            verbose=verbose)
    else:
        xfe0 = xfe0_fast_nuz(x0, h0=h0, bases=bases, nuz_est=nuz_est, nbs=nbs,
                             nbp=nbp, nbs_loop=nbs_loop,
                             radius=radius, kappa=kappa,
                             S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                             pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                             NNBP=NNBP, c=c, e_loop=e_loop, T=T,
                             spacing=spacing, min_stepsize=min_stepsize,
                             boltzmann_factor=boltzmann_factor,
                             verbose=verbose)

    # with cnps.cn_plot('notebook') as cnp:
    if axes is None:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
        # ax2 = cnps.second_ax(link_ax=ax)
        # ax2.xaxis.set_visible(False)
    else:
        ax, ax2 = axes
        fig = ax.get_figure()
    nuz = xfe0['NUZ0']
    energy = xfe0['E0']
    min_e = energy.min()
    # normalize energy relative to min_e in unzits of kT
    energy -= min_e
    energy /= kB*T

    # displacement = xfe0['D0']
    boltzmann_factor = xfe0['W0'] / np.sum(xfe0['W0'])
    cumsum = np.cumsum(xfe0['W0']) / np.sum(xfe0['W0'])
    # if cnp is not None:
    #     ax.plot(nuz, energy, c=cnp.color)
    #     # ax.axhline(xfe0['D0_avg'], c=cnp.color)
    #     ax2.plot(nuz, boltzmann_factor, c=cnp.color)
    #     ax2.plot(nuz, cumsum, c=cnp.color)
    # else:
    ax.plot(nuz, energy)
    # ax.axhline(xfe0['D0_avg'])
    ax2.plot(nuz, boltzmann_factor)
    ax2.plot(nuz, cumsum)
    # ax.axvline(xfe0['NUZ0_min'], c='cyan')
    ax.axvline(xfe0['NUZ0_avg'], c='magenta')
    # ax.axvline(xfe0['NUZ0_max'], c='cyan')

    ax.set_xlabel('Number of unzipped basepairs')
    ax.set_ylabel('Energy difference ($k_{B}*T$)')
    ax2.set_ylabel('Boltzmann factor')

    return fig, ax, ax2


def equilibrium_xfe0_rot(A0, bases='', nuz=0, nbs=0, nbp=0, nbs_loop=0,
                         r0_sph=None, kappa=None, k_rot=None,
                         S=None, L_p_ssDNA=None, z=None,
                         pitch=None, L_p_dsDNA=None,
                         NNBP=False, c=None, e_loop=0.0, T=298.2, verbose=False):
    """
    Calculate the equilibrium extension, force, and energy for a given stage
    displacement `x0` and a fixed set of the following parameters.

    Parameters
    ----------
    A0 : np.ndarray of type float
        Position (m) of the DNA attachment point on the glass surface relative
        to the trap center: [x, y, z].
    nuz : int
        Number of unzipped basepairs
    nbs : int
        Number of extra ssDNA bases in the construct
    nbp : int
        Number of basepairs of dsDNA spacer
    nbs_loop : int
        Number of extra ssDNA bases in the hairpin
    r0_sph : np.ndarray of type float
        Radius vector of the bead [radius, theta, phi] ([m, rad, rad]).
    kappa : np.ndarray of type float
        Stiffness for [x, y, z] of lever (handle) attached to DNA in N/m.
    k_rot : np.ndarray of type float, optional
        Rotational stiffness (N/rad) of the bead relative to `r0_sph`, with two
        components pointing in the direction of θ (latitude) and φ (longitude):
        [θ, φ]. Defaults to [0, 0].
    """
    # One unzipped basepair leads to 2 free ssDNA bases
    nbs = 2*nuz + nbs

    # If unzipping fork has reached the last basepair and end loop of unzipping
    # construct should be unzipped, elongates the ssDNA by nbs_loop bases
    if nbs_loop > 0 and nuz >= len(bases) + 1:
        nbs += nbs_loop

    # Calculate most probable force for
    #   number of unzipped bases nbs and
    #   number of basepairs nbp and
    #   stage displacement x0
    f0, d, d_angles, ext_app = \
        F_0_3D(A0, nbs=nbs, S=S, L_p_ssDNA=L_p_ssDNA, z=z, T=T,
               nbp=nbp, pitch=pitch, L_p_dsDNA=L_p_dsDNA,
               r0_sph=r0_sph, kappa=kappa, k_rot=k_rot,
               verbose=verbose)

    # Calculate most probable extension for most probable force for
    #   both of the two ssDNA strands and
    #   one dsDNA strand for
    #   number of unzipped base pairs j
    if r0_sph is None:
        radius = 0.0
    else:
        radius = r0_sph[0]
    if k_rot is None:
        k_rot = np.array([0, 0])

    x0_ss = ext_ssDNA(f0, nbs=nbs, S=S, L_p=L_p_ssDNA, z=z, T=T)
    x0_ds = ext_dsDNA_wlc(f0, nbp=nbp, pitch=pitch, L_p=L_p_dsDNA, T=T)
    e0 = E_tot(bases=bases, nuz=nuz, nbs=nbs, x_ss=x0_ss, nbp=nbp, x_ds=x0_ds,
               d=d, kappa=kappa, d_angles=d_angles, k_rot=k_rot, radius=radius,
               S=S, L_p_ssDNA=L_p_ssDNA, z=z,
               pitch=pitch, L_p_dsDNA=L_p_dsDNA,
               NNBP=NNBP, c=c, e_loop=e_loop, T=T, verbose=verbose)

    if verbose:
        template = "nuz: {:03d}, f0: {:.3e}, e0: {:.3e}"
        print(template.format(nuz, f0, e0))

    return x0_ss, x0_ds, d, d_angles, ext_app, f0, e0


def approx_eq_nuz_rot(A0, bases='', nbs=0, nbp=0,
                      r0_sph=None, kappa=None, k_rot=None,
                      S=None, L_p_ssDNA=None, z=None,
                      pitch=None, L_p_dsDNA=None,
                      NNBP=False, c=None, T=298.2,
                      spacing=5, min_stepsize=10, verbose=False):
    """
    Find the approximate number of unzipped basepairs the unzipping construct
    automatically adjust itself when in equilibrium.
    The search is performed in a binary mode, i.e. the number of calculations
    to find the number of unzipped basepairs is of class O(log(n)), where n is
    the number of basepairs in the unzipping seagment.
    """
    # maximal number of unzipped basepairs
    nuz_max = len(bases)

    # verify sppacing and set limits for nuz
    spacing = min(spacing, nuz_max)
    minimum = 0
    maximum = nuz_max - spacing

    # initialize step size and starting nuz
    step = int(round((maximum - minimum) / 2))
    nuz = int(round((maximum - minimum) / 2))

    def unzip_for_eq(nuz=0):
        """
        Calculate the gradient of the energy.
        Return True, if unzipping construct has to be further unzipped, to
        reach equilibrium. Return False, if unziping construct has to be
        further annealed, to reach equilibrium. Ignore the opening of the
        endloop (nbs_loop=0, e_loop=0.0) for finding the minimum of the total
        energy, to avoid falsly high numbers of unzipped basepairs, due to
        energy jump upon opening of the end loop.
        """
        nuzl = nuz
        nuzr = nuz + spacing

        _, _, _, _, _, f0l, e0l = \
            equilibrium_xfe0_rot(A0, bases=bases, nuz=nuzl, nbs=nbs, nbp=nbp,
                                 nbs_loop=0,
                                 r0_sph=r0_sph, kappa=kappa, k_rot=k_rot,
                                 S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                                 pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                                 NNBP=NNBP, c=c, e_loop=0.0, T=T, verbose=verbose)
        _, _, _, _, _, f0r, e0r = \
            equilibrium_xfe0_rot(A0, bases=bases, nuz=nuzr, nbs=nbs, nbp=nbp,
                                 nbs_loop=0,
                                 r0_sph=r0_sph, kappa=kappa, k_rot=k_rot,
                                 S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                                 pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                                 NNBP=NNBP, c=c, e_loop=0.0, T=T, verbose=verbose)
        return e0l > e0r

    # Search for the approximate number of unzipped basepairs, to be in
    # equilibrium
    i = 0
    while step > min_stepsize:
        i += 1
        if unzip_for_eq(nuz=nuz):
            if verbose:
                print('nuz + step -> new: {} + {}'.format(nuz, step),
                      end=' -> ')
            nuz += step
        else:
            if verbose:
                print('nuz - step -> new: {} - {}'.format(nuz, step),
                      end=' -> ')
            nuz -= step
        if verbose:
            print(nuz)
        if nuz < minimum or nuz > maximum:
            # unzipping construct has to be either fully closed or fully opened
            # to be in equilibrium -> stop the loop and return either 0 or
            # nuz_max
            step = 0
            nuz = max(0, nuz)
            nuz = min(nuz, nuz_max)
        # half the stepsize
        step = int(round(step / 2))
    if verbose:
        print('Number of iterations to find approximation of eq nuz: {}'
              ''.format(i))
    return nuz


def xfe0_fast_nuz_rot(A0, bases='', nuz_est=-1, nbs=0, nbp=0, nbs_loop=0,
                      r0_sph=None, kappa=None, k_rot=None,
                      S=None, L_p_ssDNA=None, z=None,
                      pitch=None, L_p_dsDNA=None,
                      NNBP=False, c=0, e_loop=0.0, T=298.2,
                      spacing=5, min_stepsize=10,
                      boltzmann_factor=1e-9, verbose=False):
    """
    Calculate the equilibrium extensions, forces and energies for a given stage
    displacement `x0` for most probable numbers of unzipped basepairs and find
    the number of unzipped bases, at which the unzipping fork will most likely
    fluctuate.

    Parameters
    ----------
    bases : str
        Sequence of sense strand of dsDNA which is (will be) unzipped
    nuz_est : int
        Estimate number of unzipped basepairs. 0 <= `nuz_est` <= `nuz_max`.
        If `nuz_est` < 0, the number is approximated automatically with a
        binary search using the function `approx_eq_nuz`.
    nbs : int
        Number of extra ssDNA bases in the construct
    nbp : int
        Number of basepairs of dsDNA spacer
    nbs_loop : int
        Number of extra ssDNA bases in the hairpin
    kappa : float
        Stiffness of lever (handle) attached to DNA in N/m
    boltzmann_factor : float
        The minimum probability each number of unzipped basepairs (nuz) state
        has to have relative to the most probable one to be considered in the
        calculation. The smaller the boltzmann_factor, the more exact the
        result is. The larger the boltzmann factor is, the faster the
        calculation.
    """
    nuz_max = len(bases)

    if boltzmann_factor <= 0:
        # All nuz will be calculated, start in the middle
        nuz_est = int(round(nuz_max / 2))
    elif nuz_est < 0:
        # Autodetermine the approximate nuz which will be in equilibrium
        nuz_est = approx_eq_nuz_rot(A0, bases=bases, nbs=nbs, nbp=nbp,
                                    r0_sph=r0_sph, kappa=kappa, k_rot=k_rot,
                                    S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                                    pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                                    NNBP=NNBP, c=c, e_loop=e_loop, T=T,
                                    spacing=spacing, min_stepsize=min_stepsize,
                                    verbose=verbose)
    else:
        # Set nuz_est to valid value
        nuz_est = max(0, nuz_est)
        nuz_est = min(nuz_est, nuz_max)

    # Go through all possible numbers of unzipped basepairs
    # and calculate the equilibrium forces and energies, while
    # ignoring the fluctuations of the extensions of the DNA
    # and the bead in the trap
    #
    # Speed up calculation, i.e. calculate only around the most likely nuz:
    # 1. If no nuz_est given, perform binary search to find most likely nuz
    NUZ0 = []
    X0_ss = []
    X0_ds = []
    EXT_APP = []
    D0 = []
    DA0 = []
    F0 = []
    E0 = []
    W0 = []

    # Calculate force, extension, and weight for given number of unzipped
    # basepairs
    def eq_few0(nuz, w0_likely):
        x0_ss, x0_ds, d, d_angles, ext_app, f0, e0 = \
            equilibrium_xfe0_rot(A0, bases=bases, nuz=nuz, nbs=nbs, nbp=nbp,
                                 nbs_loop=nbs_loop,
                                 r0_sph=r0_sph, kappa=kappa, k_rot=k_rot,
                                 S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                                 pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                                 NNBP=NNBP, c=c, e_loop=e_loop, T=T, verbose=verbose)
        NUZ0.append(nuz)
        X0_ss.append(x0_ss)
        X0_ds.append(x0_ds)
        EXT_APP.append(ext_app)
        D0.append(d)
        DA0.append(d_angles)
        F0.append(f0)
        E0.append(e0)
        w0 = mpmath.exp(- e0 / (kB*T))
        W0.append(w0)

        # 3a. Set new minimum energy, if energy is smaller than all previous
        # calculated energies, and if energy is not calculated from max nuz
        # opened, which could result in a huge drop of the energy, due to the
        # loop opening and therefore in too many falsly neglected calculated
        # energies, due to boltzmann_factor selection
        if w0 > w0_likely and nuz < nuz_max:
            w0_likely = w0
        stop = False
        # 3b. if energy difference > 20 kT stop calculation
        if w0_likely != 0 and w0 / w0_likely < boltzmann_factor:
            stop = True
        return w0_likely, stop

    # 2. Calculate energy for most likely nuz
    w0_likely, _ = eq_few0(nuz_est, 0)

    # 3. Calculate neighbouring nuzes (nuz_left / nuz_right)
    nuz_left = nuz_est - 1
    nuz_right = nuz_est + 1
    stop_left = nuz_left < 0
    stop_right = nuz_right > nuz_max
    while not (stop_left and stop_right):
        if not stop_left:
            w0_likely, stop_left = eq_few0(nuz_left, w0_likely)
            nuz_left -= 1
            # stop, if nuz_left is negative
            stop_left = stop_left or nuz_left < 0
        if not stop_right:
            w0_likely, stop_right = eq_few0(nuz_right, w0_likely)
            nuz_right += 1
            # stop, if nuz_right is larger than number of unzippable basepairs
            stop_right = stop_right or nuz_right > nuz_max

    # Select nuz datapoints that are at least equally likely as
    # `boltzmann_factor`
    # boltzmann_factor = 1e-9
    # mpmath.exp(-20) > 1e-9 -> corresponds to more than 20 kT difference
    W0 = np.array(W0)
    idx_vld = W0 / w0_likely >= boltzmann_factor

    # Sort nuz in ascending order
    NUZ0 = np.array(NUZ0)[idx_vld]
    idx_srt = np.argsort(NUZ0)
    NUZ0 = NUZ0[idx_srt]

    X0_ss = np.array(X0_ss)[idx_vld][idx_srt]
    X0_ds = np.array(X0_ds)[idx_vld][idx_srt]
    EXT_APP = np.array(EXT_APP)[idx_vld][idx_srt]
    D0 = np.array(D0)[idx_vld][idx_srt]
    DA0 = np.array(DA0)[idx_vld][idx_srt]
    F0 = np.array(F0)[idx_vld][idx_srt]
    E0 = np.array(E0)[idx_vld][idx_srt]
    W0 = W0[idx_vld][idx_srt]

    # Calculate weighted averages of:
    W0_sum = W0.sum()
    P0 = W0 / W0_sum
    #   unzipped basepairs
    NUZ0_avg = (NUZ0 * W0).sum() / W0_sum
    #   bead displacements
    D0_avg = (D0 * W0[np.newaxis].T).sum(axis=0) / W0_sum
    #   bead rotations
    DA0_avg = (DA0 * W0[np.newaxis].T).sum(axis=0) / W0_sum
    # DNA extension
    EXT_APP_avg = (EXT_APP * W0).sum() / W0_sum
    #   force
    F0_avg = (F0 * W0).sum() / W0_sum
    #   extension of the construct
    ext_avg = ((X0_ss + X0_ds) * W0).sum() / W0_sum

    # Automatically detect the number of unzipped bases, at which
    # the unzipping fork will fluctuate, i.e. select the
    # number of basepairs which have significant weights (weights
    # with a minimum probability compared to the largest weight).
    # Calculate later used variables.
    idx_max = W0.argmax()
    NUZ0_max_W0 = NUZ0[idx_max]
    F0_max_W0 = F0[idx_max]
    W0_max = W0[idx_max]

    # boltzmann_factor = 1e-9
    # mpmath.exp(-20) > 1e-9 -> corresponds to more than 20 kT difference
    NUZ0_min = NUZ0.min()
    NUZ0_max = NUZ0.max()

    r = {
        'x': ext_avg,
        'A0': A0,
        'NUZ0': NUZ0,
        'D0': D0,
        'DA0': DA0,
        'EXT_APP': EXT_APP,
        'F0': F0,
        'E0': E0,
        'W0': W0,
        'P0': P0,
        'NUZ0_avg': NUZ0_avg,
        'D0_avg': D0_avg,
        'DA0_avg': DA0_avg,
        'EXT_APP_avg': EXT_APP_avg,
        'F0_avg': F0_avg,
        'W0_max': W0_max,
        'NUZ0_max_W0': NUZ0_max_W0,
        'F0_max_W0': F0_max_W0,
        'NUZ0_min': NUZ0_min,
        'NUZ0_max': NUZ0_max,
        'settings': {
            'bases': bases,
            'nbs': nbs,
            'nbp': nbp,
            'nbs_loop': nbs_loop,
            'r0_sph': r0_sph,
            'kappa': kappa,
            'k_rot': k_rot,
            'S': S,
            'L_p_ssDNA': L_p_ssDNA,
            'z': z,
            'pitch': pitch,
            'L_p_dsDNA': L_p_dsDNA,
            'NNBP': NNBP,
            'c': c,
            'e_loop': e_loop,
            'T': T,
            'spacing': spacing,
            'min_stepsize': min_stepsize,
            'boltzmann_factor': boltzmann_factor
        }
    }
    return r


class _xfe0_fast_nuz_rot_chained(object):
    """Speed up calculation of xfe0_fast_nuz_rot by taking the nuz_est from
    previous calculation for next calculation

    The object of this class is a drop in replacement for the original
    `xfe0_fast_nuz_rot` function, if using the the multiprocessing package.

    Each process gets its own copy of the a _xfe0_fast_nuz_chained object,
    which is initialized with nuz_est = -1. Upon each call nuz_est is set to
    the previous outcome of the calculated NUZ0_avg.
    """
    def __init__(self):
        self.nuz_est = -1

    def __call__(self, A0, bases='', nuz_est=-1, nbs=0, nbp=0, nbs_loop=0,
                 r0_sph=None, kappa=None, k_rot=None,
                 S=None, L_p_ssDNA=None, z=None,
                 pitch=None, L_p_dsDNA=None,
                 NNBP=False, c=0, e_loop=0.0, T=298.2,
                 spacing=5, min_stepsize=10,
                 boltzmann_factor=1e-9, verbose=False):
        if nuz_est == -1:
            nuz_est = self.nuz_est
        # print('x0 {}, nuz_est {}'.format(x0, nuz_est))
        r = xfe0_fast_nuz_rot(A0, bases=bases, nuz_est=nuz_est, nbs=nbs,
                              nbp=nbp, nbs_loop=nbs_loop,
                              r0_sph=r0_sph, kappa=kappa, k_rot=k_rot,
                              S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                              pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                              NNBP=NNBP, c=c, e_loop=e_loop, T=T,
                              boltzmann_factor=boltzmann_factor,
                              verbose=verbose)
        self.nuz_est = int(round(r['NUZ0_avg']))
        return r


def unzipping_force_energy_rot(x0_min, x0_max, y0=0.0, h0=0.0, resolution=1e-9,
                               processes=8,
                               bases='', nbs=0, nbp=0, nbs_loop=0,
                               radius=0.0, angles_r0=None, kappa=None,
                               k_rot=None,
                               S=None, L_p_ssDNA=None, z=None,
                               pitch=None, L_p_dsDNA=None,
                               NNBP=False, c=0, e_loop=0.0, T=298.2,
                               spacing=5, min_stepsize=10,
                               boltzmann_factor=1e-9,
                               individual_points=False, verbose=False,
                               F_ssDNA_mod=None, E_ext_ssDNA_mod=None,
                               ext_dsDNA_wlc_mod=None):
    # Assign DNA model functions to the variables of the global (module) scope,
    # such that `multiprocessing.Pool` will see these variables.
    global F_ssDNA
    global E_ext_ssDNA
    global ext_dsDNA_wlc

    # Set DNA model functions to the unbuffered default functions
    F_ssDNA = F_ssDNA_mod or _F_ssDNA
    E_ext_ssDNA = E_ext_ssDNA_mod or _E_ext_ssDNA
    # Initialize the approximations of the ssDNA/dsDNA model functions with
    # fixed model function parameters and substitute the original DNA model
    # functions
    # F_ssDNA is implicitly buffered with `ext_dsDNA_wlc`.
    # Buffered `E_ext_ssDNA` does not speed up calculation.
    # E_ext_ssDNA = \
    #     init_buf_E_ext_ssDNA(read=False, write=False, filename='E_ext_ssDNA',
    #                          processes=processes,
    #                          bases=bases, nbs=nbs, nbs_loop=nbs_loop,
    #                          S=S, L_p=L_p_ssDNA, z=z, T=T)
    ext_dsDNA_wlc = ext_dsDNA_wlc_mod or \
        init_buf_ext_dsDNA_wlc(nbp=nbp, pitch=pitch, L_p=L_p_dsDNA, T=T)

    resolution = int(np.round((x0_max - x0_min) / resolution + 1))
    X0 = np.linspace(x0_min, x0_max, resolution)

    # calculate bead radius vector
    if angles_r0 is None:
        angles_r0 = [0, 0]  # theta, phi
    r0_sph = np.array([radius, *angles_r0])

    # Speed up calculation with the multiprocessing package,
    # by taking the nuz_est from previous calculation for
    # each subsequent calculation
    xfe0_fast_nuz_rot = _xfe0_fast_nuz_rot_chained()

    # Define a closure to be executed by the pool
    def f(x0):
        print('\rCalculating equilibrium for stage displacement x0 = {:.3e}'
              '...'.format(x0), end='', flush=True)

        A0 = attachment_point(x0, y0=y0, h0=h0, radius=radius)
        # return xfe0_fast_nuz(x0, bases=bases, nuz_est=nuz_est, nbs=nbs,
        #                       nbp=nbp, nbs_loop=nbs_loop, kappa=kappa,
        return xfe0_fast_nuz_rot(A0, bases=bases, nbs=nbs, nbp=nbp,
                                 nbs_loop=nbs_loop,
                                 r0_sph=r0_sph, kappa=kappa, k_rot=k_rot,
                                 S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                                 pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                                 NNBP=NNBP, c=c, e_loop=e_loop, T=T,
                                 spacing=spacing, min_stepsize=min_stepsize,
                                 boltzmann_factor=boltzmann_factor)
        # nuz_est = int(round(r['NUZ0_avg']))
    f = unboundfunction(f)

    # Process function in pool with 8 parallelly executed processes
    with Pool(processes=processes) as pool:
        start = time.time()
        XFE0 = pool.map(f, X0)
        stop = time.time()
        print('\nDone, elapsed time: {:.1f} s'.format(stop - start))

    # combine all individually simulated points into one array
    XFE = {
        'X': np.array([xfe0['x'] for xfe0 in XFE0]),
        'A0': np.array([xfe0['A0'] for xfe0 in XFE0]),
        'D0_avg': np.array([xfe0['D0_avg'] for xfe0 in XFE0]),
        'DA0_avg': np.array([xfe0['DA0_avg'] for xfe0 in XFE0]),
        'EXT_APP_avg': np.array([xfe0['EXT_APP_avg'] for xfe0 in XFE0]),
        'NUZ0_avg': np.array([xfe0['NUZ0_avg'] for xfe0 in XFE0]),
        'NUZ0_max_W0': np.array([xfe0['NUZ0_max_W0'] for xfe0 in XFE0]),
        'F0_avg': np.array([xfe0['F0_avg'] for xfe0 in XFE0]),
        'F0_max_W0': np.array([xfe0['F0_max_W0'] for xfe0 in XFE0]),
        'settings': {
            'x0_min': x0_min,
            'x0_max': x0_max,
            'resolution': resolution,
            'bases': bases,
            'nbs': nbs,
            'nbp': nbp,
            'nbs_loop': nbs_loop,
            'r0_sph': r0_sph,
            'kappa': kappa,
            'k_rot': k_rot,
            'S': S,
            'L_p_ssDNA': L_p_ssDNA,
            'z': z,
            'pitch': pitch,
            'L_p_dsDNA': L_p_dsDNA,
            'NNBP': NNBP,
            'c': c,
            'e_loop': e_loop,
            'T': T,
            'spacing': spacing,
            'min_stepsize': min_stepsize,
            'boltzmann_factor': boltzmann_factor
        }
    }

    if individual_points:
        return XFE, XFE0
    return XFE


def simulation_settings(simulation_file, **kwargs):
    # Get simulation settings
    with open(simulation_file,'rb') as f:
        simulation = pickle.load(f)
    simulation['settings'].update(kwargs)
    return {'settings': simulation['settings']}


def get_unzipping_simulation(simulation_settings_file, simulations_file=None,
                             save=True, **kwargs):
    simulations_file = simulations_file or 'simulations.p'
    # Get simulation settings
    simulation = simulation_settings(simulation_settings_file, **kwargs)
    hash_key = get_key(**simulation['settings'])
    try:
        with open(simulations_file, 'rb') as f:
            simulations = pickle.load(f)
    except FileNotFoundError:
        simulations = {}
    if hash_key in simulations:
        return simulations[hash_key]
    else:
        # Do the simulation
        simulation = simulate_unzipping(simulation)
    # Save the simulation
    if save:
        try:
            with open(simulations_file, 'rb+') as f:
                simulations = pickle.load(f)
                f.seek(0)
                simulations[hash_key] = copy.deepcopy(simulation)
                pickle.dump(simulations, f)
        except FileNotFoundError:
            with open(simulations_file, 'wb') as f:
                simulations = {}
                simulations[hash_key] = copy.deepcopy(simulation)
                pickle.dump(simulations, f)
    return simulation


def simulate_unzipping(simulation_settings, processes=8):
    simulation = simulation_settings
    # Set the unzipping construct parameters
    bases = simulation['settings']['bases']
    nbs = simulation['settings']['nbs']
    nbp = simulation['settings']['nbp']
    nbs_loop = simulation['settings']['nbs_loop']
    S = simulation['settings']['S']
    L_p_ssDNA = simulation['settings']['L_p_ssDNA']
    z = simulation['settings']['z']
    pitch = simulation['settings']['pitch']
    L_p_dsDNA = simulation['settings']['L_p_dsDNA']

    # Set other experimental parameters
    radius = simulation['settings']['radius']
    angles_r0 = simulation['settings']['angles_r0']
    kappa = simulation['settings']['kappa']
    k_rot = simulation['settings']['k_rot']
    c = simulation['settings']['c']
    e_loop = simulation['settings']['e_loop']
    T = simulation['settings']['T']

    # Set parameters for the simulation
    NNBP = simulation['settings']['NNBP']
    x0_min = simulation['settings']['x0_min']
    x0_max = simulation['settings']['x0_max']
    h0 = simulation['settings']['h0']
    y0 = simulation['settings']['y0']
    resolution = simulation['settings']['resolution']
    boltzmann_factor = simulation['settings']['boltzmann_factor']

    # If no rotation
    if angles_r0 is None and k_rot is None and y0 == 0.0 and len(kappa) == 2:
        # Simulate in 2D only (3x as fast as 3D without rotation)
        XFE, XFE0 = unzipping_force_energy(x0_min, x0_max, h0=h0,
                                           resolution=resolution,
                                           processes=processes,
                                           bases=bases, nbs=nbs, nbp=nbp,
                                           nbs_loop=nbs_loop,
                                           radius=radius, kappa=kappa,
                                           S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                                           pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                                           NNBP=NNBP, c=c, e_loop=e_loop, T=T,
                                           boltzmann_factor=boltzmann_factor,
                                           individual_points=True)
    else:
        # Simulate in 3D
        XFE, XFE0 = unzipping_force_energy_rot(x0_min, x0_max, y0=y0, h0=h0,
                                               resolution=resolution,
                                               processes=processes,
                                               bases=bases, nbs=nbs, nbp=nbp,
                                               nbs_loop=nbs_loop,
                                               radius=radius,
                                               angles_r0=angles_r0,
                                               kappa=kappa, k_rot=k_rot,
                                               S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                                               pitch=pitch,
                                               L_p_dsDNA=L_p_dsDNA,
                                               NNBP=NNBP, c=c, e_loop=e_loop,
                                               T=T,
                                               boltzmann_factor=boltzmann_factor,
                                               individual_points=True)

    simulation['XFE'] = XFE
    simulation['XFE0'] = XFE0

    return simulation


def plot_unzip_energy_rot(x0, y0=0.0, h0=0.0, bases='', nuz_est=-1, nbs=0,
                          nbp=0, nbs_loop=0,
                          radius=0.0, angles_r0=None, kappa=None, k_rot=None,
                          S=None, L_p_ssDNA=None, z=None,
                          pitch=None, L_p_dsDNA=None,
                          NNBP=False, c=0, e_loop=0.0, T=298.2,
                          spacing=5, min_stepsize=10,
                          boltzmann_factor=1e-9,
                          verbose=False, axes=None):
    A0 = attachment_point(x0, y0=y0, h0=h0, radius=radius)
    # calculate bead radius vector
    if angles_r0 is None:
        angles_r0 = [0, 0]  # theta, phi
    r0_sph = np.array([radius, *angles_r0])
    xfe0 = xfe0_fast_nuz_rot(A0, bases=bases, nuz_est=nuz_est, nbs=nbs,
                             nbp=nbp, nbs_loop=nbs_loop,
                             r0_sph=r0_sph, kappa=kappa, k_rot=k_rot,
                             S=S, L_p_ssDNA=L_p_ssDNA, z=z,
                             pitch=pitch, L_p_dsDNA=L_p_dsDNA,
                             NNBP=NNBP, c=c, e_loop=e_loop, T=T,
                             spacing=spacing, min_stepsize=min_stepsize,
                             boltzmann_factor=boltzmann_factor,
                             verbose=verbose)

    # with cnps.cn_plot('notebook') as cnp:
    if axes is None:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler
        # ax2 = cnps.second_ax(link_ax=ax)
        # ax2.xaxis.set_visible(False)
    else:
        ax, ax2 = axes
        fig = ax.get_figure()
    nuz = xfe0['NUZ0']
    energy = xfe0['E0']
    min_e = energy.min()
    # normalize energy relative to min_e in unzits of kT
    energy -= min_e
    energy /= kB*T

    # displacement = xfe0['D0']
    boltzmann_factor = xfe0['W0'] / np.sum(xfe0['W0'])
    cumsum = np.cumsum(xfe0['W0']) / np.sum(xfe0['W0'])
    # if cnp is not None:
    #     ax.plot(nuz, energy, c=cnp.color)
    #     # ax.axhline(xfe0['D0_avg'], c=cnp.color)
    #     ax2.plot(nuz, boltzmann_factor, c=cnp.color)
    #     ax2.plot(nuz, cumsum, c=cnp.color)
    # else:
    ax.plot(nuz, energy)
    # ax.axhline(xfe0['D0_avg'])
    ax2.plot(nuz, boltzmann_factor)
    ax2.plot(nuz, cumsum)
    # ax.axvline(xfe0['NUZ0_min'], c='cyan')
    ax.axvline(xfe0['NUZ0_avg'], c='magenta')
    # ax.axvline(xfe0['NUZ0_max'], c='cyan')

    ax.set_xlabel('Number of unzipped basepairs')
    ax.set_ylabel('Energy difference ($k_{B}*T$)')
    ax2.set_ylabel('Boltzmann factor')

    return fig, ax, ax2


def plot_simulated_force_extension(simulation, x=None, y=None, yXYZ=None,
                                   axes=None, ylim=None):
    # Set the unzipping construct parameters
    #bases = simulation['settings']['bases']
    #nbs = simulation['settings']['nbs']
    #nbp = simulation['settings']['nbp']
    #nbs_loop = simulation['settings']['nbs_loop']
    #S = simulation['settings']['S']
    #L_p_ssDNA = simulation['settings']['L_p_ssDNA']
    #z = simulation['settings']['z']
    #pitch = simulation['settings']['pitch']
    #L_p_dsDNA = simulation['settings']['L_p_dsDNA']

    # Set other experimental parameters
    #radius = simulation['settings']['radius']
    #angles_r0 = simulation['settings']['angles_r0']
    kappa = simulation['settings']['kappa']
    k_rot = simulation['settings']['k_rot']
    #c = simulation['settings']['c']
    #T = simulation['settings']['T']

    # Set parameters for the simulation
    #NNBP = simulation['settings']['NNBP']
    #x0_min = simulation['settings']['x0_min']
    #x0_max = simulation['settings']['x0_max']
    #h0 = simulation['settings']['h0']
    #y0 = simulation['settings']['y0']
    #resolution = simulation['settings']['resolution']
    #boltzmann_factor = simulation['settings']['boltzmann_factor']

    # Set variables of simulated data
    XFE, XFE0 = simulation['XFE'], simulation['XFE0']

    # Assign variables to be plotted
    # extension, number of unzipped basepairs, force
    # extension of the construct
    X = XFE['X']
    # 3D position of the stage
    # X0 = XFE['A0'][:,0]
    # average number of unzipped basepairs
    NUZ0_avg = XFE['NUZ0_avg']
    # most probable number of unzipped basepairs
    # NUZ0_max = XFE['NUZ0_max_W0']
    # Apparent extension (taking into consideration rotation of bead)
    try:
        EXT_APP_avg = XFE['EXT_APP_avg']
    except KeyError:
        # Fallback to extension of the construct
        EXT_APP_avg = X

    # Average force acting on the construct
    # F0_avg = XFE['F0_avg']
    # Most probable force acting on the construct
    # F0_max = XFE['F0_max_W0']
    # Average force acting on the bead
    F0d_avg = np.array([np.sqrt(((xfe0['D0_avg'] * kappa)**2).sum()) for xfe0
                        in XFE0])
    # should be the same as:
    # F0d_avg = np.array([(np.sqrt(((xfe0['D0'] * kappa)**2).sum(axis=1)) * xfe0['W0'] / xfe0['W0'].sum()).sum() for xfe0 in XFE0])
    F0XYZ_avg = np.array([np.sqrt(((xfe0['D0_avg'] * kappa)**2)) for xfe0
                          in XFE0])
    # Bead rotation in difference of angles of vectors r0 and r of the bead
    try:
        THETA_DIFF = XFE['DA0_avg'][:,0]
    except KeyError:
        THETA_DIFF = np.zeros_like(X)

    # Select data which was properly fitted
    if k_rot is None or np.all(k_rot == 0):
        idx_valid = (X != 0)
    else:
        idx_valid = np.logical_and(abs(THETA_DIFF) > 0*math.pi/180,
                                   abs(THETA_DIFF) < 45*math.pi/180)

    if axes is None:
        fig, axes = plt.subplots(2, 1)
    else:
        fig = axes[0].get_figure()

    ax = axes[0]
    ax2 = ax.twinx()
    ax2._get_lines.prop_cycler = ax._get_lines.prop_cycler

    # Plot simulated unzipping curve
    ax.plot(EXT_APP_avg[idx_valid]*1e9, F0d_avg[idx_valid]*1e12,
            label='Force bead')

    # Plot measured unzipping curve
    if x is not None and y is not None:
        ax.plot(x, y)

    # Plot number of simulated unzipped basepairs
    ax2.plot(EXT_APP_avg[idx_valid]*1e9, NUZ0_avg[idx_valid], color='cyan')

    ax.grid(True)

    ax.set_xlabel('(Apparent) extension of the construct (nm)')
    ax.set_ylabel('Force (pN)')
    ax2.set_ylabel('Number of unzipped basepairs')

    ylim = ylim or (-1, 18)
    ax.set_ylim(ylim)

    ax = axes[1]
    ax2 = plt.twinx(ax=ax)
    ax2.xaxis.set_visible(False)

    # Plot simulated unzipping curves
    ax.plot(EXT_APP_avg[idx_valid]*1e9, F0XYZ_avg[idx_valid]*1e12)

    # Plot measured unzipping curves
    if x is not None and yXYZ is not None:
        ax.plot(x, np.abs(yXYZ))

    # Plot plot differenc of angle r0 and r
    ax2.plot(EXT_APP_avg[idx_valid]*1e9, THETA_DIFF[idx_valid]*180/math.pi,
             color='cyan')

    ax.grid(True)

    ax.set_xlabel('(Apparent) extension of the construct (nm)')
    ax.set_ylabel('Force (pN)')
    ax2.set_ylabel('Angle theta difference (°)')

    ax.set_ylim(ylim)
    return fig, axes


def get_key_OLD(radius, angles_r0, kappa, k_rot):
    hasher = hashlib.md5()
    hasher.update(bytes(str(radius), 'ASCII'))
    hasher.update(angles_r0)
    hasher.update(kappa)
    hasher.update(k_rot)
    key = hasher.hexdigest()
    return key


def get_key(x0_min, x0_max, y0, h0, resolution,
            bases, nbs, nbp, nbs_loop,
            radius, angles_r0, kappa, k_rot,
            S, L_p_ssDNA, z,
            pitch, L_p_dsDNA,
            NNBP, c, e_loop, T, boltzmann_factor):
    hasher = hashlib.md5()
    for c in [x0_min, x0_max, y0, h0, resolution,
              bases.capitalize(), nbs, nbp, nbs_loop,
              radius,
              S, L_p_ssDNA, z,
              pitch, L_p_dsDNA,
              NNBP, c, e_loop, T, boltzmann_factor]:
        hasher.update(bytes(str(c), 'ASCII'))
    if angles_r0 is not None:
        hasher.update(angles_r0)
    hasher.update(kappa)
    if k_rot is not None:
        hasher.update(k_rot)
    key = hasher.hexdigest()
    return key


# Set DNA model functions to unbuffered versions per default
F_ssDNA = _F_ssDNA
E_ext_ssDNA = _E_ext_ssDNA
ext_dsDNA_wlc = _ext_dsDNA_wlc

# timeit results:
#x0 = 800e-9
#xfe0 = xfe0_fast_nuz(x0, bases=bases, nbs=nbs, nbp=nbp, nbs_loop=nbs_loop,
#                     kappa=kappa,
#                     S=S, L_p_ssDNA=L_p_ssDNA, z=z,
#                     pitch=pitch, L_p_dsDNA=L_p_dsDNA,
#                     NNBP=NNBP, c=c, T=T,
#                     boltzmann_factor=1e-5)

# Time needed with current settings:
# RS: 140 s
# R0: 884 s
# RI: 180 s
# RP: 12368 s

# Calculation of E_ext_ssDNA buffer on multiple cores
#   121.9 s

# Calculation times for force extension curve with differing boltzmann_factor on mulitple cpu cores
# boltzmann_factor
# 1e-2   35 s, STD = 55.043 fN (compared to gold standard), deviation seen of up to ~ 500 fN and partly distorted force/extension curve
# 1e-3   40 s, STD = 4.525 fN (compared to gold standard), deviation seen of up to 30 fN
# 1e-4   45 s, STD = 2.519 fN (compared to gold standard), deviation seen of only up to sub fN
# 1e-5   50 s, STD = 2.513 fN (compared to gold standard)
# 1e-6   54 s, STD = 1.363 fN (compared to gold standard)
# 1e-7   57 s, STD = 64.751 aN (compared to gold standard)
# 1e-8   62 s, STD = 3.281 aN (compared to gold standard)
# 1e-9   64 s, STD = 0.170 aN (compared to gold standard)
# 0e-9   806 s, gold standard

# Calculation times for force extension curve with boltzmann_factor = 1e-5 and iterative on one cpu
#   F_ssDNA buffer calculation would need roughly the same amount of time as E_ext_ssDNA calculation
#   F_ssDNA would be only called once for every call of E_ext_ssDNA -> not reasonable to do buffer F_ssDNA
# nuz_est iterative
#   E_ext_ssDNA buffered
#     ext_dsDNA_wlc buffered
#       F_ssDNA buffered
# + + + - elapsed time:  138.7 s  -> 34.675 s + 121.9 s ->  156.575 s   371 %  -> only feasable, if calculated ~ 12 x
# + - + - elapsed time:  168.6 s  ->                         42.15 s    100 %  -> only feasable, if 10 s per calculation important
# + + - - elapsed time: 2853.1 s  -> 713.275 s + 121.9 s -> 835.175 s  1981 %
# + - - - elapsed time: 2872.2 s  ->                        718.05 s   1704 %
# - + + - elapsed time:  173.3 s  -> 43.325 s + 121.9 s ->  165.225 s   392 %  -> only feasable, if calculated ~ 12 x
# - - + - elapsed time:  215.1 s  ->                         53.775 s   128 %  -> most feasable settings
# - + - - elapsed time: 3XXX.X s  -> not measured, only estimated
# - - - - elapsed time: 3641.0 s  ->                        910.25 s   2160 %