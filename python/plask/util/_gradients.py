# This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
# Copyright (c) 2022 Lodz University of Technology
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# 2021-02-19 by Michał Dobrski

import numpy as np
import scipy.special as sp
import scipy.optimize as opt


_1P34 = (-1)**0.75
_1P14 = (-1)**0.25


def _kBfromkA(kA, r):
    return np.sqrt((kA**2 * (1 - r[1,1]**2) - r[1,0]**2) / (r[0,0]**2 - 1 + (kA**2 * r[0,1]**2)))


def _kexpr(k, r):
    return (k * r[0,1] + r[1,0] / k) / (r[0,0] - r[1,1])


def _maineq(kA, r):
    return ((1./kA) * np.arctan(-1./_kexpr(kA, r)) + 1./_kBfromkA(kA, r) * np.arctan(1./_kexpr(_kBfromkA(kA, r), r)) - 1.0)


def _pcfd(n, z):
    # return sp.pbdv(n.real, z)[0]  # this doesn't work for complex z
    a = -n - 0.5
    zeta = 0.5 * a + 0.25
    z2 = z**2
    y1 = np.exp(-0.25 * z2) * sp.hyp1f1(zeta, 0.5, 0.5 * z2)
    y2 = z * np.exp(-0.25 * z2) * sp.hyp1f1(zeta + 0.5, 1.5, 0.5 * z2)
    return 0.5641895835477563 * 0.5**zeta * (np.cos(np.pi*zeta) * sp.gamma(0.5-zeta) * y1 \
        -1.4142135623730951 * np.sin(np.pi*zeta) * sp.gamma(1-zeta) * y2)


class Layer:
    def __init__(self, ninit):
        self.ninit = ninit

    def twoLayers(self, lambda0):
        r = self.Tmatrix(lambda0)
        kinit = 2*np.pi * np.array([self.ninit.real, self.ninit.imag]) / lambda0

        def func(x):
            res = _maineq(x[0] + 1j * x[1], r)
            return np.array([res.real, res.imag])

        kA = opt.fsolve(func, kinit)
        kA = kA[0] + 1j * kA[1]

        nA = kA * lambda0 / (2 * np.pi)
        kB = _kBfromkA(kA, r)
        nB = kB * lambda0 / (2 * np.pi)
        dA = np.arctan(-1. / _kexpr(kA, r)) / kA
        dB = 1.0 - dA
        return (nA, dA.real), (nB, dB.real)


class GradLayer(Layer):
    def __init__(self, nin, nout, ninit=None):
        if ninit is None:
            ninit = nin + 0.333333 * (nout - nin)
        super().__init__(ninit)
        self.nin = nin
        self.nout = nout


class AiryLayer(GradLayer):
    def Tmatrix(self, lambda0):
        kin = 2 * np.pi * self.nin / lambda0
        kout = 2 * np.pi * self.nout / lambda0
        A = kout**2 - kin**2
        B = kin**2

        a0, ap0, b0, bp0 = sp.airy(-(B / (-A)**0.666666666667))
        a1, ap1, b1, bp1 = sp.airy(-((A+B) / (-A)**0.666666666667))

        return np.array([[-(np.pi * ap0 * b1) + np.pi * a1 * bp0, (np.pi * (-(a1 * b0) + a0 * b1)) / (-A)**0.333333333333],
                         [(-A)**0.333333333333 * np.pi * (ap1 * bp0 - ap0 * bp1), -(np.pi * ap1 * b0) + np.pi * a0 * bp1]])

class PcfdLayer(GradLayer):
    def Tmatrix(self, lambda0):
        kin = 2 * np.pi * self.nin / lambda0
        kout = 2 * np.pi * self.nout / lambda0
        A = kout - kin
        B = kin
        VA = A**0.5
        AB = A + B

        pcfd1 = _pcfd(0.5, ((1+1j) * B) / VA)
        pcfd2 = _pcfd(-0.5, ((1+1j) * B) / VA)
        pcfd3 = _pcfd(0.5, ((-1+1j) * B) / VA)
        pcfd4 = _pcfd(-0.5, ((-1+1j) * B) / VA)
        pcfd5 = _pcfd(-0.5, ((1+1j) * AB) / VA)
        pcfd6 = _pcfd(-0.5, ((-1+1j) * AB) / VA)
        pcfd7 = _pcfd(0.5, ((-1+1j) * AB) / VA)
        pcfd8 = _pcfd(0.5, ((1+1j) * AB) / VA)

        return np.array([[
            (
                1j * (
                    pcfd6 * (2 * _1P34 * VA * pcfd1 + 1.4142135623730951 * B * pcfd2) +
                    (2 * _1P14 * VA * pcfd3 + 1.4142135623730951 * B * pcfd4) * pcfd5
                )
            ) / (2 * _1P34 * VA * pcfd3 * pcfd2 + 2j * pcfd4 * (_1P34 * VA * pcfd1 + 1.4142135623730951 * B * pcfd2)),

            ((1+1j) * (pcfd6 * pcfd2 - pcfd4 * pcfd5)) / (2 * VA * pcfd3 * pcfd2 + pcfd4 * (2j * VA * pcfd1 + (2-2j) * B * pcfd2))
        ],
        [
            (
                AB * pcfd6 * ((-1+1j) * VA * pcfd1 + B * pcfd2) + pcfd7 * (-2 * A * pcfd1 + (1+1j) * VA * B * pcfd2) -
                ((1+1j) * VA * pcfd3 + B * pcfd4) * ((-1+1j) * VA * pcfd8 + AB * pcfd5)
            ) / ((-1+1j) * VA * pcfd3 * pcfd2 + pcfd4 * ((-1-1j) * VA * pcfd1 + 2j * B * pcfd2)),

            (
                1j * (
                    (2 * _1P14 * VA * pcfd7 + 1.4142135623730951 * AB * pcfd6) * pcfd2 +
                    pcfd4 * (2 * _1P34 * VA * pcfd8 + 1.4142135623730951 * AB * pcfd5)
                )
            ) / (2 * _1P34 * VA * pcfd3 * pcfd2 + 2j * pcfd4 * (_1P34 * VA * pcfd1 + 1.4142135623730951 * B * pcfd2))
        ]])


__cache = {}


def simplify_gradient_nr(d, n0, n1, lam, linear='nr', ninit=None):
    """Compute thicknesses and refractive indices of two layers simplifying a gradient layer.

    Args:
        d (float): layer thickness
        n0 (complex): bottom refractive index
        n1 (complex): top refractive index
        lam (float): wavelength
        linear ('nr' or 'eps'): which parameter should be linear
        ninit (float or None): initial refractive index of the first simplified sub-layer
    """
    key = n0, n1, lam, linear
    if key in __cache:
        (nA, fA), (nB, fB) = __cache[key]
    else:
        if linear == 'eps':
            layer = AiryLayer(n0, n1, ninit)
        elif linear.lower() == 'nr':
            layer = PcfdLayer(n0, n1, ninit)
        else:
            raise ValueError("'linear' argument must be either 'eps' or 'nr'")
        (nA, fA), (nB, fB) = __cache[key] = layer.twoLayers(lam)
    return (nA, d * fA), (nB, d * fB)


__all__ = 'simplify_gradient_nr'
