# -*- coding: utf-8 -*-
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


def _maineq(kA, r, d):
    return ((1 / kA) * np.arctan(-1. / _kexpr(kA, r)) + 1 / _kBfromkA(kA, r) * np.arctan(1. / _kexpr(_kBfromkA(kA, r), r)) - d)


def _fsolve(func, z0, *args, **kwargs):
    def func2(z, *a):
        res = func(z[0] + 1j * z[1], *a)
        return np.array([res.real, res.imag])

    result = opt.fsolve(func2, np.array([z0.real, z0.imag]), *args, **kwargs)
    return result[0] + 1j * result[1]


def _pcfd(n, z):
    # return sp.pbdv(n.real, z)[0]  # this doesn't work for complex z
    a = -n - 0.5
    zeta = 0.5 * a + 0.25
    y1 = np.exp(-0.25 * (z**2.0)) * sp.hyp1f1(zeta, 0.5, 0.5 * (z**2.0))
    y2 = z * np.exp(-0.25 * (z**2.0)) * sp.hyp1f1(zeta + 0.5, 1.5, 0.5 * (z**2.0))
    return 0.5641895835477563 * 0.5**zeta * (np.cos(np.pi*zeta) * sp.gamma(0.5-zeta) * y1 \
        -1.4142135623730951 * np.sin(np.pi*zeta) * sp.gamma(1-zeta) * y2)


class Layer:
    def __init__(self, d, ninit):
        self.d = d
        self.ninit = ninit

    def twoLayers(self, lambda0):
        r = self.Tmatrix(lambda0)
        kA = _fsolve(lambda kAx: _maineq(kAx, r, self.d), 2 * np.pi * self.ninit / lambda0)
        nA = kA * lambda0 / (2 * np.pi)
        kB = _kBfromkA(kA, r)
        nB = kB * lambda0 / (2 * np.pi)
        dA = np.arctan(-1. / _kexpr(kA, r)) / kA
        dB = self.d - dA
        return (nA, dA.real), (nB, dB.real)


class GradLayer(Layer):
    def __init__(self, nin, nout, d, ninit=None):
        super().__init__(d, ninit if ninit is not None else nin)
        self.nin = nin
        self.nout = nout


class airyLayer(GradLayer):
    def Tmatrix(self, lambda0):
        x0 = 0
        x1 = x0 + self.d
        kin = 2 * np.pi * self.nin / lambda0
        kout = 2 * np.pi * self.nout / lambda0
        A = (kout**2 - kin**2) / self.d
        B = kin**2 - x0 * (kout**2 - kin**2) / self.d

        a0, ap0, b0, bp0 = sp.airy(-((B + A * x0) / (-A)**(2 / 3)))
        a1, ap1, b1, bp1 = sp.airy(-((B + A * x1) / (-A)**(2 / 3)))

        return np.array([[-(np.pi * ap0 * b1) + np.pi * a1 * bp0, (np.pi * (-(a1 * b0) + a0 * b1)) / (-A)**0.333333333333],
                         [(-A)**0.333333333333 * np.pi * (ap1 * bp0 - ap0 * bp1), -(np.pi * ap1 * b0) + np.pi * a0 * bp1]])

class pcfdLayer(GradLayer):
    def Tmatrix(self, lambda0):
        kin = 2 * np.pi * self.nin / lambda0
        kout = 2 * np.pi * self.nout / lambda0
        A = (kout - kin) / self.d
        B = kin
        VA = A**0.5

        pcfd1 = _pcfd(0.5, ((1+1j) * B) / VA)
        pcfd2 = _pcfd(-0.5, ((1+1j) * B) / VA)
        pcfd3 = _pcfd(0.5, ((-1+1j) * B) / VA)
        pcfd4 = _pcfd(-0.5, ((-1+1j) * B) / VA)
        pcfd5 = _pcfd(-0.5, ((1+1j) * (B + A * self.d)) / VA)
        pcfd6 = _pcfd(-0.5, ((-1+1j) * (B + A * self.d)) / VA)
        pcfd7 = _pcfd(0.5, ((-1+1j) * (B + A * self.d)) / VA)
        pcfd8 = _pcfd(0.5, ((1+1j) * (B + A * self.d)) / VA)

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
                (B + A * self.d) * pcfd6 * ((-1+1j) * VA * pcfd1 + B * pcfd2) +
                pcfd7 * (-2 * A * pcfd1 + (1+1j) * VA * B * pcfd2) -
                ((1+1j) * VA * pcfd3 + B * pcfd4) * ((-1+1j) * VA * pcfd8 +
                (B + A * self.d) * pcfd5)
            ) / ((-1+1j) * VA * pcfd3 * pcfd2 + pcfd4 * ((-1-1j) * VA * pcfd1 + 2j * B * pcfd2)),

            (
                1j * (
                    (2 * _1P14 * VA * pcfd7 + 1.4142135623730951 * (B + A * self.d) * pcfd6) * pcfd2 +
                    pcfd4 * (2 * _1P34 * VA * pcfd8 + 1.4142135623730951 * (B + A * self.d) * pcfd5)
                )
            ) / (2 * _1P34 * VA * pcfd3 * pcfd2 + 2j * pcfd4 * (_1P34 * VA * pcfd1 + 1.4142135623730951 * B * pcfd2))
        ]])


def simplify_gradient_nr(d, n0, n1, lam, linear='nr', ninit=None):
    """Compute thiknesses and refactive indices of two layers simplifying a gradient layer.

    Args:
        d (float): layer thickness
        n0 (complex): bottom refractive index
        n1 (complex): top refractive index
        lam (float): wavelength
        linear ('nr' or 'eps'): which parameter should be linear
        ninit (float or None): initial refractive index of the first simplified sub-layer
    """
    if linear == 'eps':
        layer = airyLayer(n0, n1, 1., ninit)
    elif linear.lower() == 'nr':
        layer = pcfdLayer(n0, n1, 1., ninit)
    else:
        raise ValueError("'linear' argument must be either 'eps' or 'nr'")
    (nA, fA), (nB, fB) = layer.twoLayers(lam)
    return (nA, d * fA), (nB, d * fB)


__all__ = 'simplify_gradient_nr'
