# -*- coding: utf-8 -*-
# 2021-02-19 by Michał Dobrski

import mpmath as mp
from mpmath import j, pi, re, mpmathify


# ArcTan[x,y] z Mathematiki, używany jako _atan(W,1) dając efektywnie arccot(W)
def _atan(x, y):
    return -j * mp.log((x + j * y) / mp.sqrt(x**2 + y**2))


def _acot(z):
    return _atan(z, 1)


def _airyaiprime(z):  # żeby mi było łatwiej Mathematiki przeniesc
    return mp.airyai(z, derivative=1)


def _airybiprime(z):  # j.w.
    return mp.airybi(z, derivative=1)


def _kBfromkA(kA, r):
    return mp.sqrt((kA**2 * (1 - r[1, 1]**2) - r[1, 0]**2) / (r[0, 0]**2 - 1 + (kA**2 * r[0, 1]**2)))


def _kexpr(k, r):
    return (k * r[0, 1] + r[1, 0] / k) / (r[0, 0] - r[1, 1])


def _maineq(kA, r, d):
    return ((1 / kA) * _acot(-_kexpr(kA, r)) + (1 / _kBfromkA(kA, r)) * _acot(_kexpr(_kBfromkA(kA, r), r)) - d)


class Layer:

    ninit = None
    d = None

    def __init__(self, d):
        self.d = mpmathify(d)

    def twoLayers(self, lambda0):
        r = self.Tmatrix(lambda0)
        kA = mp.findroot(lambda kAx: _maineq(kAx, r, self.d), 2 * pi * self.ninit / lambda0, solver='newton')
        nA = kA * lambda0 / (2 * pi)
        kB = _kBfromkA(kA, r)
        nB = kB * lambda0 / (2 * pi)
        dA = _acot(-_kexpr(kA, r)) / kA
        dB = self.d - dA
        dA = mp.re(dA)
        dB = mp.re(dB)
        return (nA, dA), (nB, dB)


class gradLayer(Layer):

    nin = None
    nout = None

    def __init__(self, nin, nout, d):
        super().__init__(d)
        self.nin = mpmathify(nin)
        self.nout = mpmathify(nout)
        self.ninit = mpmathify(nin)


class airyLayer(gradLayer):

    def Tmatrix(self, lambda0, kt=0):
        x0 = 0
        x1 = x0 + self.d
        kin = 2 * mp.pi * self.nin / lambda0
        kout = 2 * mp.pi * self.nout / lambda0
        A = (kout**2 - kin**2) / self.d
        B = kin**2 - x0 * (kout**2 - kin**2) / self.d - kt**2

        return mp.matrix([[
                              -(mp.pi * _airyaiprime(-((B + A * x0) / (-A)**(2/3))) * mp.airybi(-((B + A * x1) / (-A)**(2/3)))) +
                              mp.pi * mp.airyai(-((B + A * x1) / (-A)**(2/3))) * _airybiprime(-((B + A * x0) / (-A)**(2/3))),
                              (
                                  mp.pi * (
                                      -(mp.airyai(-((B + A * x1) / (-A)**(2/3))) * mp.airybi(-((B + A * x0) / (-A)**(2/3)))) +
                                      mp.airyai(-((B + A * x0) / (-A)**(2/3))) * mp.airybi(-((B + A * x1) / (-A)**(2/3)))
                                  )
                              ) / (-A)**(1 / 3)
                          ],
                          [(-A)**(1 / 3) * mp.pi * (
                              _airyaiprime(-((B + A * x1) / (-A)**(2/3))) * _airybiprime(-((B + A * x0) / (-A)**(2/3))) -
                              _airyaiprime(-((B + A * x0) / (-A)**(2/3))) * _airybiprime(-((B + A * x1) / (-A)**(2/3)))
                          ), -(mp.pi * _airyaiprime(-((B + A * x1) / (-A)**(2/3))) * mp.airybi(-((B + A * x0) / (-A)**(2/3)))) +
                           mp.pi * mp.airyai(-((B + A * x0) / (-A)**(2/3))) * _airybiprime(-((B + A * x1) / (-A)**(2/3)))]])


class pcfdLayer(gradLayer):

    def Tmatrix(self, lambda0, kt=0):
        kin = 2 * mp.pi * self.nin / lambda0
        kout = 2 * mp.pi * self.nout / lambda0
        A = (kout - kin) / self.d
        B = kin

        return mp.matrix(
            [[
                (
                    j * (
                        mp.pcfd(-(A + j * kt**2) / (2 * A),
                                ((-1 + j) * (B + A * self.d)) / mp.sqrt(A)) * (
                                    2 * (-1)**(3 / 4) * mp.sqrt(A) * mp.pcfd((A + j * kt**2) / (2 * A),
                                                                             ((1 + j) * B) / mp.sqrt(A)) +
                                    mp.sqrt(2) * B * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * B) / mp.sqrt(A))
                                ) + (
                                    2 * (-1)**(1 / 4) * mp.sqrt(A) * mp.pcfd((A - j * kt**2) / (2 * A),
                                                                             ((-1 + j) * B) / mp.sqrt(A)) +
                                    mp.sqrt(2) * B * mp.pcfd(-(A + j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A))
                                ) * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * (B + A * self.d)) / mp.sqrt(A))
                    )
                ) / (
                    2 * (-1)**(3 / 4) * mp.sqrt(A) * mp.pcfd((A - j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) *
                    mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * B) / mp.sqrt(A)) +
                    (2 * j) * mp.pcfd(-(A + j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) *
                    ((-1)**(3 / 4) * mp.sqrt(A) * mp.pcfd((A + j * kt**2) / (2 * A), ((1 + j) * B) / mp.sqrt(A)) +
                     mp.sqrt(2) * B * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * B) / mp.sqrt(A)))
                ),
                ((1 + j) * (
                    mp.pcfd(-(A + j * kt**2) / (2 * A),
                            ((-1 + j) *
                             (B + A * self.d)) / mp.sqrt(A)) * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * B) / mp.sqrt(A)) -
                    mp.pcfd(-(A + j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) *
                    mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * (B + A * self.d)) / mp.sqrt(A))
                )) / (
                    2 * mp.sqrt(A) * mp.pcfd((A - j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) *
                    mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A,
                            ((1 + j) * B) / mp.sqrt(A)) + mp.pcfd(-(A + j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) *
                    ((2 * j) * mp.sqrt(A) * mp.pcfd((A + j * kt**2) / (2 * A), ((1 + j) * B) / mp.sqrt(A)) +
                     (2 - 2 * j) * B * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * B) / mp.sqrt(A)))
                )
            ],
             [((B + A * self.d) * mp.pcfd(-(A + j * kt**2) / (2 * A), ((-1 + j) * (B + A * self.d)) / mp.sqrt(A)) *
               ((-1 + j) * mp.sqrt(A) * mp.pcfd((A + j * kt**2) / (2 * A), ((1 + j) * B) / mp.sqrt(A)) +
                B * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A,
                            ((1 + j) * B) / mp.sqrt(A))) + mp.pcfd((A - j * kt**2) / (2 * A),
                                                                   ((-1 + j) * (B + A * self.d)) / mp.sqrt(A)) *
               (
                   -2 * A * mp.pcfd((A + j * kt**2) / (2 * A), ((1 + j) * B) / mp.sqrt(A)) +
                   (1 + j) * mp.sqrt(A) * B * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * B) / mp.sqrt(A))
               ) - ((1 + j) * mp.sqrt(A) * mp.pcfd((A - j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) +
                    B * mp.pcfd(-(A + j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A))) *
               ((-1 + j) * mp.sqrt(A) * mp.pcfd((A + j * kt**2) / (2 * A), ((1 + j) * (B + A * self.d)) / mp.sqrt(A)) +
                (B + A * self.d) * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * (B + A * self.d)) / mp.sqrt(A)))) /
              ((-1 + j) * mp.sqrt(A) * mp.pcfd((A - j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) *
               mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A,
                       ((1 + j) * B) / mp.sqrt(A)) + mp.pcfd(-(A + j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) *
               ((-1 - j) * mp.sqrt(A) * mp.pcfd((A + j * kt**2) / (2 * A), ((1 + j) * B) / mp.sqrt(A)) +
                (2 * j) * B * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * B) / mp.sqrt(A)))),
              (
                  j * ((
                      2 * (-1)**(1 / 4) * mp.sqrt(A) * mp.pcfd((A - j * kt**2) / (2 * A),
                                                               ((-1 + j) * (B + A * self.d)) / mp.sqrt(A)) + mp.sqrt(2) *
                      (B + A * self.d) * mp.pcfd(-(A + j * kt**2) / (2 * A), ((-1 + j) * (B + A * self.d)) / mp.sqrt(A))
                  ) * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A,
                              ((1 + j) * B) / mp.sqrt(A)) + mp.pcfd(-(A + j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) *
                       (
                           2 * (-1)**(3 / 4) * mp.sqrt(A) * mp.pcfd((A + j * kt**2) / (2 * A),
                                                                    ((1 + j) * (B + A * self.d)) / mp.sqrt(A)) + mp.sqrt(2) *
                           (B + A * self.d) * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * (B + A * self.d)) / mp.sqrt(A))
                       ))
              ) / (
                  2 * (-1)**(3 / 4) * mp.sqrt(A) * mp.pcfd((A - j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) *
                  mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A,
                          ((1 + j) * B) / mp.sqrt(A)) + (2 * j) * mp.pcfd(-(A + j * kt**2) / (2 * A), ((-1 + j) * B) / mp.sqrt(A)) *
                  ((-1)**(3 / 4) * mp.sqrt(A) * mp.pcfd((A + j * kt**2) / (2 * A), ((1 + j) * B) / mp.sqrt(A)) +
                   mp.sqrt(2) * B * mp.pcfd(-1 / 2 + ((j / 2) * kt**2) / A, ((1 + j) * B) / mp.sqrt(A)))
              )]]
        )


def simplify_gradient_nr(d, n0, n1, lam, linear='nr'):
    """Compute thiknesses and refactive indices of two layers simplifying a gradient layer.

    Args:
        d (float): layer thickness
        n0 (complex): bottom refractive index
        n1 (complex): top refractive index
        lam (float): wavelength
        linear ('nr' or 'eps'): which parameter should be linear
    """
    if linear == 'eps':
        layer = airyLayer(mpmathify(n0), mpmathify(n1), mpmathify(1.))
    elif linear.lower() == 'nr':
        layer = pcfdLayer(mpmathify(n0), mpmathify(n1), mpmathify(1.))
    else:
        raise ValueError("'linear' argument must be either 'eps' or 'nr'")
    (nA, fA), (nB, fB) = layer.twoLayers(lam)
    return (complex(nA), d * float(fA)), (complex(nB), d * float(fB))
