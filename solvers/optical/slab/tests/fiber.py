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

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv, kv
from scipy.optimize import fsolve


class Analytic:

    def __init__(self, R=1., nr=3.5, m=1, lam=1000.):
        self.R = R
        self.nr = nr
        self.m = m
        self.J = lambda z: jv(m, z)
        self.K = lambda z: kv(m, z)
        self.D_J = lambda z: 0.5 * (jv(m-1,z) - jv(m+1, z))
        self.D_K = lambda z: - kv(m-1, z) - m/z * kv(m, z)
        self.k0 = 2e3*np.pi / lam
        self.V = R * self.k0 * (nr**2 - 1)**0.5

        nn = np.linspace(1.00, nr, 4001)[1:-1]
        dets = np.abs(self.fun(nn))
        self._minima = nn[1:-1][(dets[1:-1] <= dets[:-2]) & (dets[1:-1] < dets[2:])][::-1]

    def fun(self, neff):
        """
        Characteristic function for finding fiber modes

        Taken from Snyder & Love, p.253
        """
        kz = neff * self.k0
        U = self.R * ((self.k0 * self.nr)**2 - kz**2)**0.5
        W = self.R * (kz**2 - self.k0**2)**0.5
        fcore = self.D_J(U) / (U * self.J(U))
        fclad = self.D_K(W) / (W * self.K(W))
        lhs = (fcore + fclad) * (fcore + fclad / self.nr**2)
        rhs = (self.m * kz / (self.k0 * self.nr))**2 * (self.V / (U * W))**4
        return lhs - rhs

    def field(self, neff, r):
        kz = neff * self.k0
        sr = np.sign(r)
        r = np.abs(r)
        m = self.m
        U = self.R * ((self.k0 * self.nr)**2 - kz**2)**0.5
        W = self.R * (kz**2 - self.k0**2)**0.5
        Δ = 0.5 * (self.nr**2 - 1) / self.nr**2
        JU = jv(m, U)
        JUp = jv(m+1, U)
        JUm = jv(m-1, U)
        KW = kv(m, W)
        KWp = kv(m+1, W)
        KWm = kv(m-1, W)
        if m != 0:
            b1 = 0.5 / U * (JUm - JUp) / JU
            b2 = - 0.5 / W * (KWm + KWp) / KW
            F2 = (self.V / (U*W))**2 * m / (b1 + b2)
            a1 = 0.5 * (F2 - 1)
            a2 = 0.5 * (F2 + 1)
            Er = - np.where(r <= self.R,
                            (a1 * jv(m-1,U*r) + a2 * jv(m+1,U*r)) / JU,
                            U/W * (a1 * kv(m-1,W*r) - a2 * kv(m+1,W*r)) / KW)
            Ep = + np.where(r <= self.R,
                            (a1 * jv(m-1,U*r) - a2 * jv(m+1,U*r)) / JU,
                            U/W * (a1 * kv(m-1,W*r) + a2 * kv(m+1,W*r)) / KW)
            Ez = - U / (kz*self.R) * np.where(r <= self.R,
                                              jv(m, U*r) / JU,
                                              kv(m, W*r) / KW)
            if m % 2 == 0:
                Er[sr < 0] = -Er[sr < 0]
                Ep[sr < 0] = -Ep[sr < 0]
            else:
                Ez[sr < 0] = -Ez[sr < 0]
        else:
            fcore = self.D_J(U) / (U * JU)
            fclad = self.D_K(W) / (W * KW)
            if abs(fcore + fclad) < 1e-6:
                Ep = - np.where(r <= R, jv(1, U*r) / JUp, kv(1, W*r) / KWp)
                Er = Ez = np.zeros(len(Ep))
            else:
                Er = np.where(r <= R, jv(1, U*r) / JUp, nr**2 * kv(1, W*r) / KWp)
                Ep = np.zeros(len(Er))
                Ez = np.where(r <= R, U / (kz*R) * jv(0, U*r) / JUp, - W / (kz*R) * nr**2 * kv(0, W*r) / KWp)
        M = np.real(Er * np.conj(Er) + Ep * np.conj(Ep) + Ez * np.conj(Ez))
        fm = 1 / np.max(M)**0.5
        Er *= fm
        Ep *= fm
        Ez *= fm
        return np.array([Ep, Er, Ez]).T

    def __len__(self):
        return len(self._minima)

    def __getitem__(self, num):
        if isinstance(num, slice):
            return [fsolve(self.fun, x)[0] for x in self._minima[num]]
        else:
            return fsolve(self.fun, self._minima[num])[0]


def make_radial_plot(Er, r=None, m=1, R=1., neff=None, c=1, axs=None):
    """Plot radial profile

    Args:
        Er (array of Er): Field to plot.
        r (array, optional): Radial positions. If None, it is take from data mesh.
        m (int): Angular mode param.
        R (float): Fiber radius.
    """
    if r is None:
        r = Er.mesh.axis0
        Er = Er.array[:,0,:]
    if axs is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for ax in ax1, ax2:
            ax.axvline(-R, color='k', ls='--', lw=0.5)
            ax.axvline(+R, color='k', ls='--', lw=0.5)
        ax1.plot((np.nan,), (np.nan,), color='k', ls='-', label="$E_r$")
        ax1.plot((np.nan,), (np.nan,), color='k', ls='-', alpha=0.35, label=r"$E_\varphi$")
        ax1.plot((np.nan,), (np.nan,), color='k', ls='--', label="$E_z$")
        ax2.set_xlabel("$r$ (µm)")
        leg1 = ax1.legend(loc='best', prop={'size': 6})
        try: leg1.set_draggable(True)
        except AttributeError: pass
        fig.set_tight_layout({'pad': 0.1})
    else:
        ax1, ax2 = axs
    ax1.plot(r, Er[:,1], color=f'C{c}', ls='-')
    ax1.plot(r, Er[:,0], color=f'C{c}', ls='-', alpha=0.5)
    ax1.plot(r, Er[:,2], color=f'C{c}', ls='--')
    ax2.plot(r, np.sum(np.real(Er * np.conj(Er)), 1), color=f'C{c}', label=f"{c}: {neff.real:.4f}" if neff is not None else None)
    leg2 = ax2.legend(loc='best', prop={'size': 6})
    try: leg2.set_draggable(True)
    except AttributeError: pass
    return ax1, ax2


def make_polar_plot(Er, r=None, m=1, R=1.):
    """Plot polar maps

    Args:
        Er (array of Er): Field to plot.
        r (array, optional): Radial positions. If None, it is take from data mesh.
        m (int): Angular mode param.
        R (float): Fiber radius.
    """
    if r is None:
        r = Er.mesh.axis0
        Er = Er.array[:,0,:2]

    p = np.linspace(0., 2*np.pi, 37)
    rr, pp = np.meshgrid(r, p)
    phi = np.linspace(0., 2*np.pi, 1001)

    E = Er[None,:,:]
    E = np.repeat(E, len(p), 0)

    mer = np.max(np.abs(E.ravel().real))
    mei = np.max(np.abs(E.ravel().imag))
    if mer >= mei: E = E.real / mer
    else: E = E.imag / mei

    if m != 0:
        E[:,:,0] *= np.sin(m * pp)
        E[:,:,1] *= np.cos(m * pp)

    F = np.sum((E * E.conj()).real, 2)

    fs = plt.rcParams['figure.figsize'][0]
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(fs,fs))
    ax.tick_params('y', colors='w')
    ax.tick_params(axis='x', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    ax.tick_params(axis='y', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(False)

    plt.contourf(pp, rr, F, 64)

    Ex = E[:,:,1] * np.cos(pp) - E[:,:,0] * np.sin(pp)
    Ey = E[:,:,1] * np.sin(pp) + E[:,:,0] * np.cos(pp)
    plt.quiver(pp, rr, Ex, Ey, pivot='mid', scale=30, color='w')

    plt.plot(phi, (R,)*len(phi), 'w', lw=0.5)
