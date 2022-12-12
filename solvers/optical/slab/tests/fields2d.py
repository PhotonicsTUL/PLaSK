#!/usr/bin/env python3
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

import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical import slab

config.axes = 'zxy'

@material.simple()
class Glass(material.Material):
    def Nr(self, wl, T=300., n=0.): return 1.3


solver = slab.Fourier2D("solver")
solver.ft = 'analytic'
wire_stack = geometry.Stack2D()
wire_stack.append(geometry.Rectangle(0.75, 0.125, Glass()))
rect = geometry.Rectangle(0.75, 0.375, Glass())
rect.role = 'interface'
wire_stack.append(rect)

shelf = geometry.Shelf2D()
shelf.append(wire_stack)
shelf.append(geometry.Rectangle(0.15, 0.50, 'air'))

solver.geometry = geometry.Cartesian2D(shelf, left="mirror", right="periodic")

solver.size = 12

det = False

start = 1.1216 # 1.1196



def plotFields(symmetry):
    solver.lam = 1000.
    solver.symmetry = symmetry
    solver.invalidate()

    if det:
        figure()
        neffs = linspace(1.0, 1.2, 1001)
        dets = solver.get_determinant(neff=neffs)
        plot(neffs, abs(dets))
        window_title(f"det {symmetry}")

    mn = solver.find_mode(neff=start)

    msh = mesh.Rectangular2D(mesh.Regular(-2., 2., 401), mesh.Regular(-0.4, 1.0, 1501))
    figure(figsize=(14, 6))

    E = solver.outLightE(mn, msh)
    m = max(abs(E))
    E /= m

    ax_mag = subplot2grid((3,3), (0,1), rowspan=3)
    plot_field(Data(sum(real(E*conj(E)),-1), E.mesh))
    plot_geometry(solver.geometry, color='w', mirror=True, periods=2, lw=0.5)
    #colorbar(use_gridspec=True)

    c = 2

    subplot2grid((3,3), (0,2), rowspan=3, sharey=ax_mag)
    msh1 = mesh.Rectangular2D([0], msh.axis1)
    E1 = solver.outLightE(mn, msh1)
    E1 /= m
    mr = max(abs(E.array[:,:,c].real).ravel())
    mi = max(abs(E.array[:,:,c].imag).ravel())
    if mi > mr: E1 = E1.imag
    else: E1 = E1.real
    axhline(0.0, ls='--', color='k', lw=0.5)
    axhline(0.5, ls='--', color='k', lw=0.5)
    plot_profile(E1, swap_axes=True, comp=c)
    Ec = E1.array[0,:,c].copy()
    y = array(msh.axis1)
    Ec[(0.0 <= y) & (y <= 0.5)] *= 1.3**2
    plot(Ec, y)
    ylim(-0.4, 1.0)
    window_title(f"Field {symmetry}")

    m = max(abs(E))
    E /= m
    levels = linspace(-1, 1, 16)
    for c in range(3):
        subplot2grid((3,3), (c, 0), sharey=ax_mag, sharex=ax_mag)
        mr = max(abs(E.array[:,:,c].real).ravel())
        mi = max(abs(E.array[:,:,c].imag).ravel())
        if mi > mr: Ec = E.imag
        else: Ec = E.real
        plot_field(Ec, levels, comp=c, cmap='bwr')
        plot_geometry(solver.geometry, color='k', mirror=True, periods=2, lw=0.5)
        xlim(msh.axis0[0], msh.axis0[-1])

    tight_layout()


plotFields('Htran')
plotFields(None)
show()
