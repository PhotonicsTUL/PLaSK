#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical import slab

config.axes = 'zxy'

@material.simple()
class Glass(material.Material):
    def Nr(self, wl, T=300., n=0.): return 1.3


solver = slab.Fourier3D("solver")
solver.ft = 'analytic'
wire_stack = geometry.Stack2D()
wire_stack.append(geometry.Rectangle(0.75, 0.125, Glass()))
rect = geometry.Rectangle(0.75, 0.375, Glass())
rect.role = 'interface'
wire_stack.append(rect)

shelf = geometry.Shelf2D()
shelf.append(wire_stack)
shelf.append(geometry.Rectangle(0.15, 0.50, 'air'))

extrusion = geometry.Extrusion(shelf, 1.)
solver.geometry = geometry.Cartesian3D(extrusion,
                                       left="mirror", right="periodic",
                                       back='mirror', front='periodic')

solver.dct = 1

solver.size = 0, 12

start = 1.1216

det = False

def plotFields(symmetry):
    solver.lam = 1000.
    solver.symmetry = None, symmetry

    k0 = 2e3*pi / solver.lam

    if det:
        figure()
        neffs = linspace(1.0, 1.2, 1001)
        dets = solver.get_determinant(klong=k0*neffs)
        plot(neffs, abs(dets))
        window_title(f"det {symmetry}")

    mn = solver.find_mode(klong=k0*start)
    print_log('result', f"neff = {(solver.modes[mn].klong/k0).real}")

    msh = mesh.Rectangular3D([0], mesh.Regular(-2., 2., 401), mesh.Regular(-0.4, 1.0, 1501))
    figure(figsize=(14, 6))

    E = solver.outLightE(mn, msh)
    m = max(abs(E))
    E /= m

    ax_mag = subplot2grid((3,3), (0,1), rowspan=3)
    plot_field(Data(sum(real(E*conj(E)),-1), E.mesh), plane='xy')
    plot_geometry(solver.geometry, color='w', mirror=True, periods=2, lw=0.5, plane='xy')
    xticks(arange(-2., 2.01, 0.5))
    c = 2

    subplot2grid((3,3), (0,2), rowspan=3, sharey=ax_mag)
    msh1 = mesh.Rectangular3D([0], [0], msh.axis2)
    E1 = solver.outLightE(mn, msh1)
    mr = max(abs(E.array[:,:,c].real).ravel())
    mi = max(abs(E.array[:,:,c].imag).ravel())
    if mi > mr: E1 = E1.imag
    else: E1 = E1.real
    axhline(0.0, ls='--', color='k', lw=0.5)
    axhline(0.5, ls='--', color='k', lw=0.5)
    plot_profile(E1, swap_axes=True, comp=c)
    Ec = E1.array[0,0,:,c].copy()
    y = array(msh.axis2)
    Ec[(0.0 <= y) & (y <= 0.5)] *= 1.3**2
    plot(Ec, y)
    ylim(-0.4, 1.0)
    window_title(f"Field {symmetry}")

    levels = linspace(-1, 1, 16)
    for c in range(3):
        subplot2grid((3,3), (c, 0), sharey=ax_mag)
        mr = max(abs(E.array[:,:,:,c].real).ravel())
        mi = max(abs(E.array[:,:,:,c].imag).ravel())
        print_log('data', f"Max E{config.axes[c]}: Re={mr} Im={mi}")
        if mi > mr: Ec = E.imag
        else: Ec = E.real
        plot_field(Ec, levels, comp=c, cmap='bwr', plane='xy')
        plot_geometry(solver.geometry, color='k', mirror=True, periods=2, lw=0.5, plane='xy')
        xlim(msh.axis0[0], msh.axis0[-1])
        xticks(arange(-2., 2.01, 0.5))

    tight_layout()


plotFields('Htran')
plotFields(None)
show()

