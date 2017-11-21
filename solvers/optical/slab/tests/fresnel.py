#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical.slab import Fourier2D, Fourier3D

plask.config.axes = 'xyz'

angles = linspace(0.0, 90.0, 11)

nr = material.GaAs().nr(1000)
print("nr = {}".format(nr))

class Refl(object):

    def __init__(self, solver, direction, polarization):
        self.solver = solver
        self.solver.lam0 = 1000.
        self.direction = direction
        self.polarization = polarization
        solver.klong = solver.ktran = 0.

    def __call__(self, a):
        setattr(self.solver, 'k'+self.direction, 2*pi * sin(a*pi/180.))
        return self.solver.compute_reflectivity(1000., self.polarization, 'top'), \
               self.solver.compute_transmittivity(1000., self.polarization, 'top')

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


def R_TE(angle):
    angle = angle * pi/180
    cos1 = cos(angle)
    cos2 = sqrt(1. - (sin(angle)/nr)**2)
    return 100. * ((cos1 - nr*cos2) / (cos1 + nr*cos2))**2

def R_TM(angle):
    angle = angle * pi/180
    cos1 = cos(angle)
    cos2 = sqrt(1. - (sin(angle)/nr)**2)
    return 100. * ((cos2 - nr*cos1) / (cos2 + nr*cos1))**2

def T_TE(angle):
    return 100. - R_TE(angle)

def T_TM(angle):
    return 100. - R_TM(angle)


def show_plots(solver, direction, title=None, sep=False):
    solver.klong = solver.ktran = 0.
    if sep: solver.polarization = 'El'
    with Refl(solver, direction, 'El') as refl:
        Rlong = array([refl(a) for a in angles]).T
    if sep: solver.polarization = 'Et'
    with Refl(solver, direction, 'Et') as refl:
        Rtran = array([refl(a) for a in angles]).T
    f = figure()
    plot(angles, Rlong[0], label='R long')
    plot(angles, Rtran[0], label='R tran')
    plot(angles, Rlong[1], label='T long')
    plot(angles, Rtran[1], label='T tran')
    xlabel('angle [deg]')
    ylabel('reflectivity/transmittivity [%]')
    ylim(0,100)
    legend(loc='best')
    f.canvas.set_window_title((direction if title is None else title).title())


class FresnelTest(unittest.TestCase):

    def testFresnel2D(self):
        top = geometry.Rectangle(1, 0.5, 'air')
        bottom = geometry.Rectangle(1, 0.5, 'GaAs')
        stack = geometry.Stack2D()
        stack.prepend(top)
        stack.prepend(bottom)
        geo = geometry.Cartesian2D(stack, left='periodic', right='periodic', top='extend', bottom='extend')

        solver = Fourier2D()
        solver.geometry = geo
        solver.size = 3

        solver.polarization = 'El'
        with Refl(solver, 'tran', 'El') as refl:
            for a in angles:
                R, T = refl(a)
                self.assertAlmostEqual(R, R_TE(a), 3)
                self.assertAlmostEqual(T, T_TE(a), 3)

        solver.polarization = 'Et'
        with Refl(solver, 'tran', 'Et') as refl:
            for a in angles:
                R, T = refl(a)
                self.assertAlmostEqual(R, R_TM(a), 3)
                self.assertAlmostEqual(T, T_TM(a), 3)

        if __name__ == '__main__':
            show_plots(solver, 'tran', 'Separated Tran 2D', True)

        solver.polarization = None
        #solver.initialize()
        with Refl(solver, 'long', 'El') as refl:
            for a in angles:
                R, T = refl(a)
                self.assertAlmostEqual(R, R_TM(a), 3)
                self.assertAlmostEqual(T, T_TM(a), 3)

        with Refl(solver, 'long', 'Et') as refl:
            for a in angles:
                R, T = refl(a)
                self.assertAlmostEqual(R, R_TE(a), 3)
                self.assertAlmostEqual(T, T_TE(a), 3)

        with Refl(solver, 'tran', 'El') as refl:
            for a in angles:
                R, T = refl(a)
                self.assertAlmostEqual(R, R_TE(a), 3)
                self.assertAlmostEqual(T, T_TE(a), 3)

        with Refl(solver, 'tran', 'Et') as refl:
            for a in angles:
                R, T = refl(a)
                self.assertAlmostEqual(R, R_TM(a), 3)
                self.assertAlmostEqual(T, T_TM(a), 3)

        if __name__ == '__main__':
            show_plots(solver, 'long', 'Long 2D')
            show_plots(solver, 'tran', 'Tran 2D')


    def testFresnel3D(self):
        top = geometry.Cuboid(1, 1, 0.5, 'air')
        bottom = geometry.Cuboid(1, 1, 0.5, 'GaAs')
        stack = geometry.Stack3D()
        stack.prepend(top)
        stack.prepend(bottom)
        geo = geometry.Cartesian3D(stack, left='periodic', right='periodic', back='periodic', front='periodic', top='extend', bottom='extend')

        solver = Fourier3D()
        solver.geometry = geo
        solver.size = 1,1

        with Refl(solver, 'long', 'El') as refl:
            for a in angles:
                R, T = refl(a)
                self.assertAlmostEqual(R, R_TM(a), 3)
                self.assertAlmostEqual(T, T_TM(a), 3)

        with Refl(solver, 'long', 'Et') as refl:
            for a in angles:
                R, T = refl(a)
                self.assertAlmostEqual(R, R_TE(a), 3)
                self.assertAlmostEqual(T, T_TE(a), 3)

        with Refl(solver, 'tran', 'El') as refl:
            for a in angles:
                R, T = refl(a)
                self.assertAlmostEqual(R, R_TE(a), 3)
                self.assertAlmostEqual(T, T_TE(a), 3)

        with Refl(solver, 'tran', 'Et') as refl:
            for a in angles:
                R, T = refl(a)
                self.assertAlmostEqual(R, R_TM(a), 3)
                self.assertAlmostEqual(T, T_TM(a), 3)

        if __name__ == '__main__':
            show_plots(solver, 'long', 'Long 3D')
            show_plots(solver, 'tran', 'Tran 3D')


if __name__ == '__main__':
    test = unittest.main(exit=False)
    show()
    sys.exit(not test.result.wasSuccessful())
