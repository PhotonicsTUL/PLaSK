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

# coding: utf-8
# log level: data

from plask import material
from plask import print_log


@material.alloy('AlGaAs')
class AlGaAs_fake(material.Material):

    def VB(self, T=300., e=0., point='G', hole='H'):
        return 2.*T

    def nr(self, lam, T=300., n=0.):
        return 3.5

    def absp(self, lam, T=300, n=0.):
        return 0.


@material.alloy()
class AlGaAsDp(material.Material):

    name = "AlGaAs:Dp"

    def __init__(self, *args, **kwargs):
        super().__init__(doping=kwargs['doping'])
        self.args = args,
        self.kwargs = kwargs
        self._composition = self.complete_composition(kwargs)

    def VB(self, T=300., e=0., point='G', hole='H'):
        return self.kwargs['doping'] * T

    def CB(self, T=300., e=0., point='G'):
        return self._composition['Ga'] * T

    def Eps(self, wl, T, n):
        return (12.25, 12.96, 13.69, 0.01)


@material.simple()
class WithChar(material.Material):
    def chi(self, T, e=0., point='G'):
        print("WithChar: %s" % point)
        return 1.5


print_log('detail', 'Module {} loaded'.format(__name__))
