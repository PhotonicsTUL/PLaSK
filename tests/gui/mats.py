#!/usr/bin/env python
# -*- coding: utf-8 -*-
from plask import material
from plask import print_log

@material.complex('AlGaAs')
class AlGaAs_fake(material.Material):
    def VB(self, T=300., e=0., point='G', hole='H'):
        return 2.*T
    def nr(self, wl, T=300., n=0.):
        return 3.5
    def absp(self, wl, T):
        return 0.

@material.complex()
class AlGaAsDp(material.Material):
    name = "AlGaAs:Dp"
    def __init__(self, *args, **kwargs):
        super(AlGaAsDp, self).__init__()
        self.args = args,
        self.kwargs = kwargs
        self.composition = self.complete_composition(kwargs, self.name);
    def VB(self, T=300., e=0., point='G', hole='H'):
        return self.kwargs['dc'] * T
    def CB(self, T=300., e=0., point='G'):
        return self.composition['Ga'] * T
    def NR(self, wl, T, n):
        return (3.5, 3.6, 3.7, 0.1)

@material.simple()
class WithChar(material.Material):
    def chi(self, T, e, p):
        print("WithChar: %s" % p)
        return 1.5

print_log('detail', 'Module {} loaded'.format(__name__))
