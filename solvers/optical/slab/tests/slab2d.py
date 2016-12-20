#!/home/maciek/Dokumenty/PLaSK/tests/plask
# -*- coding: utf-8 -*-

config.axes = 'xy'

import optical

wn = 1.2
wa = 2.0 - wn
h = 0.10

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

symmetric = True
periodic = True

size = 16
refine = 8
oversampling = 1

smooth = 0.01

dct = 2

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@plask.material.simple()
class Glass(plask.material.Material):
    def nr(self, w, T=300., n=0.):
        return 1.3

@plask.material.simple()
class Asym(plask.material.Material):
    def nr(self, w, T=300., n=0.):
        return 1.1

shelf = geometry.Shelf2D()
if symmetric:
    stack = geometry.Stack2D(left=0)
else:
    stack = geometry.Stack2D(xcenter=0)

core = geometry.Block2D(wn, h, Glass())
air =  geometry.Block2D(wa, h, material.air)

if not symmetric:
    shelf.append(air)
    shelf.append(core)
    shelf.append(core)
    shelf.append(air)
else:
    shelf.append(core)
    shelf.append(air)

stack.append(shelf)
p = stack.append(shelf)

if periodic:
    if symmetric:
        main = geometry.Cartesian2D(stack, left='mirror', right='periodic')
    else:
        main = geometry.Cartesian2D(stack, left='periodic', right='periodic')
else:
    if symmetric:
        main = geometry.Cartesian2D(stack, left='mirror', right='extend')
    else:
        main = geometry.Cartesian2D(stack, right='extend')

opt = optical.Fourier2D("opt")
opt.geometry = main
opt.wavelength = 980.
opt.smooth = smooth
opt.size = size
opt.refine = refine

opt.oversampling = oversampling

opt.dct = dct

opt.set_interface(shelf, p)

opt.pml.dist = 0.
opt.pml.order = 1
opt.pml.factor = 1-2j

right = 3.5
left = -3.5

if symmetric:
    opt.symmetry = 'Ex'

#XX = linspace(left, right, 10)
XX = linspace(left, right, 24000)

msh = mesh.Rectangular2D(XX, [0.5*h])

for i in range(-4, 5):
    axvline((wn+wa)*i, color='0.75')

NR = [main.get_material(x, 0.5*h).nr(opt.wavelength.real).real for x in XX]
plot(XX, NR, '--k')

NR = opt.outRefractiveIndex(msh)
plot(XX, NR.array[:,0,2].real, 'r', label='Fourier')



xlim(XX[0], XX[-1])
ylim(0.95, 1.35)

#legend(loc='best')

tight_layout()

#import os
#if os.getcwd().split(os.sep)[-2] == 'plask':
    #title = "Old symmetric method"
#else:
    #title = "New symmetric method"
#gcf().canvas.set_window_title(title)
gcf().canvas.set_window_title("%s %s" % ('Symmetric' if symmetric else 'Asymmetric', 'periodic' if periodic else ''))

del opt

show()

