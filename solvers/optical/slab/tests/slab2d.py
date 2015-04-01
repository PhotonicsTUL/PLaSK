#!/home/maciek/Dokumenty/PLaSK/tests/plask
# -*- coding: utf-8 -*-

config.axes = 'xy'

import optical

wn = 1.2
wa = 2.0 - wn
h = 0.10

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

symmetric = True
periodic = False

size = 12
refine = 1
dct = 1

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
    stack = geometry.Stack2D()
else:
    stack = geometry.Stack2D(left=0)

core = geometry.Block2D(wn, h, Glass())
air =  geometry.Block2D(wa, h, material.air)

if not symmetric:
    shelf.append(core)
    shelf.append(air)
    shelf.append(air)
    shelf.append(core)
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
        main = geometry.Cartesian2D(stack, left='mirror')
    else:
        main = geometry.Cartesian2D(stack)

opt = optical.Fourier2D("opt")
opt.geometry = main
opt.wavelength = 980.
opt.smooth = 0.0
opt.size = size
opt.refine = refine

opt.dct = dct

opt.set_interface(shelf, p)

opt.pml.shift = 0.
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

NR = opt.outRefractiveIndex(msh, 'spline')
plot(XX, NR.array[:,0,2].real, 'b', label='spline')

print " ".join("{:.4f}".format(x) for x in opt.material_mesh)

mm = opt.material_mesh
#mm = {
    #(8, True): [0.0000, 0.0531, 0.1062, 0.1594, 0.2125, 0.2656, 0.3187, 0.3719, 0.4250, 0.4781, 0.5312, 0.5844, 0.6375, 0.6906, 0.7438, 0.7969, 0.8500],
    #(8, False): [0.0000, 0.1469, 0.2938, 0.4406, 0.5875, 0.7344, 0.8813, 1.0281, 1.1750, 1.3219, 1.4688, 1.6156, 1.7625, 1.9094, 2.0562, 2.2031, 2.3500],
#}[opt.size, periodic]
NR = opt.outRefractiveIndex(mesh.Rectangular2D(mm, [0.5*h]), 'nearest')
plot(mm, NR.array[:,0,2].real, 'ob', ms=4, alpha=0.5)


xlim(XX[0], XX[-1])
#ylim(0.95, 1.35)

legend(loc='best')

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

