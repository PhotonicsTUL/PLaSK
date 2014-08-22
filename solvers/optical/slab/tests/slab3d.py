#!/home/maciek/Dokumenty/PLaSK/tests/plask
# -*- coding: utf-8 -*-

config.axes = 'xyz'

import optical

depth = 1.0

w = 0.20
wa = 0.10
h = 0.10

d = 0.

axis = 0

symmetric = False
periodic = False



@plask.material.simple
class Glass(plask.material.Material):
    def nr(self, w, T=300., n=0.):
        return 1.3

@plask.material.simple
class Asym(plask.material.Material):
    def nr(self, w, T=300., n=0.):
        return 1.1


def dims(w, h):
    if axis == 0: return vec(w, depth, h)
    elif axis == 1: return vec(depth, w, h)
    raise ValueError("wrong axis")

def pos(w, z=0):
    if axis == 0: return vec(w, 0, z)
    elif axis == 1: return vec(0, w, z)
    raise ValueError("wrong axis")


shelf = geometry.Container3D()

if symmetric:
    stack = geometry.Stack3D(left=0, back=0)
else:
    stack = geometry.Stack3D(xcenter=0, ycenter=0)

core = geometry.Block3D(dims(w, h), Glass())
asym = geometry.Block3D(dims(wa, h), material.air)
air =  geometry.Block3D(dims(w, h), material.air)

air1 =  geometry.Block3D(dims(w-d, h), material.air)
air2 =  geometry.Block3D(dims(w+d, h), material.air)

if not symmetric:
    shelf.append(air1, pos(-2*w-d))
    shelf.append(core, pos(-w))
    shelf.append(core, pos(0))
    shelf.append(air2, pos(w))
else:
    shelf.append(core, pos(0))
    shelf.append(air, pos(w))

stack.append(shelf)
p = stack.append(shelf)


borders = dict(left='periodic', right='periodic', back='periodic', front='periodic')
b0, b1 = (('back', 'front'), ('left', 'right'))[axis]
if not periodic: borders[b0] = borders[b1] = None
if symmetric: borders[b0] = 'mirror'

main = geometry.Cartesian3D(stack, **borders)

opt = optical.Fourier3D("opt")
opt.geometry = main
opt.wavelength = 980.
opt.smooth = 0.00025
opt.size = 0
opt.size[axis] = 24
opt.smooth = 0.
#opt.refine = 8

opt.set_interface(shelf, p)

opt.pmls.tran.dist = 0.5
opt.pmls.tran.size = 0.5
opt.pmls.tran.order = 1
opt.pmls.tran.factor = 1-2j

opt.pmls.long = opt.pmls.tran

right = 2.0
left = -2.0

#AX = linspace(left, right, 10)
AX = linspace(left, right, 4000)

if axis == 0:
    msh = mesh.Rectilinear3D(AX, [0], [0.5*h])
elif axis == 1:
    msh = mesh.Rectilinear3D([0], AX, [0.5*h])
else:
    raise ValueError("wrong axis")

NR = [main.get_material(pos(ax, 0.5*h)).nr(opt.wavelength.real).real for ax in AX]
plot(AX, NR, '--k')

NR = opt.outRefractiveIndex(msh, 'fourier')
plot(AX, NR.array[:,:,0,2].ravel().real, 'r', label='Fourier')

#NR = [main.get_material(pos(ax, 0.5*h)).nr(opt.wavelength.real).real for ax in opt.tran_mesh]
#plot(opt.tran_mesh, NR, 'b.')

#opt.invalidate()
#opt.symmetry = 'Ex'
#NR = opt.outRefractiveIndex(msh)
#plot(AX, NR.array[:,:,0,2].ravel().real, 'r', label='Symmetric')

#NR = [main.get_material(pos(ax, 0.5*h)).nr(opt.wavelength.real).real for ax in opt.tran_mesh]
#plot(opt.tran_mesh, NR, 'r.')

NR = opt.outRefractiveIndex(msh, 'nearest')
plot(AX, NR.array[:,:,0,2].ravel().real, 'b', label='nearest')

xlim(AX[0], AX[-1])
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

