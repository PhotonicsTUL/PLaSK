# coding=utf8

import sys

config.axes = 'zxy'
config.log.level = 'detail'

from optical.slab import Fourier3D

@material.simple()
class Glass(material.Material):
    def Nr(self, wl, T, n): return 1.3

solver = Fourier3D("solver")


h_top = 0.1
L = 3.0
depth = 1.0


#sym = None
#sym = 'Ez'
sym = 'Ex'

#interp = 'fourier'
interp = 'nearest'


provider = solver.outElectricField
#provider = solver.outMagneticField
#provider = solver.outLightMagnitude
comp = 2


solver.size = 1, 32
solver.smooth = 1e-3

solver.vpml.dist = 1.00
solver.vpml.size = 0.50
solver.vpml.factor = 1.-2j

solver.pmls.tran.dist = L/2 - 0.25
solver.pmls.tran.size = 0.5
solver.pmls.tran.factor = 1.-2j
print solver.pmls.tran, solver.vpml


start = {'Ez': 1.144, 'Ex': 1.115, None: 1.144}[sym]


solver.root.method = 'broyden'


rect_top = geometry.Cuboid(depth, 0.25 if sym else 0.5, h_top, 'Glass')
rect_bottom = geometry.Cuboid(depth, 0.25 if sym else 0.5, 1.5-h_top, 'Glass')

stack = geometry.Stack3D(shift=-rect_bottom.height, back=0, **({'left':0} if sym else {'xcenter':0}))
stack.prepend(rect_top)
stack.prepend(rect_bottom)

spacer = geometry.Cuboid(depth, L/2 if sym else L, 0., None)
#stack.append(spacer)

space = geometry.Cartesian3D(stack,
                             back='periodic', front='periodic',
                             left='mirror' if sym else 'air', right='air',
                             bottom='air', top='air')

solver.geometry = space
solver.set_interface(rect_top)
solver.symmetry = None, sym

solver.wavelength = 1000.


#nn = linspace(1.001, 1.299, 200)
#dets = solver.get_determinant(klong=nn*solver.k0)
#figure()
#plot(nn, abs(dets), 'r', label='PLaSK')
#xlabel('neff')
#legend(loc='best')
#gcf().canvas.set_window_title('Determinant')
#show()
#sys.exit(0)

mod = solver.find_mode(klong=start*solver.k0)

solver.modes[mod].power = 1.
print solver.modes[mod].klong/solver.modes[mod].k0, solver.modes[mod]

lx = (solver.pmls.tran.size + L/2.)
ly_b = - rect_bottom.height - solver.vpml.dist - solver.vpml.size
ly_t = rect_top.height + solver.vpml.dist + solver.vpml.size

XX = linspace(-lx, lx, 201)
YY = linspace(ly_b, ly_t, 201)
ZZ = [0.]

h = h_top - stack.bbox.height / 2

figure()
msh = mesh.Rectangular3D(ZZ, XX, YY)
field = solver.outLightMagnitude(mod, msh, interp)
#field = Data(sum(provider(mod, msh, interp).array[0,:,:,:]**2, 2), msh)
im = plot_field(field, plane='xy')
colorbar(im, use_gridspec=True)
plot_geometry(solver.geometry, 'w', mirror=True, plane='xy')
axvline(0, color='w', ls=':')
axhline(h, color='w', ls=':')
gcf().canvas.set_window_title('2D intensity [PLaSK]')
tight_layout()

figure()
msh = mesh.Rectangular3D(ZZ, [0.], YY)
field = provider(mod, msh, interp).array[0,0,:,comp]
plot(field.real, YY, 'r')
plot(field.imag, YY, 'b')
#plot(abs(field), YY, 'r')
bboxes = space.get_leafs_bboxes()
lines = set()
for box in bboxes:
    lines.add(box.lower.y)
    lines.add(box.upper.y)
for line in lines:
    axhline(line, ls=':', color='k')
axhline(space.bbox.lower.y-solver.vpml.dist, ls=':', color='y')
axhline(space.bbox.upper.y+solver.vpml.dist, ls=':', color='y')
gcf().canvas.set_window_title("E{} (y) [PLaSK]".format(config.axes[comp]))
ylim(min(YY), max(YY));
tight_layout()

figure()
msh = mesh.Rectangular3D(ZZ, XX, [h])
field = provider(mod, msh, interp).array[0,:,0,comp]
plot(XX, field.real, 'r')
plot(XX, field.imag, 'b')
#plot(XX, abs(field), 'r')
xlim(XX[0],XX[-1])
bboxes = space.get_leafs_bboxes()
lines = set()
for box in bboxes:
   lines.add(box.lower.x)
   lines.add(box.upper.x)
for line in lines:
   axvline(line, ls=':', color='k')
gcf().canvas.set_window_title("E{} (x) [PLaSK]".format(config.axes[comp]))
tight_layout()


show()
