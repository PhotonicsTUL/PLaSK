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
L = 6.0
depth = 1.0


symmetric = True

interp = 'fourier'
#interp = 'linear'


provider = solver.outLightE
#provider = solver.outLightH
#provider = solver.outLightMagnitude
comp = 1


solver.size = 1, 32
solver.smooth = 1e-3

solver.vpml.dist = 1.00
solver.vpml.size = 0.50
solver.vpml.factor = 1-2j

solver.pmls.tran.dist = L/2 - 0.25
solver.pmls.tran.size = 0.5
solver.pmls.tran.factor = 1-2j
print_log('data', 'Tran. PML:', solver.pmls.tran)
print_log('data', 'Vert. PML:', solver.vpml)


solver.root.method = 'broyden'

rect_top = geometry.Cuboid(depth, 0.25 if symmetric else 0.5, h_top, 'Glass')
rect_bottom = geometry.Cuboid(depth, 0.25 if symmetric else 0.5, 1.5-h_top, 'Glass')

stack = geometry.Stack3D(shift=-rect_bottom.height, back=0, **({'left':0} if symmetric else {'xcenter':0}))
stack.prepend(rect_top)
stack.prepend(rect_bottom)

spacer = geometry.Cuboid(depth, L/2 if symmetric else L, 2., None)
#stack.prepend(spacer)

space = geometry.Cartesian3D(stack,
                             back='periodic', front='periodic',
                             left='mirror' if symmetric else 'air', right='air',
                             bottom='air', top='air')

solver.geometry = space
solver.set_interface(rect_top)
solver.wavelength = 1000.

def compute(sym):

    if symmetric:
        solver.symmetry = None, sym
    else:
        solver.symmetry = None, None

    start = {'Ez': 1.1433, 'Ex': 1.115}[sym]

    #nn = linspace(1.001, 1.299, 200)
    #dets = solver.get_determinant(klong=nn*solver.k0)
    #figure()
    #plot(nn, abs(dets), 'r', label='PLaSK')
    #xlabel('neff')
    #legend(loc='best')
    #gcf().canvas.set_window_title('Determinant [{}]'.format(sym))
    #show()
    #sys.exit(0)

    mod = solver.find_mode(klong=start*solver.k0)

    solver.modes[mod].power = 1.
    print_log('result', f"neff: {solver.modes[mod].klong/solver.modes[mod].k0}", solver.modes[mod])

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
    gcf().canvas.set_window_title('2D intensity [{}]'.format(sym))
    tight_layout()

    figure()
    msh = mesh.Rectangular3D(ZZ, [0.], YY)
    field = provider(mod, msh, interp).array[0,0,:,comp]
    #plot(field.real, YY)
    #plot(field.imag, YY)
    plot(abs(field), YY)
    bboxes = space.get_leafs_bboxes()
    lines = set()
    for box in bboxes:
        lines.add(box.lower.y)
        lines.add(box.upper.y)
    for line in lines:
        axhline(line, ls=':', color='k')
    axhline(space.bbox.lower.y-solver.vpml.dist, ls=':', color='y')
    axhline(space.bbox.upper.y+solver.vpml.dist, ls=':', color='y')
    gcf().canvas.set_window_title("E{} (y) [{}]".format(config.axes[comp], sym))
    ylim(min(YY), max(YY));
    tight_layout()

    figure()
    msh = mesh.Rectangular3D(ZZ, XX, [h])
    field = provider(mod, msh, interp).array[0,:,0,comp]
    #plot(XX, field.real)
    #plot(XX, field.imag)
    plot(XX, abs(field))
    xlim(XX[0],XX[-1])
    bboxes = space.get_leafs_bboxes()
    lines = set()
    for box in bboxes:
        lines.add(box.lower.x)
        lines.add(box.upper.x)
    for line in lines:
        axvline(line, ls=':', color='k')
    gcf().canvas.set_window_title("E{} (x) [{}]".format(config.axes[comp], sym))
    tight_layout()

    box = space.bbox
    integral_mesh = mesh.Rectangular3D([-1, 1],
                                       #mesh.Regular(-box.right if sym else box.left, box.right, 20001),
                                       mesh.Regular(-lx, lx, 1001),
                                       mesh.Regular(box.bottom, box.top, 1001))
    dx, dy = integral_mesh.axis1[1] - integral_mesh.axis1[0], integral_mesh.axis2[1] - integral_mesh.axis2[0]
    integral_mesh = integral_mesh.elements.mesh

    E = solver.outLightE(mod, integral_mesh).array
    E2 = sum(real(E*conj(E)), -1)
    print_log('result', "E:", 0.5 * sum(E2) * dx * dy / solver.integrateEE(mod, box.bottom, box.top))

    H = solver.outLightH(mod, integral_mesh).array
    H2 = sum(real(H*conj(H)), -1)
    print_log('result', "H:", 0.5 * sum(H2) * dx * dy / solver.integrateHH(mod, box.bottom, box.top))


compute('Ez')
compute('Ex')

for mode in solver.modes:
    print_log('important', f"neff: {mode.klong/mode.k0}", mode)

show()
