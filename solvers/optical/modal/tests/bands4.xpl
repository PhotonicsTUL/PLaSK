<plask loglevel="detail">

<defines>
  <define name="r" value="0.3"/>
  <define name="h" value="0.6"/>
  <define name="nk" value="12"/>
  <define name="N" value="2"/>
  <define name="step" value="0.0005"/>
  <define name="start" value="step"/>
  <define name="end" value="0.7500"/>
</defines>

<materials>
  <material name="mat" base="semiconductor">
    <Nr>3.5</Nr>
  </material>
</materials>

<geometry>
  <cartesian3d name="main" axes="x,y,z" back="periodic" front="periodic" left="periodic" right="periodic" bottom="air" top="air">
    <stack x="0" y="0">
      <clip name="layer" back="-0.5" front="0.5" left="-0.5" right="0.5">
        <align xcenter="0" ycenter="0" zcenter="0">
          <cuboid material="mat" dx="2.0" dy="2.0" dz="{h/2}"/>
          <cylinder material="air" radius="{r}" height="{h/2}"/>
        </align>
      </clip>
      <zero/>
      <again ref="layer"/>
    </stack>
  </cartesian3d>
</geometry>

<solvers>
  <optical name="OPTICAL" solver="Fourier3D" lib="modal">
    <geometry ref="main"/>
    <expansion lam0="980." size="{N}"/>
    <interface position="0."/>
  </optical>
</solvers>

<script><![CDATA[
from mpi4py.MPI import COMM_WORLD as mpi

msize = mpi.Get_size()
mrank = mpi.Get_rank()


if mrank == 0:
    msh = mesh.Rectangular3D(linspace(-1.5, 1.5, 301), linspace(-1.5, 1.5, 301), [0.])
    epsilon = OPTICAL.outEpsilon(msh, interpolation='nearest')
    plot_field(epsilon, None, plane='yx', comp='x', cmap='Blues')
    plot_geometry(OPTICAL.geometry, plane='yx', color='w', periods=3)
    aspect('equal')
    window_title('Refractive Index')
    xlim(-1.5, 1.5)
    ylim(-1.5, 1.5)


k0s = arange(2*pi*start, 2*pi*end+1e-12, 2*pi*step)


def browse():
    dets = abs(OPTICAL.get_determinant(k0=k0s))
    mins = [k0s[i] for i in range(1, len(dets)-1) if dets[i-1] >= dets[i] and dets[i] < dets[i+1]]
    print_log('data', 'Approx. bands:', ', '.join('{:.2f}'.format(x) for x in mins))
    return mins


def find_mode(k0):
    try:
        return OPTICAL.find_mode(k0=k0)
    except ComputationError:
        return None


# Rectangular lattice: X = [0.5, 0] M = [0.5, 0.5]
sqrt2 = sqrt(2)

# Γ -> X
n = int(nk)
wavevectors = [ (pi * k/n, 0., pi * k/n) for k in range(n) ]

# X -> M
n = int(nk)
wavevectors.extend([ (pi, pi * k/n, pi * (1 + k/n)) for k in range(n) ])

# M -> Γ
n = int(round(nk * sqrt2))
wavevectors.extend( [ (pi * (1-k/n), pi * (1-k/n), pi * (2 + sqrt2*k/n)) for k in range(n+1) ] )

wavevectors = array(wavevectors)

## Do the computations ##

mwavevectors = array_split(wavevectors, msize)

results = []

print_log('data', 'K =', [(k[0], k[1]) for k in mwavevectors[mrank]])

for i, K in enumerate(mwavevectors[mrank]):
    OPTICAL.ktran, OPTICAL.klong = K[:2]

    modes = set(find_mode(k0) for k0 in browse())

    # test for spurious modes (not optimal method, but should work)
    bands = [k0 for k0 in (OPTICAL.modes[m].k0.real for m in modes if m is not None) if k0 >= 0]

    for k0 in bands:
        results.append((K[2], K[0], K[1], k0))

    print_log('important', "{}/{} {:.1f}%".format(mrank, msize, 100.*(i+1)/len(mwavevectors[mrank])), K, ':', ' '.join(str(k0) for k0 in bands))

results = mpi.gather(results)


if mrank == 0:
    results = concatenate(results)

    with open('bands4.out', 'w') as out:
        out.write("#_po__ _kx__ _ky__  omega/c\n")
        for row in results:
            out.write("{:6.3f} {:5.3f} {:5.3f}  {:.4f}\n".format(*row))

    results = results.T

    figure()
    wavevectors = array(wavevectors)
    plot(wavevectors[:,2], sqrt(sum(wavevectors[:,:2]**2, 1)), 'k--', lw=0.7)

    try:
        reference = loadtxt('bands6.dat', unpack=True)
    except IOError:
        pass
    else:
        plot(reference[0], reference[3], '.', color='0.7')

    plot(results[0], results[3], '.', color='maroon', ms=2)

    xticks([0., pi, 2*pi, pi*(2+sqrt2)], ['$\\Gamma$', 'X', 'M', '$\\Gamma$'])
    xlim(0., pi*(2+sqrt2))
    grid(axis='x')

    ylabel("$\\omega/c$")

    tight_layout()
    window_title('Photonic Bands')
    savefig('bands4.png')


    def show_field(kt, kl, k0):
        OPTICAL.ktran, OPTICAL.klong = kt, kl
        m = OPTICAL.find_mode(k0=k0)

        msh = mesh.Rectangular3D(linspace(-1.5, 1.5, 301), linspace(-1.5, 1.5, 301), [0.])
        field = OPTICAL.outLightMagnitude(m, msh)

        figure()
        plot_field(field, None, plane='yx')
        plot_geometry(OPTICAL.geometry, plane='yx', color='w', periods=3)
        aspect('equal')
        gcf().canvas.set_window_title('Field @ {:.3f}'.format(OPTICAL.modes[m].k0.real))

        ka = sqrt(OPTICAL.klong**2 + OPTICAL.ktran**2).real

        plot([0., -OPTICAL.klong.real/ka], [0., OPTICAL.ktran.real/ka], color='m')
        xlim(-1.5, 1.5)
        ylim(1.5, -1.5)

    # show_field(2.033, 1.174, 1.1967)
    # show_field(2.033, 1.174, 2.2450)

    show()
]]></script>

</plask>
