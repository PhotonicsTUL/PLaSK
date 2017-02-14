<plask loglevel="detail">

<defines>
  <define name="r" value="0.3"/>
  <define name="h" value="0.6"/>
  <define name="nk" value="20"/>
  <define name="N" value="5"/>
  <define name="start" value="0.01"/>
  <define name="end" value="0.500"/>
  <define name="step" value="0.005&#10;"/>
</defines>

<materials>
  <material name="mat" base="semiconductor">
    <Nr>3.5</Nr>
  </material>
</materials>

<geometry>
  <cartesian3d name="main" axes="x,y,z" back="periodic" front="periodic" left="periodic" right="periodic" bottom="air" top="air">
    <stack>
      <align name="layer" xcenter="0" ycenter="0" zcenter="0">
        <cuboid material="mat" dx="{sqrt(3)}" dy="1.0" dz="{h/2}"/>
        <clip back="{-sqrt(3)/2}" front="{sqrt(3)/2}" left="-0.5" right="0.5">
          <lattice ax="0.0" ay="1.0" az="0.0" bx="{sqrt(3)/2}" by="0.5" bz="0.0">
            <segments>0 -1; -1 1; 0 1; 1 -1</segments>
            <cylinder material="air" radius="{r}" height="{h/2}"/>
          </lattice>
        </clip>
      </align>
      <zero/>
      <again ref="layer"/>
    </stack>
  </cartesian3d>
</geometry>

<solvers>
  <optical name="OPTICAL" solver="Fourier3D" lib="slab">
    <geometry ref="main"/>
    <expansion lam0="980." oversampling="8" refine="0" size="2"/>
    <interface position="0."/>
    <lattice long0="0.0" long1="{sqrt(3)/2}" tran0="1.0" tran1="0.5"/>
  </optical>
</solvers>

<script><![CDATA[
# OPTICAL.klong = 2.218
# OPTICAL.ktran = 1.280
# OPTICAL.find_mode(k0=1.2556)
# OPTICAL.find_mode(k0=2.4986)

k0s = arange(2*pi*start, 2*pi*end+1e-12, 2*pi*step)

def browse():
    dets = abs(OPTICAL.get_determinant(k0=k0s))
    mins = [k0s[i] for i in range(1, len(dets)-1) if dets[i-1] >= dets[i] and dets[i] < dets[i+1]]
    print_log('data', 'Approx. bands:', ', '.join('{:.2f}'.format(x) for x in mins))
    return mins


# Triangular lattice: K = [4*pi/(3),0] M = [pi, pi/(sqrt(3))]
sqr3 = sqrt(3)

# Gamma -> M
n = int(sqr3/2 * nk)
wavevectors = [ (pi * k/n, pi/sqr3 * k/n) for k in range(n) ]
graphpos = [ pi * sqr3/2 * k/n for k in range(n) ]

bb = len(graphpos) + 1

# M -> K
n = int(nk/2.)
wavevectors.extend( [ (pi * ( 1 + 1./3.*k/n), pi/sqr3 * (n-k)/n) for k in range(n) ] )
graphpos.extend( [ pi * (sqr3/2 + 0.5 * k/n) for k in range(n) ] )

bb = bb, len(graphpos) + 1


# K -> Gamma
n = nk
wavevectors.extend( [ (4*pi/3 * (n-k)/nk, 0) for k in range(n+1) ] )
graphpos.extend( [ pi * (sqr3/2 + 0.5 + 1. * k/n) for k in range(n+1) ] )



## Do the computations ##

results = []

for K,pos in zip(wavevectors, graphpos):
    OPTICAL.ktran, OPTICAL.klong = K

    modes = set(OPTICAL.find_mode(k0=k0) for k0 in browse())

    # test for spurious modes (not optimal method, but should work)
    bands = [k0 for k0 in (OPTICAL.modes[m].k0.real for m in modes) if k0 >= 0]

    for k0 in bands:
        results.append((pos, K[0], K[1], k0))
    print_log('result', K, ':', ' '.join(str(k0) for k0 in bands))

results = array(results).T

try:
    reference = loadtxt('bands.dat', unpack=True)
except IOError:
    pass
else:
    plot(reference[0], reference[3], '.', color='0.7')
plot(results[0], results[3], '.', color='maroon')
xticks([0., graphpos[bb[0]], graphpos[bb[1]], graphpos[-1]], ['$\\Gamma$', '$M$', '$K$', '$\\Gamma$'])
xlim(0., graphpos[-1])
grid(axis='x')
ylabel("$\\omega/c$")
tight_layout(0.1)
savefig('bandgap.png')
show()
]]></script>

</plask>
