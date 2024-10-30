<plask loglevel="detail">

<defines>
  <define name="m" value="1"/>
  <define name="num" value="3"/>
  <define name="size" value="4*ARRAYID if  ARRAYID is not None else 72"/>
  <define name="domain" value="'infinite'"/>
  <define name="rule" value="'direct'"/>
  <define name="kspacing" value="'nonuniform'"/>
  <define name="kscale" value="0.7"/>
  <define name="kmax" value="4"/>
  <define name="pml_dist" value="10"/>
  <define name="pml_size" value="5"/>
  <define name="pml_factor" value="1-2j"/>
  <define name="plot_radial" value="True"/>
  <define name="plot_polar" value="False"/>
  <define name="rmax" value="2"/>
  <define name="R" value="1."/>
  <define name="Rtot" value="1."/>
  <define name="nr" value="3.5"/>
  <define name="lam" value="1000."/>
  <define name="integral_error" value="1e-12"/>
  <define name="integral_points" value="4000"/>
</defines>

<materials>
  <material name="core" base="dielectric">
    <Nr>nr</Nr>
  </material>
</materials>

<geometry>
  <cylindrical2d name="optical" axes="rz" outer="air" bottom="extend" top="extend">
    <shelf>
      <rectangle name="disk" material="core" dr="{R}" dz="2.0"/>
      <rectangle material="air" dr="{Rtot-R}" dz="2.0"/>
    </shelf>
  </cylindrical2d>
</geometry>

<grids>
  <mesh name="diff" type="ordered">
    <axis start="0.0" stop="6.0" num="{size+1}"/>
  </mesh>
</grids>

<solvers>
  <optical name="BESSEL" solver="BesselCyl" lib="modal">
    <geometry ref="optical"/>
    <expansion lam0="{lam}" domain="{domain}" size="{size}" integrals-error="{integral_error}" integrals-points="{integral_points}" k-method="{kspacing}" k-max="{kmax}" k-scale="{kscale}" rule="{rule}"/>
    <mode lam="{lam}" emission="top"/>
    <pml factor="{pml_factor}" shape="2" dist="{pml_dist}" size="{pml_size}"/>
  </optical>
</solvers>

<script><![CDATA[
import unittest

from fiber import Analytic


BESSEL.m = m
eigenmodes = BESSEL.layer_eigenmodes(1)

modes = list(eigenmodes)
for i,mode in enumerate(modes):
    mode.idx = i
    mode.neff = mode.kvert / BESSEL.k0
modes.sort(key=lambda m: - m.kvert.real)
# modes.sort(key=lambda m: - m.kvert.imag)

neffs_numeric = [mode.neff for mode in modes[:num]]

analytic = Analytic(R, nr, m, lam)
neffs_analytic = analytic[:num]


print("Exact:  ", end="")
for neff in neffs_analytic:
    print(f"{neff.real:8.6f}", end="  ")
print()

print("Result: ", end="")
for neff in neffs_numeric:
    print(f"{neff.real:8.6f}", end="  ")
print()



class FiberTest(unittest.TestCase):

    def testModes(self):
        global neffs_numeric, neffs_analytic
        for n, a in zip(neffs_numeric, neffs_analytic):
            self.assertAlmostEqual(n, a, 2)


if __name__ == '__main__':
    from fiber import make_radial_plot, make_polar_plot

    if rmax is None:
        rmax = R + BESSEL.pml.dist + BESSEL.pml.size
    rr = linspace(-rmax, rmax, 10001)
    z0 = 1.

    if plot_radial:
        pmsh = mesh.Rectangular2D(rr, [z0])
        axs = None
        for i, mode in enumerate(modes[:num]):
            field = eigenmodes.outLightE(mode.idx, pmsh)
            magnitude = eigenmodes.outLightMagnitude(mode.idx, pmsh)
            fm = max(magnitude)
            if fm != 0:
                field /= fm**0.5
            axs = make_radial_plot(field, m=m, R=R, neff=mode.neff, c=i, axs=axs)
        window_title("Mode Profiles")

        axs = None
        for i, neff in enumerate(neffs_analytic):
            field = analytic.field(neff, rr)
            axs = make_radial_plot(field, rr, m=m, R=R, neff=neff, c=i, axs=axs)
        window_title("Analytic Profiles")


    if plot_polar:
        r = linspace(0., rmax, 21)
        rmesh0 = mesh.Rectangular2D(r, [z0])

        for mode in modes[:num]:
            Er = eigenmodes.outLightE(mode.idx, rmesh0)
            make_polar_plot(Er)
            tight_layout(pad=0.1)
            window_title(f"Optical Field {mode.neff.real:.4f}{mode.neff.imag:+.2e}j")

        if analytic:
            for neff in neffs_analytic:
                Er = analytic.field(neff, r)[:,:2]
                make_polar_plot(Er, r)
                tight_layout(pad=0.1)
                window_title(f"Analytic Optical Field {neff:.4f}")

    show()
]]></script>

</plask>
