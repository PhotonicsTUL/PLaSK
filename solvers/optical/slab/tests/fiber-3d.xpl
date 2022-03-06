<plask loglevel="detail">

<defines>
  <define name="m" value="1"/>
  <define name="symx" value="'Ex'"/>
  <define name="symy" value="'Ex'"/>
  <define name="num" value="3"/>
  <define name="size" value="ARRAYID if  ARRAYID is not None else 19"/>
  <define name="rule" value="'new'"/>
  <define name="pml_dist" value="2"/>
  <define name="pml_size" value="1"/>
  <define name="pml_factor" value="1-2j"/>
  <define name="plot_radial" value="True"/>
  <define name="plot_polar" value="False"/>
  <define name="rmax" value="2"/>
  <define name="R" value="1."/>
  <define name="Rtot" value="1."/>
  <define name="period" value="3."/>
  <define name="nr" value="3.5"/>
  <define name="lam" value="1000."/>
</defines>

<materials>
  <material name="core" base="dielectric">
    <Nr>nr</Nr>
  </material>
</materials>

<geometry>
  <cartesian3d name="extended3d" axes="xyz" back="mirror" front="extend" left="mirror" right="extend">
    <clip back="0" left="0">
      <align xcenter="0" ycenter="0">
        <cuboid material="air" dx="{2*Rtot}" dy="{2*Rtot}" dz="2.0"/>
        <cylinder material="core" radius="{R}" height="2.0"/>
      </align>
    </clip>
  </cartesian3d>
  <cartesian3d name="periodic3d" axes="xyz" back="mirror" front="periodic" left="mirror" right="periodic">
    <clip back="0" left="0">
      <align xcenter="0" ycenter="0">
        <cuboid material="air" dx="{period}" dy="{period}" dz="2.0"/>
        <cylinder material="core" radius="{R}" height="2.0"/>
      </align>
    </clip>
  </cartesian3d>
</geometry>

<grids>
  <mesh name="diff" type="ordered">
    <axis start="0.0" stop="6.0" num="{size+1}"/>
  </mesh>
</grids>

<solvers>
  <optical name="FOURIER" solver="Fourier3D" lib="slab">
    <geometry ref="extended3d"/>
    <expansion lam0="{lam}" size="{size}" rule="{rule}"/>
    <mode lam="{lam}" symmetry-long="{symx}" symmetry-tran="{symy}" emission="top"/>
    <pmls factor="{pml_factor}" dist="{pml_dist}" size="{pml_size}"/>
  </optical>
</solvers>

<script><![CDATA[
import unittest

from fiber import Analytic


FOURIER.m = m
eigenmodes = FOURIER.layer_eigenmodes(1)

modes = list(eigenmodes)
for i,mode in enumerate(modes):
    mode.idx = i
    mode.neff = mode.kvert / FOURIER.k0
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


# if __name__ == '__main__':
#     from fiber import make_radial_plot, make_polar_plot
#
#     if rmax is None:
#         rmax = R + FOURIER.pml.dist + FOURIER.pml.size
#     rr = linspace(-rmax, rmax, 10001)
#     z0 = 1.
#
#     if plot_radial:
#         pmsh = mesh.Rectangular2D(rr, [z0])
#         axs = None
#         for i, mode in enumerate(modes[:num]):
#             field = eigenmodes.outLightE(mode.idx, pmsh)
#             magnitude = eigenmodes.outLightMagnitude(mode.idx, pmsh)
#             fm = max(magnitude)
#             if fm != 0:
#                 field /= fm**0.5
#             axs = make_radial_plot(field, m=m, R=R, neff=mode.neff, c=i, axs=axs)
#         window_title("Mode Profiles")
#
#         axs = None
#         for i, neff in enumerate(neffs_analytic):
#             field = analytic.field(neff, rr)
#             axs = make_radial_plot(field, rr, m=m, R=R, neff=neff, c=i, axs=axs)
#         window_title("Analytic Profiles")
#
#
#     if plot_polar:
#         r = linspace(0., rmax, 21)
#         rmesh0 = mesh.Rectangular2D(r, [z0])
#
#         for mode in modes[:num]:
#             Er = eigenmodes.outLightE(mode.idx, rmesh0)
#             make_polar_plot(Er)
#             tight_layout(pad=0.1)
#             window_title(f"Optical Field {mode.neff.real:.4f}{mode.neff.imag:+.2e}j")
#
#         if analytic:
#             for neff in neffs_analytic:
#                 Er = analytic.field(neff, r)[:,:2]
#                 make_polar_plot(Er, r)
#                 tight_layout(pad=0.1)
#                 window_title(f"Analytic Optical Field {neff:.4f}")
#
#     show()
]]></script>

</plask>
