<plask loglevel="data">

<defines>
  <define name="lam" value="850."/>
  <define name="symt" value="True"/>
  <define name="syml" value="False"/>
  <define name="L" value="3.0"/>
  <define name="Nt" value="8"/>
  <define name="Nl" value="8"/>
  <define name="ft" value="0.5"/>
  <define name="fl" value="0.9"/>
  <define name="h" value="0.3"/>
</defines>

<materials>
  <material name="slab" base="semiconductor">
    <Nr>3.5</Nr>
  </material>
  <material name="absorber" base="semiconductor">
    <Nr>3.5 - 0.02j</Nr>
  </material>
  <material name="clad" base="semiconductor">
    <Nr>1.5</Nr>
  </material>
</materials>

<geometry>
  <cartesian3d name="struct" axes="x,y,z" back="mirror" front="periodic" left="mirror" right="periodic" bottom="clad">
    <clip back="0" left="0">
      <stack xcenter="0" ycenter="0">
        <stack name="grating">
          <cuboid material="slab" dx="{fl*L}" dy="{ft*L}" dz="0.4"/>
          <cuboid material="slab" dx="{L}" dy="{L}" dz="0.4"/>
        </stack>
        <cuboid name="absorber" material="absorber" dx="{L}" dy="{L}" dz="0.2"/>
        <cuboid material="slab" dx="{L}" dy="{L}" dz="0.4"/>
      </stack>
    </clip>
  </cartesian3d>
</geometry>

<solvers>
  <optical name="FOURIER3D" solver="Fourier3D" lib="slab">
    <geometry ref="struct"/>
    <expansion size-long="{Nl-1}" size-tran="{Nt-1}" rule="semi"/>
    <mode lam="{lam}" symmetry-long="{'Ey' if syml else 'none'}" symmetry-tran="{'Ey' if symt else 'none'}"/>
  </optical>
</solvers>

<script><![CDATA[
import unittest

side = 'bottom'

gbox = FOURIER3D.geometry.bbox
abox = GEO.struct.get_object_bboxes(GEO.absorber)[0]
ibox = GEO.struct.get_object_bboxes(GEO.grating)[0]

nx = 20
ny = 20
nz = 80

amesh = mesh.Rectangular3D(
    mesh.Regular(-gbox.front, gbox.front, nx+1),
    mesh.Regular(-gbox.right, gbox.right, ny+1),
    mesh.Regular(abox.bottom, abox.top, nz+1)
).elements.mesh

imesh = mesh.Rectangular3D(
    mesh.Regular(-gbox.front, gbox.front, nx+1),
    mesh.Regular(-gbox.right, gbox.right, ny+1),
    mesh.Regular(ibox.bottom, ibox.top, nz+1)
).elements.mesh

FOURIER3D.lam = lam
comp = FOURIER3D.scattering(side, 'Et')


class IntegralsTest(unittest.TestCase):

  def test_integrals_E(self):
      EEn = sum(comp.outLightMagnitude(imesh)) * (ibox.top - ibox.bottom) / (nx * ny * nz)
      # En = comp.outLightE(imesh).array
      # EEn = 0.5/phys.eta0 * sum(real(En * conj(En))) * (ibox.top - ibox.bottom) / (nx * ny * nz)
      EEa = 0.25/phys.eta0 * comp.integrateEE(ibox.bottom, ibox.top) / (ibox.front * ibox.right)
      self.assertAlmostEqual(EEn, EEa, 3)

  def test_integrals_H(self):
      Hn = comp.outLightH(imesh).array
      HHn = 0.5*phys.eta0 * sum(real(Hn * conj(Hn))) * (ibox.top - ibox.bottom) / (nx * ny * nz)
      HHa = 0.25*phys.eta0 * comp.integrateHH(ibox.bottom, ibox.top) / (ibox.front * ibox.right)
      self.assertAlmostEqual(HHn, HHa, 2)

  def test_absorption_numeric(self):
      EEn = sum(comp.outLightMagnitude(amesh)) * (abox.top - abox.bottom) / (nx * ny * nz)
      eps = material.get('absorber').Nr(lam)**2
      nclad = material.get('clad').Nr(lam).real
      A = 1. - comp.R - comp.T
      An = 2e3 * pi / lam * abs(eps.imag) * EEn / nclad
      self.assertAlmostEqual(A, An, 3)

  def test_absorption_analytic(self):
      EEa = 0.25/phys.eta0 * comp.integrateEE(abox.bottom, abox.top) / (abox.front * abox.right)
      eps = material.get('absorber').Nr(lam)**2
      nclad = material.get('clad').Nr(lam).real
      A = 1. - comp.R - comp.T
      Aa = 2e3 * pi / lam * abs(eps.imag) * EEa / nclad
      self.assertAlmostEqual(A, Aa, 4)


if __name__ == '__main__':
    import __main__
    __main__.IntegralsTest = IntegralsTest
    unittest.main()
]]></script>

</plask>
