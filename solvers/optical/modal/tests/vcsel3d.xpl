<plask loglevel="detail">

<defines>
  <define name="nxy" value="50"/>
  <define name="nz" value="20"/>
</defines>

<materials>
  <material name="GaAs" base="semiconductor">
    <nr>3.53</nr>
    <absp>0.</absp>
  </material>
  <material name="AlGaAs" base="semiconductor">
    <nr>3.08</nr>
    <absp>0.</absp>
  </material>
  <material name="AlAs" base="semiconductor">
    <nr>2.95</nr>
    <absp>0.</absp>
  </material>
  <material name="AlOx" base="semiconductor">
    <nr>1.53</nr>
    <absp>0.</absp>
  </material>
  <material name="InGaAs" base="semiconductor">
    <nr>3.53</nr>
    <absp>0.</absp>
  </material>
</materials>

<geometry>
  <cartesian3d name="vcsel" axes="xyz" back="mirror" front="extend" left="mirror" right="extend" bottom="GaAs">
    <clip back="0" left="0">
      <stack name="layers" xcenter="0" ycenter="0">
        <cuboid material="GaAs" dx="10" dy="10" dz="0.06949"/>
        <stack name="top-dbr" repeat="24">
          <cuboid material="AlGaAs" dx="10" dy="10" dz="0.07955"/>
          <cuboid material="GaAs" dx="10" dy="10" dz="0.06949"/>
        </stack>
        <cuboid name="x1" material="AlGaAs" dx="10" dy="10" dz="0.06371"/>
        <align name="oxide-layer" bottom="0">
          <item xcenter="0" ycenter="0">
            <cuboid material="AlOx" dx="10" dy="10" dz="0.01593"/>
          </item>
          <item xcenter="0" ycenter="0">
            <cylinder material="AlAs" radius="4" height="0.01593"/>
          </item>
        </align>
        <cuboid name="x" material="AlGaAs" dx="10" dy="10" dz="0.00000"/>
        <cuboid material="GaAs" dx="10" dy="10" dz="0.13649"/>
        <align name="QW" bottom="0">
          <item xcenter="0" ycenter="0">
            <cuboid material="InGaAs" dx="10" dy="10" dz="0.00500"/>
          </item>
          <item xcenter="0" ycenter="0">
            <cylinder name="active" role="gain" material="InGaAs" radius="4" height="0.00500"/>
          </item>
        </align>
        <zero/>
        <cuboid material="GaAs" dx="10" dy="10" dz="0.13649"/>
        <stack name="bottom-dbr" repeat="29">
          <cuboid material="AlGaAs" dx="10" dy="10" dz="0.07955"/>
          <cuboid material="GaAs" dx="10" dy="10" dz="0.06949"/>
        </stack>
        <cuboid material="AlGaAs" dx="10" dy="10" dz="0.07955"/>
      </stack>
    </clip>
  </cartesian3d>
</geometry>

<solvers>
  <optical name="FOURIER3D" solver="Fourier3D" lib="modal">
    <geometry ref="vcsel"/>
    <expansion lam0="980"/>
    <interface object="QW"/>
  </optical>
</solvers>

<script><![CDATA[
import unittest

from numpy import *

from plask import *
from plask import material, geometry, mesh
from optical import modal

plask.config.axes = "xyz"


class VCSEL:

    def setUp(self):
        self.profile = StepProfile(FOURIER3D.geometry)
        FOURIER3D.inGain = self.profile.outGain
        FOURIER3D.refine = 32
        FOURIER3D.emission = 'top'
        FOURIER3D.size = 5
        FOURIER3D.root.method = 'broyden'
        FOURIER3D.symmetry = 'Ex', 'Ex'


class DirectRule(VCSEL, unittest.TestCase):

    def testComputations(self):    # 979.686-0.0210j
        FOURIER3D.rule = 'direct'
        m = FOURIER3D.find_mode(lam=979.75)
        self.assertEqual(m, 0)
        self.assertEqual(len(FOURIER3D.modes), 1)
        self.assertAlmostEqual(FOURIER3D.modes[m].lam, 979.678-0.0208j, 3)

    def testIntegrals(self):
        FOURIER3D.rule = 'direct'
        FOURIER3D.invalidate()
        FOURIER3D.find_mode(lam=979.75)

        z1 = GEO.vcsel.get_object_bboxes(GEO.bottom_dbr)[0].top
        z2 = GEO.vcsel.get_object_bboxes(GEO.x1)[0].top

        imesh = mesh.Rectangular3D(
            mesh.Regular(-5, 5, nxy+1),
            mesh.Regular(-5, 5, nxy+1),
            mesh.Regular(z1, z2, nz+1)
        ).elements.mesh

        EEn = phys.eta0 * sum(FOURIER3D.outLightMagnitude(imesh)) * (z2 - z1) / (nxy**2 * nz)
        EEa = FOURIER3D.integrateEE(z1, z2) / 100.
        self.assertAlmostEqual(EEa / EEn, 1., 3)

        Hn = FOURIER3D.outLightH(imesh).array
        HHn = 0.5 * sum(real(Hn * conj(Hn))) * (z2 - z1) / (nxy**2 * nz)
        HHa = FOURIER3D.integrateHH(z1, z2) / 100.
        self.assertAlmostEqual(HHa / HHn, 1., 3)


class InverseRule(VCSEL, unittest.TestCase):

    def testComputations(self):    # 979.686-0.0210j
        FOURIER3D.rule = 'inverse'
        m = FOURIER3D.find_mode(lam=979.75)
        self.assertEqual(m, 0)
        self.assertEqual(len(FOURIER3D.modes), 1)
        self.assertAlmostEqual(FOURIER3D.modes[m].lam, 979.671-0.0209j, 3)


class CombinedInverseRule(VCSEL, unittest.TestCase):

    def testComputations(self):
        FOURIER3D.rule = 'combined'
        m = FOURIER3D.find_mode(lam=979.75)
        self.assertEqual(m, 0)
        self.assertEqual(len(FOURIER3D.modes), 1)
        self.assertAlmostEqual(FOURIER3D.modes[m].lam, 979.691-0.0263j, 3)


class OldRule(VCSEL, unittest.TestCase):

    def testComputations(self):
        FOURIER3D.rule = 'old'
        m = FOURIER3D.find_mode(lam=979.75)
        self.assertEqual(m, 0)
        self.assertEqual(len(FOURIER3D.modes), 1)
        self.assertAlmostEqual(FOURIER3D.modes[m].lam, 979.678-0.0210j, 3)


if __name__ == "__main__":
    vcsel = VCSEL()
    vcsel.setUp()

    FOURIER3D.initialize()
    z0 = 0.

    box = FOURIER3D.geometry.bbox
    msh = mesh.Rectangular3D([0.], mesh.Regular(0., 7., 201), mesh.Regular(box.lower.z, box.upper.z, 1001))
    mshr = mesh.Rectangular3D([0.], mesh.Regular(0., 7., 201), [z0])

    #lams = linspace(977., 981., 201)
    #dets = FOURIER3D.get_determinant(lam=lams)
    #plot(lams, abs(dets))
    #yscale('log')

    modes = FOURIER3D.find_mode(lam=979.75), FOURIER3D.find_mode(lam=978.48),

    for m in modes:
        print_log('result', FOURIER3D.modes[m])

        figure()
        plot_field(FOURIER3D.outLightMagnitude(m, msh, 'fourier'), plane='yz')

        figure()
        plot_profile(FOURIER3D.outLightMagnitude(m, mshr, 'fourier') / FOURIER3D.modes[m].power)
        axvline(box.upper.y, ls=':', color='k')
        axvline(box.upper[1] + FOURIER3D.pmls[1].dist, ls=':', color='k')
        axvline(box.upper[1] + FOURIER3D.pmls[1].dist + FOURIER3D.pmls[1].size, ls=':', color='k')
        xlim(mshr.axis1[0], mshr.axis1[-1])

    show()
]]></script>

</plask>
