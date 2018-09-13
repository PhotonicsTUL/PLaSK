<plask loglevel="detail">

<defines>
  <define name="theta" value="60"/>
  <define name="phi" value="30"/>
  <define name="side" value="'bottom'"/>
  <define name="dim" value="2"/>
</defines>

<materials>
  <material name="GaAs" base="GaAs">
    <NR>3.5, 3.6, 3.5, 0</NR>
  </material>
</materials>

<geometry>
  <cartesian2d name="main" axes="xyz" left="mirror" right="periodic">
    <stack name="structure" left="0">
      <shelf>
        <rectangle material="air" dy="0.6" dz="0.28"/>
        <rectangle material="GaAs" dy="0.4" dz="0.28"/>
      </shelf>
    </stack>
  </cartesian2d>
  <cartesian3d name="main3d" axes="x,y,z" back="mirror" front="periodic" left="mirror" right="periodic">
    <extrusion length="1">
      <again ref="structure"/>
    </extrusion>
  </cartesian3d>
</geometry>

<solvers>
  <optical name="OPTICAL2D" solver="Fourier2D" lib="slab">
    <geometry ref="main"/>
    <expansion lam0="980.0" size="4" smooth="8e-5"/>
    <mode wavelength="980.0"/>
    <vpml dist="1500"/>
  </optical>
  <optical name="OPTICAL3D" solver="Fourier3D" lib="slab">
    <geometry ref="main3d"/>
    <expansion lam0="980.0" size-long="4" size-tran="4" smooth="8e-5"/>
    <mode wavelength="980.0"/>
    <vpml dist="1500"/>
  </optical>
</solvers>

<script><![CDATA[
import unittest

plask.config.axes = 'xyz'


class TiltedTest(unittest.TestCase):

    def doTests(self, solver):
        for pol in ('Ex', 'Ey'):
            s = solver.scattering(side, pol)
            R = s.R
            T = s.T
            self.assertAlmostEqual(R+T, 1.000, 3)

    def test2D(self):
        self.doTests(OPTICAL2D)

    def test3D(self):
        self.doTests(OPTICAL2D)


if __name__ == '__main__':
    import __main__
    __main__.TiltedTest = TiltedTest
    unittest.main()
]]></script>

</plask>
