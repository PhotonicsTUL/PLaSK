<plask loglevel="detail">

<defines>
  <define name="ns" value="3.48"/>
  <define name="nh" value="3.48"/>
  <define name="nl" value="1.00"/>
  <define name="L" value="1.00"/>
  <define name="tg" value="0.20"/>
  <define name="tl" value="0.83"/>
  <define name="fill" value="0.4"/>
  <define name="N" value="12"/>
</defines>

<materials>
  <material name="Subs" base="dielectric">
    <nr>{ns}</nr>
  </material>
  <material name="Hi" base="dielectric">
    <nr>{nh}</nr>
  </material>
  <material name="Lo" base="dielectric">
    <nr>{nl}</nr>
  </material>
</materials>

<geometry>
  <cartesian2d name="grating" axes="xy" left="mirror" right="periodic" bottom="Subs">
    <clip left="0" right="{L/2}">
      <stack xcenter="0">
        <rectangle name="bar" material="Hi" dx="{fill*L}" dy="{tg}"/>
        <rectangle material="Lo" dx="{2*L}" dy="{tl}"/>
      </stack>
    </clip>
  </cartesian2d>
</geometry>

<grids>
  <mesh name="plot" type="rectangular2d">
    <axis0 start="{-1.5*L}" stop="{1.5*L}" num="501"></axis0>
    <axis1 start="-0.5" stop="{tl+tg+2.0}" num="501"></axis1>
  </mesh>
</grids>

<solvers>
  <optical name="solver" solver="Fourier2D" lib="slab">
    <geometry ref="grating"/>
    <expansion lam0="1000." size="{N}" smooth="0"/>
    <interface object="bar"/>
  </optical>
</solvers>

<script><![CDATA[
import unittest


class GratingTest(unittest.TestCase):

    def setUp(self):
        solver.lam0 = 1500.

    def testTE_separated_asymmetric(self):
        solver.polarization = 'El'
        solver.symmetry = None
        r_te = solver.compute_reflectivity(array([1500., 1600.]), 'top', 'El')
        self.assertAlmostEqual( r_te[0], 99.934, 2 )
        self.assertAlmostEqual( r_te[1], 98.878, 2 )

    def testTE_separated_symmetric(self):
        solver.polarization = 'El'
        solver.symmetry = 'El'
        r_te = solver.compute_reflectivity(array([1500., 1600.]), 'top', 'El')
        self.assertAlmostEqual( r_te[0], 99.934, 2 )
        self.assertAlmostEqual( r_te[1], 98.864, 2 )

    def testTM_separated_asymmetric(self):
        solver.polarization = 'Et'
        solver.symmetry = None
        r_tm = solver.compute_reflectivity(array([1298., 1344.]), 'top', 'Et')
        self.assertAlmostEqual( r_tm[0], 98.504, 2 )
        self.assertAlmostEqual( r_tm[1], 28.228, 2 )

    def testTM_separated_symmetric(self):
        solver.polarization = 'Et'
        solver.symmetry = 'Et'
        r_tm = solver.compute_reflectivity(array([1298., 1344.]), 'top', 'Et')
        self.assertAlmostEqual( r_tm[0], 98.640, 2 )
        self.assertAlmostEqual( r_tm[1], 28.228, 2 )

    def testTE_asymmetric(self):
        solver.polarization = None
        solver.symmetry = None
        r_te = solver.compute_reflectivity(array([1500., 1600.]), 'top', 'El')
        self.assertAlmostEqual( r_te[0], 99.934, 2 )
        self.assertAlmostEqual( r_te[1], 98.878, 2 )

    def testTE_symmetric(self):
        solver.polarization = None
        solver.symmetry = 'El'
        r_te = solver.compute_reflectivity(array([1500., 1600.]), 'top', 'El')
        self.assertAlmostEqual( r_te[0], 99.934, 2 )
        self.assertAlmostEqual( r_te[1], 98.864, 2 )

    def testTM_asymmetric(self):
        solver.polarization = None
        solver.symmetry = None
        r_tm = solver.compute_reflectivity(array([1298., 1344.]), 'top', 'Et')
        self.assertAlmostEqual( r_tm[0], 98.504, 2 )
        self.assertAlmostEqual( r_tm[1], 28.228, 2 )

    def testTM_symmetric(self):
        solver.polarization = None
        solver.symmetry = 'Et'
        r_tm = solver.compute_reflectivity(array([1298., 1344.]), 'top', 'Et')
        self.assertAlmostEqual( r_tm[0], 98.640, 2 )
        self.assertAlmostEqual( r_tm[1], 28.228, 2 )


if __name__ == '__main__':
    import __main__
    __main__.GratingTest = GratingTest
    unittest.main()
]]></script>

</plask>
