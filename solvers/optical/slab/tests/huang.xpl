<plask loglevel="info">

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
  <cartesian2d name="grating" axes="xy" left="periodic" right="periodic" bottom="Subs">
    <clip left="{-L/2}" right="{L/2}">
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

    def testComputations(self):
        l_te = array([1500., 1600.])
        solver.lam0 = 1500.
        r_te = solver.compute_reflectivity(l_te, 'El', 'top')
        self.assertAlmostEqual( r_te[0], 99.934, 2 )
        self.assertAlmostEqual( r_te[1], 98.878, 2 )

        l_tm = array([1298., 1344.])
        solver.lam0 = 1500.
        r_tm = solver.compute_reflectivity(l_tm, 'Et', 'top')
        self.assertAlmostEqual( r_tm[0], 99.097, 2 )
        self.assertAlmostEqual( r_tm[1], 26.911, 2 )


if __name__ == '__main__':
    import __main__
    __main__.GratingTest = GratingTest
    unittest.main()
]]></script>

</plask>
