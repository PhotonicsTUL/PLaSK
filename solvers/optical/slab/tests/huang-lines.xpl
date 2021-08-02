<plask loglevel="detail">

<defines>
  <define name="ns" value="3.48"/>
  <define name="nh" value="3.48"/>
  <define name="nl" value="1.00"/>
  <define name="L" value="1.00"/>
  <define name="tg" value="0.20"/>
  <define name="tl" value="0.83"/>
  <define name="fill" value="0.4"/>
  <define name="dx" value="0.2"/>
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
  <cartesian2d name="grating" axes="xy" left="mirror" right="extend" bottom="Subs">
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
    <axis0 start="{-0.5*L-2.0}" stop="{0.5*L+2.0}" num="2001"/>
    <axis1 start="-0.5" stop="{tl+tg+2.0}" num="501"/>
  </mesh>
</grids>

<solvers>
  <optical name="solver" solver="Lines2D" lib="slab">
    <geometry ref="grating"/>
    <discretization lam0="1000." density="{dx}"/>
    <interface object="bar"/>
    <pml factor="2"/>
  </optical>
</solvers>

<script><![CDATA[
# import unittest
#
#
# class GratingTest(unittest.TestCase):
#
#     def setUp(self):
#         solver.lam0 = 1500.
#
#     def testTE_separated_asymmetric(self):
#         solver.polarization = 'El'
#         solver.symmetry = None
#         r_te = solver.compute_reflectivity(array([1500., 1600.]), 'top', 'El')
#         self.assertAlmostEqual( r_te[0], 99.934, 2 )
#         self.assertAlmostEqual( r_te[1], 98.878, 2 )
#
#     def testTE_separated_symmetric(self):
#         solver.polarization = 'El'
#         solver.symmetry = 'El'
#         r_te = solver.compute_reflectivity(array([1500., 1600.]), 'top', 'El')
#         self.assertAlmostEqual( r_te[0], 99.934, 2 )
#         self.assertAlmostEqual( r_te[1], 98.864, 2 )
#
#     def testTM_separated_asymmetric(self):
#         solver.polarization = 'Et'
#         solver.symmetry = None
#         r_tm = solver.compute_reflectivity(array([1298., 1344.]), 'top', 'Et')
#         self.assertAlmostEqual( r_tm[0], 98.504, 2 )
#         self.assertAlmostEqual( r_tm[1], 28.228, 2 )
#
#     def testTM_separated_symmetric(self):
#         solver.polarization = 'Et'
#         solver.symmetry = 'Et'
#         r_tm = solver.compute_reflectivity(array([1298., 1344.]), 'top', 'Et')
#         self.assertAlmostEqual( r_tm[0], 98.640, 2 )
#         self.assertAlmostEqual( r_tm[1], 28.228, 2 )
#
#     def testTE_asymmetric(self):
#         solver.polarization = None
#         solver.symmetry = None
#         r_te = solver.compute_reflectivity(array([1500., 1600.]), 'top', 'El')
#         self.assertAlmostEqual( r_te[0], 99.934, 2 )
#         self.assertAlmostEqual( r_te[1], 98.878, 2 )
#
#     def testTE_symmetric(self):
#         solver.polarization = None
#         solver.symmetry = 'El'
#         r_te = solver.compute_reflectivity(array([1500., 1600.]), 'top', 'El')
#         self.assertAlmostEqual( r_te[0], 99.934, 2 )
#         self.assertAlmostEqual( r_te[1], 98.864, 2 )
#
#     def testTM_asymmetric(self):
#         solver.polarization = None
#         solver.symmetry = None
#         r_tm = solver.compute_reflectivity(array([1298., 1344.]), 'top', 'Et')
#         self.assertAlmostEqual( r_tm[0], 98.504, 2 )
#         self.assertAlmostEqual( r_tm[1], 28.228, 2 )
#
#     def testTM_symmetric(self):
#         solver.polarization = None
#         solver.symmetry = 'Et'
#         r_tm = solver.compute_reflectivity(array([1298., 1344.]), 'top', 'Et')
#         self.assertAlmostEqual( r_tm[0], 98.640, 2 )
#         self.assertAlmostEqual( r_tm[1], 28.228, 2 )
#
#
# if __name__ == '__main__':
#     import __main__
#     __main__.GratingTest = GratingTest
#     unittest.main()



# figure()
# plot_field(solver.outRefractiveIndex(MSH.plot), comp='yy')
# plot_geometry(solver.geometry, lw=0.7, color='k', alpha=0.35, mirror=True)
# colorbar(use_gridspec=True)
# tight_layout()

figure()
axvline(-L/2., color='k', lw=0.7)
axvline(+L/2., color='k', lw=0.7)
axvline(-L/2. - solver.actual_pml.dist, color='k', lw=0.7, ls='--')
axvline(+L/2. + solver.actual_pml.dist, color='k', lw=0.7, ls='--')
axvline(-L/2. - solver.actual_pml.dist - solver.actual_pml.size, color='k', lw=0.7)
axvline(+L/2. + solver.actual_pml.dist + solver.actual_pml.size, color='k', lw=0.7)
nr1, nr2 = (solver.outRefractiveIndex(mesh.Rectangular2D(solver.actual_mesh, y))**2 for y in (-1., 1.))
plot_profile(nr1, comp='xx', color='C0', marker='.', ls='')
plot_profile(nr1, comp='yy', color='C0', marker='.', ls='')
plot_profile(nr1, comp='zz', color='C0', marker='.', ls='')
plot_profile(nr2, comp='xx', color='C1', marker='.', ls='')
plot_profile(nr2, comp='yy', color='C1', marker='.', ls='')
plot_profile(nr2, comp='zz', color='C1', marker='.', ls='')
nr1, nr2 = (solver.outRefractiveIndex(mesh.Rectangular2D(MSH.plot.axis0, y))**2 for y in (-1., 1.))
plot_profile(nr1, comp='xx', color='C0', ls='-')
plot_profile(nr1, comp='yy', color='C0', ls='--')
plot_profile(nr1, comp='zz', color='C0', ls=':')
plot_profile(nr2, comp='xx', color='C1', ls='-')
plot_profile(nr2, comp='yy', color='C1', ls='--')
plot_profile(nr2, comp='zz', color='C1', ls=':')

show()
]]></script>

</plask>
