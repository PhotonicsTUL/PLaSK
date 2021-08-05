<plask loglevel="detail">

<defines>
  <define name="ns" value="3.48"/>
  <define name="nh" value="3.48"/>
  <define name="nl" value="1.00"/>
  <define name="L" value="1.00"/>
  <define name="tg" value="0.20"/>
  <define name="tl" value="0.83"/>
  <define name="fill" value="0.4"/>
  <define name="dx" value="0.005"/>
  <define name="ktran" value="0."/>
  <define name="pol" value="'El'"/>
  <define name="spol" value="pol"/>
  <define name="lam" value="1500."/>
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
    <axis0 start="{-0.5*L-2.0}" stop="{0.5*L+2.0}" num="2001"/>
    <axis1 start="-0.5" stop="{tl+tg+1.5}" num="501"/>
  </mesh>
</grids>

<solvers>
  <optical name="fourier" solver="Fourier2D" lib="slab">
    <geometry ref="grating"/>
    <expansion lam0="1000."/>
    <mode k-tran="{ktran}"/>
  </optical>
  <optical name="solver" solver="Lines2D" lib="slab">
    <geometry ref="grating"/>
    <discretization lam0="1000." density="{dx}"/>
    <mode k-tran="{ktran}"/>
    <interface object="bar"/>
    <pml factor="2"/>
  </optical>
</solvers>

<script><![CDATA[

# figure()
# plot_field(solver.outRefractiveIndex(MSH.plot), comp='yy')
# plot_geometry(solver.geometry, lw=0.7, color='k', alpha=0.35, mirror=True)
# colorbar(use_gridspec=True)
# tight_layout()

# figure()
# axvline(-L/2., color='k', lw=0.7)
# axvline(+L/2., color='k', lw=0.7)
# axvline(-L/2. - solver.actual_pml.dist, color='k', lw=0.7, ls='--')
# axvline(+L/2. + solver.actual_pml.dist, color='k', lw=0.7, ls='--')
# axvline(-L/2. - solver.actual_pml.dist - solver.actual_pml.size, color='k', lw=0.7)
# axvline(+L/2. + solver.actual_pml.dist + solver.actual_pml.size, color='k', lw=0.7)
# nr1, nr2 = (solver.outRefractiveIndex(mesh.Rectangular2D(solver.actual_mesh, y))**2 for y in (-1., 1.))
# plot_profile(nr1, comp='xx', color='C0', marker='.', ls='')
# plot_profile(nr1, comp='yy', color='C0', marker='.', ls='')
# plot_profile(nr1, comp='zz', color='C0', marker='.', ls='')
# plot_profile(nr2, comp='xx', color='C1', marker='.', ls='')
# plot_profile(nr2, comp='yy', color='C1', marker='.', ls='')
# plot_profile(nr2, comp='zz', color='C1', marker='.', ls='')
# nr1, nr2 = (solver.outRefractiveIndex(mesh.Rectangular2D(MSH.plot.axis0, y))**2 for y in (-1., 1.))
# plot_profile(nr1, comp='xx', color='C0', ls='-')
# plot_profile(nr1, comp='yy', color='C0', ls='--')
# plot_profile(nr1, comp='zz', color='C0', ls=':')
# plot_profile(nr2, comp='xx', color='C1', ls='-')
# plot_profile(nr2, comp='yy', color='C1', ls='--')
# plot_profile(nr2, comp='zz', color='C1', ls=':')


fourier.polarization = pol
solver.polarization = spol

solver.initialize()

# lams = linspace(800., 1700., 91)
# plot(lams, fourier.compute_reflectivity(lams, 'top', pol), label='Fourier')
# r = solver.compute_reflectivity(lams, 'top', pol)
# r[r > 100.] = 10000. / r[r > 100.]
# plot(lams, r, label='lines')
# ylim(0., 100.)
# legend()

fig, ax = subplots()
solver.lam = lam
s = solver.scattering('top', pol)
print(100 * s.R, fourier.compute_reflectivity(lam, 'top', pol))
# plot_field(s.outLightMagnitude(MSH.plot))
plot_field(s.reflected.outLightE(MSH.plot).real, None, comp=pol[-1], cmap='bwr')
plot_geometry(GEO.grating)

fig, ax = subplots()
fourier.lam = lam
s = fourier.scattering('top', pol)
print(100 * s.R, fourier.compute_reflectivity(lam, 'top', pol))
# plot_field(s.outLightMagnitude(MSH.plot))
plot_field(s.reflected.outLightE(MSH.plot).real, None, comp=pol[-1], cmap='bwr')
plot_geometry(GEO.grating)


show()
]]></script>

</plask>
