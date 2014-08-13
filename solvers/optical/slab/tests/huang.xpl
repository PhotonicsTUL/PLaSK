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
  <cartesian2d axes="xy" name="grating" bottom="Subs" left="periodic" right="periodic">
    <stack trancenter="0">
      <block dx="{fill*L}" dy="{tg}" material="Hi" name="bar"/>
      <block dx="{L}" dy="{tl}" material="Lo"/>
    </stack>
  </cartesian2d>
</geometry>

<grids>
  <mesh name="plot" type="rectangular2d">
    <axis0 start="{-1.5*L}" stop="{1.5*L}" num="501"/>
    <axis1 start="-0.5" stop="{tl+tg+2.0}" num="501"/>
  </mesh>
</grids>

<solvers>
  <optical name="solver" solver="Fourier2D">
    <geometry ref="grating"/>
    <interface object="bar"/>
    <expansion size="{N}" smooth="0"/>
  </optical>
</solvers>


<script><![CDATA[
rc('figure', figsize = (4,3))
rc('figure.subplot', left=0.1, bottom=0.05, right=0.99, top=0.92)
rc('figure', dpi=150)
rc('savefig', dpi=150)
rc('font', family='serif')
#rc('font', **{'serif': 'ZapfHumnstDmPL'})


lams = linspace(1000., 2200., 1201)

R_TE = solver.compute_reflectivity(lams, 'El', 'top', dispersive=False)
R_TM = solver.compute_reflectivity(lams, 'Et', 'top', dispersive=False)

plot(lams, R_TE/100., 'r', label='TE')
plot(lams, R_TM/100., 'b', label='TM')
legend(loc='best')
xlabel("Wavelength [nm]")
ylabel("Reflectivity")
ylim(-0.01,1.01)
axhline(1.000, color='#888888')
axhline(0.990, ls=':', color='#888888')
axhline(0.998, ls=':', color='#888888')
tight_layout(0)

figure()
field = solver.reflected(1060., 'El', 'top').outLightMagnitude(MSH.plot)
plot_field(field, 256, vmin=0.)
plot_geometry(GEO.grating, color='w')
gca().set_aspect('equal')
tight_layout(0)
gcf().canvas.set_window_title("TE @ 1060 nm")

figure()
field = solver.reflected(1510., 'El', 'top').outLightMagnitude(MSH.plot)
plot_field(field, 256, vmin=0.)
plot_geometry(GEO.grating, color='w')
gca().set_aspect('equal')
tight_layout(0)
gcf().canvas.set_window_title("TE @ 1510 nm")

figure()
field = solver.reflected(1298., 'Et', 'top').outLightMagnitude(MSH.plot)
plot_field(field, 256, vmin=0.)
plot_geometry(GEO.grating, color='w')
gca().set_aspect('equal')
tight_layout(0)
gcf().canvas.set_window_title("TM @ 1298 nm")

figure()
field = solver.reflected(1330., 'Et', 'top').outLightMagnitude(MSH.plot)
plot_field(field, 256, vmin=0.)
plot_geometry(GEO.grating, color='w')
gca().set_aspect('equal')
tight_layout(0)
gcf().canvas.set_window_title("TM @ 1330 nm")

figure()
field = solver.reflected(1600., 'Et', 'top').outLightMagnitude(MSH.plot)
plot_field(field, 256, vmin=0.)
plot_geometry(GEO.grating, color='w')
gca().set_aspect('equal')
tight_layout(0)
gcf().canvas.set_window_title("TM @ 1600 nm")

show()

]]></script>

</plask>
