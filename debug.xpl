<plask loglevel="detail">

<defines>
  <define name="aperture" value="8."/>
  <define name="mesa" value="{4 * aperture}"/>
  <define name="U" value="1.6"/>
</defines>

<materials>
  <material name="InGaAsQW" base="In(0.22)GaAs">
    <nr>3.621</nr>
    <absp>0</absp>
    <A>110000000</A>
    <B>7e-011-1.08e-12*(T-300)</B>
    <C>1e-029+1.4764e-33*(T-300)</C>
    <D>10+0.01667*(T-300)</D>
  </material>
</materials>

<geometry>
  <cylindrical2d name="GeoE" axes="r,z">
    <stack>
      <item right="{mesa/2-1}">
        <rectangle name="n-contact" material="Au" dr="4" dz="0.0500"/>
      </item>
      <stack name="VCSEL">
        <rectangle material="GaAs:Si=2e+18" dr="{mesa/2}" dz="0.0700"/>
        <stack name="top-DBR" repeat="24">
          <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesa/2}" dz="0.0795"/>
          <rectangle material="GaAs:Si=2e+18" dr="{mesa/2}" dz="0.0700"/>
        </stack>
        <shelf>
          <rectangle name="aperture" material="AlAs:Si=2e+18" dr="{aperture/2}" dz="0.0160"/>
          <rectangle name="oxide" material="AlOx" dr="{(mesa-aperture)/2}" dz="0.0160"/>
        </shelf>
        <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesa/2}" dz="0.0635"/>
        <rectangle material="GaAs:Si=5e+17" dr="{mesa/2}" dz="0.1160"/>
        <stack name="junction" role="active">
          <stack repeat="4">
            <rectangle name="QW" role="QW" material="InGaAsQW" dr="{mesa/2}" dz="0.0050"/>
            <rectangle material="GaAs" dr="{mesa/2}" dz="0.0050"/>
          </stack>
          <again ref="QW"/>
        </stack>
        <rectangle material="GaAs:C=5e+17" dr="{mesa/2}" dz="0.1160"/>
        <stack name="bottom-DBR" repeat="30">
          <rectangle material="Al(0.73)GaAs:C=2e+18" dr="{mesa/2}" dz="0.0795"/>
          <rectangle material="GaAs:C=2e+18" dr="{mesa/2}" dz="0.0700"/>
        </stack>
      </stack>
      <zero/>
      <rectangle name="p-contact" material="GaAs:C=2e+18" dr="{mesa/2}" dz="5."/>
    </stack>
  </cylindrical2d>
  <cylindrical2d name="GeoT" axes="r,z">
    <stack name="full">
      <item right="{mesa/2-1}">
        <rectangle material="Au" dr="4" dz="0.0500"/>
      </item>
      <again ref="VCSEL"/>
      <zero/>
      <rectangle material="GaAs:C=2e+18" dr="2500." dz="150."/>
      <rectangle material="Cu" dr="2500." dz="5000."/>
    </stack>
  </cylindrical2d>
  <cylindrical2d name="GeoO" axes="r,z" outer="extend" bottom="GaAs" top="air">
    <again ref="VCSEL"/>
  </cylindrical2d>
  <cartesian3d name="GeoO3D" axes="x,y,z" back="mirror" front="extend" left="mirror" right="extend" bottom="GaAs">
    <clip back="0" front="{0.3 * mesa}" left="0" right="{0.3 * mesa}" bottom="0">
      <revolution name="rev">
        <again ref="full"/>
      </revolution>
    </clip>
  </cartesian3d>
</geometry>

<grids>
  <generator name="default" type="rectangular2d" method="divide">
    <postdiv by0="3" by1="2"/>
    <refinements>
      <axis0 object="oxide" at="-0.1"/>
      <axis0 object="oxide" at="-0.05"/>
      <axis0 object="aperture" at="0.1"/>
    </refinements>
    <warnings outside="no"/>
  </generator>
  <generator name="optical" type="ordered" method="divide">
    <prediv by="10"/>
  </generator>
</grids>

<solvers>
  <meta name="SOLVER" solver="ThresholdSearchCyl" lib="shockley">
    <geometry thermal="GeoT" electrical="GeoE" optical="GeoO"/>
    <mesh thermal="default" electrical="default" optical="optical"/>
    <optical lam0="980." maxlam="980." dlam="0.01" vat="0"/>
    <root bcond="0" vmin="1.4" vmax="1.6"/>
    <voltage>
      <condition value="{U}">
        <place side="bottom" object="p-contact"/>
      </condition>
      <condition value="0.0">
        <place side="top" object="n-contact"/>
      </condition>
    </voltage>
    <temperature>
      <condition value="300.">
        <place side="bottom"/>
      </condition>
    </temperature>
    <junction beta0="11" js0="1"/>
    <diffusion fem-method="parabolic" accuracy="0.005"/>
    <gain lifetime="0.5" matrix-elem="10"/>
    <optical-root determinant="full"/>
  </meta>
  <optical name="OPT3D" solver="Fourier3D" lib="slab">
    <geometry ref="GeoO3D"/>
    <expansion lam0="980." size="5"/>
    <mode symmetry-long="Etran" symmetry-tran="Etran"/>
    <interface object="junction"/>
  </optical>
</solvers>

<connects>
  <connect out="SOLVER.outGain" in="OPT3D.inGain"/>
  <connect out="SOLVER.outTemperature" in="OPT3D.inTemperature"/>
</connects>

<script><![CDATA[
config.axes = 'xyz'

SOLVER.electrical.compute(6)
SOLVER.thermal.compute(1)

# We can perform computations normally
# SOLVER.compute_thermoelectric()
# mn = OPT3D.find_mode(lam=980)
# print(OPT3D.modes[mn])


# Let's prove the gain and themperature are transformed correctly by plotting them

temp_mesh_xz = mesh.Rectangular3D(mesh.Regular(-0.6*mesa, 0.6*mesa, 501), [0.], mesh.Regular(-1, GEO.GeoO3D.bbox.top, 2001))
temp = OPT3D.inTemperature(temp_mesh_xz)
figure()
plot_field(temp, plane='xz')
plot_geometry(GEO.GeoE, mirror='true', color='k', alpha=0.2, lw=1)
colorbar()
xlim(-0.6*mesa, 0.6*mesa)
ylim(temp_mesh_xz.axis2[0], temp_mesh_xz.axis2[-1])
window_title("Temperature xz")

jbox = GEO.GeoE.get_object_bboxes(GEO.junction)[0]

temp_mesh_xy = mesh.Rectangular3D(mesh.Regular(-0.6*mesa, 0.6*mesa, 501), mesh.Regular(-0.6*mesa, 0.6*mesa, 501), [jbox.center.z])
temp = OPT3D.inTemperature(temp_mesh_xy)
figure()
plot_field(temp, plane='xy')
plot_geometry(GEO.GeoO3D, plane='xy', color='k', alpha=0.2, lw=1)
colorbar()

heat_mesh_xy = mesh.Rectangular3D(mesh.Regular(-0.6*mesa, 0.6*mesa, 51), mesh.Regular(-0.6*mesa, 0.6*mesa, 51), [jbox.center.z])
heat_tran = flow.HeatFluxReceiver3D()
heat_tran.attach(SOLVER.outHeatFlux)
heat = heat_tran(heat_mesh_xy)
plot_vectors(heat, color='w', plane='xy')

xlim(-0.6*mesa, 0.6*mesa)
ylim(-0.6*mesa, 0.6*mesa)
aspect('equal')
window_title("Temperature xy")



# gain_mesh_xz = mesh.Rectangular3D(mesh.Regular(-0.6*mesa, 0.6*mesa, 501), [0.], mesh.Regular(jbox.bottom-0.2, jbox.top+0.2, 201))
# gain = OPT3D.inGain(gain_mesh_xz, 980.)
# figure()
# plot_field(gain, 256, plane='xz', comp=0)
# plot_geometry(GEO.GeoE, mirror='true', color='k', alpha=0.2, lw=1)
# colorbar()
# xlim(-0.6*mesa, 0.6*mesa)
# ylim(gain_mesh_xz.axis2[0], gain_mesh_xz.axis2[-1])
# window_title("Gain xz")

# show()
]]></script>

</plask>
