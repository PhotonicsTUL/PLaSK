<plask>

<defines>
  <define name="mesa" value="10."/>
  <define name="aperture" value="{mesa-6.}"/>
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
  <cylindrical2d name="GeoTE" axes="r,z">
    <stack>
      <shelf>
        <gap total="{mesa-1}"/>
        <rectangle name="n-contact" material="Au" dr="4" dz="0.0500"/>
      </shelf>
      <stack name="VCSEL">
        <rectangle material="GaAs:Si=2e+18" dr="{mesa}" dz="0.0700"/>
        <stack name="top-DBR" repeat="24">
          <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesa}" dz="0.0795"/>
          <rectangle material="GaAs:Si=2e+18" dr="{mesa}" dz="0.0700"/>
        </stack>
        <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesa}" dz="0.0318"/>
        <shelf>
          <rectangle name="aperture" material="AlAs:Si=2e+18" dr="{aperture}" dz="0.0160"/>
          <rectangle name="oxide" material="AlOx" dr="{mesa-aperture}" dz="0.0160"/>
        </shelf>
        <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesa}" dz="0.0318"/>
        <rectangle material="GaAs:Si=5e+17" dr="{mesa}" dz="0.1176"/>
        <stack name="junction" role="active">
          <stack repeat="4">
            <rectangle name="QW" role="QW" material="InGaAsQW" dr="{mesa}" dz="0.0050"/>
            <rectangle material="GaAs" dr="{mesa}" dz="0.0050"/>
          </stack>
          <again ref="QW"/>
        </stack>
        <rectangle material="GaAs:C=5e+17" dr="{mesa}" dz="0.1176"/>
        <stack name="bottom-DBR" repeat="30">
          <rectangle material="Al(0.73)GaAs:C=2e+18" dr="{mesa}" dz="0.0795"/>
          <rectangle material="GaAs:C=2e+18" dr="{mesa}" dz="0.0700"/>
        </stack>
      </stack>
      <zero/>
      <rectangle material="GaAs:C=2e+18" dr="200." dz="150."/>
      <rectangle name="p-contact" material="Cu" dr="2500." dz="5000."/>
    </stack>
  </cylindrical2d>
  <cylindrical2d name="GeoO" axes="r,z" outer="extend" bottom="GaAs" top="air">
    <again ref="VCSEL"/>
  </cylindrical2d>
</geometry>

<grids>
  <generator method="divide" name="default" type="rectangular2d">
    <postdiv by0="3" by1="2"/>
    <refinements>
      <axis1 object="p-contact" at="50"/>
      <axis0 object="oxide" at="-0.1"/>
      <axis0 object="oxide" at="-0.05"/>
      <axis0 object="aperture" at="0.1"/>
    </refinements>
    <warnings outside="no"/>
  </generator>
  <mesh name="diffusion" type="regular">
    <axis start="0" stop="{mesa}" num="2000"></axis>
  </mesh>
  <generator method="divide" name="optical" type="rectangular2d">
    <prediv by0="10" by1="3"/>
  </generator>
  <generator method="divide" name="plots" type="rectangular2d">
    <postdiv by="30"/>
  </generator>
</grids>

<solvers>
  <thermal name="THERMAL" solver="StaticCyl" lib="fem">
    <geometry ref="GeoTE"/>
    <mesh ref="default"/>
    <temperature>
    <condition value="300." place="bottom"/>
  </temperature>
  </thermal>
  <electrical name="ELECTRICAL" solver="ShockleyCyl" lib="fem">
    <geometry ref="GeoTE"/>
    <mesh ref="default"/>
    <junction beta0="11" js0="1"/>
    <voltage>
    <condition value="1.4">
      <place object="p-contact" side="bottom"/>
    </condition>
    <condition value="0.0">
      <place object="n-contact" side="top"/>
    </condition>
  </voltage>
  </electrical>
  <electrical name="DIFFUSION" solver="DiffusionCyl" lib="diffusion">
    <geometry ref="GeoO"/>
    <mesh ref="diffusion"/>
    <config accuracy="0.005" fem-method="parabolic"/>
  </electrical>
  <gain name="GAIN" solver="FermiCyl" lib="simple">
    <geometry ref="GeoO"/>
    <config lifetime="0.5" matrix-elem="8"/>
  </gain>
  <optical name="OPTICAL" solver="EffectiveFrequencyCyl" lib="effective">
    <geometry ref="GeoO"/>
    <mesh ref="optical"/>
    <mode lam0="981." vat="0."/>
  </optical>
</solvers>

<connects>
  <connect in="ELECTRICAL.inTemperature" out="THERMAL.outTemperature"/>
  <connect in="THERMAL.inHeat" out="ELECTRICAL.outHeat"/>
  <connect in="DIFFUSION.inTemperature" out="THERMAL.outTemperature"/>
  <connect in="DIFFUSION.inCurrentDensity" out="ELECTRICAL.outCurrentDensity"/>
  <connect in="GAIN.inTemperature" out="THERMAL.outTemperature"/>
  <connect in="GAIN.inCarriersConcentration" out="DIFFUSION.outCarriersConcentration"/>
  <connect in="OPTICAL.inTemperature" out="THERMAL.outTemperature"/>
  <connect in="OPTICAL.inGain" out="GAIN.outGain"/>
</connects>

<script><![CDATA[
plot_geometry(GEO.GeoTE, margin=0.01)
defmesh = MSG.default(GEO.GeoTE.item)
plot_mesh(defmesh, color="0.75")
plot_boundary(ELECTRICAL.voltage_boundary, defmesh,
              ELECTRICAL.geometry, color="b", marker="D")
plot_boundary(THERMAL.temperature_boundary, defmesh,
              THERMAL.geometry, color="r")
gcf().canvas.set_window_title("Default mesh")

task = algorithm.ThresholdSearch(THERMAL, ELECTRICAL, DIFFUSION,
                                 GAIN, OPTICAL, 0, 1.4, 981.5)
threshold_voltage = task.run()
threshold_current = task.threshold_current
print("Vth = {:.3f}V, Ith = {:.3f}mA"
    .format(threshold_voltage, threshold_current))

figure()
task.plot_optical_field()
axvline(GEO.aperture.dr, color='0.75', ls=":", linewidth=1)


# Find threshold for a new aperture

new_aperture = 3.
GEO.aperture.dr = new_aperture
GEO.oxide.dr = mesa - new_aperture

threshold_voltage = task.run()
threshold_current = task.threshold_current
print("New aperture: Vth = {:.3f}V, Ith = {:.3f}mA"
    .format(threshold_voltage, threshold_current))

figure()
task.plot_optical_field()
axvline(GEO.aperture.dr, color='0.75', ls=":", linewidth=1)
gcf().canvas.set_window_title("Light Intensity (new aperture)")


show()

]]></script>

</plask>
