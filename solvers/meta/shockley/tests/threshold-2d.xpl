<plask loglevel="detail">

<defines>
  <define name="ridge" value="4."/>
  <define name="width" value="20."/>
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
  <cartesian2d name="GeoE" axes="x,y">
    <stack>
      <rectangle name="n-contact" material="Au" dx="{ridge/2}" dy="0.0500"/>
      <stack name="laser">
        <rectangle material="GaAs:Si=2e+17" dx="{ridge/2}" dy="0.0700"/>
        <rectangle material="GaAs:Si=5e+17" dx="{width/2}" dy="0.1160"/>
        <stack name="junction" role="active">
          <stack repeat="4">
            <rectangle name="QW" role="QW" material="InGaAsQW" dx="{width/2}" dy="0.0050"/>
            <rectangle material="GaAs" dx="{width/2}" dy="0.0050"/>
          </stack>
          <again ref="QW"/>
        </stack>
        <rectangle material="GaAs:C=5e+17" dx="{width/2}" dy="0.1160"/>
      </stack>
      <zero/>
      <rectangle name="p-contact" material="GaAs:C=2e+18" dx="{width/2}" dy="5."/>
    </stack>
  </cartesian2d>
  <cartesian2d name="GeoT" axes="x,y">
    <stack>
      <rectangle material="Au" dx="4" dy="0.0500"/>
      <again ref="laser"/>
      <zero/>
      <rectangle material="GaAs:C=2e+18" dx="2500." dy="150."/>
      <rectangle material="Cu" dx="2500." dy="5000."/>
    </stack>
  </cartesian2d>
  <cartesian2d name="GeoO" axes="x,y" right="extend" bottom="GaAs" top="air">
    <again ref="laser"/>
  </cartesian2d>
</geometry>

<grids>
  <generator method="divide" name="default" type="rectangular2d">
    <postdiv by0="3" by1="2"/>
    <refinements>
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
    <prediv by0="10"/>
  </generator>
</grids>

<solvers>
  <meta name="SOLVER" solver="ThresholdSearchCyl" lib="shockley">
    <geometry electrical="GeoE" optical="GeoO" thermal="GeoT"/>
    <mesh diffusion="diffusion" electrical="default" thermal="default"/>
    <optical dlam="0.005" lam0="980." maxlam="980."/>
    <root bcond="0" vmax="1.8" vmin="1.5"/>
    <voltage>
      <condition value="1.4">
        <place side="bottom" object="p-contact"/>
      </condition>
      <condition value="0.0">
        <place side="top" object="n-contact"/>
      </condition>
    </voltage>
    <temperature>
      <condition place="bottom" value="300."/>
    </temperature>
    <junction beta0="11" js0="1"/>
    <diffusion accuracy="0.005" fem-method="parabolic"/>
    <gain lifetime="0.5" matrix-elem="10"/>
  </meta>
</solvers>

<script><![CDATA[
lams = linspace(975., 980., 501)
plot(lams, abs(SOLVER.optical_determinant(lams)))
xlabel("Wavelength [nm]")
window_title("Determinant")

threshold_voltage = SOLVER.compute()
threshold_current = SOLVER.threshold_current
print("Vth = {:.3f} V,  Ith = {:.3f} 4mA"
    .format(threshold_voltage, threshold_current))

figure()
SOLVER.plot_optical_field()
axvline(GEO.aperture.dr, color='0.75', ls=":", linewidth=1)
window_title("Light Intensity")

show()
]]></script>

</plask>
