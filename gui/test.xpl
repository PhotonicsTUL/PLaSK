<plask>

<defines>
  <define name="mesa" value="10"/>
  <define name="aprt" value="4"/>
</defines>

<materials>
  <material name="active" base="In(0.13)GaAs">
    <nr>3.53</nr>
    <absp>0.</absp>
  </material>
  <material name="inactive" base="active">
    <absp>1000.</absp>
  </material>
</materials>

<geometry>
  <cylindrical2d axes="rz" name="main" top="air" bottom="AlAs" outer="extend">
    <stack>
      <stack name="top-DBR" repeat="24">
        <block dr="{mesa}" dz="0.07" material="GaAs"/>
        <block dr="{mesa}" dz="0.07945" material="Al(0.73)GaAs"/>
      </stack>
      <block dr="{mesa}" dz="0.07003" material="GaAs"/>
      <block dr="{mesa}" dz="0.03178" material="Al(0.73)GaAs"/>
      <shelf>
        <block dr="{aprt}" dz="0.01603" material="AlAs"/>
        <block dr="{mesa-aprt}" dz="0.01603" material="AlxOy"/>
      </shelf>
      <block dr="{mesa}" dz="0.03178" material="Al(0.73)GaAs"/>
      <block dr="{mesa}" dz="0.13756" material="GaAs"/>
      <shelf>
        <block dr="{aprt}" dz="0.005" role="gain" material="active" name="gain-region"/>
        <block dr="{mesa-aprt}" dz="0.005" material="inactive"/>
      </shelf>
      <block dr="{mesa}" dz="0.13756" material="GaAs"/>
      <stack name="bottom-DBR" repeat="29">
        <block dr="{mesa}" dz="0.07945" material="Al(0.73)GaAs"/>
        <block dr="{mesa}" dz="0.07003" material="GaAs"/>
      </stack>
      <block dr="10" dz="0.07945" material="Al(0.73)GaAs"/>
    </stack>
  </cylindrical2d>
</geometry>

<grids>
  <mesh name="plot" type="rectangular2d">
    <axis0 start="{-mesa}" stop="{mesa}" num="501"></axis0>
    <axis1 start="2" stop="7" num="2001"></axis1>
  </mesh>
  <generator method="divide" name="test" type="rectangular2d">
    <prediv by0="0" by1="2"/>
  </generator>
  <mesh name="plot2" type="rectilinear2d">
    <axis0 start="{-mesa}" stop="{mesa}" num="501"/>
    <axis1 start="2" stop="7" num="2001"/>
  </mesh>
</grids>

<solvers>
  <optical solver="EffectiveFrequencyCyl" name="efm">
    <geometry ref="main"/>
  </optical>
</solvers>

<connects/>

<script><![CDATA[
import sys
import scipy.optimize

profile = StepProfile(GEO.main, default=0.)
profile[GEO.gain_region] = 500.

efm.inGain = profile.outGain

def loss_on_gain(gain):
    global profile, efm
    profile[GEO.gain_region] = gain
    mode_number = efm.find_mode(980.)
    return efm.outLoss(mode_number)

efm.lam0 = 980.

# threshold_gain = scipy.optimize.brentq(loss_on_gain, 0., 2500., xtol=0.1)
threshold_gain = scipy.optimize.fsolve(loss_on_gain, 1000., xtol=0.1)[0]

profile[GEO.gain_region] = threshold_gain
mode_number = efm.find_mode(980.)
mode_wavelength = efm.outWavelength(mode_number)
print_log(LOG_INFO,
          "Threshold material gain is {:.0f}/cm with resonant wavelength {:.2f}nm"
          .format(threshold_gain, mode_wavelength))

plot_geometry(efm.geometry, color='0.5', mirror=True)
efm.modes[0].power = 10.
plot_field(efm.outLightMagnitude(0, MSH.plot))
ylim(2,7)
show()]]></script>

</plask>