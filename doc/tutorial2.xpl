<plask>

<materials>
  <material name="active" base="semiconductor">
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
        <block dr="10" dz="0.07" material="GaAs"/>
        <block dr="10" dz="0.07945" material="Al(0.73)GaAs"/>
      </stack>
      <block dr="10" dz="0.07003" material="GaAs"/>
      <block dr="10" dz="0.03178" material="Al(0.73)GaAs"/>
      <shelf>
        <block dr="4" dz="0.01603" material="AlAs"/>
        <block dr="6" dz="0.01603" material="AlOx"/>
      </shelf>
      <block dr="10" dz="0.03178" material="Al(0.73)GaAs"/>
      <block dr="10" dz="0.13756" material="GaAs"/>
      <shelf>
        <block dr="4" dz="0.005" role="gain" material="active" name="gain-region"/>
        <block dr="6" dz="0.005" material="inactive"/>
      </shelf>
      <block dr="10" dz="0.13756" material="GaAs"/>
      <stack name="bottom-DBR" repeat="29">
        <block dr="10" dz="0.07945" material="Al(0.73)GaAs"/>
        <block dr="10" dz="0.07003" material="GaAs"/>
      </stack>
      <block dr="10" dz="0.07945" material="Al(0.73)GaAs"/>
    </stack>
  </cylindrical2d>
</geometry>

<solvers>
  <optical solver="EffectiveFrequencyCyl" name="efm">
    <geometry ref="main"/>
    <mode lam0="980."/>
  </optical>
</solvers>

<script>
import scipy.optimize

profile = plask.StepProfile(GEO.main, default=0.)
profile[GEO.gain_region] = 500.

efm.inGain = profile.outGain

def loss_on_gain(gain):
    profile[GEO.gain_region] = gain
    mode_number = efm.find_mode(980.5)
    return efm.outLoss(mode_number)

threshold_gain = scipy.optimize.fsolve(loss_on_gain, 2000., xtol=0.1)[0]

profile[GEO.gain_region] = threshold_gain
mode_number = efm.find_mode(980.5)
mode_wavelength = efm.outWavelength(mode_number)
print_log(LOG_INFO,
          "Threshold material gain is {:.0f}/cm with resonant wavelength {:.2f}nm"
          .format(threshold_gain, mode_wavelength))
</script>

</plask>
