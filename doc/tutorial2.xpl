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
  <cylindrical2d name="main" axes="r,z" outer="extend" bottom="GaAs">
    <stack>
      <rectangle material="GaAs" dr="10" dz="0.0700"/>
      <stack name="top-DBR" repeat="24">
        <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
        <rectangle material="GaAs" dr="10" dz="0.0700"/>
      </stack>
      <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0318"/>
      <shelf>
        <rectangle material="AlAs" dr="4" dz="0.0160"/>
        <rectangle material="AlOx" dr="6" dz="0.0160"/>
      </shelf>
      <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0318"/>
      <rectangle material="GaAs" dr="10" dz="0.1376"/>
      <shelf>
        <rectangle name="gain-region" role="gain" material="active" dr="4" dz="0.0050"/>
        <rectangle material="inactive" dr="6" dz="0.0050"/>
      </shelf>
      <rectangle material="GaAs" dr="10" dz="0.1376"/>
      <stack name="bottom-DBR" repeat="30">
        <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
        <rectangle material="GaAs" dr="10" dz="0.0700"/>
      </stack>
    </stack>
  </cylindrical2d>
</geometry>

<solvers>
  <optical name="efm" solver="EffectiveFrequencyCyl" lib="effective">
    <geometry ref="main"/>
    <mode lam0="980."/>
  </optical>
</solvers>

<script><![CDATA[
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
]]></script>

</plask>
