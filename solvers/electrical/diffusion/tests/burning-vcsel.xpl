<plask loglevel="debug">

<defines>
  <define name="V" value="1.6"/>
  <define name="P" value="0.701"/>
  <define name="lam" value="977.94"/>
  <define name="aperture" value="8."/>
  <define name="mesa" value="{4 * aperture}"/>
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
  <material name="GaAs" base="GaAs">
    <absp>0</absp>
  </material>
  <material name="GaAs:Si" base="GaAs:Si">
    <absp>0</absp>
  </material>
  <material name="AlAs:Si" base="AlAs:Si">
    <absp>0</absp>
  </material>
  <material name="AlOx" base="AlOx">
    <absp>0</absp>
  </material>
  <material name="AlGaAs:Si" base="AlGaAs:Si" alloy="yes">
    <absp>0</absp>
  </material>
  <material name="GaAs:C" base="GaAs:C">
    <absp>0</absp>
  </material>
  <material name="AlGaAs:C" base="AlGaAs:C" alloy="yes">
    <absp>0</absp>
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
        <stack name="bottom-DBR" repeat="60">
          <rectangle material="Al(0.73)GaAs:C=2e+18" dr="{mesa/2}" dz="0.0795"/>
          <rectangle material="GaAs:C=2e+18" dr="{mesa/2}" dz="0.0700"/>
        </stack>
      </stack>
      <zero/>
      <rectangle name="p-contact" material="GaAs:C=2e+18" dr="{mesa/2}" dz="5."/>
    </stack>
  </cylindrical2d>
  <cylindrical2d name="GeoO" axes="r,z" outer="extend" bottom="GaAs" top="air">
    <again ref="VCSEL"/>
  </cylindrical2d>
</geometry>

<grids>
  <generator name="default" type="rectangular2d" method="divide">
    <postdiv by0="3" by1="2"/>
    <refinements>
      <axis0 object="oxide" at="-0.1"/>
      <axis0 object="oxide" at="-0.05"/>
      <axis0 object="aperture" at="0.1"/>
    </refinements>
  </generator>
  <generator name="optical" type="ordered" method="divide">
    <prediv by="10"/>
  </generator>
  <generator name="dense" type="ordered" method="regular">
    <spacing every="0.001"/>
  </generator>
</grids>

<solvers>
  <electrical name="ELECTRICAL" solver="ShockleyCyl" lib="shockley">
    <geometry ref="GeoE"/>
    <mesh ref="default"/>
    <voltage>
      <condition value="{V}">
        <place side="bottom" object="p-contact"/>
      </condition>
      <condition value="0.0">
        <place side="top" object="n-contact"/>
      </condition>
    </voltage>
    <junction beta0="11" js0="1"/>
  </electrical>
  <optical name="OPTICAL" solver="EffectiveFrequencyCyl" lib="effective">
    <geometry ref="GeoO"/>
    <mesh ref="optical"/>
    <mode lam0="980."/>
    <root determinant="full"/>
  </optical>
  <gain name="GAIN" solver="FreeCarrierCyl" lib="freecarrier">
    <geometry ref="GeoE"/>
    <config lifetime="0.5" matrix-elem="10"/>
  </gain>
  <electrical name="DIFFUSION" solver="DiffusionCyl" lib="diffusion">
    <geometry ref="GeoE"/>
  </electrical>
</solvers>

<connects>
  <connect out="ELECTRICAL.outCurrentDensity" in="DIFFUSION.inCurrentDensity"/>
  <connect out="DIFFUSION.outCarriersConcentration" in="GAIN.inCarriersConcentration"/>
  <connect out="GAIN.outGain" in="OPTICAL.inGain"/>
  <connect out="GAIN.outGain" in="DIFFUSION.inGain"/>
  <connect out="OPTICAL.outLightE" in="DIFFUSION.inLightE"/>
  <connect out="OPTICAL.outWavelength" in="DIFFUSION.inWavelength"/>
</connects>

<script><![CDATA[
pmesh = mesh.Rectangular2D(mesh.Regular(0., mesa / 2, 1001), [GEO.GeoE.get_object_bboxes(GEO.junction)[0].center.z])


def plot_optical(scale, **kwargs):
    mag = OPTICAL.outLightMagnitude(pmesh)
    plot_profile(Data(scale * mag.array / max(mag), mag.mesh), color='C2', lw=0.5, **kwargs)


ELECTRICAL.compute()

try:
    DIFFUSION.compute()
except AttributeError:
    DIFFUSION.compute_threshold()

conc = DIFFUSION.outCarriersConcentration(pmesh)
plot_profile(conc)
scale = max(conc)

OPTICAL.find_mode(lam=lam)
plot_optical(scale)


def burn(P):
    OPTICAL.modes[0].power = P
    try:
        DIFFUSION.compute(shb=True)
    except AttributeError:
        DIFFUSION.compute_overthreshold()
    OPTICAL.invalidate()
    OPTICAL.find_mode(lam=lam)
    plot_optical(scale, ls='--')
    loss = OPTICAL.modes[0].loss
    print_log('result', f"P = {P:.3f}  loss = {loss}")
    return loss


# from scipy.optimize import brentq
# P = brentq(burn, 0.0, 1.0)
# print_log('important', f"P = {P}")

burn(P)

print_log('result', f"P[SHB] = {DIFFUSION.get_total_burning()}")
plot_profile(DIFFUSION.outCarriersConcentration(pmesh))

ylabel("Carriers concentration (cm$^{-3}$)")
xlim(0., mesa / 2)
show()

]]></script>

</plask>
