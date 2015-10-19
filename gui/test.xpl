<plask loglevel="detail">

<defines>
  <define name="mesaRadius" value="10."/>
  <define name="aperture" value="{mesaRadius-6.}"/>
  <define name="beta_def" value="19"/>
  <define name="js_def" value="1"/>
</defines>

<materials>
  <material name="test" base="semiconductor">
    <A>0.1 * T + 0.02 * (T-300)**2</A>
  </material>
  <material name="InGaAsQW:Si" base="In(0.2)GaAs:Si=1e18">
    <nr>3.621</nr>
    <absp>0</absp>
    <A>110000000</A>
    <B>7e-011-1.08e-12*(T-300)</B>
    <C>1e-029+1.4764e-33*(T-300)</C>
    <D>10+0.016670*(T-300)</D>
  </material>
</materials>

<geometry>
  <cartesian2d name="geo2d">
    <stack>
      <arrange dtran="0.4" dvert="0" count="3">
        <stack>
          <rectangle name="rr" material="InN" dtran="0.2" dvert="0.1"/>
          <rectangle material="In(0.5)GaN:Si=1e18" dtran="0.2" dvert="0.1"/>
        </stack>
      </arrange>
      <shelf flat="no">
        <rectangle material="Al(0.02)GaAs" dtran="1" dvert="0.2"/>
        <gap total="2"/>
        <triangle material="AlOx" atran="-1" avert="0" btran="0" bvert="0.2"/>
      </shelf>
      <rectangle material="Al(0.2)GaN" dtran="2" dvert="0.5"/>
      <rectangle material="GaN" dtran="2" dvert="1"/>
    </stack>
  </cartesian2d>
  <cartesian2d name="simple">
    <stack>
      <rectangle name="one" material="Al(0.73)GaAs:C=2e+18" dtran="1" dvert="1"/>
      <rectangle material="Al(0.73)GaAs:Si=1e18" dtran="1" dvert="1"/>
      <shelf/>
    </stack>
  </cartesian2d>
  <cylindrical2d name="GeoTE" axes="rz">
    <stack name="stack">
      <zero/>
      <item path="abs123" right="{mesaRadius-1}">
        <rectangle name="n-contact" material="Au" dr="4" dz="0.0500"/>
      </item>
      <stack name="VCSEL">
        <stack name="top-DBR" repeat="24">
          <rectangle material="GaAs:Si=2e+18" dr="{mesaRadius}" dz="0.07003"/>
          <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesaRadius}" dz="0.07945"/>
        </stack>
        <rectangle material="GaAs:Si=2e+18" dr="{mesaRadius}" dz="0.07003"/>
        <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesaRadius}" dz="0.03178"/>
        <shelf>
          <rectangle name="aperture" material="AlAs:Si=2e+18" dr="{aperture}" dz="0.01603"/>
          <rectangle name="oxide" material="AlOx" dr="{mesaRadius-aperture}" dz="0.01603"/>
          <gap size="12"/>
        </shelf>
        <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesaRadius}" dz="0.03178"/>
        <rectangle material="GaAs:Si=5e+17" dr="{mesaRadius}" dz="0.11756"/>
        <stack name="junction" role="active">
          <rectangle role="QW" material="InGaAsQW:Si=1e18" dr="{mesaRadius}" dz="0.005"/>
          <stack repeat="4">
            <rectangle material="GaAs" dr="{mesaRadius}" dz="0.005"/>
            <rectangle role="QW" material="InGaAsQW:Si=1e18" dr="{mesaRadius}" dz="0.005"/>
          </stack>
        </stack>
        <rectangle material="GaAs:C=5e+17" dr="{mesaRadius}" dz="0.11756"/>
        <stack name="bottom-DBR" repeat="29">
          <rectangle material="Al(0.73)GaAs:C=2e+18" dr="{mesaRadius}" dz="0.07945"/>
          <rectangle material="GaAs:C=2e+18" dr="{mesaRadius}" dz="0.07003"/>
        </stack>
        <rectangle material="Al(0.73)GaAs:C=2e+18" dr="{mesaRadius}" dz="0.07945"/>
      </stack>
      <rectangle material="GaAs:C=2e+18" dr="200." dz="150."/>
      <rectangle name="p-contact" material="Cu" dr="2500." dz="5000."/>
    </stack>
  </cylindrical2d>
  <cylindrical2d name="GeoO" axes="x,y,z" outer="extend" bottom="GaAs" top="air">
    <align left="0" bottom="0">
      <clip>
        <again ref="VCSEL"/>
      </clip>
    </align>
  </cylindrical2d>
  <cartesian3d name="geo3d-1" axes="x,y,z">
    <align>
      <revolution>
        <again ref="VCSEL"/>
      </revolution>
    </align>
  </cartesian3d>
  <cartesian3d name="geo3d2" axes="long,tran,vert">
    <align back="0" tran="0" top="0">
      <extrusion length="10">
        <again ref="stack"/>
      </extrusion>
      <cuboid material="GaAs" dlong="1" dtran="100" dvert="100"/>
    </align>
  </cartesian3d>
  <cartesian3d name="l3cavity">
    <stack front="0">
      <lattice along="-{sqrt(3)/2}" atran="0.5" avert="0" blong="{sqrt(3)/2}" btran="0.5" bvert="0">
        <segments>-4 0; 0 4; 4 4; 4 0; 0 -4; -4 -4 ^ 2 1; 1 2; -2 -1; -1 -2</segments>
        <cylinder material="GaAs" radius="0.35" height="1.0"/>
      </lattice>
    </stack>
  </cartesian3d>
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
  </generator>
  <mesh name="diffusion" type="regular">
    <axis start="0" stop="{mesaRadius}" num="200"></axis>
  </mesh>
  <generator method="divide" name="optical" type="rectangular2d">
    <prediv by0="10" by1="3"/>
    <options aspect="100" gradual="no"/>
  </generator>
  <generator method="smooth" name="smoothie" type="rectangular2d">
    <steps small="0.005" factor="1.2"/>
  </generator>
  <mesh name="plots" type="rectangular2d">
    <axis0 start="0" stop="10" num="20"></axis0>
    <axis1 start="0" stop="1" num="10"></axis1>
  </mesh>
  <generator method="smooth" name="sss" type="rectangular3d">
    <steps small="0.005" factor="1.2"/>
  </generator>
</grids>

<solvers>
  <thermal name="THERMAL" solver="StaticCyl" lib="static">
    <geometry ref="GeoTE"/>
    <mesh ref="default"/>
    <temperature>
    <condition value="300." place="bottom"/>
  </temperature>
  </thermal>
  <electrical name="ELECTRICAL" solver="ShockleyCyl" lib="fem">
    <geometry ref="GeoTE"/>
    <mesh ref="default"/>
    <junction beta0="{beta_def}" beta1="{beta_def - 1.2}" js0="{js_def}" js1="{js_def + 0.1}"/>
    <voltage>
    <condition value="2.0">
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
    <mode emission="bottom" lam0="980"/>
  </optical>
  <optical name="fourier" solver="Fourier3D" lib="slab">
    <geometry ref="geo3d2"/>
    <pmls>
      <long factor="23"/>
      <tran factor="45"/>
    </pmls>
  </optical>
  <filter for="Temperature" geometry="GeoTE" name="filtr"/>
  <optical name="efm" solver="EffectiveFrequencyCyl" lib="effective">
    <geometry ref="GeoO"/>
    <mesh ref="optical"/>
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
  <connect in="DIFFUSION.inGain" out="GAIN.outGain"/>
  <connect in="DIFFUSION.inGainOverCarriersConcentration" out="GAIN.outGainOverCarriersConcentration"/>
</connects>

<script><![CDATA[
from scipy import optimize

print mesaRadius + 0

print_log('data', "łóżko")
print_log('info', "informacja")

# OPTICAL.find_mode

class A(object):
    
    val = property()
    """
    ppp
    """

    @property
    def prop(self):
        """
        Prop
        :rtype: RootParams
        """
        pass

    def fun(self):
        """
        Fun fun fun
        :rtype: RootParams
        """
        pass

a = A()
a.prop
a.fun()

config.axes = 'rz'

cyl = geometry.Cylinder(2, 1, None)
cyl.get_object_positions

def loss_on_voltage(voltage):
    ELECTRICAL.invalidate()
    ELECTRICAL.voltage_boundary[0].value = voltage[0]
    verr = ELECTRICAL.compute(1)
    terr = THERMAL.compute(1)
    iters=0
    while (terr > THERMAL.maxerr or verr > ELECTRICAL.maxerr) and iters < 15:
        verr = ELECTRICAL.compute(8)
        terr = THERMAL.compute(1)
        iters += 1
    DIFFUSION.compute_threshold()
    det_lams = linspace(OPTICAL.lam0-2, OPTICAL.lam0+2, 401)+0.2j*(voltage-0.5)/1.5
    det_vals = abs(OPTICAL.get_determinant(det_lams, m=0))
    det_mins = np.r_[False, det_vals[1:] < det_vals[:-1]] & \
               np.r_[det_vals[:-1] < det_vals[1:], False] & \
               np.r_[det_vals[:] < 1]
    mode_number = OPTICAL.find_mode(max(det_lams[det_mins]))
    mode_loss = OPTICAL.outLoss(mode_number)
    print_log(LOG_RESULT,
              'V = {:.3f}V, I = {:.3f}mA, lam = {:.2f}nm, loss = {}/cm'
              .format(voltage[0], ELECTRICAL.get_total_current(), OPTICAL.outWavelength(mode_number), mode_loss))
    return mode_loss
                
OPTICAL.lam0 = 981.5
OPTICAL.vat = 0

threshold_voltage = optimize.fsolve(loss_on_voltage, 1.5, xtol=0.01)
loss_on_voltage(threshold_voltage)
threshold_current = abs(ELECTRICAL.get_total_current())
print_log(LOG_WARNING, "Vth = {:.3f}V    Ith = {:.3f}mA"
                       .format(threshold_voltage, threshold_current))

geometry_width = GEO.GeoO.bbox.upper[0]
geometry_height = GEO.GeoO.bbox.upper[1]
RR = linspace(-geometry_width, geometry_width, 200)
ZZ = linspace(0, geometry_height, 500)
intensity_mesh = mesh.Rectangular2D(RR, ZZ)

IntensityField = OPTICAL.outLightMagnitude(len(OPTICAL.outWavelength)-1, intensity_mesh)
figure()
plot_field(IntensityField, 100)
plot_geometry(GEO.GeoO, mirror=True, color="w")
gcf().canvas.set_window_title('Light Intensity Field ({0} micron aperture)'.format(GEO["aperture"].dr))
axvline(x=GEO["aperture"].dr, color='w', ls=":", linewidth=1)
axvline(x=-GEO["aperture"].dr, color='w', ls=":", linewidth=1)
xticks(append(xticks()[0], [-GEO["aperture"].dr, GEO["aperture"].dr]))
xlabel(u"r [\xb5m]")
ylabel(u"z [\xb5m]")

new_aperture = 3.
GEO["aperture"].dr = new_aperture
GEO["oxide"].dr = DEF["mesaRadius"] - new_aperture

OPTICAL.lam0=982.
threshold_voltage = scipy.optimize.brentq(loss_on_voltage, 0.5, 2., xtol=0.01)
loss_on_voltage(threshold_voltage)
threshold_current = abs(ELECTRICAL.get_total_current())
print_log(LOG_WARNING, "Vth = {:.3f}V    Ith = {:.3f}mA"
                       .format(threshold_voltage, threshold_current))

IntensityField = OPTICAL.outLightMagnitude(len(OPTICAL.outWavelength)-1, intensity_mesh)
figure()
plot_field(IntensityField, 100)
plot_geometry(GEO.GeoO, mirror=True, color="w")
gcf().canvas.set_window_title('Light Intensity Field ({0} micron aperture)'.format(GEO["aperture"].dr))
axvline(x=GEO["aperture"].dr, color='w', ls=":", linewidth=1)
axvline(x=-GEO["aperture"].dr, color='w', ls=":", linewidth=1)
xticks(append(xticks()[0], [-GEO["aperture"].dr, GEO["aperture"].dr]))
xlabel(u"r [\xb5m]")
ylabel(u"z [\xb5m]")

figure()
plot_geometry(GEO.GeoTE, margin=0.01)
gcf().canvas.set_window_title("GEO TE")

figure()
plot_geometry(GEO.GeoTE, margin=0.01)
defmesh = MSG.default(GEO.GeoTE.item)
plot_mesh(defmesh, color="0.75")
plot_boundary(ELECTRICAL.voltage_boundary, defmesh, ELECTRICAL.geometry, color="b", marker="D")
plot_boundary(THERMAL.temperature_boundary, defmesh, THERMAL.geometry, color="r")
gcf().canvas.set_window_title("Default mesh")

show()

sys.exit()

GEO.junction

class A(object):
    def __init__(self):
        self.a = 1
        
a = A()
print a.a
]]></script>

</plask>
