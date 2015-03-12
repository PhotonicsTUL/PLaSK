<plask>

<defines>
  <define name="mesaRadius" value="10."/>
  <define name="aperture" value="{mesaRadius-6.}"/>
  <define name="beta_def" value="19"/>
  <define name="js_def" value="1"/>
</defines>

<materials>
  <material name="InGaAsQW:Si" base="In(0.2)GaAs:Si">
    <nr>3.621</nr>
    <absp>0</absp>
    <A>110000000</A>
    <B>7e-011-1.08e-12*(T-300)</B>
    <C>1e-029+1.4764e-33*(T-300)</C>
    <D>10+0.016670*(T-300)</D>
  </material>
</materials>

<geometry>
  <cylindrical2d name="GeoTE" axes="rz">
    <stack name="stack">
      <item path="abs123" right="{mesaRadius-1}">
        <rectangle name="n-contact" material="Au" dr="4" dz="0.0500"/>
      </item>
      <item>
        <stack name="VCSEL">
          <item>
            <stack name="top-DBR" repeat="24">
              <rectangle material="GaAs:Si=2e+18" dr="{mesaRadius}" dz="0.07003"/>
              <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesaRadius}" dz="0.07945"/>
            </stack>
          </item>
          <item>
            <rectangle material="GaAs:Si=2e+18" dr="{mesaRadius}" dz="0.07003"/>
          </item>
          <rectangle material="Al(0.73)GaAs:Si=2e+18" dr="{mesaRadius}" dz="0.03178"/>
          <shelf>
            <rectangle name="aperture" material="AlAs:Si=2e+18" dr="{aperture}" dz="0.01603"/>
            <rectangle name="oxide" material="AlxOy" dr="{mesaRadius-aperture}" dz="0.01603"/>
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
          <item>
            <stack name="bottom-DBR" repeat="29">
              <rectangle material="Al(0.73)GaAs:C=2e+18" dr="{mesaRadius}" dz="0.07945"/>
              <rectangle material="GaAs:C=2e+18" dr="{mesaRadius}" dz="0.07003"/>
            </stack>
          </item>
          <rectangle material="Al(0.73)GaAs:C=2e+18" dr="{mesaRadius}" dz="0.07945"/>
        </stack>
      </item>
      <zero/>
      <item>
        <rectangle material="GaAs:C=2e+18" dr="200." dz="150."/>
      </item>
      <item>
        <rectangle name="p-contact" material="Cu" dr="2500." dz="5000."/>
      </item>
    </stack>
  </cylindrical2d>
  <cylindrical2d name="GeoO" axes="x,y,z" outer="extend" bottom="GaAs" top="air">
    <align left="0" bottom="0">
      <item>
        <clip>
          <again ref="VCSEL"/>
        </clip>
      </item>
    </align>
  </cylindrical2d>
  <cartesian3d axes="x,y,z">
    <align>
      <item>
        <revolution>
          <again ref="VCSEL"/>
        </revolution>
      </item>
    </align>
  </cartesian3d>
  <cartesian2d>
    <stack>
      <item>
        <rectangle material="GaN" dtran="1" dvert="0.2"/>
      </item>
      <item>
        <rectangle material="Al(0.2)GaN" dtran="2" dvert="0.5"/>
      </item>
      <item>
        <rectangle material="GaN" dtran="2" dvert="1"/>
      </item>
    </stack>
  </cartesian2d>
  <cartesian3d axes="z,x,y">
    <align back="0" x="0" top="0">
      <extrusion length="10">
        <again ref="stack"/>
      </extrusion>
      <item>
        <cuboid material="GaAs" dz="1" dx="100" dy="100"/>
      </item>
    </align>
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
    <axis start="0" stop="{mesaRadius}" num="2000"></axis>
  </mesh>
  <generator method="divide" name="optical" type="rectangular2d">
    <options aspect="100" gradual="no"/>
    <prediv by0="10" by1="3"/>
  </generator>
  <mesh name="plots" type="rectangular2d">
    <axis0></axis0>
    <axis1></axis1>
  </mesh>
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
    <mode lam0="980"/>
  </optical>
  <optical name="fourier" solver="Fourier3D" lib="slab">
    <mode symmetry="Etran,Etran"/>
  </optical>
  <filter for="Temperature" geometry="GeoTE" name="filtr"/>
  <optical name="efm" solver="EffectiveFrequencyCyl" lib="effective"/>
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
from scipy import optimize

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

class A(object):
    def __init__(self):
        self.a = 1
        
a = A()
print a.a

]]></script>

</plask>