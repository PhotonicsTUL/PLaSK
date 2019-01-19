<plask loglevel="detail">

<defines>
  <define name="aperture" value="8"/>
  <define name="mesa" value="{4. * aperture}"/>
  <define name="beta_def" value="19"/>
  <define name="js_def" value="1"/>
  <define name="L" value="4.0"/>
  <define name="d" value="0.5"/>
  <define name="X" value="{(6.1*sqrt(3)/2+d)*L}"/>
  <define name="Y" value="{(6.1+d)*L}"/>
  <define name="h_start" value="0"/>
  <define name="h_end" value="24"/>
  <define name="f" value="0"/>
  <define name="lineto" value="200."/>
  <define name="tt" value="30"/>
</defines>

<materials>
  <material name="test" base="semiconductor">
    <A>0.1 * T + 0.02 * (T-300)**2</A>
    <NR>3.6, 3.6, 3.4, 0.0</NR>
    <thermk>10.+ 0.001 * T**2</thermk>
  </material>
  <material name="InGaAsQW" base="In(0.2)GaAs"/>
  <material name="InGaAs_QW:Si" base="InGaAs:Si" alloy="yes">
    <nr>3.621</nr>
    <thermk>self.In</thermk>
    <A>110000000</A>
    <B>7e-011-1.08e-12*(T-300)</B>
    <C>1e-029+1.4764e-33*(T-300)</C>
    <D>10+0.016670*(T-300)</D>
  </material>
  <material name="name" base="semiconductor"/>
  <module name="mats"/>
  <material name="GaAs2:Si" base="GaAs:Si"/>
  <material name="mat" base="semiconductor">
    <nr>5</nr>
  </material>
  <material name="a" base="semiconductor">
    <thermk>100 + T/{tt}</thermk>
  </material>
  <material name="b" base="a"/>
  <material name="GaAs" base="GaAs">
    <y1>10</y1>
    <y2>20</y2>
    <y3>30</y3>
  </material>
</materials>

<geometry>
  <cartesian2d name="geo2d" bottom="extend" length="1000">
    <stack name="stack2d">
      <shelf flat="no">
        <stack name="new">
          <arrange name="Pilars" dtran="0.4" dvert="0" count="3">
            <stack name="STOS">
              <rectangle name="rr" material="InN" dtran="0.2" dvert="0.1"/>
              <item left="0">
                <rectangle material="In(0.5)GaN:Si=1e18" dtran="0.2" dvert="0.1"/>
              </item>
            </stack>
          </arrange>
          <again ref="Pilars"/>
          <rectangle material="Al(0.9)GaN:Si=2e18" dtran="1" dvert="0.1"/>
          <rectangle material="Al(0.5)GaN:Si=2e18" dtran="1" dvert="0.2"/>
        </stack>
        <gap total="2"/>
        <stack name="stos2">
          <item path="tadam" right="0.8">
            <triangle material="AlOx" atran="-0.4" avert="0" btran="0" bvert="0.2"/>
          </item>
          <rectangle material="AlN" dtran="0.8" dvert="0.1"/>
        </stack>
        <stack name="pikusik">
          <rectangle name="kwadrat" material="AlGa(0.1)N" dtran="0.1" dvert="0.1"/>
          <rectangle material="AlGa(0.5)N" dtran="0.1" dvert="0.2"/>
        </stack>
      </shelf>
      <rectangle name="posredni" material="Al(0.2)GaN" dtran="2" dvert="0.5"/>
      <rectangle role="substrate" material="GaN" dtran="2" dvert="1"/>
    </stack>
  </cartesian2d>
  <cartesian2d name="geo2d-copy">
    <copy from="stack2d">
      <toblock object="new" material="GaAs" name="blok2" role="rola1"/>
      <replace object="stos2">
        <shelf2d>
          <rectangle material="GaAs" dtran="0.4" dvert="0.5"/>
          <rectangle material="AlAs" dtran="0.4" dvert="0.5"/>
        </shelf2d>
      </replace>
      <delete object="posredni"/>
      <replace object="pikusik" with="kwadrat"/>
    </copy>
  </cartesian2d>
  <cartesian2d name="simple">
    <stack>
      <again ref="blok2"/>
      <stack repeat="3">
        <rectangle material="Al(0.9)GaAs" dtran="1" dvert="0.3"/>
        <rectangle material="Al(0.9)GaN" dtran="1" dvert="0.2"/>
      </stack>
      <rectangle name="one" material="Al(0.73)GaAs:C=2e+18" dtran="1" dvert="1.0"/>
      <rectangle material="Al(0.73)GaN:Si=1e18" dtran="1" dvert="1.0"/>
    </stack>
  </cartesian2d>
  <cartesian3d name="l3cavity">
    <stack front="0">
      <lattice along="{-sqrt(3)/2}" atran="0.5" blong="{sqrt(3)/2}" btran="0.5">
        <segments>-4 0; 0 4; 4 4; 4 0; 0 -4; -4 -4 ^ -1 -2; -2 -2; -2 -1; 1 2; 2 2; 2 1</segments>
        <cylinder material="GaAs" radius="0.35" height="1.0"/>
      </lattice>
    </stack>
  </cartesian3d>
  <cartesian3d name="vcsel" axes="x,y,z" back="mirror" front="extend" left="mirror" right="extend" bottom="GaAs">
    <clip>
      <align x="0" y="0" top="0">
        <item xcenter="0" ycenter="0">
          <stack>
            <stack name="top-dbr" repeat="24">
              <cuboid material="GaAs" dx="{X}" dy="{Y}" dz="0.06940"/>
              <cuboid material="Al(0.7)GaAs" dx="{X}" dy="{Y}" dz="0.07955"/>
            </stack>
            <stack name="cavity">
              <cuboid material="GaAs" dx="{X}" dy="{Y}" dz="0.12171"/>
              <stack name="active">
                <align name="qw" xcenter="0" ycenter="0" bottom="0">
                  <cuboid material="In(0.7)GaAs" dx="{X}" dy="{Y}" dz="0.00800"/>
                  <cylinder name="gain" role="gain" material="In(0.7)GaAs" radius="{L/2}" height="0.00800"/>
                </align>
                <cuboid name="interface" material="GaAs" dx="{X}" dy="{Y}" dz="0.00500"/>
                <again ref="qw"/>
                <cuboid material="GaAs" dx="{X}" dy="{Y}" dz="0.00500"/>
                <again ref="qw"/>
              </stack>
              <cuboid material="GaAs" dx="{X}" dy="{Y}" dz="0.12171"/>
            </stack>
            <stack name="bottom-dbr" repeat="29">
              <cuboid material="Al(0.2)GaAs" dx="{X}" dy="{Y}" dz="0.07955"/>
              <cuboid material="GaAs" dx="{X}" dy="{Y}" dz="0.06940"/>
            </stack>
          </stack>
        </item>
        <item top="{-h_start*(0.06940+0.07955)}">
          <lattice ax="0" ay="{L}" az="0" bx="{L*sqrt(3)/2}" by="{L/2}" bz="0">
            <segments>-1 -3; 4 -3; 4 -2; 3 0; 2 2; 1 3; -4 3; -4 2; -3 0; -2 -2 ^ 0 -1; 1 -1; 1 0; 0 1; -1 1; -1 0</segments>
            <cylinder material="air" radius="{0.5*d*L}" height="{(h_end-h_start)*(0.06940+0.07955)}"/>
          </lattice>
        </item>
      </align>
    </clip>
  </cartesian3d>
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
    <stack>
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
  <cartesian2d name="main" axes="x,y">
    <rectangle material="mat" dx="1" dy="{wl('mat', 1000)}"/>
  </cartesian2d>
  <cartesian3d name="prismatic" axes="x,y,z">
    <stack>
      <prism material="AlAs" ax="0.9" ay="-0.5" bx="-0.1" by="1.5" height="1.0"/>
      <cuboid material="GaAs" dx="1.0" dy="2.0" dz="0.5"/>
    </stack>
  </cartesian3d>
  <cartesian2d name="roads">
    <stack>
      <shelf>
        <item path="bl,blsl,blsr">
          <stack name="big">
            <shelf>
              <item path="sl,blsl,brsl">
                <rectangle name="small" material="AlAs" dtran="0.333" dvert="0.2"/>
              </item>
              <gap total="1.0"/>
              <item path="sr,blsr,brsr">
                <again ref="small"/>
              </item>
            </shelf>
            <rectangle material="AlN" dtran="1.0" dvert="0.5"/>
          </stack>
        </item>
        <gap total="3.0"/>
        <item path="br,brsl,brsr">
          <again ref="big"/>
        </item>
      </shelf>
      <rectangle material="GaN" dtran="3.0" dvert="0.5"/>
    </stack>
  </cartesian2d>
</geometry>

<grids>
  <generator method="divide" name="default" type="rectangular2d">
    <postdiv by0="2" by1="1"/>
    <refinements>
      <axis1 object="p-contact" at="50"/>
      <axis0 object="oxide" at="-0.1"/>
      <axis0 object="oxide" at="-0.05"/>
      <axis0 object="aperture" at="0.1"/>
    </refinements>
  </generator>
  <mesh name="diffusion" type="regular">
    <axis start="0" stop="{mesa}" num="200"></axis>
  </mesh>
  <generator method="divide" name="optical" type="rectangular2d">
    <prediv by0="10" by1="3"/>
    <options aspect="100" gradual="no"/>
  </generator>
  <generator method="smooth" name="smoothie" type="rectangular2d">
    <steps small0="0.005" small1="0.01" large0="0.05" factor="1.2"/>
  </generator>
  <generator method="divide" name="oned" type="ordered">
    <refinements>
      <axis0 object="bottom-DBR" at="1"/>
    </refinements>
  </generator>
  <mesh name="plots" type="rectangular2d">
    <axis0 start="0" stop="10" num="20"></axis0>
    <axis1 start="0" stop="1" num="10"></axis1>
  </mesh>
  <generator method="smooth" name="sss" type="rectangular3d">
    <steps small0="0.005" small1="0.05" small2="0.05" factor="1.2"/>
  </generator>
  <generator method="regular" name="reg" type="rectangular2d">
    <spacing every0="0.2" every1="1"/>
  </generator>
  <generator method="simple" name="spl" type="rectangular2d">
    <boundaries split="yes"/>
  </generator>
  <mesh name="fine" type="rectangular3d">
    <axis0 start="0.5" stop="1.0" num="2001"></axis0>
    <axis1 start="0" stop="2" num="4001"></axis1>
    <axis2>0.0 0.5 1.5</axis2>
  </mesh>
</grids>

<solvers>
  <thermal name="THERMAL" solver="StaticCyl" lib="static">
    <geometry ref="GeoT"/>
    <mesh ref="default"/>
    <temperature>
      <condition value="320.">
        <place line="horizontal" at="10" start="0" stop="{lineto}"/>
      </condition>
      <condition place="bottom" value="300."/>
    </temperature>
  </thermal>
  <optical name="fourier2" solver="Fourier2D" lib="slab">
    <geometry ref="geo2d"/>
  </optical>
  <gain name="gain2" solver="FreeCarrierCyl" lib="freecarrier">
    <geometry ref="GeoO"/>
    <config T0="300" matrix-elem="10"/>
  </gain>
  <electrical name="ELECTRICAL" solver="ShockleyCyl" lib="shockley">
    <geometry ref="GeoE"/>
    <mesh ref="default"/>
    <voltage>
      <condition value="1">
        <place side="top" object="n-contact"/>
      </condition>
      <condition place="bottom" value="0"/>
    </voltage>
    <matrix algorithm="cholesky" itererr="2"/>
    <junction beta0="{beta_def}" beta1="19.2" js0="{js_def}" js1="1.1"/>
  </electrical>
  <electrical name="DIFFUSION" solver="DiffusionCyl" lib="diffusion">
    <geometry ref="GeoO"/>
    <mesh ref="diffusion"/>
  </electrical>
  <gain name="GAIN" solver="FreeCarrierCyl" lib="freecarrier">
    <geometry ref="GeoO"/>
    <config lifetime="0.5" matrix-elem="8"/>
  </gain>
  <optical name="OPTICAL" solver="EffectiveFrequencyCyl" lib="effective">
    <geometry ref="GeoO"/>
    <mesh ref="optical"/>
    <mode emission="bottom" lam0="980"/>
  </optical>
  <filter for="Temperature" geometry="GeoT" name="filtr"/>
  <optical name="efm" solver="EffectiveFrequencyCyl" lib="effective">
    <geometry ref="GeoO"/>
  </optical>
  <electrical name="DDM" solver="DriftDiffusion2D" lib="ddm2d">
    <geometry ref="geo2d"/>
    <mesh ref="optical"/>
    <voltage>
      <condition value="0">
        <place side="bottom" object="stack2d"/>
      </condition>
    </voltage>
  </electrical>
  <meta name="meta2" solver="ThermoElectric2D" lib="shockley">
    <geometry electrical="roads" thermal="geo2d-copy"/>
    <mesh electrical="default" thermal="default"/>
    <voltage>
      <condition value="1.">
        <place side="top" object="small" path="sr"/>
      </condition>
      <condition value="0.">
        <place line="horizontal" at="0" start="1" stop="2"/>
      </condition>
    </voltage>
  </meta>
  <meta name="bessel" solver="ThresholdSearchBesselCyl" lib="shockley">
    <geometry electrical="GeoE" optical="GeoO" thermal="GeoT"/>
    <mesh diffusion="diffusion" electrical="default" thermal="default"/>
    <root bcond="0"/>
  </meta>
  <meta name="threshold" solver="ThresholdSearchBesselCyl" lib="shockley">
    <geometry electrical="GeoE" optical="GeoO" thermal="GeoT"/>
    <mesh electrical="default" thermal="default"/>
    <root bcond="1"/>
    <voltage>
      <condition place="bottom" value="0"/>
      <condition value="1">
        <place side="top" object="n-contact"/>
      </condition>
    </voltage>
    <temperature>
      <condition place="bottom" value="300"/>
    </temperature>
  </meta>
  <optical name="F3D" solver="Fourier3D" lib="slab">
    <geometry ref="l3cavity"/>
  </optical>
  <thermal name="solver" solver="Static3D" lib="static">
    <geometry ref="prismatic"/>
    <mesh ref="fine"/>
  </thermal>
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
</connects>

<script><![CDATA[
from __future__ import print_function

GaAs = material.get('GaAs')
print(GaAs.y1(), GaAs.y2(), GaAs.y3())

print(material.get('b').thermk(300))

print_log('result', """\
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure \
dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non \
proident, sunt in culpa qui officia deserunt mollit anim id est laborum.""")

# def p():
#     print 1
#     print(2); print 3

import os
import sys


csys = 1
cmap = 2

print_log('info', "START")

print_log('data', os.environ.get('DISPLAY'))

figure()
xx = linspace(0., 12., 1001)
plot(xx, sin(xx))
figure()
xx = linspace(0., 12., 1001)
plot(xx, cos(xx), color='C1')
show()

figure()
xx = linspace(0., 12., 1001)
plot(xx, sin(xx)/xx, color='C2')
show()


from scipy import optimize
import sys

print_log(LOG_RESULT, sys.executable)

print_log(LOG_RESULT, "DEFINES")
for item in list(DEF.items()):
    print_log(LOG_RESULT, "{} = {}".format(*item))

print_log(LOG_RESULT, "ARGUMENTS")
for arg in sys.argv[1:]:
    print_log(LOG_RESULT, arg)

# print ur"Python2 style"

print(mesa + 0, )

print_log('data', "łóżko")
print_log('info', "informacja")

# OPTICAL.find_mode

print(f, file=sys.stderr)

class A(object):
    
    def __init__(self):
        pass
    
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
        return 0xff

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
    while (terr >= THERMAL.maxerr or verr >= ELECTRICAL.maxerr) and iters <= 15:
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
xlabel("r [\xb5m]")
ylabel("z [\xb5m]")

new_aperture = 3.
GEO["aperture"].dr = new_aperture
GEO["oxide"].dr = DEF["mesa"] - new_aperture

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
xlabel("r [\xb5m]")
ylabel("z [\xb5m]")

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
print(a.a, file=sys.stderr)
]]></script>

</plask>
