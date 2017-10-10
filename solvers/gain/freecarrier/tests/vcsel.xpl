<plask loglevel="detail">

<defines>
  <define name="U" value="1.2"/>
  <define name="m" value="0"/>
  <define name="n" value="1"/>
  <define name="U0" value="{U}"/>
  <define name="UU" value="arange(1.2, 3.01, 0.1)"/>
  <define name="oapprox" value="None"/>
  <define name="presentation" value="True"/>
  <define name="save_fig" value="True"/>
  <define name="save_h5" value="False"/>
  <define name="screen" value="True"/>
  <define name="h_oxide" value="0.020"/>
  <define name="oxide_loc" value="1.0 # 0.0 or 1.0 = antinode, 0.5 = node"/>
  <define name="n_QW" value="6"/>
  <define name="h_QW" value="0.0060"/>
  <define name="h_bar" value="0.0067"/>
  <define name="aprt" value="6."/>
  <define name="aprtc" value="16."/>
  <define name="mesa" value="40."/>
  <define name="r_substr" value="1000."/>
  <define name="gN" value="0.3"/>
  <define name="gT" value="300."/>
  <define name="heat_t" value="100."/>
  <define name="cool_t" value="{500.-heat_t}"/>
  <define name="hc_cycles" value="20"/>
  <define name="u_t" value="5"/>
  <define name="adt" value="False"/>
  <define name="Tamb" value="300."/>
  <define name="PP" value="None"/>
  <define name="P" value="1e-6"/>
  <define name="modes" value="(0,1), (1,1), (2,1), (0,2), (3,1)"/>
  <define name="THERMAL" value="'T_STATIC'"/>
  <define name="ELECTRICAL" value="'E_SCHOCKLEY'"/>
  <define name="DIFFUSION" value="'D_PARABOLIC'"/>
  <define name="GAIN" value="'G_FREE'"/>
  <define name="OPTICAL" value="'O_BESSEL'"/>
</defines>

<materials>
  <material name="SiO2" base="semiconductor">
    <cp>700</cp>
    <dens>2200</dens>
    <nr>1.55</nr>
  </material>
  <material name="Si" base="semiconductor">
    <cp>712</cp>
    <dens>2329</dens>
    <nr>3.52</nr>
  </material>
  <material name="InAlGaAs" base="semiconductor">
    <thermk>4.0, 4.0</thermk>
    <cond>1e-06, 0.2</cond>
    <cp>380</cp>
    <dens>5000</dens>
    <A>7e7+(T-300)*1.4e5</A>
    <B>1.1e-10-(T-300)*-2.2e-13</B>
    <C>1.130976e-28-(T-300)*1.028743e-30+(T-300)*(T-300)*8.694142e-32</C>
    <D>10</D>
    <VB>-0.75</VB>
    <Eg>1.30-0.0003*(T-300)</Eg>
    <Dso>0.3548</Dso>
    <Me>0.103</Me>
    <Mhh>0.6</Mhh>
    <Mlh>0.14</Mlh>
    <nr>3.3 + (wl-1310)*5.00e-4 + (T-300)*7e-4</nr>
    <absp>50. + (T-300)*7e-2</absp>
  </material>
  <material name="InAlGaAs-QW" base="InAlGaAs">
    <thermk>4.0, 4.0</thermk>
    <cond>1e-06, 0.2</cond>
    <cp>1000</cp>
    <dens>1000</dens>
    <A>7e7+(T-300)*1.4e5</A>
    <B>1.1e-10-(T-300)*(-2.2e-13)</B>
    <C>1.130976e-28-(T-300)*1.028743e-30+(T-300)**2*8.694142e-32</C>
    <D>10</D>
    <VB>0.14676+0.000033*(T-300)-0.75</VB>
    <Eg>0.87-0.0002*(T-300)</Eg>
    <Dso>0.3548</Dso>
    <Me>0.052</Me>
    <Mhh>0.477</Mhh>
    <Mlh>0.103</Mlh>
    <nr>3.6 + (wl-1310)*5.00e-4 + (T-300)*7e-4</nr>
    <absp>1000.</absp>
  </material>
  <material name="P++" base="GaAs">
    <thermk>1.4, 1.4</thermk>
    <cond>1000000.0, 1000000.0</cond>
  </material>
  <material name="TJ" base="P++">
    <cond>5.0, 5.0</cond>
  </material>
  <material name="TJB" base="GaAs">
    <thermk>1.4, 1.4</thermk>
    <cond>1e-06, 0.0001</cond>
  </material>
  <material name="AlOx" base="AlOx">
    <cp>880</cp>
    <dens>3690</dens>
  </material>
  <material name="Cu" base="Cu">
    <cp>386</cp>
    <dens>8960</dens>
  </material>
  <material name="In" base="In">
    <cp>233</cp>
    <dens>7310</dens>
  </material>
  <material name="Au" base="Au">
    <cp>126</cp>
    <dens>19300</dens>
  </material>
</materials>

<geometry>
  <cylindrical2d name="main" axes="rz" bottom="GaAs">
    <clip right="120">
      <stack name="vcsel">
        <shelf flat="false">
          <stack name="vcsel-layers">
            <shelf flat="false">
              <gap size="{aprtc/2}"/>
              <stack>
                <rectangle name="top-contact" material="Au" dr="{(mesa-aprtc)/2-2}" dz="0.035"/>
                <rectangle role="p-contact" material="Au" dr="{(mesa-aprtc)/2-2}" dz="0.005"/>
              </stack>
            </shelf>
            <stack repeat="{24}">
              <rectangle material="GaAs:C=3e+18" dr="{mesa/2}" dz="0.0958"/>
              <rectangle material="AlAs:C=3e+18" dr="{mesa/2}" dz="0.1124"/>
            </stack>
            <rectangle material="GaAs:C=3e+18" dr="{mesa/2}" dz="{0.5*oxide_loc*0.3831-0.85*h_oxide/2}"/>
            <shelf>
              <rectangle name="aperture" material="AlAs:C=3e+18" dr="{aprt/2}" dz="{h_oxide}"/>
              <rectangle material="AlOx" dr="{(mesa-aprt)/2}" dz="{h_oxide}"/>
            </shelf>
            <rectangle material="GaAs:C=3e+18" dr="{mesa/2}" dz="{(1.0-0.5*oxide_loc)*0.3831-(0.85*h_oxide+1.06*n_QW*h_QW+0.97*(n_QW+1)*h_bar)/2}"/>
            <stack name="active" role="active">
              <rectangle name="barrier" material="InAlGaAs" dr="{mesa/2}" dz="{h_bar}"/>
              <stack repeat="{n_QW}">
                <rectangle name="QW" role="QW" material="InAlGaAs-QW" dr="{mesa/2}" dz="{h_QW}"/>
                <again ref="barrier"/>
              </stack>
            </stack>
            <rectangle material="GaAs:Si=2e+18" dr="{mesa/2}" dz="{0.5*0.3831-(1.06*n_QW*h_QW+0.97*(n_QW+1)*h_bar)/2}"/>
            <stack repeat="{35}">
              <rectangle material="AlAs:Si=2e+18" dr="{mesa/2}" dz="0.1124"/>
              <rectangle material="GaAs:Si=2e+18" dr="{mesa/2}" dz="0.0958"/>
            </stack>
          </stack>
          <gap size="30"/>
          <stack>
            <rectangle name="bottom-contact" material="Au" dr="50" dz="0.035"/>
            <rectangle role="n-contact" material="Au" dr="50" dz="0.005"/>
          </stack>
        </shelf>
        <rectangle material="GaAs:Si=3e+18" dr="{r_substr}" dz="0.500"/>
      </stack>
    </clip>
  </cylindrical2d>
  <cylindrical2d name="thermal" axes="rz">
    <stack>
      <again ref="vcsel"/>
      <zero/>
      <rectangle material="GaAs" dr="{r_substr}" dz="299.5"/>
      <rectangle material="In" dr="{r_substr}" dz="3.0"/>
      <rectangle material="Cu" dr="{r_substr}" dz="1000."/>
    </stack>
  </cylindrical2d>
  <cylindrical2d name="optical" axes="rz" outer="extend" bottom="GaAs">
    <clip right="{mesa/2-0.001}">
      <again ref="vcsel"/>
    </clip>
  </cylindrical2d>
</geometry>

<grids>
  <generator method="divide" name="default" type="rectangular2d">
    <postdiv by0="4" by1="2"/>
    <refinements>
      <axis0 object="aperture" at="{aprt/2-0.2}"/>
    </refinements>
  </generator>
  <mesh name="active" type="regular">
    <axis start="0" stop="{mesa/2}" num="2001"></axis>
  </mesh>
  <mesh name="gain" type="regular">
    <axis start="0" stop="{mesa/2}" num="80"></axis>
  </mesh>
  <generator method="divide" name="optical" type="ordered">
    <options gradual="no"/>
    <refinements>
      <axis0 object="active" every="0.25"/>
    </refinements>
  </generator>
</grids>

<solvers>
  <thermal name="T_STATIC" solver="StaticCyl" lib="static">
    <geometry ref="thermal"/>
    <mesh ref="default"/>
    <temperature>
      <condition place="bottom" value="{Tamb}"/>
    </temperature>
  </thermal>
  <thermal name="T_DYNAMIC" solver="DynamicCyl" lib="dynamic">
    <geometry ref="thermal"/>
    <mesh ref="default"/>
    <loop logfreq="10" rebuildfreq="0" timestep="0.1"/>
    <matrix lumping="yes"/>
  </thermal>
  <electrical name="E_SCHOCKLEY" solver="ShockleyCyl" lib="shockley">
    <geometry ref="main"/>
    <mesh ref="default"/>
    <voltage>
      <condition value="{U}">
        <place side="top" object="top-contact"/>
      </condition>
      <condition value="0">
        <place side="top" object="bottom-contact"/>
      </condition>
    </voltage>
    <junction beta0="10." js0="500." pnjcond="0.02"/>
    <contacts ncond="8.34" pcond="8.34"/>
  </electrical>
  <electrical name="D_PARABOLIC" solver="DiffusionCyl" lib="diffusion">
    <geometry ref="main"/>
    <mesh ref="active"/>
    <config accuracy="0.005" fem-method="parabolic"/>
  </electrical>
  <gain name="G_FREE" solver="FreeCarrierCyl" lib="freecarrier">
    <geometry ref="main"/>
    <config T0="300." lifetime="0.1" matrix-elem="16"/>
  </gain>
  <optical name="O_EFM" solver="EffectiveFrequencyCyl" lib="effective">
    <geometry ref="optical"/>
    <mesh ref="optical"/>
    <mode emission="top" lam0="1300" vat="0"/>
  </optical>
  <optical name="O_BESSEL" solver="BesselCyl" lib="slab">
    <geometry ref="optical"/>
    <expansion lam0="1300" size="20"/>
    <interface object="active"/>
    <pml dist="40"/>
  </optical>
</solvers>

<script><![CDATA[
import sys
import os
import fcntl
import collections

from scipy.optimize import newton, brentq
from scipy.optimize import fsolve, broyden1, broyden2

from thermal.dynamic import DynamicCyl
from optical.slab import BesselCyl

THERMAL    = eval(THERMAL)
ELECTRICAL = eval(ELECTRICAL)
DIFFUSION  = eval(DIFFUSION)
GAIN       = eval(GAIN)
OPTICAL    = eval(OPTICAL)

try:
    task = sys.argv[1]
except IndexError:
    task = 'CO'
task = task.upper()

print_log('data', "Task:", task)

if task == 'TT'  and not isinstance(THERMAL, DynamicCyl):
    print_log('warning', "Temperature time evolution requested. Setting THERMAL to T_DYNAMIC")
    THERMAL = T_DYNAMIC

solvers = THERMAL.id[2] + ELECTRICAL.id[2] + DIFFUSION.id[2] + GAIN.id[2] + OPTICAL.id[2]

desc = "{}-o{:.2f}-a{:.1f}um-{}".format(task.lower(), oxide_loc, aprt, solvers.lower())


# Connect solvers
ELECTRICAL.inTemperature = THERMAL.outTemperature
DIFFUSION.inTemperature = THERMAL.outTemperature
GAIN.inTemperature = THERMAL.outTemperature
OPTICAL.inTemperature = THERMAL.outTemperature
THERMAL.inHeat = ELECTRICAL.outHeat
DIFFUSION.inCurrentDensity = ELECTRICAL.outCurrentDensity
GAIN.inCarriersConcentration = DIFFUSION.outCarriersConcentration
OPTICAL.inGain = GAIN.outGain
DIFFUSION.inGain = GAIN.outGain
DIFFUSION.inLightMagnitude = OPTICAL.outLightMagnitude
DIFFUSION.inWavelength = OPTICAL.outWavelength


if isinstance(OPTICAL, BesselCyl):
    m += 1


try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


if presentation:
    rc.figure.figsize = 4.0, 3.0
    rc.font.serif = 'Iwona'
    rc.font.sans = 'Iwona'
    rc.mathtext.fontset = 'custom'
    rc.mathtext.cal = 'Iwona'
    rc.mathtext.rm  = 'Iwona'
    rc.mathtext.it  = 'Iwona:italic'
    rc.mathtext.bf  = 'Iwona:bold'
    rc.mathtext.sf  = 'Iwona'
    try:
        style.use('dark_background')
    except NameError:
        pass

if not screen:
    switch_backend('Agg')

color_cycle = [c['color'] for c in rc.axes.prop_cycle]

class SaveFigure(object):
    def __init__(self):
        self.nr = 1
    def __call__(self, title=None):
        if not save_fig: return
        if title is None:
            title = self.nr
            self.nr += 1
        savefig('{}-{}.svg'.format(desc, title))
save_figure = SaveFigure()


T = None

zqw = GEO.main.get_object_bboxes(GEO.QW)[0].center.z

olams = linspace(1280, 1320, 241)
glams = linspace(1280, 1320, 241)

OPTICAL.mesh = MSG.optical


# desc = os.path.join('Results', 'tajwan-cav{cavity}-re{aperture:.2f}-ta{Ta:.0f}'.format(**DEF))
#
# if task == 'TE':
#     desc += '-lp{}{}-v{:.3f}'.format(m, n, U)
# elif task == 'TH':
#     desc += '-lp{}{}-threshold'.format(m, n)
# elif task == 'T0':
#     desc += '-lp{}{}-t0-v{:.3f}'.format(m, n, U)
# elif task == 'L1':
#     desc += '-lp{}{}-v{:.3f}{}'.format(m, n, U, '-at' if adt else '')
# elif task == 'LO':
#     desc += '-lo-v{:.3f}{}'.format(U, '-at' if adt else '')
# elif task == 'LI':
#     desc += '-li'
# else:
#     desc += '-lp{}{}'.format(m, n)


class ApproximateMode(object):
    def __init__(self, m=m, n=1, lams=olams, imags=linspace(-0.05, 0.05, 41)):
        self.lams = lams
        self.no = -n
        self.m = m-1 if isinstance(OPTICAL, BesselCyl) else m
        self.imags = 1j*imags
    def _minimag(self, lam):
        lams = lam + self.imags
        i = abs(O_EFM.get_determinant(lam=lams, m=self.m)).argmin()
        return lams[i]
    def __call__(self):
        dets = abs(O_EFM.get_determinant(lam=self.lams, m=self.m))
        mins = [self.lams[i] for i in range(1, len(dets)-1) if dets[i-1] >= dets[i] and dets[i] < dets[i+1]]
        print_log('data', 'Approx. modes:', ', '.join('{:.2f}'.format(x) for x in reversed(mins)), ' ({})'.format(mins[self.no]))
        return self._minimag(mins[self.no])
    def __iter__(self):
        dets = abs(O_EFM.get_determinant(lam=self.lams, m=self.m))
        mins = [self.lams[i] for i in range(1, len(dets)-1) if dets[i-1] >= dets[i] and dets[i] < dets[i+1]]
        print_log('data', 'Approx. modes:', ', '.join('{:.2f}'.format(x) for x in reversed(mins)))
        return (self._minimag(m) for m in reversed(mins))
def approximate_mode(*args, **kwargs):
    if oapprox is None:
        return ApproximateMode(*args, **kwargs)()
    else:
        return oapprox


def plot_rs(geo):
    simplemesh = plask.mesh.Rectangular2D.SimpleGenerator()(geo)
    for r in simplemesh.axis0:
        axvline(r, ls=":", lw=0.5, color='0.7' if presentation else 'k')


def plot_thermoelectric(save=True, suffix=''):
    if suffix and not suffix.endswith('-'):
        suffix += '-'

    global T

    # figure()
    # plot_geometry(GEO.thermal, margin=0.01)
    # plot_mesh(THERMAL.mesh, color="0.75")
    # plot_boundary(ELECTRICAL.voltage_boundary, ELECTRICAL.mesh, ELECTRICAL.geometry, color="b", marker="D")
    # plot_boundary(THERMAL.temperature_boundary, THERMAL.mesh, THERMAL.geometry, color="r")
    # plot_boundary(THERMAL.convection_boundary, THERMAL.mesh, THERMAL.geometry, color="g")
    # plot_boundary(THERMAL.radiation_boundary, THERMAL.mesh, THERMAL.geometry, color="y")
    # gcf().canvas.set_window_title("Default mesh")

    figure()
    work.plot_temperature(cmap='inferno', geometry_color=(0.75, 0.75, 0.75, 0.1))
    xlim(0., 0.55*mesa)
    ylim(GEO.main.bbox.bottom, GEO.main.bbox.top)
    tight_layout(0.2)
    save_figure(suffix+'temp2d')

    if not presentation:
        fig = figure()
        subplot(221)
    else:
        figure()

    jtemp = THERMAL.outTemperature(mesh.Rectangular2D(THERMAL.mesh.get_midpoints().axis0, [zqw]), 'spline')
    T = max(jtemp)
    plot_profile(jtemp, color=color_cycle[3], label='PLaSK')
    plot_rs(GEO.thermal)
    if presentation:
        gcf().canvas.set_window_title("Junction Temperature")
    else:
        title("Junction Temperature [K]")
    xlim(0., 0.55*mesa)

    if presentation:
        tight_layout(0.2)
        save_figure(suffix+'temp')
        figure()
    else:
        subplot(222)

    work.plot_junction_current(color=color_cycle[4], label='PLaSK')
    plot_rs(GEO.main)
    if presentation:
        gcf().canvas.set_window_title("Junction Current")
    else:
        title("Junction Current")
    xlim(0., 0.55*mesa)

    if presentation:
        tight_layout(0.2)
        save_figure(suffix+'curr')
        figure()
    else:
        subplot(223)

    plot_profile(1e-18 * DIFFUSION.outCarriersConcentration(mesh.Rectangular2D(DIFFUSION.mesh, [zqw]), 'spline'),
                 label='PLaSK', color=color_cycle[2])
    plot_rs(GEO.main)
    ylabel("Carriers Concentration [10$^{18}$/cm$^3$]")
    if presentation:
        gcf().canvas.set_window_title("Carriers Concentration")
    else:
        title("Carriers Concentration")
    xlim(0., 0.55*mesa)

    if presentation:
        tight_layout(0.2)
        save_figure(suffix+'conc')
    else:
        suptitle(u"$U = {U:.3f}$V  $J = {J:.2f}$mA  $T_{{\\mathrm{{max}}}} = {T:.1f}$K".format(**globals()), fontsize=16)

        gcf().canvas.set_window_title("Thermoelectric")
        subplots_adjust(top=0.89)
        tight_layout(0.2)
        save_figure()


def plot_gain(sub=None, suffix=''):
    if suffix and not suffix.endswith('-'):
        suffix += '-'
    if sub is None:
        fig = figure()
        fig.canvas.set_window_title("Gain Profile")
    else:
        subplot(sub)
        title("Gain Profile")
    plot_profile(GAIN.outGain(mesh.Rectangular2D(DIFFUSION.mesh, [zqw]), 1310., 'spline'), color=color_cycle[5])
    gca().ticklabel_format(useOffset=False)
    plot_rs(GEO.main)
    ylabel("Gain Profile [1/cm]")
    xlim(0., 0.55*mesa)
    tight_layout(0.2)
    subplots_adjust(top=0.89)
    save_figure(suffix+'gain')


def plot_gain_spectrum(new=False, label='PLaSK'):
    spectrum = GAIN.spectrum(0, zqw)(glams)
    print(spectrum.shape)
    print(sys.getrefcount(spectrum)-1)
    if new:
        figure()
        plot(glams, spectrum, '#7A68A6', label=label)
    else:
        plot(glams, spectrum, label=label)
    xlabel("Wavelength [nm]")
    ylabel("Gain [1/cm]")
    gcf().canvas.set_window_title("Gain Spectrum")


def plot_optical(suffix=''):
    if suffix and not suffix.endswith('-'):
        suffix += '-'

    figure()
    if not presentation:
        subplot(221)
    plot(olams, abs(OPTICAL.get_determinant(lam=olams, m=m)))
    xlabel("Wavelength [nm]")
    ylabel("Det [a.u.]")
    yscale('log')
    if presentation:
        tight_layout(0.2)
        save_figure(suffix+'det')
        gcf().canvas.set_window_title("Determinant")
    else:
        title("Determinant")
        tight_layout(0.2)

    try:
        opt = OPTICAL.outLightMagnitude(mesh.Rectangular2D(mesh.Regular(-0.3125*aprtc, 0.3125*aprtc, 1001), mesh.Regular(0., 14., 1401)))
        # opt = OPTICAL.outLightMagnitude(mesh.Rectangular2D(mesh.Regular(-5.0, 5.0, 501), mesh.Regular(3.5, 6.0, 1401)))
    except Exception as err:
        print_log('error', err)
    else:
        hm = opt.mesh.axis0
        vm = opt.mesh.axis1

        if not presentation:
            subplot(222)
        else:
            figure()

        # plot_field(opt, cmap='YlGnBu_r')
        imshow(opt.array.T, cmap='YlGnBu_r', extent=(opt.mesh.axis0[0], opt.mesh.axis0[-1], opt.mesh.axis1[0], opt.mesh.axis1[-1]),
               origin='lower', aspect='auto')
        plot_geometry(GEO.optical, color='0.5', alpha=0.2, mirror=True)

        if presentation:
            tight_layout(0.2)
            save_figure(suffix+'opt')
            gcf().canvas.set_window_title("Optical Field")
            figure()
        else:
            title("Optical Field")
            subplot(223)

        opt = OPTICAL.outLightMagnitude(mesh.Rectangular2D(hm, [zqw]))
        opt = Data(opt.array/max(opt), opt.mesh)
        plot_profile(opt, color=color_cycle[1])
        if presentation:
            ylabel("Light intensity [a.u.]")
        plot_rs(GEO.optical)
        xlim(0., hm[-1])

        if presentation:
            tight_layout(0.2)
            save_figure(suffix+'optr')
            gcf().canvas.set_window_title("Lateral")
            figure()
        else:
            title("Lateral")
            subplot(224)

        opt = OPTICAL.outLightMagnitude(mesh.Rectangular2D([0.01], vm))
        opt = Data(opt.array/max(opt), opt.mesh)
        plot_profile(opt, color=color_cycle[1])
        if presentation:
            ylabel("Light intensity [a.u.]")
        twinx()
        nr = OPTICAL.outRefractiveIndex(mesh.Rectangular2D([0.01], vm))
        plot_profile(nr, color=color_cycle[4], comp='rr', alpha=0.5)
        ylim(0.95, 3.55)

        if presentation:
            ylabel("Refractive Index")
            tight_layout(0.2)
            save_figure(suffix+'optz')
            gcf().canvas.set_window_title("Vertical")
        else:
            title("Vertical")

    if not presentation:
        suptitle(u"LP$_{{{m},{n}}}$".format(**DEF), fontsize=16)

        gcf().canvas.set_window_title("Optical")
        tight_layout(0.)
        subplots_adjust(top=0.89)
        save_figure()


try:

    if task == 'BC':
        fig = figure()
        plot_geometry(ELECTRICAL.geometry, margin=0.1, color='w')
        plot_mesh(ELECTRICAL.mesh, color='0.1')
        plot_boundary(ELECTRICAL.voltage_boundary, ELECTRICAL.mesh, ELECTRICAL.geometry, cmap='bwr', s=40)
        fig.canvas.set_window_title('Voltage Boundary Conditions')

    elif task == 'TE':
        work = algorithm.ThermoElectric(THERMAL, ELECTRICAL)
        work.run(save=save_h5 and '{}.h5'.format(desc))
        J = abs(ELECTRICAL.get_total_current())
        print_log(LOG_INFO, "Total current: {:.3g} mA".format(J))
        DIFFUSION.compute_threshold()
        plot_thermoelectric(False)
        plot_gain(None if presentation else 224)
        try:
            OPTICAL.find_mode(approximate_mode(m,n), m=m)
        except (ComputationError, IndexError):
            figure()
            plot(olams, abs(OPTICAL.get_determinant(lam=olams, m=m)), 'c')
            xlabel("Wavelength [nm]")
            ylabel("Det [a.u.]")
            yscale('log')
            title("Determinant")
            gcf().canvas.set_window_title("Optical")
            tight_layout(0.1)
            lam_th = NAN
            loss_th = NAN
        else:
            lam_th = OPTICAL.outWavelength()
            loss_th = OPTICAL.outLoss()
            print_log('result', 'lambda = {lam_th:6.2f}nm  loss = {loss_th:g}/cm\n'.format(**locals()))
            plot_optical()
        with open('te.out', 'a') as output:
            fcntl.lockf(output.fileno(), fcntl.LOCK_EX)  # lock file between other processes
            output.write('{oxide_loc:.2f} {aprt:.1f}  {solvers}  {DEF[m]:02d} {n:02d}  {U:6.3f} {J:6.3f}  {lam_th:6.2f} {loss_th:7.3f}  {T:5.1f}\n'.format(**locals()))


    elif task == 'TT':
        if u_t != 0:
            hn = int(round(heat_t / u_t))
            if hn: u_t = heat_t / hn
            cn = int(round(cool_t / u_t))
            if cn: c_t = cool_t / cn
        else:
            hn = cn = 1
            u_t = heat_t
            c_t = cool_t

        T = THERMAL.inittemp

        times = []
        temps = []
        currents = []

        max_times = []
        max_temps = []
        min_times = [0.]
        min_temps = [T]

        with open("{}-u{:.2f}v-h{:.2f}-time-temp.out".format(desc, U, heat_t/(heat_t+cool_t)), 'w') as out:
            out.write("#  t[ns]     T[K]  J[mA]\n")
            for hc in xrange(hc_cycles):
                ELECTRICAL.voltage_boundary[0].value = U
                ELECTRICAL.compute()
                J = abs(ELECTRICAL.get_total_current())
                times.append(T_DYNAMIC.time)
                temps.append(T)
                currents.append(J)
                out.write("{:8.2f}  {:6.3f}  {:5.2f}\n".format(T_DYNAMIC.time, T, J))
                out.flush()
                for _ in xrange(hn):
                    # Heating
                    THERMAL.compute(u_t)
                    T = max(THERMAL.outTemperature(THERMAL.mesh))
                    ELECTRICAL.compute()
                    J = abs(ELECTRICAL.get_total_current())
                    times.append(T_DYNAMIC.time)
                    temps.append(T)
                    currents.append(J)
                    out.write("{:8.2f}  {:6.3f}  {:5.2f}\n".format(T_DYNAMIC.time, T, J))
                    out.flush()
                ELECTRICAL.voltage_boundary[0].value = 0
                ELECTRICAL.compute()
                times.append(T_DYNAMIC.time)
                temps.append(T)
                currents.append(0.)
                out.write("{:8.2f}  {:6.3f}  {:5.2f}\n".format(T_DYNAMIC.time, T, 0.))
                out.flush()
                if hn:
                    max_times.append(THERMAL.time)
                    max_temps.append(T)
                for _ in xrange(cn):
                    # Cooling
                    THERMAL.compute(c_t)
                    T = max(THERMAL.outTemperature(THERMAL.mesh))
                    times.append(THERMAL.time)
                    temps.append(T)
                    currents.append(0.)
                    out.write("{:8.2f}  {:6.3f}  {:5.2f}\n".format(T_DYNAMIC.time, T, 0.))
                    out.flush()
                if cn:
                    min_times.append(THERMAL.time)
                    min_temps.append(T)

        fig = figure()
        fig.set_figwidth(8.)
        plot(min_times, min_temps, '--', color=color_cycle[3], alpha=0.5)
        plot(max_times, max_temps, '--', color=color_cycle[3], alpha=0.5)
        plot(times, temps, color=color_cycle[3])
        xlim(0, T_DYNAMIC.time)
        xlabel("Time [ns]")
        ylabel("Max. Temperature [K]")
        twinx()
        plot(times, currents, color=color_cycle[4])
        xlim(0, T_DYNAMIC.time)
        ylabel("Total Current [mA]")
        tight_layout(0.2)
        gcf().canvas.set_window_title("Temperature/Current Evolution ({:.1f})".format(heat_t/(heat_t+cool_t)))
        save_figure("u{:.2f}v-h{:.2f}-time-temp".format(U, heat_t/(heat_t+cool_t)))


    elif task == 'TH':
        if U is None:
            U = ELECTRICAL.voltage_boundary[0].value
        work = algorithm.ThresholdSearch(THERMAL, ELECTRICAL, DIFFUSION, GAIN, OPTICAL, 0, U0,
                                         ApproximateMode(m,n) if oapprox is None else oapprox,
                                         optargs={'m': m}, loss='wavelength', invalidate=False)
        work.run(save=save_h5 and '{}.h5'.format(desc))
        U = ELECTRICAL.voltage_boundary[0].value
        J = abs(ELECTRICAL.get_total_current())
        print_log(LOG_INFO, "Total current: {:.3g} mA".format(J))

        suffix = "lp{}{}".format(DEF['m'],n)

        lam_th = OPTICAL.outWavelength()
        loss_th = OPTICAL.outLoss()
        plot_thermoelectric(False, suffix=suffix)
        plot_gain(None if presentation else 224, suffix=suffix)
        with open('th.out', 'a') as output:
            fcntl.lockf(output.fileno(), fcntl.LOCK_EX) # lock file between other processes
            output.write('{oxide_loc:.2f} {aprt:.1f}  {solvers}  {DEF[m]:02d} {n:02d}  {U:6.3f} {J:6.3f}  {lam_th:6.2f} {loss_th:7.3f}  {T:5.1f}\n'.format(**locals()))
        plot_optical(suffix=suffix)
        print_log('result', 'lambda = {lam_th:6.2f}nm  loss = {loss_th:g}/cm\n'.format(**locals()))


    elif task == 'CO':
        OPTICAL.inGain = 0.
        if OPTICAL is not O_EFM:
            O_EFM.inGain = 0.
        try:
            OPTICAL.find_mode(approximate_mode(m,n), m=m)
        except (ComputationError, IndexError):
            lam = loss = NAN
            figure()
            plot(olams, abs(OPTICAL.get_determinant(lam=olams, m=m)), 'c')
            xlabel("Wavelength [nm]")
            ylabel("Det [a.u.]")
            yscale('log')
            title("Determinant")
            gcf().canvas.set_window_title("Optical")
            tight_layout(0.1)
            save_figure('lp{}x-det'.format(DEF['m']))
        else:
            lam = OPTICAL.outWavelength()
            loss = OPTICAL.outLoss()
            print_log('result', 'lambda = {lam:6.2f}nm  loss = {loss:g}/cm\n'.format(**locals()))
            plot_optical(suffix="lp{}{}".format(DEF['m'],n))
        with open(os.path.join('co.out'), 'a') as output:
            fcntl.lockf(output.fileno(), fcntl.LOCK_EX) # lock file between other processes
            output.write('{oxide_loc:.2f} {aprt:.1f}  {solvers}  {DEF[m]:02d} {n:02d}  {lam:6.2f} {loss:7.3f}\n'.format(**locals()))


    elif task == 'GS':
        GAIN.inCarriersConcentration = 1e19 * gN
        if isinstance(gT, collections.Sequence):
            figure()
            for T in gT:
                GAIN.inTemperature = T
                plot_gain_spectrum(False, label="{:.0f} K".format(T))
                new = False
            legend(loc='best')
        else:
            GAIN.inTemperature = gT
            plot_gain_spectrum(True)
        axvline(1310., color='0.7' if presentation else 'k', lw=1)
        tight_layout(0.2)
        save_figure()

    elif task == 'EB':

        box = GEO.main.get_object_bboxes(GEO.active)[0]
        zz = linspace(box.lower.z-0.002, box.upper.z+0.002, 1001)
        CC = [GEO.main.get_material(0.,z).CB() for z in zz]
        VV = [GEO.main.get_material(0.,z).VB() for z in zz]
        plot(1e3*zz, CC, color=color_cycle[0])
        plot(1e3*zz, VV, color=color_cycle[1])
        xlim(1e3*zz[0], 1e3*zz[-1])
        xlabel("$z$ [nm]")
        ylabel("Band Edges [eV]")
        levels = None
        if levels is not None:
            for l in levels['el']:
                axhline(co+l, color=el_color, ls='--')
            for l in levels['hh']:
                axhline(vo+l, color=hh_color, ls='--')
            for l in levels['lh']:
                axhline(vo+l, color=lh_color, ls='--')
        gcf().canvas.set_window_title("Band edges")
        tight_layout(0.2)
        save_figure()

    elif task == 'T0':

        work = algorithm.ThermoElectric(THERMAL, ELECTRICAL)
        work.run(save=False)
        DIFFUSION.compute_threshold()
        J0 = abs(ELECTRICAL.get_total_current())

        lam = approximate_mode(m,n)

        def func(volt):
            try: volt = volt[0]
            except TypeError: pass
            ELECTRICAL.voltage_boundary[0].value = volt
            ELECTRICAL.invalidate()
            ELECTRICAL.compute()
            DIFFUSION.compute_threshold()
            OPTICAL.invalidate()
            modeno = OPTICAL.find_mode(lam, m=m)
            val = OPTICAL.modes[modeno].loss
            plask.print_log('result', "ThresholdSearch: V = {:.4f} V, loss = {:g} / cm".format(volt, val))
            return val

        fsolve(func, 8.0, xtol=1e-6)[0]
        U = ELECTRICAL.voltage_boundary[0].value
        J = abs(ELECTRICAL.get_total_current())
        print_log(LOG_INFO, "Total current: {:.3g} mA".format(J))
        lam_th = OPTICAL.outWavelength()
        loss_th = OPTICAL.outLoss()
        plot_thermoelectric(False)
        plot_gain(None if presentation else 224)
        with open(os.path.join('t0.out'), 'a') as output:
            fcntl.lockf(output.fileno(), fcntl.LOCK_EX) # lock file between other processes
            output.write('{oxide_loc:.2f} {aprt:.1f}  {solvers}  {DEF[m]:02d} {n:02d}  {U:6.3f} {J:6.3f}  {lam_th:6.2f} {loss_th:7.3f}  {T:5.1f}  {J0:6.3f}\n'.format(**locals()))
        plot_optical()
        print_log('result', 'lambda = {lam_th:6.2f}nm  loss = {loss_th:g}/cm\n'.format(**locals()))

        if save:
            work.save('{}.h5'.format(desc))


    elif task == 'LT':
        work = algorithm.ThermoElectric(THERMAL, ELECTRICAL)
        work.run(save=save_h5 and '{}.h5'.format(desc))
        J = abs(ELECTRICAL.get_total_current())
        print_log(LOG_INFO, "Total current: {:.3g} mA".format(J))
        jtemp = THERMAL.outTemperature(mesh.Rectangular2D(THERMAL.mesh.get_midpoints().axis0, [zqw]), 'spline')
        T = max(jtemp)
        print_log(LOG_INFO, "Junction temperature: {:.2g} K".format(T))

        DIFFUSION.compute_threshold()
        optstart = approximate_mode(m,n)
        
        def func(volt):
            """Function to search zero of"""
            ELECTRICAL.voltage_boundary[0].value = volt
            ELECTRICAL.compute()
            DIFFUSION.compute_threshold()
            mn = OPTICAL.find_mode(optstart, m=m)
            val = (4e7*pi / OPTICAL.modes[mn].lam).imag
            plask.print_log('result', "ThresholdSearch: V = {:.4f} V, loss = {:g} / cm"
                            .format(volt, val))
            return val
        
        brentq(func, 1.0, U)
        U = ELECTRICAL.voltage_boundary[0].value
        Jth = abs(ELECTRICAL.get_total_current())
        print_log(LOG_INFO, "Total threshold current: {:.3g} mA".format(J))

        suffix = "lp{}{}".format(DEF['m'],n)

        lam_th = OPTICAL.outWavelength()
        loss_th = OPTICAL.outLoss()

        netto = -OPTICAL.get_total_absorption() / OPTICAL.modes[0].power
        brutto = OPTICAL.get_gain_integral() / OPTICAL.modes[0].power
        eta = netto / brutto

        P = 1e9 * eta * phys.h.J * phys.c / phys.qe / lam_th / n_QW * (J-Jth)

        plot_thermoelectric(False, suffix=suffix)
        plot_gain(None if presentation else 224, suffix=suffix)

        with open('lt.out', 'a') as output:
            fcntl.lockf(output.fileno(), fcntl.LOCK_EX) # lock file between other processes
            output.write('{oxide_loc:.2f} {aprt:.1f}  {solvers}   {J:6.3f}  {DEF[m]:02d} {n:02d}  {U:6.3f} {Jth:6.3f}  {lam_th:6.2f} {loss_th:7.3f}  {T:5.1f}  {P:.4g}\n'.format(**locals()))
        plot_optical(suffix=suffix)
        print_log('result', 'lambda = {lam_th:6.2f}nm  loss = {loss_th:g}/cm\n'.format(**locals()))
        

    elif task == 'L1':
        work = algorithm.ThermoElectric(THERMAL, ELECTRICAL)
        work.run(save=save_h5 and '{}.h5'.format(desc))
        DIFFUSION.compute_threshold()

        lam = approximate_mode(m,n)

        OPTICAL.find_mode(lam, m=m)

        gmsh = OPTICAL.mesh.get_midpoints().axis0
        GAIN.mesh = gmsh

        def func(arg):
            global lam
            OPTICAL.modes[0].power = arg
            print_log('data', "P = {:.4g} mW".format(arg))
            DIFFUSION.invalidate()
            DIFFUSION.compute_overthreshold()
            try:
                OPTICAL.find_mode(lam, m=m)
            except ComputationError:
                lam = approximate_mode(m,n)
                OPTICAL.find_mode(lam, m=m)
            OPTICAL.modes[0].power = max(arg, 1e-9)
            gain = - (4e7*pi / OPTICAL.modes[0].lam).imag
            print_log('result', "P = {:.4g} mW  gain = {:.4g} / cm".format(arg, gain))

            U = ELECTRICAL.voltage_boundary[0].value
            J = abs(ELECTRICAL.get_total_current())
            lo, hi = GEO.optical.item.bbox.lower[1], GEO.optical.item.bbox.upper[1]
            fld = OPTICAL.outLightMagnitude(mesh.Rectangular2D([0], [lo, hi]))
            fac = 1. + GEO.optical.get_material(0, lo-0.0001).nr(1310.) * fld[0] / fld[1]
            TP = fac * OPTICAL.modes[0].power
            netto = -OPTICAL.get_total_absorption()
            brutto = OPTICAL.get_gain_integral()
            burn = DIFFUSION.get_total_burning()
            print_log('data', "Emitted power: {:.3f}mW".format(TP))
            print_log('data', "Gained power netto: {:.3f}mW".format(netto))
            print_log('data', "Gained power brutto: {:.3f}mW".format(brutto))
            print_log('data', "Totally burned: {:.3f}mW".format(burn))
            print_log('data', "Netto error: {:.2f}%".format(200. * (netto-TP) / (netto+TP)))
            print_log('data', "Brutto error: {:.2f}%\n".format(200. * (brutto-burn) / (brutto+burn)))

            return gain

        if adt:
            print_log('info', "Computing power with temperature adjustement")
            Tprev = 0.
            T = max(THERMAL.outTemperature(mesh.Rectangular2D(THERMAL.mesh.get_midpoints().axis0, [zqw]), 'spline'))
            THERMAL.inHeat = ELECTRICAL.outHeat + OPTICAL.outHeat
            while abs(T - Tprev) > 0.01:
                Pprev = P
                if PP is not None:
                    P = brentq(func, *PP, xtol=0.001)
                else:
                    P = newton(func, P, tol=0.001)
                OPTICAL.modes[0].power = P
                work.run(save=False)
                Tprev = T
                T = max(THERMAL.outTemperature(mesh.Rectangular2D(THERMAL.mesh.get_midpoints().axis0, [zqw]), 'spline'))
                print_log('result', "LI: P = {:.4g} mW  T = {:.2f} K   (previously: P = {:.4g} mW  T = {:.2f} K)".format(P, T, Pprev, Tprev))
            OPTICAL.find_mode(lam, m=m)
            OPTICAL.modes[0].power = P
        else:
            if PP is not None:
                P = brentq(func, *PP, xtol=0.001)
            else:
                P = newton(func, P, tol=0.001)
            OPTICAL.modes[0].power = P

        plot_thermoelectric(False)
        plot_gain(None if presentation else 224)
        plot_optical()

        U = ELECTRICAL.voltage_boundary[0].value
        J = abs(ELECTRICAL.get_total_current())

        lam_th = OPTICAL.outWavelength()
        loss_th = OPTICAL.outLoss()
        rname = 'l1{}.out'.format('-a' if adt else '')
        with open(rname, 'a') as output:
            fcntl.lockf(output.fileno(), fcntl.LOCK_EX) # lock file between other processes
            output.write('{oxide_loc:.2f} {aprt:.1f}  {solvers}  {DEF[m]:02d} {n:02d}  {U:6.3f} {J:6.3f}  {P:1.3f}  {T:5.1f}  {lam_th:6.2f} {loss_th:7.3f}\n'.format(**locals()))

    elif task == 'LA':

        #rname = 'tajwan-li{}.out'.format('a' if adt else '')
        rname = 'la.out'
        output = open(rname, 'a')

        work = algorithm.ThermoElectric(THERMAL, ELECTRICAL)

        class Func(object):
            def __init__(self, modes):
                self.modes = modes
            def __call__(self, powers):
                if powers is None:
                    burns = DIFFUSION.mode_burns
                    print_log('data', "B = {} mW".format(list(burns)))
                    for mode,burn in zip(OPTICAL.modes, burns):
                        mode.gain_integral = burn
                    powers = [mode.power for mode in OPTICAL.modes]
                else:
                    for i in range(l):
                        OPTICAL.modes[i].power = powers[i]
                print_log('data', "P = {} mW".format(list(powers)))
                l = len(powers)
                assert len(OPTICAL.modes) == l, "number of modes {} != {}".format(len(OPTICAL.modes), l)
                powers[powers < 0.] = 0.
                DIFFUSION.compute_overthreshold()
                gains = zeros(l)
                lams = [OPTICAL.modes[i].lam for i in range(l)]
                OPTICAL.invalidate()
                for j,(m,n) in enumerate(self.modes):
                    try:
                        i = OPTICAL.find_mode(lams[j], m=m)
                    except ComputationError:
                        i = OPTICAL.find_mode(approximate_mode(m,n), m=m)
                    gains[i] = - (4e7*pi / OPTICAL.modes[i].lam).imag
                    if powers[i] == 0. and gains[i] < 0.:      # make sure we can have modes with no power and not gaining
                        gains[i] = 0.
                print_log('result', "P = {} mW  gain = {} / cm".format(list(powers), list(gains)))
                self.result = gains
                return gains

        ELECTRICAL.voltage_boundary[0].value = U
        work.run(save=False, noinit=True)
        J = abs(ELECTRICAL.get_total_current())

        DIFFUSION.compute_threshold()

        gmsh = OPTICAL.mesh.get_midpoints().axis0
        GAIN.mesh = gmsh

        for m,n in modes:
            i = OPTICAL.find_mode(approximate_mode(m,n), m=m)

        func = Func(modes)

        if PP is None:
            PP = zeros(len(modes))

        ## Adam's algorithm
        DIFFUSION.compute_overthreshold()
        err = 1.
        while err > 5e-5:
            err = max(abs(func(None)))
            PP = [mode.power for mode in OPTICAL.modes]
        result = func.result

        plot_thermoelectric(suffix=suffix)
        plot_gain(None if presentation else 224, suffix=suffix)

        res = "{U:6.3f} {J:6.3f} {T:5.1f}   ".format(**locals()) + " ".join("{:.3f}".format(p) for p in PP) + \
              "  # " + " ".join(str(x) for x in result)
        print(res)
        output.write(res + '\n')
        output.flush()


    elif task == 'LI':
        ##
        ## This is quite advanced although linear task script
        ##

        if PP is not None:
            PD = dict(zip(modes, PP))
        else:
            PD = {}

        #rname = 'tajwan-li{}.out'.format('a' if adt else '')
        rname = 'li.out'
        output = open(rname, 'a')

        gmsh = OPTICAL.mesh.get_midpoints().axis0
        GAIN.mesh = gmsh

        work = algorithm.ThermoElectric(THERMAL, ELECTRICAL)

        class Func(object):
            def __init__(self, modes):
                self.modes = modes
            def __call__(self, powers):
                print_log('data', "P = {} mW".format(list(powers)))
                l = len(powers)
                assert len(OPTICAL.modes) == l, "number of modes {} != {}".format(len(OPTICAL.modes), l)
                powers[powers < 0.] = 0.
                for i in range(l):
                    OPTICAL.modes[i].power = powers[i]
                DIFFUSION.compute_overthreshold()
                gain = zeros(l)
                lams = [OPTICAL.modes[i].lam for i in range(l)]
                OPTICAL.invalidate()
                for j,(m,n) in enumerate(self.modes):
                    try:
                        i = OPTICAL.find_mode(lams[j], m=m)
                    except ComputationError:
                        i = OPTICAL.find_mode(approximate_mode(m,n), m=m)
                    gain[i] = - (4e7*pi / OPTICAL.modes[i].lam).imag
                    if powers[i] == 0. and gain[i] < 0.:      # make sure we can have modes with no power and not gaining
                        gain[i] = 0.
                print_log('result', "P = {} mW  gain = {} / cm".format(list(powers), list(gain)))
                self.result = gain
                return gain

        for U in UU:
            ELECTRICAL.voltage_boundary[0].value = U
            work.run(save=False, noinit=True)
            J = abs(ELECTRICAL.get_total_current())

            ## Find modes above threshold for current voltage
            DIFFUSION.compute_threshold()
            suffix = 'j{:.3f}-'.format(J)
            OPTICAL.invalidate()
            mods = []
            lams = []
            had = set()
            for m,n in modes:
                try:
                    i = OPTICAL.find_mode(approximate_mode(m,n), m=m)
                except ComputationError as err:
                    print_log(LOG_CRITICAL_ERROR, 'LP{}{} filter:'.format(m,n), err)
                else:
                    if i not in had and OPTICAL.modes[i].lam.imag > 0:
                        mods.append((m,n))
                        lams.append(OPTICAL.modes[i].lam)
                        had.add(i)
            PP = [max(PD.get(mn, P), P) for mn in mods]
            OPTICAL.clear_modes()
            for i in range(len(mods)):
                OPTICAL.set_mode(lams[i], m=mods[i][0])

            ## Do the overthreshold computations
            if PP:
                func = Func(mods)
                try:
                    PP = [max(p, 0.) for p in broyden1(func, PP, f_tol=5e-5, x_tol=1e-4)]
                except ComputationError as err:
                    print_log(LOG_CRITICAL_ERROR, 'LP{}{}:'.format(m,n), err)
                    continue
                else:
                    result = func.result
            else:
                result = []
            PD = dict(zip(mods, PP))

            plot_thermoelectric(suffix=suffix)
            plot_gain(None if presentation else 224, suffix=suffix)

            res = "{U:6.3f} {J:6.3f} {T:5.1f}   ".format(**locals()) + " ".join("{:.3f}".format(PD.get(mn, 0.)) for mn in modes) + \
                  "  # " + " ".join(str(x) for x in result)
            print(res)
            output.write(res + '\n')
            output.flush()


    elif task == 'LO':

        work = algorithm.ThermoElectric(THERMAL, ELECTRICAL)
        work.run(save=save_h5 and '{}.h5'.format(desc))
        J = abs(ELECTRICAL.get_total_current())
        DIFFUSION.compute_threshold()

        if PP is not None:
            PP = array(PP0)
        else:
            PP = P0 * ones(len(modes))

        def get_modes():
            OPTICAL.invalidate()
            for m,n in modes:
                OPTICAL.find_mode(approximate_mode(m,n), m=m)

        get_modes()

        for iter in range(500):
            for i,mode in enumerate(OPTICAL.modes):
                mode.power = PP[i]
            if adt:
                work.run(save=False)
                J = abs(ELECTRICAL.get_total_current())
            DIFFUSION.compute_overthreshold()
            get_modes()
            alpha = array([-(4e7*pi / mode.lam).imag for mode in OPTICAL.modes])
            err = max(abs(alpha))
            PP *= exp(alpha*dt)
            print("iteration: {}  J = {:6.3f}mA".format(iter+1, J))
            for i,a in enumerate(alpha):
                print("LP {}:  {:.6f}mW   {:.6}/cm".format(modes[i], PP[i], a))
            print("err = {}/cm".format(err))
            if err < 1e-4: break

        rname = 'lo{}.out'.format('-at' if adt else '')
        with open(rname, 'a') as output:
            fcntl.lockf(output.fileno(), fcntl.LOCK_EX) # lock file between other processes
            output.write('{oxide_loc:.2f} {aprt:.1f}  {solvers}  {DEF[m]:02d} {n:02d}  {U:6.3f} {J:6.3f}  {T:5.1f}  '.format(**locals()))
            res = " ".join("{:.3f}".format(*PP)) + "  # " + " ".join(str(a) for a in alpha)


    else:
        raise ValueError("Invalid task {}".format(task))


finally:
    if not presentation:
        pdf.close()


show()

]]></script>

</plask>
