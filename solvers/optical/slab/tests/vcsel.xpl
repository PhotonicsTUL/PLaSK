<plask loglevel="debug">

<defines>
  <define name="m" value="0"/>
  <define name="mesa" value="10."/>
  <define name="x" value="0.03185 #3"/>
  <define name="aprt" value="2."/>
  <define name="estart" value="978.5"/>
  <define name="bstart" value="{estart}"/>
  <define name="pml" value="1."/>
</defines>

<materials>
  <material name="GaAs" base="semiconductor">
    <nr>3.53</nr>
    <absp>0.</absp>
  </material>
  <material name="AlGaAs" base="semiconductor">
    <nr>3.08</nr>
    <absp>0.</absp>
  </material>
  <material name="AlAs" base="semiconductor">
    <nr>2.95</nr>
    <absp>0.</absp>
  </material>
  <material name="AlOx" base="semiconductor">
    <nr>1.60</nr>
    <absp>0.</absp>
  </material>
  <material name="InGaAs" base="semiconductor">
    <Nr>3.53-0.01j</Nr>
  </material>
</materials>

<geometry>
  <cylindrical2d name="vcsel" axes="rz" outer="extend" bottom="GaAs">
    <stack name="layers">
      <stack name="top-dbr" repeat="24">
        <rectangle name="dbr-gaas" material="GaAs" dr="{mesa}" dz="0.06949"/>
        <rectangle name="dbr-algaas" material="AlGaAs" dr="{mesa}" dz="0.07963"/>
      </stack>
      <again ref="dbr-gaas"/>
      <rectangle name="x1" material="AlGaAs" dr="{mesa}" dz="{0.06371-x}"/>
      <shelf name="oxide-layer">
        <rectangle material="AlAs" dr="{aprt/2}" dz="0.01593"/>
        <rectangle name="alox" material="AlOx" dr="{mesa-aprt/2}" dz="0.01593"/>
      </shelf>
      <rectangle name="x" material="AlGaAs" dr="{mesa}" dz="{x}"/>
      <rectangle name="cavity" material="GaAs" dr="{mesa}" dz="0.13649"/>
      <shelf name="QW">
        <rectangle name="active" role="gain" material="InGaAs" dr="{aprt/2}" dz="0.00500"/>
        <rectangle name="inactive" material="InGaAs" dr="{mesa-aprt/2}" dz="0.00500"/>
      </shelf>
      <zero/>
      <again ref="cavity"/>
      <stack name="bottom-dbr" repeat="29">
        <again ref="dbr-algaas"/>
        <again ref="dbr-gaas"/>
      </stack>
      <again ref="dbr-algaas"/>
    </stack>
  </cylindrical2d>
</geometry>

<solvers>
  <optical name="efm" solver="EffectiveFrequencyCyl" lib="effective">
    <geometry ref="vcsel"/>
    <mode emission="top" lam0="980." vat="0" vlam="981."/>
  </optical>
  <optical name="bessel" solver="BesselCyl" lib="slab">
    <geometry ref="vcsel"/>
    <expansion lam0="980." size="100"/>
    <interface object="QW"/>
    <pml dist="30." factor="{pml}" size="2."/>
  </optical>
</solvers>

<script><![CDATA[
from timeit import timeit

rcParams['axes.color_cycle'] = cc = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']
profile = StepProfile(GEO.vcsel)
profile[GEO.active] = 0.

bessel.inGain = profile.outGain
efm.inGain = profile.outGain

em = efm.find_mode(estart, m=m)
elam = efm.modes[em].lam

box = GEO.vcsel.bbox
z = GEO.vcsel.get_object_bboxes(GEO.QW)[0].center.z
rmsh = mesh.Rectangular2D(linspace(-10., 10., 2001), [z])

desc = u"aprt:{:.1f}µm PML:{}".format(aprt, pml)

efield = efm.outLightMagnitude(em, rmsh)

N0 = bessel.size

NN = range(20, 151, 10)
lams = []
times = []

figure()
for N in NN:
    bessel.size = N
    try:
        t = timeit(lambda: bessel.find_mode(bstart, m=m+1), number=1)
    except ComputationError:
        NN = NN[:len(lams)]
        break
    else:
        times.append(t)
        lams.append(bessel.modes[0].lam)
        mag = bessel.outLightMagnitude(rmsh)
        plot_profile(mag/max(mag), label=str(N))

plot_profile(efield/max(efield), ls='--', color='0.8', label="EFM")

legend(loc='best')
ylabel("Light magnitude [a.u.]")
gcf().canvas.set_window_title(u"Mode profiles — base" + desc)
tight_layout(0.2)

lams = array(lams)

figure()
plot(NN, real(lams), '-', color=cc[0])
plot(NN, real(lams), '.', color=cc[0])
axhline(real(elam), ls='--', color=cc[0])
gca().ticklabel_format(useOffset=False)
xlabel("Base size")
ylabel("Resonant wavelength [nm]", color=cc[0])
twinx()
Q = - 0.5 * real(lams) / imag(lams)
plot(NN, Q, '-', color=cc[1])
plot(NN, Q, '.', color=cc[1])
axhline(-0.5*real(elam)/imag(elam), ls='--', color=cc[1])
gca().ticklabel_format(useOffset=False)
ylabel("Q-factor [-]", color=cc[1])
gcf().canvas.set_window_title(u"Convergence — base" + desc)
tight_layout(0.2)

figure()
plot(NN, times, '-', color=cc[2])
plot(NN, times, '.', color=cc[2])
xlabel("Base size")
ylabel("Computation time [s]")
gcf().canvas.set_window_title(u"Computation time — base" + desc)
tight_layout(0.2)


bessel.size = N0

mesas = arange(10., 81., 5.)

lams = []

figure()
for mesa in mesas:
    GEO.dbr_gaas.width = mesa
    GEO.dbr_algaas.width = mesa
    GEO.x1.width = mesa
    GEO.x.width = mesa
    GEO.cavity.width = mesa
    GEO.alox.width = mesa - aprt/2
    GEO.inactive.width = mesa - aprt/2
    try:
        bessel.find_mode(bstart, m=m+1)
    except ComputationError:
        mesas = mesas[:len(lams)]
        break
    else:
        lams.append(bessel.modes[0].lam)
        mag = bessel.outLightMagnitude(rmsh)
        plot_profile(mag/max(mag), label=u"{} µm".format(mesa))

plot_profile(efield/max(efield), ls='--', color='0.8', label="EFM")

legend(loc='best')
ylabel("Light magnitude [a.u.]")
gcf().canvas.set_window_title(u"Mode profiles — mesa" + desc)
tight_layout(0.2)

lams = array(lams)

figure()
plot(mesas, real(lams), '-', color=cc[0])
plot(mesas, real(lams), '.', color=cc[0])
axhline(real(elam), ls='--', color=cc[0])
gca().ticklabel_format(useOffset=False)
xlabel(u"Mesa size [µm]")
ylabel("Resonant wavelength [nm]", color=cc[0])
twinx()
Q = - 0.5 * real(lams) / imag(lams)
plot(mesas, Q, '-', color=cc[1])
plot(mesas, Q, '.', color=cc[1])
axhline(-0.5*real(elam)/imag(elam), ls='--', color=cc[1])
gca().ticklabel_format(useOffset=False)
ylabel("Q-factor [-]", color=cc[1])
gcf().canvas.set_window_title(u"Convergence — mesa" + desc)
tight_layout(0.2)


show()
]]></script>

</plask>
