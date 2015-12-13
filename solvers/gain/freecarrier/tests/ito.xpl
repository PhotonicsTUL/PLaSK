<plask loglevel="detail">

<defines>
  <define name="cavity" value="3"/>
  <define name="aperture" value="7.0"/>
  <define name="gT" value="300."/>
  <define name="gN" value="3.96"/>
  <define name="mesa1" value="15."/>
  <define name="mesa2" value="8."/>
  <define name="ind" value="0.13"/>
  <define name="eg_n" value="-2e-8"/>
  <define name="n_qw" value="4e19"/>
  <define name="blue_qw" value="0.050"/>
  <define name="cito" value="0.5"/>
  <define name="cti" value="1.0"/>
  <define name="hnAu" value="0.5964"/>
  <define name="width" value="100."/>
  <define name="ewidth" value="40."/>
  <define name="Si1" value="1.0"/>
  <define name="Si2" value="0.5"/>
</defines>

<materials>
  <material name="InGaN_QW" base="In({ind})GaN">
    <A>#6e+7 * (T/300.)
6e+7 - 98460 * (T-300.) + 9660 * (T-300.)**2</A>
    <B>#2e-11 * (300./T)**1.5
2e-11 - 8.568e-014 * (T-300.) + 1.6854e-016 * (T-300.)**2</B>
    <C>#1.5e-30 * (T/300.)
1.5e-30 + 1.7385e-034 * (T-300.) + 2.7165e-036 * (T-300.)**2</C>
    <D>#2. * (T/300.)
2. - 0.006194 * (T-300.) + 8.066e-006 * (T-300.)**2</D>
    <nr>2.70617 + 7.36e-5 * (T-300.)</nr>
    <absp>0.</absp>
    <Eg>self.base.Eg(T,e) + {eg_n * n_qw**(1/3)} + {blue_qw}</Eg>
    <VB>0.3 * ( 3.510 - 0.914e-3*T**2 / (T+825.) - self.Eg(T,e) )</VB>
    <thermk>5.0</thermk>
  </material>
  <material name="GaN_barrier" base="GaN">
    <nr>2.506361 + 1.241616e-4 * (T-300.)</nr>
    <absp>10. + 3.611592e-3 * (T-300.)</absp>
    <thermk>5.0</thermk>
  </material>
  <material name="Al2O3" base="AlOx">
    <thermk>40.0</thermk>
    <cond>1e-05</cond>
  </material>
  <material name="Au" base="Au">
    <nr>1.63501-0.0021881*(wl-414.0)</nr>
    <absp>594050.0+1434.9*(wl-414.0)</absp>
  </material>
  <material name="ITO" base="metal">
    <thermk>3.2</thermk>
    <cond>1000000.0</cond>
    <nr>2.095379 - 5.25e-6 * (T-300.)</nr>
    <absp>1001.</absp>
  </material>
  <material name="ITOct" base="ITO">
    <cond>{cito}</cond>
  </material>
  <material name="PbSn" base="metal">
    <thermk>50.0</thermk>
    <cond>6700000.0</cond>
  </material>
  <material name="Ta2O5" base="dielectric">
    <thermk>1.5</thermk>
    <cond>0.0001</cond>
    <nr>2.15 - 5.25e-6  *  (T-300.)</nr>
    <absp>1. + 0.073  *  (T-300.)</absp>
  </material>
  <material name="SiO2" base="dielectric">
    <thermk>1.5</thermk>
    <cond>0.0001</cond>
    <nr>1.556061 - 5.25e-6 * (T-300.)</nr>
    <absp>1. + 0.073 * (T-300.)</absp>
  </material>
  <material name="SiNx" base="dielectric">
    <thermk>2.0</thermk>
    <cond>1e-05</cond>
    <nr>2.067 - 5.25e-6 * (T-300.)</nr>
    <absp>1. + 0.073 * (T-300.)</absp>
  </material>
  <material name="Ti" base="metal">
    <thermk>21.9</thermk>
    <cond>{cti}</cond>
  </material>
  <material name="AlN_DBR" base="AlN:Si=1e16">
    <nr>2.12961 - 4.13812e-4 * (wl-414.0) + 1.241616e-4 * (T-300.)</nr>
    <thermk>self.base.thermk(T, 1.)[1], self.base.thermk(T, h)[1]</thermk>
  </material>
  <material name="GaN_DBR" base="GaN:Si=1e16">
    <thermk>self.base.thermk(T, 1.)[1], self.base.thermk(T, h)[1]</thermk>
  </material>
  <material name="GaN_cavity:Mg" base="GaN:Mg">
    <thermk>self.base.thermk(T, 10.)[1], self.base.thermk(T, h)[1]</thermk>
  </material>
  <material name="GaN_cavity:Si" base="GaN:Si">
    <VB>material.get('Al(0.20)GaN:Mg=4e17').VB(T, e, point)</VB>
    <Eg>material.get('Al(0.20)GaN:Mg=4e17').Eg(T, e, point)</Eg>
    <thermk>self.base.thermk(T, 10.)[1], self.base.thermk(T, {0.1652*cavity-0.1834})[1]</thermk>
  </material>
</materials>

<geometry>
  <cylindrical2d name="electrical" axes="r,z">
    <clip right="{ewidth}">
      <stack name="vcsel">
        <shelf flat="no">
          <stack name="mesa">
            <stack repeat="8">
              <rectangle material="Ta2O5" dr="{mesa2}" dz="0.0481"/>
              <rectangle material="SiO2" dr="{mesa2}" dz="0.0665"/>
            </stack>
            <shelf>
              <stack>
                <shelf>
                  <stack>
                    <stack repeat="2">
                      <rectangle material="Ta2O5" dr="{aperture-0.5}" dz="0.0481"/>
                      <rectangle material="SiO2" dr="{aperture-0.5}" dz="0.0665"/>
                    </stack>
                    <rectangle name="cap" material="Ta2O5" dr="{aperture-0.5}" dz="0.0190"/>
                  </stack>
                  <rectangle material="Au" dr="0.5" dz="{0.1146+0.1636-0.0300}"/>
                </shelf>
                <rectangle name="ITO" material="ITO" dr="{aperture}" dz="0.0250"/>
                <rectangle material="ITOct" dr="{aperture}" dz="0.0050"/>
              </stack>
              <stack>
                <shelf>
                  <rectangle material="Au" dr="{mesa2-aperture}" dz="0.1146"/>
                  <rectangle name="p-contact" material="Au" dr="{8.0+aperture-mesa2}" dz="0.1146"/>
                </shelf>
                <rectangle material="SiNx" dr="{mesa1-aperture}" dz="0.1636"/>
              </stack>
            </shelf>
            <rectangle material="In(0.10)GaN:Mg=1.2e18" dr="{mesa1}" dz="0.0020"/>
            <rectangle material="GaN_cavity:Mg=7e17" dr="{mesa1}" dz="0.0790"/>
            <rectangle name="EBL" material="Al(0.20)GaN:Mg=4e17" dr="{mesa1}" dz="0.0240"/>
            <stack name="active" role="active">
              <stack repeat="2">
                <rectangle material="GaN_barrier" dr="{mesa1}" dz="0.0100"/>
                <rectangle name="QW" role="QW" material="InGaN_QW" dr="{mesa1}" dz="0.0030"/>
              </stack>
              <rectangle material="GaN_barrier" dr="{mesa1}" dz="0.0100"/>
            </stack>
            <rectangle material="GaN_cavity:Si={Si2*1e18}" dr="{mesa1}" dz="{0.1652 * cavity - 0.3134}"/>
          </stack>
          <stack>
            <rectangle material="SiNx" dr="0.2000" dz="{0.1652 * cavity - 0.2088}"/>
            <rectangle material="SiNx" dr="10" dz="0.2000"/>
          </stack>
          <stack>
            <rectangle name="n-contact" material="Au" dr="10" dz="{hnAu}"/>
            <rectangle material="Ti" dr="10" dz="0.0050"/>
          </stack>
        </shelf>
        <rectangle material="GaN_cavity:Si={Si1*1e18}" dr="{width}" dz="0.1300"/>
        <stack name="bottomDBR">
          <stack repeat="3">
            <stack name="bottom-pair">
              <rectangle material="AlN_DBR" dr="{width}" dz="0.0486"/>
              <rectangle material="GaN_DBR" dr="{width}" dz="0.0413"/>
            </stack>
          </stack>
          <stack name="superlattice">
            <stack repeat="5">
              <rectangle material="AlN_DBR" dr="{width}" dz="0.0052"/>
              <rectangle material="GaN_DBR" dr="{width}" dz="0.0029"/>
            </stack>
            <rectangle material="AlN_DBR" dr="{width}" dz="0.0052"/>
            <rectangle material="GaN_DBR" dr="{width}" dz="0.0413"/>
          </stack>
          <stack repeat="2">
            <again ref="bottom-pair"/>
          </stack>
          <again ref="superlattice"/>
          <stack repeat="2">
            <again ref="bottom-pair"/>
          </stack>
          <stack repeat="4">
            <again ref="superlattice"/>
            <stack repeat="4">
              <again ref="bottom-pair"/>
            </stack>
          </stack>
        </stack>
        <rectangle role="substrate" material="GaN:Si=1e16" dr="{width}" dz="2"/>
      </stack>
    </clip>
  </cylindrical2d>
  <cylindrical2d name="thermal" axes="r,z">
    <stack>
      <again ref="vcsel"/>
      <zero/>
      <rectangle material="Al2O3" dr="{width}" dz="100"/>
      <rectangle material="PbSn" dr="{width}" dz="1"/>
      <rectangle name="heatsink" material="Cu" dr="2500" dz="5000"/>
    </stack>
  </cylindrical2d>
  <cylindrical2d name="optical" axes="r,z" outer="extend" bottom="GaN">
    <clip right="{mesa2-0.001}">
      <again ref="vcsel"/>
    </clip>
  </cylindrical2d>
</geometry>

<grids>
  <mesh name="diffusion" type="regular">
    <axis start="0" stop="{mesa1}" num="1001"></axis>
  </mesh>
  <generator method="divide" name="electrical" type="rectangular2d">
    <postdiv by0="2" by1="1"/>
    <refinements>
      <axis0 object="ITO" every="0.1"/>
      <axis0 object="n-contact" at="0.1"/>
    </refinements>
  </generator>
  <generator method="divide" name="thermal" type="rectangular2d">
    <postdiv by0="2" by1="1"/>
    <refinements>
      <axis0 object="ITO" at="0.5"/>
      <axis0 object="ITO" at="{aperture-1.00}"/>
    </refinements>
  </generator>
  <generator method="divide" name="optical" type="ordered">
    <refinements>
      <axis0 object="cap" every="0.25"/>
    </refinements>
  </generator>
</grids>

<solvers>
  <gain name="GAIN1" solver="FermiCyl" lib="simple">
    <geometry ref="electrical"/>
    <config lifetime="0.1" matrix-elem="5" strained="no"/>
  </gain>
  <gain name="GAIN2" solver="FreeCarrierCyl" lib="freecarrier">
    <geometry ref="electrical"/>
    <config lifetime="0.1" matrix-elem="5" strained="no"/>
  </gain>
  <gain name="GAIN3" solver="FermiNewCyl" lib="complex">
    <geometry ref="electrical"/>
    <config matrix-elem="5" strained="no"/>
  </gain>
</solvers>

<script><![CDATA[
colors = rc.axes.color_cycle

zqw = GEO.electrical.get_object_bboxes(GEO.QW)[0].center.z
msh = mesh.Rectangular2D([0.], [zqw])

GAIN1.inTemperature = gT
GAIN2.inTemperature = gT
GAIN3.inTemperature = gT

GAIN1.inCarriersConcentration = 1e19 * gN
GAIN2.inCarriersConcentration = 1e19 * gN
GAIN3.inCarriersConcentration = 1e19 * gN



glams = linspace(350., 500., 1201)

def plot_gain(sub=None, suffix=''):
    if sub is None:
        fig = figure()
        fig.canvas.set_window_title("Gain Profile")
    else:
        subplot(sub)
        title("Gain Profile")
    plot_profile(GAIN1.outGain(mesh.Rectangular2D(DIFFUSION.mesh, [zqw]), 414., 'spline'), color='#7A68A6')
    plot_rs(GEO.electrical)
    ylabel("Gain Profile [1/cm]")
    xlim(0., mesa1+0.2)
    tight_layout(0.2)
    subplots_adjust(top=0.89)
    save_figure(suffix+'gain')


def plot_gain_spectrum(solver, new=False, label='PLaSK'):
    spectrum = solver.spectrum(0, zqw)
    if new:
        figure()
        plot(glams, spectrum(glams), label=label)
    else:
        plot(glams, spectrum(glams), label=label)
    xlabel("Wavelength [nm]")
    ylabel("Gain [1/cm]")
    gcf().canvas.set_window_title("Gain Spectrum")


def plot_bands(levels=None, co=0., vo=0., title="Levels", el_color=colors[0], hh_color=colors[1], lh_color=colors[2]):
    box = GEO.electrical.get_object_bboxes(GEO.active)[0]
    zz = linspace(box.lower.z-0.002, box.upper.z+0.002, 1001)
    CC = [GEO.electrical.get_material(0.,z).CB() for z in zz]
    VV = [GEO.electrical.get_material(0.,z).VB() for z in zz]
    plot(1e3*zz, CC, color=colors[0])
    plot(1e3*zz, VV, color=colors[1])
    xlim(1e3*zz[0], 1e3*zz[-1])
    xlabel("$z$ [nm]")
    ylabel("Band Edges [eV]")
    if levels is not None:
        for l in levels['el']:
            axhline(co+l, color=el_color, ls='--')
        print("EL: {}".format(', '.join(str(x) for x in co+array(levels['el']))))
        for l in levels['hh']:
            axhline(vo+l, color=hh_color, ls='--')
        print("HH: {}".format(', '.join(str(x) for x in vo+array(levels['hh']))))
        for l in levels['lh']:
            axhline(vo+l, color=lh_color, ls='--')
        print("LH: {}".format(', '.join(str(x) for x in vo+array(levels['lh']))))
    gcf().canvas.set_window_title(title)
    #plt.get_current_fig_manager().window.showMaximized()
    tight_layout(0.5)

levels1 = GAIN1.determine_levels(300., 1e18)[0]
levels1['el'] = [-l for l in levels1['el']]
mat = material.db.get("GaN_barrier")
cbo = mat.CB()
vbo = mat.VB()
mat = material.db.get("InGaN_QW")
cbq = mat.CB()
vbq = mat.VB()

figure()
plot_bands(levels1, co=cbo, vo=vbo, title=u"Levels: Michał")
yl = -0.4, 0.2

GAIN3.outGain(msh, 430.)
mat = material.db.get("GaN_cavity:Si={}".format(Si2*1e18))
cbo = mat.CB()
vbo = mat.VB()
levels3 = GAIN3.get_levels()[0]
plot_bands(levels3, co=cbo, vo=vbo, title=u"Levels: Michał", el_color='#00cccc', hh_color='#0000ff', lh_color='#008800')

twiny()
cc = logspace(16, 20, 65)
ffc, ffv = map(array, zip(*((f['Fc'], f['Fv']) for f in (GAIN1.determine_levels(300., c)[0] for c in cc))))
plot(cc, cbq+ffc, color=colors[0], ls='-', lw=1.5)
plot(cc, vbq+ffv, color=colors[1], ls='-', lw=1.5)
axvline(1e19*gN, color='0.75')
xlabel(u'Carriers concentation [1/cm³]')
xscale('log')
xlim(1e16, 1e20)
ylim(*yl)
tight_layout(0.2)    

levels2 = GAIN2.get_energy_levels()[0]

figure()
plot_bands(levels2, title=u"Levels: Maciek")

plot_bands(levels3, co=cbo, vo=vbo, title=u"Levels: Maciek", el_color='#00cccc', hh_color='#0000ff', lh_color='#008800')

twiny()
try:
    ff = linspace(yl[0], yl[1], 1001)
    nn = GAIN2.getN(ff)
    pp = GAIN2.getP(ff)
except AttributeError:
    pass
else:
    plot(nn, ff, color=colors[0], ls='-', lw=1.5)
    plot(pp, ff, color=colors[1], ls='-', lw=1.5)

cc = logspace(16, 20, 65)
ffc, ffv = zip(*(GAIN2.get_fermi_levels(c) for c in cc))
plot(cc, ffc, color=colors[0], ls='-', lw=1.5)
plot(cc, ffv, color=colors[1], ls='-', lw=1.5)
axvline(1e19*gN, color='0.75')

ylim(*yl)
xlabel(u'Carriers concentation [1/cm³]')
xscale('log')
xlim(1e16, 1e20)
tight_layout(0.2)    

GAIN1.inTemperature = gT
GAIN2.inTemperature = gT
GAIN3.inTemperature = gT

GAIN1.inCarriersConcentration = 1e19 * gN
GAIN2.inCarriersConcentration = 1e19 * gN
GAIN3.inCarriersConcentration = 1e19 * gN

plot_gain_spectrum(GAIN1, True, u"Michał")
plot_gain_spectrum(GAIN2, False, u"Maciek")
plot_gain_spectrum(GAIN3, False, u"Michał2")
legend(loc='best')

tight_layout(0.2)    

msh = mesh.Rectangular2D([0.], [zqw])
print(GAIN2.outGain(msh, 400.))[0]
    
show()
]]></script>

</plask>
