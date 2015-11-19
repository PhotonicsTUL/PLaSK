<plask loglevel="debug">

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
  <define name="n_QW" value="5"/>
  <define name="h_QW" value="0.0060"/>
  <define name="qwo" value="0.000000"/>
  <define name="bo" value="0.000000"/>
  <define name="h_bar" value="0.0067"/>
  <define name="aprt" value="6."/>
  <define name="aprtc" value="16."/>
  <define name="mesa" value="40."/>
  <define name="r_substr" value="1000."/>
  <define name="heat_t" value="100."/>
  <define name="cool_t" value="{500.-heat_t}"/>
  <define name="hc_cycles" value="20"/>
  <define name="u_t" value="5"/>
  <define name="adt" value="False"/>
  <define name="Tamb" value="300."/>
  <define name="PP" value="None"/>
  <define name="P" value="1e-6"/>
  <define name="eta" value="0.4"/>
  <define name="modes" value="(0,1), (1,1), (2,1), (0,2), (3,1)"/>
  <define name="THERMAL" value="'T_STATIC'"/>
  <define name="ELECTRICAL" value="'E_SCHOCKLEY'"/>
  <define name="DIFFUSION" value="'D_PARABOLIC'"/>
  <define name="GAIN" value="'G_SINGLE'"/>
  <define name="OPTICAL" value="'O_EFM'"/>
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
              <rectangle name="barrier1" material="InAlGaAs" dr="{mesa/2}" dz="{h_bar}"/>
              <rectangle name="QW1" role="QW,active" material="InAlGaAs-QW" dr="{mesa/2}" dz="{h_QW}"/>
              <again ref="barrier1"/>
              <rectangle name="QW2" role="QW,active" material="InAlGaAs-QW" dr="{mesa/2}" dz="{h_QW+qwo}"/>
              <again ref="barrier1"/>
              <again ref="QW1"/>
              <rectangle name="barrier2" material="InAlGaAs" dr="{mesa/2}" dz="{h_bar+bo}"/>
              <again ref="QW1"/>
              <again ref="barrier1"/>
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
  <gain name="GAIN1" solver="FermiCyl">
    <geometry ref="main"/>
    <config lifetime="0.3" matrix-elem="10"/>
  </gain>
  <gain name="GAIN2" solver="XFermiCyl">
    <geometry ref="main"/>
    <config lifetime="0.3" matrix-elem="10"/>
  </gain>
  <gain name="GAIN3" solver="FermiNewCyl" lib="complex">
    <geometry ref="main"/>
    <config lifetime="0.3" matrix-elem="10" adjust-layers="yes"/>
  </gain>
</solvers>

<script><![CDATA[

coff = material.get('GaAs').CB()
voff = material.get('GaAs').VB()
cbo = material.get('InAlGaAs').CB()
vbo = material.get('InAlGaAs').VB()
cbq = material.get('InAlGaAs-QW').CB()
vbq = material.get('InAlGaAs-QW').VB()

pos = GEO.main.get_object_bboxes(GEO.QW1)[0].center

colors = rc.axes.color_cycle

GAIN1.inCarriersConcentration = 3e19
GAIN2.inCarriersConcentration = 3e19
GAIN3.inCarriersConcentration = 3e19

levels1 = GAIN1.determine_levels(300., 3e19)[0]
levels1['el'] = [-l for l in levels1['el']]

cc = logspace(16, 20, 65)
fc, fv = zip(*((l['Fc']+cbq, l['Fv']+vbq) for l in (GAIN1.determine_levels(300., c)[0] for c in cc)))

levels2 = GAIN2.get_energy_levels()[0]

# GAIN3.outGain(mesh.Rectangular2D([pos[0]], [pos[1]]), 1300.)
# levels3 = GAIN3.get_levels()[0]

CE = linspace(0.25, 0.65, 10001)
VE = linspace(-0.80, -0.60, 10001)

nqw = 4

def plot_edges(attr):
    axvline(getattr(material.get('GaAs'), attr)(), ls='-', color='k')
    axvline(getattr(material.get('InAlGaAs'), attr)(), ls='-', color='k')
    axvline(getattr(material.get('InAlGaAs-QW'), attr)(), ls='-', color='k')

try:
    els = GAIN2.det_El(CE)
    hhs = GAIN2.det_Hh(VE)
    lhs = GAIN2.det_Lh(VE)
except AttributeError:
    pass
else:
    figure()
    plot(CE, els, color=colors[0], label="Electrons")
    for l in levels1['el']:
        axvline(l+cbo, ls=':', color='0.75')
#     for l in levels3['el']:
#         axvline(l+coff, ls=':', color='0.35')
    for l in levels2['el']:
        axvline(l, ls=':', color=colors[0])
    for w in range(1, nqw+1):
        plot(CE, GAIN2.det_El(CE, well=w), color=colors[0], ls='--')
    plot_edges('CB')
    xlim(CE[0], CE[-1])
    legend(loc='best')
    tight_layout(0.1)
    axhline(0., color='k')
    yscale('symlog')
    plt.get_current_fig_manager().window.showMaximized()
    tight_layout(0.5)
    gcf().canvas.set_window_title("Electrons")

    figure()
    plot(VE, hhs, color=colors[1], label="Heavy holes")
    for l in levels1['hh']:
        axvline(l+vbo, ls=':', color='0.75')
#     for l in levels3['hh']:
#         axvline(l+voff, ls=':', color='0.35')
    for l in levels2['hh']:
        axvline(l, ls=':', color=colors[1])
    for w in range(1, nqw+1):
        plot(VE, GAIN2.det_Hh(VE, well=w), color=colors[1], ls='--')
    plot_edges('VB')
    xlim(VE[0], VE[-1])
    axhline(0., color='k')
    yscale('symlog')
    plt.get_current_fig_manager().window.showMaximized()
    tight_layout(0.5)
    gcf().canvas.set_window_title("Heavy Holes")

    figure()
    plot(VE, lhs, color=colors[2], label="Light holes")
    for l in levels1['lh']:
        axvline(l+vbo, ls=':', color='0.75')
#     for l in levels3['lh']:
#         axvline(l+voff, ls=':', color='0.35')
    for l in levels2['lh']:
        axvline(l, ls=':', color=colors[2])
    for w in range(1, nqw+1):
        plot(VE, GAIN2.det_Lh(VE, well=w), color=colors[2], ls='--')
    plot_edges('VB')
    xlim(VE[0], VE[-1])
    axhline(0., color='k')
    yscale('symlog')
    plt.get_current_fig_manager().window.showMaximized()
    tight_layout(0.5)
    gcf().canvas.set_window_title("Light Holes")


def plot_bands(levels=None, co=0., vo=0., title="Levels", el_color=colors[0], hh_color=colors[1], lh_color=colors[2]):
    box = GEO.main.get_object_bboxes(GEO.active)[0]
    zz = linspace(box.lower.z-0.002, box.upper.z+0.002, 1001)
    CC = [GEO.main.get_material(0.,z).CB() for z in zz]
    VV = [GEO.main.get_material(0.,z).VB() for z in zz]
    plot(1e3*zz, CC, color=colors[0])
    plot(1e3*zz, VV, color=colors[1])
    xlim(1e3*zz[0], 1e3*zz[-1])
    xlabel("$z$ [nm]")
    ylabel("Band Edges [eV]")
    if levels is not None:
        for l in levels['el']:
            axhline(co+l, color=el_color, ls='--')
        for l in levels['hh']:
            axhline(vo+l, color=hh_color, ls='--')
        for l in levels['lh']:
            axhline(vo+l, color=lh_color, ls='--')
    gcf().canvas.set_window_title(title)
    plt.get_current_fig_manager().window.showMaximized()
    tight_layout(0.5)

yl = -0.8, 0.9

figure()
plot_bands(levels1, co=cbo, vo=vbo, title=u"Levels: Michał old")
yl = ylim()
twiny()
plot(cc, fc)
plot(cc, fv)
ylim(*yl)
xlabel(u'Carriers concentation [1/cm³]')
xscale('log')
xlim(1e16, 1e20)

# figure()
# plot_bands(levels3, co=coff, vo=voff, title=u"Levels: Michał new")

#ffc, ffv = zip(*(GAIN2.get_fermi_levels(c) for c in cc))

figure()
plot_bands(levels2, title=u"Levels: Maciek")
ff = linspace(yl[0], yl[1], 1001)
try:
    nn = GAIN2.getN(ff)
    pp = GAIN2.getP(ff)
except AttributeError:
    pass
else:
    twiny()
    plot(nn, ff)
    plot(pp, ff)
#     plot(cc, ffc, color=colors[0], ls='--', lw=1.5)
#     plot(cc, ffv, color=colors[1], ls='--', lw=1.5)
    plot(cc, fc, color='0.75')
    plot(cc, fv, color='0.75')
    ylim(*yl)
    xlabel(u'Carriers concentation [1/cm³]')
    xscale('log')
    xlim(1e16, 1e20)

show()

]]></script>

</plask>
