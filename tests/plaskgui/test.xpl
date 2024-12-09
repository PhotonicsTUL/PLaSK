<plask loglevel="detail">

<!-- My Defines -->

<defines>
  <define name="aperture" value="8"/>
  <define name="mesa" value="{4. * aperture}"/>
  <define name="beta_def" value="19"/>
  <define name="js_def" value="1"/>
  <!-- comment 1 -->
  <define name="L" value="4.0"/>
  <!-- comment 2 -->
  <define name="d" value="0.5"/>
  <define name="X" value="{(6.1*sqrt(3)/2+d)*L}"/>
  <define name="Y" value="{(6.1+d)*L}"/>
  <define name="h_start" value="0"/>
  <define name="h_end" value="24"/>
  <define name="f" value="0"/>
  <define name="lineto" value="200."/>
  <!-- comment 3 -->
  <define name="tt" value="30"/>
  <define name="doping" value="2e+18"/>
</defines>

<!-- My Materials -->

<materials>
  <material name="test" base="semiconductor">
    <cond><![CDATA[
      param = aperture * T
      if param > 550:
          return param - 500
      else:
          return 50
    ]]></cond>
    <A><![CDATA[0.1 * T + 0.02 * (T-300)**2 if T < 400. else 1000.]]></A>
    <Eps>12.96, 12.96, 11.56</Eps>
    <thermk>10.+ 0.001 * T**2</thermk>
  </material>
  <material name="AlGaAs_xxx" base="AlGaAs" alloy="yes">
    <nr>3 * self.Al</nr>
  </material>
  <material name="InGaAsQW" base="In(0.2)GaAs"/>
  <material name="AlGaAs_2" base="Al(0.2)GaAs_xxx"/>
  <material name="InGaAs_QW:Si" base="InGaAs:Si" alloy="yes">
    <nr>3.621</nr>
    <thermk>1.5 * self.In</thermk>
    <A>110000000</A>
    <!--B-->
    <B>7e-011-1.08e-12*(T-300)</B>
    <!--C-->
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
    <thermk>100 + T/tt</thermk>
  </material>
  <material name="b" base="a"/>
  <material name="GaAs" base="Al(0.2)GaAs:Dp=1e19">
    <y1>10</y1>
    <y2>20</y2>
    <y3>30</y3>
  </material>
  <material name="AlGaSb_md" base="AlGaSb" alloy="yes">
    <nr>self.Ga</nr>
  </material>
</materials>

<geometry>
  <cartesian2d name="simple" left="periodic" right="periodic" bottom="Au" top="extend">
    <stack name="simple-stack">
      <python name="aaa">
          h = 0.2
          return geometry.Rectangle(1, h, 'GaAs')
        </python>
      <stack repeat="3">
        <rectangle material="Al(0.9)GaAs" dtran="1" dvert="0.3"/>
        <rectangle material="Al(0.9)GaN" dtran="1" dvert="0.2"/>
      </stack>
      <rectangle name="one" material="Al(0.73)GaAs:C={doping}" dtran="1" dvert="1.0"/>
      <rectangle material="Al(0.73)GaN:Si=1e18" dtran="1" dvert="1.0"/>
    </stack>
  </cartesian2d>
  <!--c1-->
  <cartesian2d name="geo2d" left="mirror" bottom="GaN" length="1000">
    <!--c2-->
    <stack name="stack2d">
      <!--c3-->
      <shelf steps-num="20" flat="no">
        <stack name="new">
          <arrange name="Pilars" dtran="0.4" dvert="0" count="3">
            <stack name="STOS">
              <!--cs1-->
              <rectangle name="rr" material="InN" dtran="0.2" dvert="0.1"/>
              <!--cs2-->
              <item>
                <!--cs2a-->
                <rectangle material="In(0.5)GaN:Si=1e18" dtran="0.2" dvert="0.1"/>
                <!--cs2b-->
              </item>
              <!--ce-->
            </stack>
          </arrange>
          <item left="0.0">
            <again ref="Pilars"/>
          </item>
          <rectangle material="Al(0.9)GaN:Si=2e18" dtran="1" dvert="0.1"/>
          <rectangle material="Al(0.5)GaN:Si=2e18" dtran="1" dvert="0.2"/>
        </stack>
        <!--cg-->
        <gap total="2"/>
        <!--cs-->
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
      <item zero="0.3">
        <rectangle name="posredni" material="Al(0.2)GaN" dtran="2" dvert="0.5"/>
      </item>
      <rectangle role="substrate" material="GaN" dtran="2" dvert="1"/>
    </stack>
  </cartesian2d>
  <!--break-->
  <cartesian2d name="geo2d-copy" left="extend" right="extend" bottom="GaN">
    <copy from="stack2d">
      <toblock object="new" material-top="Al(1.0)GaAs" material-bottom="Al(0.0)GaAs" name="blok2" role="rola1"/>
      <replace object="stos2">
        <shelf2d>
          <rectangle material="GaAs" dtran="0.4" dvert="0.5"/>
          <rectangle material="AlAs" dtran="0.4" dvert="0.5"/>
        </shelf2d>
      </replace>
      <delete object="posredni"/>
      <replace object="pikusik" with="kwadrat"/>
      <simplify-gradients lam="980"/>
    </copy>
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
    <clip back="0" left="0">
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
  <cylindrical2d name="GeoE" axes="r,z" bottom="GaAs:C=2e+18">
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
  <cartesian3d name="tridi" axes="x,y,z">
    <stack>
      <triangular-prism material="AlAs" ax="0.9" ay="-0.5" bx="-0.1" by="1.5" height="1.0"/>
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
  <cartesian2d name="clip" axes="xy">
    <stack x="0">
      <clip left="0" right="-1">
        <align xcenter="0" bottom="0">
          <rectangle material="AlN" dx="2" dy="1"/>
        </align>
      </clip>
      <align xcenter="0" bottom="0">
        <rectangle material="InN" dx="2" dy="1"/>
      </align>
    </stack>
  </cartesian2d>
  <cartesian3d name="revolved" axes="x,y,z" left="extend" right="extend">
    <revolution rev-steps-num="20">
      <again ref="simple-stack"/>
    </revolution>
  </cartesian3d>
  <cartesian3d name="original3d" axes="x,y,z " left="extend" right="extend">
    <stack>
      <cuboid material="GaAs" dx="5." dy="5." dz="1.0"/>
      <cuboid material-top="Al(0.0)GaAs" material-bottom="Al(1.0)GaAs" dx="5." dy="5." dz="0.2"/>
      <cuboid material="AlAs" dx="5." dy="5." dz="1.0"/>
      <cuboid role="grad" material-top="Al(1.0)GaAs" material-bottom="Al(0.0)GaAs" dx="5." dy="5." dz="0.5"/>
      <cuboid material="GaAs" dx="5." dy="5." dz="1.0"/>
    </stack>
  </cartesian3d>
  <cartesian3d name="simplified3d" axes="x,y" left="extend" right="extend">
    <copy from="original3d">
      <simplify-gradients lam="980" linear="eps" only-role="grad"/>
    </copy>
  </cartesian3d>
  <cartesian2d name="original2d" axes="x,y">
    <stack>
      <rectangle material="GaAs" dx="5." dy="1.0"/>
      <rectangle material-top="Al(0.0)GaAs" material-bottom="Al(1.0)GaAs" dx="5." dy="0.2"/>
      <rectangle material="AlAs" dx="5." dy="1.0"/>
      <rectangle material-top="Al(1.0)GaAs" material-bottom="Al(0.0)GaAs" dx="5." dy="0.5"/>
      <rectangle material="GaAs" dx="5." dy="1.0"/>
    </stack>
  </cartesian2d>
  <cartesian2d name="simplified2d" axes="x,y">
    <copy from="original2d">
      <simplify-gradients lam="980" linear="eps"/>
    </copy>
  </cartesian2d>
  <cartesian3d name="cuboids" axes="x,y,z">
    <stack xcenter="0" ycenter="0">
      <align name="align" order="reverse" xcenter="0" ycenter="0" bottom="0">
        <cuboid material="AlOx" dx="0.05" dy="0.5" dz="0.1"/>
        <cuboid name="bar" material="GaAs" dx="0.2" dy="0.8" dz="0.1" angle="30"/>
      </align>
      <cuboid material="air" dx="2" dy="2" dz="0.1"/>
    </stack>
  </cartesian3d>
  <cartesian3d name="tubic" axes="x,y,z">
    <stack x="0" y="0">
      <tube material="GaAs" inner-radius="1" outer-radius="2" height="3"/>
    </stack>
  </cartesian3d>
  <cartesian2d name="polygons" axes="x,y">
    <polygon name="polygon" material="GaN">{'; '.join(f'{sin(i*1.2*pi)} {cos(i*1.2*pi)}' for i in range(5))}</polygon>
  </cartesian2d>
  <cartesian3d name="prismatic" axes="x,y,z">
    <prism material="GaAs" height="2.0">0 0; 1 0; 1 1; 2 1; 2 0; 3 0; 3 2; 0 2</prism>
  </cartesian3d>
</geometry>

<grids>
  <!--G1-->
  <generator name="default" type="rectangular2d" method="divide">
    <postdiv by0="2" by1="1"/>
    <options gradual1="no"/>
    <refinements>
      <axis1 object="p-contact" at="50"/>
      <axis0 object="oxide" at="-0.1"/>
      <axis0 object="oxide" at="-0.05"/>
      <axis0 object="aperture" at="0.1"/>
    </refinements>
  </generator>
  <generator name="tridi" type="rectangular3d" method="divide">
    <options gradual1="no"/>
  </generator>
  <!--M1-->
  <mesh name="diffusion" type="regular">
    <!--ax-->
    <axis start="0" stop="{mesa}" num="200"/>
  </mesh>
  <!--G2-->
  <generator name="optical" type="rectangular2d" method="divide">
    <!--prediv-->
    <prediv by0="10" by1="3"/>
    <!--options-->
    <options gradual="no" aspect="100"/>
    <!--end-->
  </generator>
  <!--G3-->
  <generator name="smoothie" type="rectangular2d" method="smooth">
    <!--steps-->
    <steps small0="0.005" small1="0.01" large0="0.05" factor="1.2"/>
    <!--end-->
  </generator>
  <!--G3½-->
  <generator name="oned" type="ordered" method="divide">
    <!--refinemenst-->
    <refinements>
      <!--axis0-->
      <axis0 object="bottom-DBR" at="1"/>
      <!--end0-->
    </refinements>
    <!--end-->
  </generator>
  <!--M2-->
  <mesh name="plots" type="rectangular2d">
    <!--axis0-->
    <axis0 start="0" stop="10" num="20"/>
    <!--axis1-->
    <axis1 start="0" stop="1" num="10"/>
    <!--no more axes-->
  </mesh>
  <!--G4-->
  <generator name="sss" type="rectangular3d" method="smooth">
    <steps small0="0.005" small1="0.05" small2="0.05" factor="1.2"/>
  </generator>
  <!--G5-->
  <generator name="reg" type="rectangular2d" method="regular">
    <spacing every0="0.2" every1="1"/>
  </generator>
  <!--G6-->
  <generator name="spl" type="rectangular2d" method="simple">
    <!--boundaries-->
    <boundaries split="yes"/>
  </generator>
  <!--M3-->
  <mesh name="fine" type="rectangular3d">
    <axis0 start="0.5" stop="1.0" num="2001"/>
    <axis1 start="0" stop="2" num="4001"/>
    <axis2>0.0 0.5 1.5</axis2>
  </mesh>
  <!--trangle-->
  <generator name="triangle" type="triangular2d" method="triangle">
    <!--options-->
    <!--end-->
  </generator>
  <generator name="revol" type="rectangular3d" method="simple"/>
</grids>

<solvers>
  <thermal name="THERMAL" solver="StaticCyl" lib="static">
    <geometry ref="GeoT"/>
    <!--mesh-->
    <mesh ref="default" empty-elements="include"/>
    <!--A-->
    <temperature>
      <!--B-->
      <condition value="320.">
        <!--C-->
        <place line="horizontal" at="10" start="0" stop="{lineto}"/>
        <!--D-->
      </condition>
      <!--E-->
      <condition value="300">
        <!--F-->
        <difference>
          <!--G-->
          <place side="bottom"/>
          <!--H-->
          <place side="left"/>
          <!--I-->
        </difference>
        <!--J-->
      </condition>
      <!--K-->
    </temperature>
    <!--L-->
  </thermal>
  <optical name="fourier2" solver="Fourier2D" lib="modal">
    <geometry ref="geo2d"/>
  </optical>
  <gain name="gain2" solver="FreeCarrierCyl" lib="freecarrier">
    <geometry ref="GeoO"/>
    <config matrix-elem="10" T0="300"/>
  </gain>
  <electrical name="ELECTRICAL" solver="ShockleyCyl" lib="shockley">
    <geometry ref="GeoE"/>
    <mesh ref="default"/>
    <voltage>
      <condition value="1">
        <place side="top" object="n-contact"/>
      </condition>
      <condition value="0">
        <place side="bottom"/>
      </condition>
    </voltage>
    <matrix algorithm="cholesky"/>
    <junction beta0="{beta_def}" beta1="19.2" js0="{js_def}" js1="1.1"/>
  </electrical>
  <electrical name="DIFFUSION" solver="DiffusionCyl" lib="diffusion">
    <geometry ref="GeoO"/>
    <mesh ref="diffusion"/>
  </electrical>
  <gain name="GAIN" solver="FreeCarrierCyl" lib="freecarrier">
    <geometry ref="GeoO"/>
    <config lifetime="0.5" matrix-elem="8" substrate="Al(0.2)GaN"/>
  </gain>
  <optical name="OPTICAL" solver="EffectiveFrequencyCyl" lib="effective">
    <geometry ref="GeoO"/>
    <mesh ref="optical"/>
    <mode lam0="980" emission="bottom"/>
  </optical>
  <!--filtr-->
  <filter name="filtr" for="Temperature" geometry="GeoT"/>
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
    <geometry thermal="geo2d-copy" electrical="roads"/>
    <mesh thermal="default" electrical="default"/>
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
    <geometry thermal="GeoT" electrical="GeoE" optical="GeoO"/>
    <mesh thermal="default" electrical="default" diffusion="diffusion"/>
    <root bcond="0"/>
  </meta>
  <meta name="threshold" solver="ThresholdSearchBesselCyl" lib="shockley">
    <geometry thermal="GeoT" electrical="GeoE" optical="GeoO"/>
    <mesh thermal="default" electrical="default"/>
    <root bcond="1"/>
    <voltage>
      <condition value="0">
        <place side="bottom"/>
      </condition>
      <condition value="1">
        <place side="top" object="n-contact"/>
      </condition>
    </voltage>
    <temperature>
      <condition value="300">
        <place side="bottom"/>
      </condition>
    </temperature>
  </meta>
  <!-- COMMENT 1 -->
  <optical name="F3D" solver="Fourier3D" lib="modal">
    <!-- COMMENT 2 -->
    <geometry ref="vcsel"/>
    <!-- COMMENT 3 -->
    <expansion size="12"/>
    <!-- COMMENT 4 -->
    <pmls>
      <!-- COMMENT 5 -->
      <long shape="1"/>
      <!-- COMMENT 6 -->
      <tran shape="1"/>
      <!-- COMMENT 7 -->
    </pmls>
    <!-- COMMENT 8 -->
  </optical>
  <thermal name="solver" solver="Static3D" lib="static">
    <geometry ref="vcsel"/>
    <mesh ref="fine"/>
  </thermal>
  <meta name="thf2d" solver="ThresholdSearchFourier2D" lib="shockley">
    <geometry thermal="geo2d" electrical="geo2d" optical="geo2d-copy"/>
    <mesh thermal="default" electrical="default" diffusion="diffusion"/>
    <optical lam="980"/>
    <root bcond="0" vmin="0" vmax="2"/>
  </meta>
  <local name="local1" solver="Generic2D" lib="solvers">
    <geometry ref="vcsel"/>
  </local>
  <local name="local2" solver="Configured2D" lib="solvers">
    <geometry ref="geo2d"/>
    <mesh ref="default"/>
    <custom attr="attr0">
      <tag1/>
      <tag2/>
    </custom>
  </local>
  <thermal name="dynamic" solver="Dynamic2D" lib="dynamic">
    <geometry ref="simple"/>
    <mesh ref="default"/>
  </thermal>
</solvers>

<connects>
  <connect out="THERMAL.outTemperature" in="ELECTRICAL.inTemperature"/>
  <connect out="ELECTRICAL.outHeat" in="THERMAL.inHeat"/>
  <connect out="THERMAL.outTemperature" in="DIFFUSION.inTemperature"/>
  <connect out="ELECTRICAL.outCurrentDensity" in="DIFFUSION.inCurrentDensity"/>
  <connect out="THERMAL.outTemperature" in="GAIN.inTemperature"/>
  <connect out="DIFFUSION.outCarriersConcentration" in="GAIN.inCarriersConcentration"/>
  <connect out="THERMAL.outTemperature" in="OPTICAL.inTemperature"/>
  <connect out="GAIN.outGain" in="OPTICAL.inGain"/>
  <connect out="GAIN.outGain" in="DIFFUSION.inGain"/>
</connects>

<!--Script-->

<script><![CDATA[
print(GEO.roads.get_object_positions(GEO.small, PTH.blsl))
print(GEO.roads.get_object_positions(GEO.small, PTH.blsr))
print(GEO.roads.get_object_positions(GEO.small, PTH.brsl))
print(GEO.roads.get_object_positions(GEO.small, PTH.brsr))

print(sys.argv)

print(material.get('InGa(0.8)As_QW:Si=1e19'))

print(material.get('test').cond(300.))

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

dictionary = {
    'a': 1,
    'b': 2,
    'c': print_log
}

import module

csys = 1
cmap = 2

print(f"""csys = {csys:03d}, cmap = {{{ {1: 'c', 2: 'a'}[cmap] }}}, function: {(lambda x: x)(3)}. And that's it! {js_def:.2f}""")

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

class A:

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
    DIFFUSION.compute()
    det_lams = linspace(OPTICAL.lam0-2, OPTICAL.lam0+2, 401)+0.2j*(voltage-0.5)/1.5
    det_vals = abs(OPTICAL.get_determinant(det_lams, m=0))
    det_mins = np.r_[False, det_vals[1:] < det_vals[:-1]] & \
               np.r_[det_vals[:-1] < det_vals[1:], False] & \
               np.r_[det_vals[:] < 1]
    mode_number = OPTICAL.find_mode(max(det_lams[det_mins]))
    mode_loss = OPTICAL.outLoss(mode_number)
    print_log(LOG_RESULT, f'V = {voltage[0]:.3f}V, I = {ELECTRICAL.get_total_current():.3f}mA, lam = {OPTICAL.outWavelength(mode_number):.2f}nm, loss = {mode_loss}/cm')
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
xlabel("r (µm)")
ylabel("z (µm)")

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
xlabel("r (µm)")
ylabel("z (µm)")

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

class A:
    def __init__(self):
        self.a = 1

a = A()
print(a.a, file=sys.stderr)
]]></script>

<!-- The End -->

</plask>
