<plask loglevel="detail">

<geometry>
  <cartesian2d name="main" axes="xy" left="mirror" length="1000">
    <stack>
      <rectangle name="top-layer" material="Al(0.3)GaAs:C=1e20" dx="1.5" dy="1.350"/>
      <rectangle material="Al(0.3)GaAs:C=1e20" dx="150" dy="0.150"/>
      <rectangle material="GaAs" dx="150" dy="0.150"/>
      <rectangle name="junction" role="active" material="In(0.2)GaAs" dx="150" dy="0.007"/>
      <rectangle material="GaAs" dx="150" dy="0.150"/>
      <rectangle material="Al(0.3)GaAs:Si=5e19" dx="150" dy="1.500"/>
      <rectangle name="substrate" material="GaAs:Si=5e19" dx="150" dy="300"/>
      <zero/>
      <rectangle material="Cu" dx="1000" dy="1000"/>
    </stack>
  </cartesian2d>
</geometry>

<grids>
  <generator method="divide" name="default" type="rectangular2d">
    <postdiv by="2"/>
  </generator>
</grids>

<solvers>
  <meta name="SOLVER" solver="ThermoElectric2D" lib="shockley">
    <geometry electrical="main" thermal="main"/>
    <mesh electrical="default" thermal="default"/>
    <voltage>
      <condition value="1.0">
        <place side="top" object="top-layer"/>
      </condition>
      <condition value="0.0">
        <place side="bottom" object="substrate"/>
      </condition>
    </voltage>
    <temperature>
      <condition place="bottom" value="300.0"/>
    </temperature>
    <junction beta0="19" js0="1"/>
  </meta>
</solvers>

<script><![CDATA[
SOLVER.tfreq = 2
print(SOLVER.tfreq, type(SOLVER.tfreq))

SOLVER.compute()

figure()
SOLVER.plot_temperature()
xlim(0., 40.)
ylim(290., ylim()[1])
tight_layout()

figure()
SOLVER.plot_voltage()
xlim(0., 40.)
ylim(290., ylim()[1])
tight_layout()

figure()
SOLVER.plot_vertical_voltage()
xlim(290., xlim()[1])
tight_layout()

figure()
SOLVER.plot_junction_current()
xlim(0., 40.)
tight_layout()

show()

]]></script>

</plask>
