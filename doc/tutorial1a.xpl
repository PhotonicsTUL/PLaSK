<plask>

<geometry>
  <cartesian2d axes="xy" left="mirror" length="1000" name="main">
    <stack>
      <block dx="1.5" dy="1.350" material="Al(0.3)GaAs:C=1e20" name="top-layer"/>
      <block dx="150" dy="0.150" material="Al(0.3)GaAs:C=1e20"/>
      <block dx="150" dy="0.150" material="GaAs"/>
      <block dx="150" dy="0.007" material="In(0.2)GaAs" role="active" name="junction"/>
      <block dx="150" dy="0.150" material="GaAs"/>
      <block dx="150" dy="1.500" material="Al(0.3)GaAs:Si=5e19"/>
      <block dx="150" dy="300" material="GaAs:Si=5e19" name="substrate"/>
      <zero/>
      <block dx="1000" dy="1000" material="Cu"/>
    </stack>
  </cartesian2d>
</geometry>

<grids>
  <generator type="rectangular2d" method="divide" name="default">
    <postdiv by="2"/>
  </generator>
</grids>

<solvers>
  <thermal solver="Static2D" name="therm">
    <geometry ref="main"/>
    <mesh ref="default"/>
    <temperature>
      <condition value="300.0" place="bottom"/>
    </temperature>
  </thermal>
  <electrical solver="Shockley2D" name="electr">
    <geometry ref="main"/>
    <mesh ref="default"/>
    <junction beta="19" js="1"/>
    <voltage>
      <condition value="1.0"><place object="top-layer" side="top"/></condition>
      <condition value="0.0"><place object="substrate" side="bottom"/></condition>
    </voltage>
  </electrical>
</solvers>

<script>

task = algorithm.ThermoElectric(therm, electr)

task.run()

figure()
task.plot_temperature()
xlim(0., 40.)
ylim(290., ylim()[1])
tight_layout()

figure()
task.plot_voltage()
xlim(0., 40.)
ylim(290., ylim()[1])
tight_layout()

figure()
task.plot_vertical_voltage()
xlim(290., xlim()[1])
tight_layout()

figure()
task.plot_junction_current()
xlim(0., 40.)
tight_layout()

show()

</script>

</plask>
