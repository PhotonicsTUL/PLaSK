<plask>

<geometry>
  <cartesian2d axes="xy" left="mirror" name="main">
    <stack shift="-3000">
      <block x="1.5" y="1.350" material="Al(0.3)GaAs:C=1e20" name="top-layer"/>
      <block x="150" y="0.150" material="Al(0.3)GaAs:C=1e20"/>
      <block x="150" y="0.150" material="GaAs"/>
      <block x="150" y="0.007" material="In(0.2)GaAs" role="active"/>
      <block x="150" y="0.150" material="GaAs"/>
      <block x="150" y="0.150" material="Al(0.3)GaAs:Si=5e19" name="substrate"/>
      <block x="150" y="1.500" material="GaAs:Si=5e19"/>
      <block x="5000" y="3000" material="Cu"/>
    </stack>
  </cartesian2d>
</geometry>

<grids>
  <generator type="rectilinear2d" method="divide" name="default">
    <postdiv by="2"/>
  </generator>
  <generator type="rectilinear2d" method="divide" name="plots">
    <postdiv by="10"/>
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
  <electrical solver="Beta2D" name="electr">
    <geometry ref="main"/>
    <mesh ref="default"/>
    <voltage>
      <condition value="1.0"><place object="top-layer" side="top"/></condition>
      <condition value="0.0"><place object="substrate" side="bottom"/></condition>
    </voltage>
  </electrical>
</solvers>

<connects>
  <connect in="electr.inTemperature" out="therm.outTemperature"/>
  <connect in="therm.inHeatDensity" out="electr.outHeatDensity"/>
</connects>

<script>

verr = electr.compute(1)
terr = therm.compute(1)

while terr > therm.corrlim or verr > electr.corrlim:
    verr = electr.compute(6)
    terr = therm.compute(1)

print_log(LOG_INFO, "Calculations finished!")

temp = therm.outTemperature(therm.mesh)

plot_field(temp, 12)
plot_geometry(GEO["main"], color='w')
colorbar()

show()

</script>

</plask>
