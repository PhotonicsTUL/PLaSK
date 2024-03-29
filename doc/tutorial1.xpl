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

<connects>
  <connect in="electr.inTemperature" out="therm.outTemperature"/>
  <connect in="therm.inHeat" out="electr.outHeat"/>
</connects>

<script>

verr = electr.compute(1)
terr = therm.compute(1)

while terr > therm.maxerr or verr > electr.maxerr:
    verr = electr.compute(6)
    terr = therm.compute(1)

print_log(LOG_INFO, "Calculations finished!")

temp = therm.outTemperature(therm.mesh)

plot_field(temp, 12)
plot_geometry(GEO["main"])
colorbar()

figure()
plot_geometry(GEO["main"], margin=0.01)
plot_mesh(electr.mesh)

pos = GEO["main"].get_object_positions(GEO["junction"])[0]
junction_mesh = mesh.Rectangular2D(linspace(-150., 150., 1000), [pos.y])
current = electr.outCurrentDensity(junction_mesh)

figure()
plot_profile(-current, comp='y')
ylabel("current density (kA/cm$^2$)")

show()

</script>

</plask>
