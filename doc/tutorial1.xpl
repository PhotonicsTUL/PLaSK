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
  <thermal solver="Rectangular2D" name="THERMAL">
    <geometry ref="main"/>
    <mesh ref="default"/>
    <temperature>
      <condition value="300.0" place="bottom"/>
    </temperature>
  </thermal>
  <electrical solver="Beta2D" name="ELECTRICAL">
    <geometry ref="main"/>
    <mesh ref="default"/>
    <junction pnjcond="1e-06,0.2"/>
    <voltage>
      <condition value="1.0"><place object="top-layer" side="top"/></condition>
      <condition value="0.0"><place object="substrate" side="bottom"/></condition>
    </voltage>
  </electrical>
</solvers>

<connects>
  <connect in="ELECTRICAL.inTemperature" out="THERMAL.outTemperature"/>
  <connect in="THERMAL.inHeatDensity" out="ELECTRICAL.outHeatDensity"/>
</connects>

<script>
ELECTRICAL.js = 1.1
ELECTRICAL.beta = 21.

verr = ELECTRICAL.compute(1)
terr = THERMAL.compute(1)
while terr > THERMAL.corrlim or verr > ELECTRICAL.corrlim:
    verr = ELECTRICAL.compute(6)
    terr = THERMAL.compute(1)

plotgrid = MSG.plots(GEO.main.child)

temperature = THERMAL.outTemperature(plotgrid)
heats = THERMAL.inHeatDensity(plotgrid)
voltage = ELECTRICAL.outPotential(plotgrid)
current = ELECTRICAL.outCurrentDensity(plotgrid)

if has_hdf5:
    import sys, os
    h5file = h5py.File(os.path.splitext(sys.argv[0])[0]+".h5", "w")
    save_field(temperature, h5file, "Temperature")
    save_field(heats, h5file, "HeatDensity")
    save_field(voltage, h5file, "Voltage")
    save_field(current, h5file, "CurrentDenstity")
    h5file.close

if has_pylab:
    plot_geometry(GEO.main, set_limits=True)
    defmesh = MSG.default(GEO.main.child)
    plot_mesh(defmesh, color="0.75")
    plot_boundary(ELECTRICAL.voltage_boundary, defmesh, color="b", marker="D")
    plot_boundary(THERMAL.temperature_boundary, defmesh, color="r")
    plot_boundary(THERMAL.convection_boundary, defmesh, color="g")
    plot_boundary(THERMAL.radiation_boundary, defmesh, color="y")
    gcf().canvas.set_window_title("Default mesh")

    figure()
    plot_field(temperature, 16)
    colorbar()
    plot_geometry(GEO.main, color="w")
    gcf().canvas.set_window_title("Temperature")

    figure()
    plot_field(heats, 16)
    colorbar()
    plot_geometry(GEO.main, color="w")
    gcf().canvas.set_window_title("Heat sources density")

    figure()
    plot_field(voltage, 16)
    colorbar()
    plot_geometry(GEO.main, color="w")
    gcf().canvas.set_window_title("Electric potential")

    #figure()
    #plot(actgrid.axis0, abs(acurrent.array[0,:,1]))
    #xlabel(u"x [\xb5m]")
    #ylabel("current density [kA/cm$^2$]")
    #simplemesh = mesh.Rectilinear2D.SimpleGenerator()(GEO.main.child)
    #for x in simplemesh.axis0:
    #    axvline(x, ls=":", color="k")
    #xlim(0., 2*simplemesh.axis0[-2])
    #gcf().canvas.set_window_title("Current density in the active region")

    show()

</script>

</plask>
