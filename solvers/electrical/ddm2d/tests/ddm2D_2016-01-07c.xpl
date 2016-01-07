<plask loglevel="detail">

<defines>
  <define name="dxLay" value="2.0000"/>
  <define name="nx" value="500"/>
  <define name="ny" value="2501"/>
</defines>

<materials>
  <material name="pCap" base="GaAs:C=1e19">
    <Ni>2.1e6</Ni>
  </material>
  <material name="pWindow" base="Al(0.85)GaAs:C=2e18">
    <Ni>2.1e6</Ni>
  </material>
  <material name="pEmitter" base="GaAs:C=4e17">
    <Ni>2.1e6</Ni>
  </material>
  <material name="nBase" base="GaAs:Si=2e16">
    <Ni>2.1e6</Ni>
  </material>
  <material name="nBufer" base="GaAs:Si=1e18">
    <Ni>2.1e6</Ni>
  </material>
  <material name="nSubstrate" base="GaAs:Si=4e18">
    <Ni>2.1e6</Ni>
  </material>
  <material name="iPowietrze" base="GaAs:Si=1e15">
    <Ni>2.1e6</Ni>
  </material>
</materials>

<geometry>
  <cartesian2d name="main" axes="xy">
    <stack name="eel">
      <shelf>
        <rectangle name="rpCap" material="pCap" dx="1.0000" dy="0.5000"/>
        <rectangle name="riPowietrze" material="iPowietrze" dx="1.0000" dy="0.5000"/>
      </shelf>
      <rectangle name="rpWindow" role="active" material="pWindow" dx="{dxLay}" dy="0.0300"/>
      <rectangle name="rpEmitter" material="pEmitter" dx="{dxLay}" dy="0.5000"/>
      <rectangle name="rnBase" material="nBase" dx="{dxLay}" dy="3.0000"/>
      <rectangle name="rnBufer" material="nBufer" dx="{dxLay}" dy="0.5000"/>
      <rectangle name="rnSubstrate" material="nSubstrate" dx="{dxLay}" dy="5.0000"/>
      <zero/>
    </stack>
  </cartesian2d>
</geometry>

<grids>
  <mesh name="siatka" type="rectangular2d">
    <axis0 start="0" stop="{dxLay}" num="{nx}"></axis0>
    <axis1 start="0" stop="9.53" num="{9531}"></axis1>
  </mesh>
  <generator method="smooth" name="genmesh2" type="rectangular2d">
    <steps small="0.001" large="0.1" factor="1.3"/>
  </generator>
</grids>

<solvers>
  <electrical name="DDM2D" solver="DriftDiffusion2D" lib="ddm2d">
    <geometry ref="main"/>
    <mesh ref="genmesh2"/>
    <loop loopsFn="3" loopsFp="3" loopsV="3" loopsV0="2000" loopsVi="10000" maxerrFn="1e-6" maxerrFp="1e-6" maxerrV="1e-6" maxerrV0="1e-10" maxerrVi="1e-12" stat="MaxwellBoltzmann"/>
  </electrical>
</solvers>

<script><![CDATA[

import scipy
from scipy import optimize
from electrical import DriftDiffusion2D
from datetime import datetime

# ustawienia solverow:
U = 0
dU = 0.002
DDM2D.voltage_boundary.append(DDM2D.mesh.TopOf(GEO.rpCap), U)
DDM2D.voltage_boundary.append(DDM2D.mesh.BottomOf(GEO.rnSubstrate), 0.0)

DDM2D.invalidate()

T = 300

# siatki do rysowania
#plotgrid = MSG.plots(GEO.main.item)
plotgrid = GEO.main.item
#actbox = GEO.main.get_object_bboxes(GEO.active)[0]
actbox = GEO.active.item
actlevel = 0.5 * (actbox.lower[1] + actbox.upper[1]) # + 0.002
actgrid = mesh.Rectangular2D(plotgrid.axis0, [actlevel])
plotgrid.axis1.insert(actlevel)

# interesujacy fragment struktury
left   = 0.
right  = 2.0000
top    = 10.0000
bottom = 0.

# linie do wykresow 1D
r = linspace(left, right, 1000)
z = linspace(bottom, top, 2000)

print str(datetime.now())

for U in arange(0., 0.5+dU/2., dU):
     print "U: %.3f V" % U
     DDM2D.voltage_boundary[0].value = U
     DDM2D.compute(200)
     print str(datetime.now())

print str(datetime.now())

print_log(LOG_INFO, "Calculations done!")
# to bedzie rysowane
potential_rz = DDM2D.outPotential(DDM2D.mesh)
potential_z = DDM2D.outPotential(mesh.Rectangular2D([0.], z), 'nearest')
#potential_z2 = DDM2D.outPotential(mesh.Rectangular2D([0.05], z), 'nearest')
#potential_z3 = DDM2D.outPotential(mesh.Rectangular2D([0.1], z), 'nearest')
Fn_z = DDM2D.outQuasiFermiEnergyLevelForElectrons(mesh.Rectangular2D([0.], z), 'nearest')
Fp_z = DDM2D.outQuasiFermiEnergyLevelForHoles(mesh.Rectangular2D([0.], z), 'nearest')
Ec_z = DDM2D.outConductionBandEdge(mesh.Rectangular2D([0.], z), 'nearest')
Ev_z = DDM2D.outValenceBandEdge(mesh.Rectangular2D([0.], z), 'nearest')
# zapis do pliku
#out_txt = open("ResultsZ.txt", 'w')
#for i in range(len(z)):
#         out_txt.write("%.6f     %.6f     %.6f     %.6f\n" % (z[i], potential_z[i], Fn_z[i], Fp_z[i]))
#out_txt.close()

# wykres E(r,z)
figure()
plot_field(potential_rz, 16)
colorbar()
plot_geometry(GEO.main, color="w")
gcf().canvas.set_window_title("Energy 2D distribution")

# wykres E(z)
#figure()
#plot(z, potential_z)
#xlabel(u"z [\xb5m]")
#ylabel("energy (eV)")
#xlim(z[0], z[-1])
#gcf().canvas.set_window_title("Energy along the laser axis")

# wykres Fn(z)
#figure()
#plot(z, Fn_z)
#xlabel(u"z [\xb5m]")
#ylabel("energy (eV)")
#xlim(z[0], z[-1])
#gcf().canvas.set_window_title("Quasi-Fermi electron level along the laser axis")

# wykres Fp(z)
#figure()
#plot(z, Fp_z)
#xlabel(u"z [\xb5m]")
#ylabel("energy (eV)")
#xlim(z[0], z[-1])
#gcf().canvas.set_window_title("Quasi-Fermi hole level along the laser axis")

# wykres E(z)
figure()
plot(z, potential_z)
#plot(z, potential_z2)
#plot(z, potential_z3)
plot(z, Fn_z)
plot(z, Fp_z)
plot(z, Ec_z)
plot(z, Ev_z)
xlabel(u"z [\xb5m]")
ylabel("energy (eV)")
xlim(z[0], z[-1])
gcf().canvas.set_window_title("Psi, Fn, Fp, Ec, Ev along the laser axis")

# zapis do pliku
#out_txt = open("Vz.txt", 'w')
#for i in range(len(z)):
#         out_txt.write("%.6f     %.6f\n" % (z[i], potential_z[i]))
#out_txt.close()

currentN = ELECTRICAL.outCurrentDensityForElectrons(plotgrid)
act_currentN = ELECTRICAL.outCurrentDensityForElectrons(actgrid)
currentP = ELECTRICAL.outCurrentDensityForHoles(plotgrid)
act_currentP = ELECTRICAL.outCurrentDensityForHoles(actgrid)

figure()
plot(actgrid.axis0, abs(act_currentN.array[:,0,1]), label="PLaSK")
legend()
xlabel(u"r [\xb5m]")
ylabel("current density  [kA/cm$^2$]")
for x in mesh.Rectangular2D.SimpleGenerator()(GEO.main.item).axis0:
    axvline(x, ls=":", color="k")
xlim(0, 10)
gcf().canvas.set_window_title("Current density for electrons in the active region")

figure()
plot(actgrid.axis0, abs(act_currentP.array[:,0,1]), label="PLaSK")
legend()
xlabel(u"r [\xb5m]")
ylabel("current density  [kA/cm$^2$]")
for x in mesh.Rectangular2D.SimpleGenerator()(GEO.main.item).axis0:
    axvline(x, ls=":", color="k")
xlim(0, 10)
gcf().canvas.set_window_title("Current density for holes in the active region")


show()

]]></script>

</plask>
