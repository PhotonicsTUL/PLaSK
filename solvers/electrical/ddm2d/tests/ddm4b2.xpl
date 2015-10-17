<plask loglevel="detail">

<defines>
  <define name="dxLay" value="0.0004"/>
  <define name="dyLay" value="0.5000"/>
  <define name="nx" value="2"/>
  <define name="ny" value="9301"/>
</defines>

<materials>
  <material name="pcap" base="GaAs:C=1e17"/>
  <material name="pcladding" base="Al(0.40)GaAs:C=1e17"/>
  <material name="pactive" base="Al(0.14)GaAs:C=5e16"/>
  <material name="ncladding" base="Al(0.40)GaAs:Si=1e17"/>
  <material name="nsubstrate" base="GaAs:Si=1e17"/>
</materials>

<geometry>
  <cartesian2d name="main" axes="xy">
    <stack name="eel">
      <rectangle name="pca" material="pcap" dx="{dxLay}" dy="{0.3000}"/>
      <rectangle name="pcl" material="pcladding" dx="{dxLay}" dy="{2.0000}"/>
      <rectangle name="pac" material="pactive" dx="{dxLay}" dy="{0.0500}"/>
      <rectangle name="ncl" material="ncladding" dx="{dxLay}" dy="{0.3000}"/>
      <rectangle name="nsu" material="nsubstrate" dx="{dxLay}" dy="{2.0000}"/>
      <zero/>
    </stack>
  </cartesian2d>
</geometry>

<grids>
  <mesh name="siatka" type="rectangular2d">
    <axis0 start="0" stop="{dxLay}" num="{nx}"></axis0>
    <axis1 start="0" stop="{2.*dyLay}" num="{ny}"></axis1>
  </mesh>
  <generator method="smooth" name="genmesh2" type="rectangular2d">
    <steps small0="0.0002" small1="0.0001" factor0="1.1" factor1="1.1"/>
  </generator>
</grids>

<script><![CDATA[

import scipy
from scipy import optimize
from electrical import DriftDiffusion2D

# ustawienia solverow:
DDM2D = DriftDiffusion2D("DDM2D")
DDM2D.maxerrPsiI = 1e-12
DDM2D.maxerrPsi0 = 1e-10
DDM2D.maxerrPsi = 1e-6
DDM2D.maxerrFn = 1e-6
DDM2D.maxerrFp = 1e-6
DDM2D.iterlimPsiI = 10000
DDM2D.iterlimPsi0 = 2000
DDM2D.iterlimPsi = 3
DDM2D.iterlimFn = 3
DDM2D.iterlimFp = 3
DDM2D.geometry = GEO.main
DDM2D.mesh = MSH.siatka
#DDM2D.mesh = MSG.genmesh2
U = 0
dU = 0.001
DDM2D.voltage_boundary.append(DDM2D.mesh.TopOf(GEO.pca), U)
DDM2D.voltage_boundary.append(DDM2D.mesh.BottomOf(GEO.nsu), 0.0)

DDM2D.invalidate()

T = 300

# interesujacy fragment struktury
left   = 0.
right  = 0.0003
top    = 10.0000
bottom = 0.

# linie do wykresow 1D
r = linspace(left, right, 1000)
z = linspace(bottom, top, 10000)

DDM2D.compute_initial_potential(1);

errorPsi0 = DDM2D.compute_potential_0(1);
print errorPsi0

while (U <= 1.8):

     U = U + dU

     print "U: %.3f V" % U

     DDM2D.voltage_boundary.append(DDM2D.mesh.TopOf(GEO.pca), U)
     DDM2D.increase_voltage();

     itersPsiFnFp = 0
     while (itersPsiFnFp<5): # 5

          error = DDM2D.compute_potential(1);
          itersPsiFnFp += 1

print_log(LOG_INFO, "Calculations done!")
# to bedzie rysowane
potential_rz = DDM2D.outPotential(DDM2D.mesh)
potential_z = DDM2D.outPotential(mesh.Rectangular2D([0.], z), 'nearest')
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

show()

]]></script>

</plask>
