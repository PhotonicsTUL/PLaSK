<plask loglevel="detail">

<defines>
  <define name="dxLay" value="0.0003"/>
  <define name="dyLay" value="0.5000"/>
</defines>

<materials>
  <material name="pcladding" base="GaAs:C=1e16"/>
  <material name="ncladding" base="GaAs:Si=1e16"/>
</materials>

<geometry>
  <cartesian2d name="main" axes="xy">
    <stack>
      <rectangle material="ncladding" dx="{dxLay}" dy="{dyLay}"/>
      <rectangle material="pcladding" dx="{dxLay}" dy="{dyLay}"/>
      <zero/>
    </stack>
  </cartesian2d>
</geometry>

<grids>
  <generator method="smooth" name="siatka" type="rectangular2d">
    <steps small0="{dxLay}" small1="0.0001" large1="0.01" factor1="1.2"/>
  </generator>
</grids>

<script><![CDATA[

import scipy
from scipy import optimize
from electrical import DriftDiffusion2D

# ustawienia solverow:
DDM2D = DriftDiffusion2D("DDM2D")
DDM2D.geometry = GEO.main
DDM2D.mesh = MSG.siatka

DDM2D.invalidate()

T = 300

DDM2D.compute(1);

print_log(LOG_INFO, "Calculations done!")

# siatki do rysowania
###plotgrid = MSH.plots(GEO.main.item)

# interesujacy fragment struktury
left   = 0.
right  = 0.0003
top    = 2. * dxLay
bottom = 0.

# linie do wykresow 1D
r = linspace(left, right, 1000)
z = linspace(bottom, top, 2000)

# to bedzie rysowane
energy_rz = DDM2D.outPotential(DDM2D.mesh)
energy_z = DDM2D.outPotential(mesh.Rectangular2D([0.], z), 'nearest')

# wykres E(r,z)
figure()
plot_field(energy_rz, 16)
colorbar()
plot_geometry(GEO.main, color="w")
gcf().canvas.set_window_title("Energy 2D distribution")

# wykres E(z)
figure()
plot(z, energy_z)
xlabel(u"z [\xb5m]")
ylabel("energy (eV)")
xlim(z[0], z[-1])
gcf().canvas.set_window_title("Energy along the laser axis")
# zapis do pliku
out_txt = open("Ez.txt", 'w')
# for i in range(len(z)):
#         out_txt.write("%.6f     %.6f\n" % (z[i], energy_z[i]))
# out_txt.close()

show()

]]></script>

</plask>
