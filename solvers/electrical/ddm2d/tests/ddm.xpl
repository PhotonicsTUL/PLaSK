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

#DDM2D.iterlimPsi0 = 0
DDM2D.invalidate()

T = 300

try:
  DDM2D.compute()
except AttributeError:
  DDM2D.compute_initial_potential(1000)
  DDM2D.compute_potential_0(1000)

print_log(LOG_INFO, "Calculations done!")

# siatki do rysowania
###plotgrid = MSH.plots(GEO.main.item)

# interesujacy fragment struktury
left   = 0.
right  = 0.0003
top    = 2. * dyLay
bottom = 0.

# linie do wykresow 1D
r = linspace(left, right, 1000)
z = linspace(bottom, top, 2000)

mshz = mesh.Rectangular2D([0.], z)

# # wykres E(r,z)
# energy_rz = DDM2D.outPotential(DDM2D.mesh)
# figure()
# plot_field(energy_rz, 16)
# colorbar()
# plot_geometry(GEO.main, color="w")
# gcf().canvas.set_window_title("Energy 2D distribution")

# wykres E(z)
energy_z = DDM2D.outPotential(mshz, 'nearest')
figure()
plot(mshz.axis1, energy_z)
xlabel(u"z [\xb5m]")
ylabel("energy (eV)")
xlim(mshz.axis1[0], mshz.axis1[-1])
gcf().canvas.set_window_title("Energy along the laser axis")
# zapis do pliku
out_txt = open("Ez.txt", 'w')
# for i in range(len(z)):
#         out_txt.write("%.6f     %.6f\n" % (z[i], energy_z[i]))
# out_txt.close()

show()

]]></script>

</plask>
