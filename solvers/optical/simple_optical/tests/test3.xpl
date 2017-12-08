<plask loglevel="detail">

<materials>
  <material name="active" base="semiconductor">
    <nr>3.53</nr>
    <absp>0.</absp>
  </material>
  <material name="inactive" base="active">
    <absp>1000.</absp>
  </material>
</materials>

<geometry>
  <cylindrical2d name="main" axes="r,z" outer="extend" bottom="GaAs">
    <stack>
      <rectangle material="GaAs" dr="10" dz="0.0700"/>
      <stack name="top-DBR" repeat="24">
        <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
        <rectangle material="GaAs" dr="10" dz="0.0700"/>
      </stack>
      <shelf>
        <rectangle material="AlAs" dr="4" dz="0.0160"/>
        <rectangle material="AlOx" dr="6" dz="0.0160"/>
      </shelf>
      <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0635"/>
      <rectangle material="GaAs" dr="10" dz="0.1376"/>
      <shelf>
        <rectangle name="gain-region" material="active" dr="4" dz="0.0050"/>
        <rectangle material="inactive" dr="6" dz="0.0050"/>
      </shelf>
      <rectangle material="GaAs" dr="10" dz="0.1376"/>
      <stack name="bottom-DBR" repeat="30">
        <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
        <rectangle material="GaAs" dr="10" dz="0.0700"/>
      </stack>
    </stack>
  </cylindrical2d>
</geometry>

<solvers>
  <optical solver="SimpleOpticalCyl" name="prosty">
    <geometry ref="main"/>
  </optical>
</solvers>

<script><![CDATA[
from scipy import ndimage

plt.rcParams.update({'font.size': 28})

wavelength = np.linspace(800, 1100, 1000)
t_bb = np.zeros(len(wavelength), dtype=complex)
for i in range(0, len(t_bb)):
    prosty.simpleVerticalSolver(wavelength[i])
    t_bb[i] = prosty.get_T_bb()
plt.plot(wavelength, np.abs(t_bb), 'b-')
plt.xlabel("wavelength [nm]")
plt.ylabel("T bb")
plt.yscale('log')
plt.show()

z = prosty.get_z()
eField = np.array(prosty.get_eField(), dtype=complex)

plt.figure()
prosty.compute_electric_field_distribution(980);
electric_field = np.array(prosty.get_eField(), dtype=complex)
z = prosty.get_z()


plt.plot(z, (electric_field), 'r-', lw=2)
plt.show()

plt.figure()
geo = prosty.geometry

p = plot_geometry(geo, fill=True, alpha=0.8)

p = p.twiny()
p.plot(np.abs(electric_field)**2, z, 'r-', lw=2)
p.set_xlabel("Electric field F")
plt.ylim([4, 6])


plt.show()




#print(prosty.geometry)


]]></script>

</plask>
