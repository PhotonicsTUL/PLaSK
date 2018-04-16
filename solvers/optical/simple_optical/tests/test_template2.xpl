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
  <cartesian2d name="main" axes="r,z" bottom="GaAs">
    <stack>
    <rectangle material="GaAs" dr="10" dz="0.0700"/>
      <stack name="top-DBR" repeat="34">
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
      <stack name="bottom-DBR" repeat="50">
        <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
        <rectangle material="GaAs" dr="10" dz="0.0700"/>
      </stack>
    </stack>
  </cartesian2d>
</geometry>

<solvers>
  <optical solver="SimpleOpticalCar2D" name="prosty" lib="simple_optical">
    <geometry ref="main"/>
  </optical>
</solvers>
<script><![CDATA[

prosty.vat = 0
prosty.root.tolx = 1e-5

mode_number = prosty.findMode(980)
Z = np.linspace(-2, 18.6, 5000)
neff = prosty.outRefractiveIndex( mesh.Rectangular2D([0], Z ))
E = prosty.outLightMagnitude(mode_number, mesh.Rectangular2D([0], Z))


fig, ax1 = plt.subplots()
ax1.plot(Z, np.abs(E), 'r-')
ax1.set_ylabel("Light Magnitude $E$", color="red")
ax1.set_yscale('log')
ax2 = ax1.twinx()
ax2.plot(Z, neff, 'b-')
ax2.set_ylabel("refractive index", color="blue")

ax2.set_ylim([0.9, 3.7])
plt.show()




 
wavelength = np.linspace(900, 1050, 100)
t_bb = np.zeros(len(wavelength), dtype=complex)

for i in range(0, len(t_bb)):
    t_bb[i] = prosty.get_vert_dteterminant(wavelength[i])

plt.plot(wavelength, np.abs(t_bb), 'b-')
plt.xlabel("wavelength [nm]")
plt.ylabel("T bb")
plt.yscale('log')
plt.axhline(color='black')
plt.ylim([10e-1, 10e4])
show()

]]></script>

</plask>
