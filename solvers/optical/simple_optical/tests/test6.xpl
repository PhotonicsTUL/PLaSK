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
      <stack name="top-DBR" repeat="6">
        <rectangle material="Al(0.73)GaAs" dr="10" dz="3.5"/>
        <rectangle material="GaAs" dr="10" dz="1.5"/>
      </stack>
      <rectangle material="Al(0.73)GaAs" dr="10" dz="3.5"/>
      <stack name="bottom-DBR" repeat="6">
        <rectangle material="Al(0.73)GaAs" dr="10" dz="1.5"/>
        <rectangle material="GaAs" dr="10" dz="3.5"/>
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
plt.rcParams.update({'font.size': 28})

prosty.vat = 0
mode_number = prosty.findMode(980) 

Z = np.linspace(0, 12.6, 5)


E = prosty.outLightMagnitude(mode_number, mesh.Rectangular2D([0], Z))
'''neff = prosty.outRefractiveIndex( mesh.Rectangular2D([0], Z ))
fig, ax1 = plt.subplots()
ax1.plot(Z, np.abs(E), 'r-')
ax1.set_ylabel("Light Magnitude $E$", color="red")
ax1.set_yscale('log')
ax2 = ax1.twinx()
ax2.plot(Z, neff, 'b-')
ax2.set_ylabel("refractive index", color="blue")
ax2.set_ylim([0.9, 3.7])
plt.show()
'''

]]></script>

</plask>
