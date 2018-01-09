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
plt.rcParams.update({'font.size': 28})



#fig, ax1 = plt.subplots()
mode_number = prosty.findMode(981) 
Z = np.linspace(0, 11, 15000)
E = prosty.outLightMagnitude(mode_number, mesh.Rectangular2D([0], Z))
E = E.array[0]
print(E)
neff = prosty.outRefractiveIndex( mesh.Rectangular2D([0], Z ))

fig, ax1 = plt.subplots()
ax1.plot(Z, np.abs(E), 'r-')
ax2 = ax1.twinx()
ax2.plot(Z, neff, 'b-')
ax2.set_ylim([3, 3.7])

plt.show()



]]></script>

</plask>