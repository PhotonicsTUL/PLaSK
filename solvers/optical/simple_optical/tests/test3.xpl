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

def get_k0(wavelength):
    return (2e3*np.pi)/wavelength


#wavelength = np.linspace(900, 1050, 100)
#t_bb = np.zeros(len(wavelength), dtype=complex)
#for i in range(0, len(t_bb)):
#    prosty.simpleVerticalSolver(wavelength[i])
#    t_bb[i] = prosty.get_T_bb()
#plt.plot(wavelength, np.abs(t_bb), 'b-')
#plt.xlabel("wavelength [nm]")
#plt.ylabel("T bb")
#plt.yscale('log')
#plt.axhline(color='black')
#plt.ylim([10e-1, 10e3])
#print(t_bb)
#Ez = prosty.getEz()
#print(Ez)
#w = np.linspace(900, 1050, 5)
#for i in w:
#    k = get_k0(i)
#    prosty.findRoot(k)
    
#prosty.findRoot(6.45)
fig, ax1 = plt.subplots()
prosty.computeField(978)

z = prosty.getZ()
eField = np.array(prosty.getEz(), dtype=complex)
nrCache = prosty.getNrCache()
print(len(z))
print(len(eField))
ax1.plot(z, np.abs((eField)), 'r-')
ax1.set_ylabel("electric field", color="red")
ax2 = ax1.twinx()
ax2.plot(z, nrCache, 'b-')
ax1.set_xlabel('z')
ax2.set_ylabel("refractive index", color="blue")

plt.figure()
geo = prosty.geometry
p = plot_geometry(geo, fill=True, alpha=0.8)
p = p.twiny()
p.plot(np.abs(eField), z, 'b-', lw=1.5)
p.set_xlabel("Electric field F")
plt.ylim([-1, 11])

plt.show()


#prosty.getLightMagnitude(2, [1,2,3])
#m = plask.mesh.Generator2D.generate(prosty.geometry())
#print(m)
#print(prosty.geometry)


]]></script>

</plask>
