<plask loglevel="detail">

<geometry>

  <cylindrical2d name="cyl">

    <stack>

      <rectangle material="AlAs" dtran="1" dvert="0.5"/>

      <rectangle material="GaAs" dtran="1" dvert="1"/>

      <rectangle material="GaAs" dtran="1" dvert="1.5"/>

    </stack>

  </cylindrical2d>

</geometry>

<solvers>

  <optical solver="SimpleOpticalCyl" name="prosty">

    <geometry ref="cyl"/>

  </optical>

</solvers>

<script><![CDATA[

plt.rcParams.update({'font.size': 28})

wavelength = np.linspace(800, 1400, 21)
t_bb = np.zeros(len(wavelength), dtype=complex)
for i in range(0, len(t_bb)):
    prosty.simpleVerticalSolver(wavelength[i])
    t_bb[i] = prosty.get_T_bb()
    print(t_bb[i])
plt.plot(wavelength, np.abs(t_bb), 'b-')
plt.xlabel("wavelength [nm]")
plt.ylabel("T bb")
plt.show()


]]></script>

</plask>
