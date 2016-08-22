<plask loglevel="detail">

<defines>
  <define name="L" value="4.0"/>
  <define name="R" value="L"/>
  <define name="d" value="0.5"/>
  <define name="totalx" value="(6.1*sqrt(3)/2+d)*L"/>
  <define name="totaly" value="(6.1+d)*L"/>
  <define name="etched" value="24"/>
  <define name="start" value="979.5"/>
  <define name="gain" value="300"/>
  <define name="threshold" value="True"/>
  <define name="N" value="12"/>
</defines>

<materials>
  <material name="GaAs" base="semiconductor">
    <Nr>3.53</Nr>
  </material>
  <material name="AlGaAs" base="semiconductor">
    <Nr>3.08</Nr>
  </material>
  <material name="QW" base="semiconductor">
    <Nr>3.56 - 0.01j</Nr>
  </material>
</materials>

<geometry>
  <cartesian3d name="main" axes="x,y,z" back="mirror" front="extend" left="mirror" right="extend" bottom="GaAs">
    <clip left="0" back="0">
      <align xcenter="0" ycenter="0" top="{0.14895*24 + 0.13471}">
        <stack>
          <stack name="top-dbr" repeat="24">
            <cuboid material="GaAs" dx="{totalx}" dy="{totaly}" dz="0.06940"/>
            <cuboid material="AlGaAs" dx="{totalx}" dy="{totaly}" dz="0.07955"/>
          </stack>
          <stack name="cavity">
            <cuboid material="GaAs" dx="{totalx}" dy="{totaly}" dz="0.12171"/>
            <align name="qw" xcenter="0" ycenter="0" bottom="0">
              <cuboid material="QW" dx="{totalx}" dy="{totaly}" dz="0.00800"/>
              <cylinder name="gain" role="gain" material="QW" radius="{R}" height="0.00800"/>
            </align>
            <cuboid name="interface" material="GaAs" dx="{totalx}" dy="{totaly}" dz="0.00500"/>
            <again ref="qw"/>
            <cuboid material="GaAs" dx="{totalx}" dy="{totaly}" dz="0.00500"/>
            <again ref="qw"/>
            <cuboid material="GaAs" dx="{totalx}" dy="{totaly}" dz="0.12171"/>
          </stack>
          <stack name="bottom-dbr" repeat="29">
            <cuboid material="AlGaAs" dx="{totalx}" dy="{totaly}" dz="0.07955"/>
            <cuboid material="GaAs" dx="{totalx}" dy="{totaly}" dz="0.06940"/>
          </stack>
        </stack>
        <lattice ax="0" ay="{L}" bx="{L*sqrt(3)/2}" by="{L/2}">
          <segments>-1 -3; 4 -3; 4 -2; 2 2; 1 3; -4 3; -4 2; -2 -2 ^ 0 -1; 1 -1; 1 0; 0 1; -1 1; -1 0</segments>
          <cylinder material="air" radius="{0.5*d*L}" height="{0.14895*etched}"/>
        </lattice>
      </align>
    </clip>
  </cartesian3d>
</geometry>

<solvers>
  <optical name="FOURIER" solver="Fourier3D" lib="slab">
    <geometry ref="main"/>
    <expansion lam0="980" size="{N}" update-gain="no"/>
    <mode symmetry-long="Etran" symmetry-tran="Etran"/>
    <interface object="interface"/>
  </optical>
</solvers>

<script><![CDATA[
from scipy.optimize import fsolve

# Here we define artificial gain in the structure. If we had added electrical and gain solvers, it could be
# calculated from the current. However, this time, we specify it manually.
profile = StepProfile(GEO.main)   # We will add gain profile to the ‘main’ geometry
profile[GEO.gain] = gain          # The gain over the geometry object ‘gain’ has value specified in defines

# As step profiles can only be defined in Python script, we must make the connection here,
# not in the ‘Connects’ section.
FOURIER.inGain = profile.outGain

# These are total sizes of computational domain. We will use them for plotting fields. In practice it is
# not necessary, but we want to make sure we did not find spurious mode in PMLs.
dx = totalx/2 + FOURIER.pmls[0].dist + FOURIER.pmls[0].size
dy = totaly/2 + FOURIER.pmls[0].dist + FOURIER.pmls[0].size

# Declare meshes for plotting the fields in xy and yz planes. Each mesh must have three axes.
xx = mesh.Regular(-dx, dx, 201)
yy = mesh.Regular(-dy, dy, 201)
zz = mesh.Regular(-5., 5., 1001)
msh = mesh.Rectangular3D(xx, yy, [GEO.main.get_object_positions(GEO.interface)[0].z])
vmsh = mesh.Rectangular3D([0.], yy, zz)


# Depending on the ‘threshold’ parameter (True of False) we can either look for the threshold (where
# imaginary part of the wavelength is always 0), or just the wavelenght (with some imaginary part, 
# denoting optical losses).

if not threshold:  # ‘threshold’ is False, so we want just an eigenmode...
    
    # For an eigenmode, the returned value of ‘FOURIER.get_determinant’ method must be 0 (both real and 
    # imaginary part). ‘FOURIER.find_mode’ uses Broyden or Muller method (depening on the configuration
    # in ‘Solvers’ tab) to find such root, altering its argument — wavelenght (lam) in this case.
    # Once it finds the eigenmode it stores its parameters in the ‘FOURIER.modes’ list and returns the index
    # of the found mode in this array.
    m = FOURIER.find_mode(lam=start)
    lam = FOURIER.modes[m].lam  # We retrieve the complex wavelength of the eignemode
    
    # Now we construct the line with interesting results (see http://thepythonguru.com/python-string-formatting/)
    line = "{L:3.1f} {d:2.1f} {part}   {N}   {lam.real:.6f} {lam.imag:6.3f}".format(lam=lam, **DEF)
    # We print the line to screen and append it to a file.
    print(line)
    output = open('losses.out', 'a')
    output.write(line+"\n")
    output.close()

else: # Looking for the threshold...

    # To save time we are not using ‘FOURIER.find_mode’ method, which looks for complex wavelength,
    # but instead we use ‘fsolve’ function from SciPy package. In each step we need to alter the gain
    # value. Then we could ‘FOURIER.find_mode’ to find the mode and repeat until the found mode has
    # imaginary part sufficiently close to 0, but it would be a waste of time. Hence, we will ignore
    # PLaSK internal root finding loop, and do it manually (actually we use external Python function
    # from SciPy package for this purpose).
    
    # First we need to create R²->R² function that takes the wavelenght (real) and the gain and return
    # Real and complex parts of the determinant.
    def fun(arg):
        profile[GEO.gain] = arg[1]  # We set the gain given in the second element of the ‘arg’ array.
        val = FOURIER.get_determinant(lam=arg[0])  # Compute the determinant for the real wavelength in arg[0]
        print_log('data', "FOURIER:optical.Fourier3D: lam={1} gain={2} det={0}".format(val, *arg))
        # As ‘fsolve’ does not browse imaginary space, but just R², we consider the complex determinant as
        # two numbers.
        return val.real, val.imag

    # We ask SciPy to find the root for us
    # (see http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html)
    result = fsolve(fun, [start, gain], xtol=1e-5)
    
    # Now, we know that we have found the mode, but the solver does not. So we need to explicitly tell it.
    # The correct profile of the gain is already set from the last call to ‘fun’ done by the ‘fsolve’ function.
    # So we need to tell the solver the eigenmode wavelenght. It will check if the determinant is sufficiently
    # close to zero and add the mode parameters to the ‘FOURIER.modes’ array.
    FOURIER.set_mode(lam=result[0])
    
    # Now print and save the results similarly as before.
    line = "{L:3.1f} {d:2.1f} {etched}   {N}   {0:.6f} {1:6.3f}".format(*result, **DEF)
    print(line)


# Below we plot the fields in xy and yz planes.

# We retrieve the field.
field = FOURIER.outLightMagnitude(msh)

# We plot the field and the geometry outlines.
fig = figure()  # Open new figure
plot_field(field, plane='xy')  # Plot the field 2D cross-section
plot_geometry(GEO.vcsel, color='w', plane='xy', mirror=True)  # Plot the geometry outline
gca().set_aspect('equal')  # Make sure both axes have the same scale
tight_layout(0.1)  # Reduce the margins of the plot
fig.canvas.set_window_title("Field")  # Set window title


# Now vertical field...

field = FOURIER.outLightMagnitude(vmsh)

fig = figure()
plot_field(field, plane='yz')
# plot_geometry(GEO.vcsel, color='w', plane='yz', mirror=True)
tight_layout(0.1)
fig.canvas.set_window_title("Vertical Field")

show()
]]></script>

</plask>
