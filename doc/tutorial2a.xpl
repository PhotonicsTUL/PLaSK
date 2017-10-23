<plask loglevel="detail">

<defines>
  <define name="m" value="3"/>
</defines>

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
        <rectangle material="AlAs" dr="8" dz="0.0160"/>
        <rectangle material="AlOx" dr="2" dz="0.0160"/>
      </shelf>
      <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0635"/>
      <rectangle material="GaAs" dr="10" dz="0.1376"/>
      <shelf>
        <rectangle name="gain-region" role="gain" material="active" dr="8" dz="0.0050"/>
        <rectangle material="inactive" dr="2" dz="0.0050"/>
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
  <optical name="efm" solver="EffectiveFrequencyCyl" lib="effective">
    <geometry ref="main"/>
    <mode lam0="980."/>
  </optical>
</solvers>

<script><![CDATA[
from matplotlib.animation import FuncAnimation

style.use('dark_background')
rc.grid.alpha = 0.1

profile = plask.StepProfile(GEO.main, default=0.)
profile[GEO.gain_region] = 10000.

efm.inGain = profile.outGain

# lams = linspace(970., 980, 1001)
# plot(lams, abs(efm.get_determinant(lams, m=m)))
# yscale('log')
# show()

mode_number = efm.find_mode(978.43, m=m)
mode_wavelength = efm.outWavelength(mode_number)
          
r = linspace(0., 20., 151)

p = linspace(0., 2*pi, 361)
R, P = meshgrid(r, p)
# X = R * cos(P)
# Y = R * sin(P)
# contourf(X, Y, F)
# gca().set_aspect('equal')

Fr = efm.outLightMagnitude(mode_number, mesh.Rectangular2D(r, [4.83]))
F = array(Fr)[None,:] * cos(m * p[:,None])**2
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
contourf(P, R, F)
# grid(False)
plot(p, [8.] * len(p), color='w')
tight_layout(0.1)
window_title("Magnitude")

Fr = efm.outElectricField(mode_number, mesh.Rectangular2D(r, [4.83]))
F = -array(Fr)[None,:,1] * cos(m * p[:,None])
F /= max(abs(F[0,:]))

rc('grid', color='k')

fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
image = pcolormesh(P, R, F.real, vmin=-1., vmax=1., cmap='RdBu')
plot(p, [8.] * len(p), color='k')
window_title("Ex")

dt = 0.02
def animate(n):
    Fn = (F * exp(2j*pi*n*dt)).real
    image.set_array(Fn[:-1, :-1].ravel())
    return ax

fr = 25
ani = FuncAnimation(fig, animate)

show()
]]></script>

</plask>
