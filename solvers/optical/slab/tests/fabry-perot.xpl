<plask loglevel="debug">

<defines>
  <define name="X" value="10"/>
  <define name="Y" value="10"/>
  <define name="start" value="980.15-0.02j"/>
  <define name="N" value="0"/>
</defines>

<materials>
  <material name="GaAs" base="semiconductor">
    <Nr>3.53</Nr>
  </material>
  <material name="AlGaAs" base="semiconductor">
    <Nr>3.08</Nr>
  </material>
  <material name="QW" base="semiconductor">
    <Nr>3.56 </Nr>
  </material>
</materials>

<geometry>
  <cartesian3d name="vcsel3d" axes="x,y,z" back="mirror" front="periodic" left="mirror" right="periodic" bottom="GaAs">
    <clip left="0" back="0">
      <stack>
        <stack repeat="24">
          <cuboid material="GaAs" dx="{X}" dy="{Y}" dz="0.06940"/>
          <cuboid material="AlGaAs" dx="{X}" dy="{Y}" dz="0.07955"/>
        </stack>
        <stack>
          <cuboid material="GaAs" dx="{X}" dy="{Y}" dz="0.12171"/>
          <stack name="active3">
            <cuboid name="qw3d" material="QW" dx="{X}" dy="{Y}" dz="0.00800"/>
            <cuboid name="interface3d" material="GaAs" dx="{X}" dy="{Y}" dz="0.00500"/>
            <again ref="qw3d"/>
            <cuboid material="GaAs" dx="{X}" dy="{Y}" dz="0.00500"/>
            <again ref="qw3d"/>
          </stack>
          <cuboid material="GaAs" dx="{X}" dy="{Y}" dz="0.12171"/>
        </stack>
        <stack repeat="29">
          <cuboid material="AlGaAs" dx="{X}" dy="{Y}" dz="0.07955"/>
          <cuboid material="GaAs" dx="{X}" dy="{Y}" dz="0.06940"/>
        </stack>
      </stack>
    </clip>
  </cartesian3d>
  <cartesian2d name="vcsel2d" axes="x,y,z" left="mirror" right="periodic" bottom="GaAs" length="{X}">
    <clip left="0">
      <stack>
        <stack repeat="24">
          <rectangle material="GaAs" dy="{Y}" dz="0.06940"/>
          <rectangle material="AlGaAs" dy="{Y}" dz="0.07955"/>
        </stack>
        <stack>
          <rectangle material="GaAs" dy="{Y}" dz="0.12171"/>
          <stack>
            <rectangle name="qw2d" material="QW" dy="{Y}" dz="0.00800"/>
            <rectangle name="interface2d" material="GaAs" dy="{Y}" dz="0.00500"/>
            <again ref="qw2d"/>
            <rectangle material="GaAs" dy="{Y}" dz="0.00500"/>
            <again ref="qw2d"/>
          </stack>
          <rectangle material="GaAs" dy="{Y}" dz="0.12171"/>
        </stack>
        <stack repeat="29">
          <rectangle material="AlGaAs" dy="{Y}" dz="0.07955"/>
          <rectangle material="GaAs" dy="{Y}" dz="0.06940"/>
        </stack>
      </stack>
    </clip>
  </cartesian2d>
</geometry>

<solvers>
  <optical name="FOURIER3D" solver="Fourier3D" lib="slab">
    <geometry ref="vcsel3d"/>
    <expansion lam0="980" size="{N}"/>
    <mode emission="top" symmetry-long="Etran" symmetry-tran="Etran"/>
    <interface object="interface3d"/>
    <pmls dist="1.0" factor="1-2j" shape="2" size="2.0"/>
  </optical>
  <optical name="FOURIER2D" solver="Fourier2D" lib="slab">
    <geometry ref="vcsel2d"/>
    <expansion lam0="980" size="{N}"/>
    <mode emission="top" symmetry="Etran"/>
    <interface object="interface2d"/>
  </optical>
</solvers>

<script><![CDATA[
# m = FOURIER3D.find_mode(lam=start)
# lam = FOURIER3D.modes[m].lam
# FOURIER3D.modes[m].power = 1000.
# print(lam)
# 
# vmsh = mesh.Rectangular3D([0.], [0.], mesh.Regular(-2., 10., 6001))
# top = mesh.Rectangular3D([0.], [0.], [GEO.vcsel3d.bbox.upper.z])
# 
# field = FOURIER3D.outLightMagnitude(vmsh)
# fig = figure()
# plot_profile(field)
# yscale('log')
# tight_layout(0.1)
# fig.canvas.set_window_title("Vertical Field")
# 
# E = FOURIER3D.outElectricField(top).array[0,0,0,:2]
# H = FOURIER3D.outMagneticField(top).array[0,0,0,1::-1]
# P = FOURIER3D.outLightMagnitude(top)[0]
# 
# S = 0.5 * (E[0] * conj(H[0]) - E[1] * conj(H[1])).real
# 
# Z0 = 376.73031346177
# 
# R = E/H / Z0
# 
# print(R[logical_not(isnan(R))])
# print(S * 1e-12*X*Y)
# print(P * 1e-12*X*Y)
# print(0.5 / Z0 * sum(E*conj(E)).real * 1e-12*X*Y)


m = FOURIER2D.find_mode(lam=start)
lam = FOURIER2D.modes[m].lam
FOURIER2D.modes[m].power = 1000.
print(lam)

vmsh = mesh.Rectangular2D([0.], mesh.Regular(-2., 10., 6001))
top = mesh.Rectangular2D([0.], [GEO.vcsel2d.bbox.upper.z])

field = FOURIER2D.outLightMagnitude(vmsh)
fig = figure()
plot_profile(field)
yscale('log')
tight_layout(0.1)
fig.canvas.set_window_title("Vertical Field")

E = FOURIER2D.outElectricField(top).array[0,0,:2]
H = FOURIER2D.outMagneticField(top).array[0,0,1::-1]
P = FOURIER2D.outLightMagnitude(top)[0]

S = 0.5 * (E * conj(H)).real

Z0 = 376.73031346177

R = E/H / Z0

print(R[logical_not(isnan(R))])
print(S * 1e-12*X*Y)
print(P * 1e-12*X*Y)
print(0.5 / Z0 * sum(E*conj(E)).real * 1e-12*X*Y)


show()
]]></script>

</plask>
