<plask loglevel="detail">

<defines>
  <define name="refl" value="True"/>
  <define name="iox" value="True"/>
  <define name="f" value="'E'"/>
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
    <stack name="full">
      <rectangle material="GaAs" dr="10" dz="0.0700"/>
      <stack name="top-DBR" repeat="24">
        <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
        <rectangle material="GaAs" dr="10" dz="0.0700"/>
      </stack>
      <shelf>
        <rectangle role="{'interface' if iox else ''}" material="AlAs" dr="4" dz="0.0160"/>
        <rectangle material="AlOx" dr="6" dz="0.0160"/>
      </shelf>
      <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0635"/>
      <rectangle material="GaAs" dr="10" dz="0.1376"/>
      <shelf>
        <rectangle name="gain-region" role="gain" material="active" dr="4" dz="0.0050"/>
        <rectangle material="inactive" dr="6" dz="0.0050"/>
      </shelf>
      <rectangle role="{'interface' if not iox else ''}" material="GaAs" dr="10" dz="0.1376"/>
      <stack name="bottom-DBR" repeat="30">
        <rectangle material="Al(0.73)GaAs" dr="10" dz="0.0795"/>
        <rectangle material="GaAs" dr="10" dz="0.0700"/>
      </stack>
    </stack>
  </cylindrical2d>
  <cartesian2d name="onedi" axes="r,z" left="mirror" right="periodic" bottom="GaAs">
    <clip right="0.5">
      <again ref="full"/>
    </clip>
  </cartesian2d>
</geometry>

<solvers>
  <optical name="F2D" solver="Fourier2D" lib="slab">
    <geometry ref="onedi"/>
    <expansion lam0="980" size="0"/>
    <mode emission="top" polarization="Etran"/>
    <transfer method="{'reflection' if refl else 'admittance'}"/>
  </optical>
</solvers>

<script><![CDATA[
F2D.find_mode(lam=980.)

def integrals(ibox):
    print(ibox.bottom, ibox.top, end=': ')
    a = getattr(F2D, 'integrate{0}{0}'.format(f))(ibox.bottom, ibox.top)
    b = sum(field)/len(field) * (ibox.top - ibox.bottom)
    print(a, b, a/b)


fbox = GEO.onedi.bbox

nmesh = mesh.Rectangular2D([0.], mesh.Rectangular2D.SimpleGenerator(split=True)(GEO.onedi).axis1)
fmesh = mesh.Rectangular2D([0.], mesh.Regular(fbox.bottom, fbox.top, 100001))

field_provider = getattr(F2D, 'outLight{0}'.format(f))

field = field_provider(fmesh)
field = Data(0.5 * sum(abs(field.array[0,:,0:2])**2, 1), fmesh)

plot_profile(F2D.outRefractiveIndex(nmesh), comp='rr', color='C0')
twinx()
plot_profile(field, color='C{}'.format({'E':1, 'H':3}[f]))

integrals(fbox)


abox = GEO.onedi.get_object_bboxes(GEO.gain_region)[0]
amesh = mesh.Rectangular2D([0.], mesh.Regular(abox.bottom, abox.top, 20001))
field = field_provider(amesh)
field = Data(0.5 * sum(abs(field.array[0,:,0:2])**2, 1), amesh)
integrals(abox)


cbox = geometry.Box2D(fbox.left, 4.44, fbox.right, 4.50)
cmesh = mesh.Rectangular2D([0.], mesh.Regular(cbox.bottom, cbox.top, 20001))
field = field_provider(cmesh)
field = Data(0.5 * sum(abs(field.array[0,:,0:2])**2, 1), cmesh)
integrals(cbox)


cbox = geometry.Box2D(fbox.left, 4.90, fbox.right, 4.95)
# cbox = geometry.Box2D(fbox.left, 4.7652, fbox.right, 4.7287)
cmesh = mesh.Rectangular2D([0.], mesh.Regular(cbox.bottom, cbox.top, 20001))
field = field_provider(cmesh)
field = Data(0.5 * sum(abs(field.array[0,:,0:2])**2, 1), cmesh)
integrals(cbox)


show()

]]></script>

</plask>
