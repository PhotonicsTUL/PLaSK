<plask loglevel="detail">

<defines>
  <define name="sep" value="True"/>
  <define name="sym" value="False"/>
  <define name="pol" value="'Etran'"/>
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
  <cylindrical2d name="cyl" axes="r,z" outer="extend" bottom="GaAs">
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
    <clip right="1.0">
      <again ref="full"/>
    </clip>
  </cartesian2d>
  <cartesian2d name="cart" left="mirror" right="periodic" bottom="GaAs">
    <again ref="full"/>
  </cartesian2d>
</geometry>

<solvers>
  <optical name="F2D" solver="Fourier2D" lib="slab">
    <geometry ref="cart"/>
    <expansion lam0="980" size="12"/>
    <mode emission="top" polarization="{pol if sep else 'none'}" symmetry="{pol if sym else 'none'}"/>
    <transfer method="{'reflection' if refl else 'admittance'}"/>
  </optical>
</solvers>

<script><![CDATA[
F2D.find_mode(lam=980.)

def integrals(ibox):
    print(ibox.bottom, ibox.top, end=': ')
    a = getattr(F2D, 'integrate{0}{0}'.format(f))(ibox.bottom, ibox.top)
    b = sum(field)/len(field) * ibox.height * 2. * F2D.geometry.bbox.width
    print(a, b, a/b)


fbox = F2D.geometry.bbox

if F2D.geometry is GEO.onedi:
    haxis = [0.]
else:
    haxis = mesh.Regular(-F2D.geometry.bbox.right, F2D.geometry.bbox.right, 201)

nmesh = mesh.Rectangular2D(haxis, mesh.Rectangular2D.SimpleGenerator(split=True)(F2D.geometry).axis1)
fmesh = mesh.Rectangular2D(haxis, mesh.Regular(fbox.bottom, fbox.top, 3001))

field_provider = getattr(F2D, 'outLight{0}'.format(f))

field = field_provider(fmesh)
field = Data(0.5 * sum(abs(field.array)**2, 2), fmesh)

if F2D.geometry is GEO.onedi:
    plot_profile(F2D.outRefractiveIndex(nmesh), comp='rr', color='C0')
    twinx()
    plot_profile(field, color='C{}'.format({'E':1, 'H':3}[f]))

integrals(fbox)


abox = F2D.geometry.get_object_bboxes(GEO.gain_region)[0]
amesh = mesh.Rectangular2D(haxis, mesh.Regular(abox.bottom, abox.top, 20001))
field = field_provider(amesh)
field = Data(0.5 * sum(abs(field.array)**2, 2), amesh)
integrals(abox)


cbox = geometry.Box2D(fbox.left, 4.44, fbox.right, 4.50)
cmesh = mesh.Rectangular2D(haxis, mesh.Regular(cbox.bottom, cbox.top, 20001))
field = field_provider(cmesh)
field = Data(0.5 * sum(abs(field.array)**2, 2), cmesh)
integrals(cbox)


cbox = geometry.Box2D(fbox.left, 4.90, fbox.right, 4.95)
# cbox = geometry.Box2D(fbox.left, 4.7652, fbox.right, 4.7287)
cmesh = mesh.Rectangular2D(haxis, mesh.Regular(cbox.bottom, cbox.top, 20001))
field = field_provider(cmesh)
field = Data(0.5 * sum(abs(field.array)**2, 2), cmesh)
integrals(cbox)


show()

]]></script>

</plask>
