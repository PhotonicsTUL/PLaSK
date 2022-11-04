<plask loglevel="detail">

<defines>
  <define name="sep" value="True"/>
  <define name="sym" value="False"/>
  <define name="pol" value="'Etran'"/>
  <define name="refl" value="False"/>
  <define name="iox" value="False"/>
  <define name="f" value="'E'"/>
  <define name="solver" value="'F2D'"/>
</defines>

<materials>
  <material name="active" base="semiconductor">
    <nr>3.53</nr>
    <absp>-3000.</absp>
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
      <shelf name="oxidation">
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
    <pml factor="1-2j"/>
    <transfer method="{'reflection' if refl else 'admittance'}"/>
  </optical>
  <optical name="BESSEL" solver="BesselCyl" lib="slab">
    <geometry ref="cyl"/>
    <expansion domain="finite" lam0="980" size="12"/>
    <mode emission="top"/>
    <transfer method="{'reflection' if refl else 'admittance'}"/>
    <vpml factor="1"/>
  </optical>
</solvers>

<script><![CDATA[
SOLVER = globals()[solver]

SOLVER.find_mode(lam=980.)

def integrals(field, mesh0):
    vert = mesh0.axis_vert
    tran = mesh0.axis_tran
    mesh1 = field.mesh
    field = field.array
    field = sum(real(field*field.conj()), 2)
    a = getattr(SOLVER, 'integrate{0}{0}'.format(f))(vert[0], vert[-1])
    if solver == 'BESSEL':
        rr, _ = meshgrid(mesh1.axis_tran, mesh1.axis_vert, indexing='ij')
        rr = abs(rr)
        rr *= 1e-6 * pi
        a *= 1e-6
    else:
        rr = ones(field.shape)
    if len(tran) < 2:
        dr = 2. * SOLVER.geometry.bbox.width
    else:
        dr = tran[1] - tran[0]
    field = 0.5 * rr * field
    b = sum(field.ravel()) * (vert[1]-vert[0]) * dr
    print_log('important', "{bot:.4f}-{top:.4f} µm: {F}{F} = {a:8.3f}/{b:8.3f} {ab}".format(bot=vert[0], top=vert[-1], a=a*1e-6, b=b*1e-6, ab=a/b, F=f))


fbox = SOLVER.geometry.bbox

if SOLVER.geometry is GEO.onedi:
    haxis = [0.]
else:
    right = SOLVER.geometry.bbox.right + SOLVER.pml.dist + SOLVER.pml.size
    haxis = mesh.Regular(-right, right, 2001)

nmesh = mesh.Rectangular2D(haxis, mesh.Rectangular2D.SimpleGenerator(split=True)(SOLVER.geometry).axis1)
fmesh = mesh.Rectangular2D(haxis, mesh.Regular(fbox.bottom, fbox.top, 3001))

field_provider = getattr(SOLVER, 'outLight{0}'.format(f))

field = field_provider(fmesh.elements.mesh)

if SOLVER.geometry is GEO.onedi:
    plot_profile(SOLVER.outRefractiveIndex(nmesh), comp='rr', color='C1')
    twinx()
    plot_profile(field, color='C{}'.format({'E':0, 'H':3}[f]))

integrals(field, fmesh)


abox = SOLVER.geometry.get_object_bboxes(GEO.gain_region)[0]
amesh = mesh.Rectangular2D(haxis, mesh.Regular(abox.bottom, abox.top, 501))
integrals(field_provider(amesh.elements.mesh), amesh)


dz = 0.005
dmesh = mesh.Rectangular2D(haxis, mesh.Regular(abox.bottom+dz, abox.top+dz, 501))
integrals(field_provider(dmesh.elements.mesh), dmesh)


obox = SOLVER.geometry.get_object_bboxes(GEO.oxidation)[0]
omesh = mesh.Rectangular2D(haxis, mesh.Regular(obox.bottom, obox.top, 501))
integrals(field_provider(omesh.elements.mesh), omesh)


cbox = geometry.Box2D(fbox.left, 4.44, fbox.right, 4.50)
cmesh = mesh.Rectangular2D(haxis, mesh.Regular(cbox.bottom, cbox.top, 501))
integrals(field_provider(cmesh.elements.mesh), cmesh)


cbox = geometry.Box2D(fbox.left, 4.90, fbox.right, 4.95)
# cbox = geometry.Box2D(fbox.left, 4.7652, fbox.right, 4.7287)
cmesh = mesh.Rectangular2D(haxis, mesh.Regular(cbox.bottom, cbox.top, 2001))
integrals(field_provider(cmesh.elements.mesh), cmesh)


show()

]]></script>

</plask>
