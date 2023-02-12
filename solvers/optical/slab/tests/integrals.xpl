<plask loglevel="data">

<defines>
  <define name="lam" value="850."/>
  <define name="L" value="3.0"/>
  <define name="Nt" value="8"/>
  <define name="Nl" value="8"/>
  <define name="side" value="'bottom'"/>
  <define name="nx" value="20"/>
  <define name="ny" value="20"/>
  <define name="nz" value="80"/>
  <define name="ft" value="0.5"/>
  <define name="fl" value="0.9"/>
  <define name="h" value="0.3"/>
</defines>

<materials>
  <material name="slab" base="semiconductor">
    <Nr>3.5</Nr>
  </material>
  <material name="absorber" base="semiconductor">
    <Nr>3.5 - 0.02j</Nr>
  </material>
  <material name="clad" base="semiconductor">
    <Nr>1.5</Nr>
  </material>
</materials>

<geometry>
  <cartesian3d name="struct3d" axes="x,y,z" back="mirror" front="periodic" left="mirror" right="periodic" bottom="clad">
    <clip back="0" left="0">
      <stack xcenter="0" ycenter="0">
        <stack name="grating3d">
          <cuboid material="slab" dx="{fl*L}" dy="{ft*L}" dz="0.4"/>
          <cuboid material="slab" dx="{L}" dy="{L}" dz="0.4"/>
        </stack>
        <cuboid name="absorber3d" material="absorber" dx="{L}" dy="{L}" dz="0.2"/>
        <cuboid material="slab" dx="{L}" dy="{L}" dz="0.4"/>
      </stack>
    </clip>
  </cartesian3d>
  <cartesian2d name="struct2d" axes="x,y,z" left="mirror" right="periodic" bottom="clad">
    <clip left="0">
      <stack ycenter="0">
        <stack name="grating2d">
          <rectangle material="slab" dy="{ft*L}" dz="0.4"/>
          <rectangle material="slab" dy="{L}" dz="0.4"/>
        </stack>
        <rectangle name="absorber2d" material="absorber" dy="{L}" dz="0.2"/>
        <rectangle material="slab" dy="{L}" dz="0.4"/>
      </stack>
    </clip>
  </cartesian2d>
</geometry>

<solvers>
  <optical name="FOURIER3D" solver="Fourier3D" lib="slab">
    <geometry ref="struct3d"/>
    <expansion size-long="{Nl-1}" size-tran="{Nt-1}" rule="direct"/>
    <mode lam="{lam}"/>
  </optical>
  <optical name="FOURIER2D" solver="Fourier2D" lib="slab">
    <geometry ref="struct2d"/>
    <expansion size="{Nt-1}"/>
    <mode lam="{lam}"/>
  </optical>
</solvers>

<script><![CDATA[
import unittest


class Integrals2DTest(unittest.TestCase):

    def __init__(self, name, sym=True, sep=True, pol='Et'):
        super().__init__(name)
        self.symmetry = pol if sym else None
        self.polarization = pol if sym else None
        self.pol = pol
        self.abox = GEO.struct2d.get_object_bboxes(GEO.absorber2d)[0]
        self.ibox = GEO.struct2d.get_object_bboxes(GEO.grating2d)[0]
        self.amesh = mesh.Rectangular2D(
            mesh.Regular(-self.abox.right, self.abox.right, ny+1),
            mesh.Regular(self.abox.bottom, self.abox.top, nz+1)
        ).elements.mesh
        self.imesh = mesh.Rectangular2D(
            mesh.Regular(-self.ibox.right, self.ibox.right, ny+1),
            mesh.Regular(self.ibox.bottom, self.ibox.top, nz+1)
        ).elements.mesh

    def setUp(self):
        FOURIER2D.symmetry = self.symmetry
        FOURIER2D.polarization = self.polarization
        FOURIER3D.klong, FOURIER3D.ktran = 0.1, 0.2
        self.scatter = FOURIER2D.scattering(side, self.pol)

    def test_integrals_E(self):
        EEn = sum(self.scatter.outLightMagnitude(self.imesh)) * (self.ibox.top - self.ibox.bottom) / (ny * nz)
        # En = self.scatter.outLightE(self.imesh).array
        # EEn = 0.5/phys.eta0 * sum(real(En * conj(En))) * (self.ibox.top - self.ibox.bottom) / (ny * nz)
        EEa = 0.5/phys.eta0 * self.scatter.integrateEE(self.ibox.bottom, self.ibox.top) / self.ibox.right
        self.assertAlmostEqual(EEa / EEn, 1., 3)

    def test_integrals_H(self):
        Hn = self.scatter.outLightH(self.imesh).array
        HHn = 0.5*phys.eta0 * sum(real(Hn * conj(Hn))) * (self.ibox.top - self.ibox.bottom) / (ny * nz)
        HHa = 0.5*phys.eta0 * self.scatter.integrateHH(self.ibox.bottom, self.ibox.top) / self.ibox.right
        self.assertAlmostEqual(HHa / HHn, 1., 3)

    def test_absorption_numeric(self):
        V = 2. * self.abox.right * (self.abox.top - self.abox.bottom)
        EEn = sum(self.scatter.outLightMagnitude(self.amesh)) / (ny * nz) * V
        eps = material.get('absorber').Nr(lam)**2
        A = 1. - self.scatter.R - self.scatter.T
        An = 2e3 * pi / lam * abs(eps.imag) * EEn
        self.assertAlmostEqual(An / A, 1., 3)

    def test_absorption_analytic(self):
        EEa = self.scatter.integrateEE(self.abox.bottom, self.abox.top) / phys.eta0
        eps = material.get('absorber').Nr(lam)**2
        A = 1. - self.scatter.R - self.scatter.T
        Aa = 2e3 * pi / lam * abs(eps.imag) * EEa
        self.assertAlmostEqual(Aa / A, 1., 4)


class Integrals3DTest(unittest.TestCase):

    def __init__(self, name, syml=True, symt=True, pol='Et'):
        super().__init__(name)
        self.symmetry = pol if syml else None, pol if symt else None
        self.pol = pol
        self.abox = GEO.struct3d.get_object_bboxes(GEO.absorber3d)[0]
        self.ibox = GEO.struct3d.get_object_bboxes(GEO.grating3d)[0]
        self.amesh = mesh.Rectangular3D(
            mesh.Regular(-self.abox.front, self.abox.front, nx+1),
            mesh.Regular(-self.abox.right, self.abox.right, ny+1),
            mesh.Regular(self.abox.bottom, self.abox.top, nz+1)
        ).elements.mesh
        self.imesh = mesh.Rectangular3D(
            mesh.Regular(-self.ibox.front, self.ibox.front, nx+1),
            mesh.Regular(-self.ibox.right, self.ibox.right, ny+1),
            mesh.Regular(self.ibox.bottom, self.ibox.top, nz+1)
        ).elements.mesh

    def setUp(self):
        FOURIER3D.symmetry = self.symmetry
        FOURIER3D.klong, FOURIER3D.ktran = 0.1, 0.2
        self.scatter = FOURIER3D.scattering(side, self.pol)

    def test_integrals_E(self):
        EEn = sum(self.scatter.outLightMagnitude(self.imesh)) * (self.ibox.top - self.ibox.bottom) / (nx * ny * nz)
        # En = self.scatter.outLightE(self.imesh).array
        # EEn = 0.5/phys.eta0 * sum(real(En * conj(En))) * (self.ibox.top - self.ibox.bottom) / (nx * ny * nz)
        EEa = 0.25/phys.eta0 * self.scatter.integrateEE(self.ibox.bottom, self.ibox.top) / (self.ibox.front * self.ibox.right)
        self.assertAlmostEqual(EEa / EEn, 1., 3)

    def test_integrals_H(self):
        Hn = self.scatter.outLightH(self.imesh).array
        HHn = 0.5*phys.eta0 * sum(real(Hn * conj(Hn))) * (self.ibox.top - self.ibox.bottom) / (nx * ny * nz)
        HHa = 0.25*phys.eta0 * self.scatter.integrateHH(self.ibox.bottom, self.ibox.top) / (self.ibox.front * self.ibox.right)
        self.assertAlmostEqual(HHa / HHn, 1., 3)

    def test_absorption_numeric(self):
        V = 4. * self.abox.front * self.abox.right * (self.abox.top - self.abox.bottom)
        EEn = sum(self.scatter.outLightMagnitude(self.amesh)) / (nx * ny * nz) * V
        eps = material.get('absorber').Nr(lam)**2
        A = 1. - self.scatter.R - self.scatter.T
        An = 2e3 * pi / lam * abs(eps.imag) * EEn
        self.assertAlmostEqual(A, An, 3)

    def test_absorption_analytic(self):
        EEa = self.scatter.integrateEE(self.abox.bottom, self.abox.top) / phys.eta0
        eps = material.get('absorber').Nr(lam)**2
        A = 1. - self.scatter.R - self.scatter.T
        Aa = 2e3 * pi / lam * abs(eps.imag) * EEa
        self.assertAlmostEqual(A, Aa, 4)


def load_tests(loader, standard_tests, pattern):
    tests = unittest.TestSuite()
    for cls in (Integrals2DTest, Integrals3DTest):
        for pol in ('Et', 'El'):
            for s1 in (False, True):
                for s2 in (False, True):
                    for test in ('integrals_E', 'integrals_H', 'absorption_analytic'):
                        tests.addTest(cls('test_'+test, s1, s2, pol))
    return tests


if __name__ == '__main__':
    unittest.main()
]]></script>

</plask>
