<plask loglevel="debug">

<defines>
  <define name="expected_gain_te" value="566.89"/>
  <define name="expected_gain_tm" value="85.943"/>
  <define name="expected_luminescence_te" value="31.212"/>
  <define name="expected_luminescence_tm" value="4.732"/>
</defines>

<materials>
  <material name="AlGaAs" base="AlGaAs" alloy="yes">
    <nr>3.5</nr>
  </material>
</materials>

<geometry>
  <cartesian2d name="main" axes="x,y">
    <stack name="stack">
      <stack name="active" role="active">
        <rectangle role="cladding" material="Al(0.40)GaAs" dx="1" dy="0.020"/>
        <rectangle material="Al(0.20)GaAs" dx="1" dy="0.010"/>
        <rectangle name="QW" role="QW" material="Al(0.01)GaAs" dx="1" dy="0.008"/>
        <rectangle material="Al(0.20)GaAs" dx="1" dy="0.010"/>
        <rectangle role="cladding" material="Al(0.40)GaAs" dx="1" dy="0.020"/>
      </stack>
      <zero/>
      <rectangle name="substrate" role="substrate" material="GaAs" dx="1" dy="1"/>
    </stack>
  </cartesian2d>
  <cartesian2d name="mod" axes="x,y">
    <copy from="stack">
      <replace object="QW">
        <rectangle role="QW" material="Al(0.02)GaAs" dx="1" dy="0.008"/>
      </replace>
    </copy>
  </cartesian2d>
</geometry>

<solvers>
  <gain name="GAIN" solver="WasiakNew2D" lib="wasiak">
    <geometry ref="main" mod="mod"/>
    <config matrix-elem="10" roughness="1" Tref="300"/>
  </gain>
</solvers>

<script><![CDATA[
import unittest


# Print materials parameters to ensure they did not change (check this first if this test fails)
mids = mesh.Rectangular2D.SimpleGenerator()(GEO.main).elements.mesh
mids = mesh.Rectangular2D(mids.axis0, mids.axis1[1:])
mats = GEO.main.get_material_field(mids)
print("                  CB(eV)   VB(eV)   Dso(eV)   Me[m0] Mhh[m0] Mlh[m0]")
for mat in mats:
    print("{mat:14s} {CB:9.6f}   {VB:6.3f}   {Dso:7.4f}   "
          "{Me:6.4f}   {Mhh:5.3f}   {Mlh:5.3f}".format(
            mat=str(mat), CB=mat.CB(), VB=mat.VB(), Dso=mat.Dso(),
            Me=mat.Me()[0], Mhh=mat.Mhh()[0], Mlh=mat.Mlh()[0]))
# layer 1 - CB: 0.947424 eV, VB: -1.012 eV, Dso: 0.3166 eV, Me: 0.0898 m0, Mhh: 0.402 m0, Mlh: 0.126 m0
# layer 2 - CB: 0.810993 eV, VB: -0.906 eV, Dso: 0.3288 eV, Me: 0.0784 m0, Mhh: 0.366 m0, Mlh: 0.108 m0
# layer 3 - CB: 0.634115 eV, VB: -0.805 eV, Dso: 0.3404 eV, Me: 0.0676 m0, Mhh: 0.332 m0, Mlh: 0.091 m0
# layer 4 - CB: 0.810993 eV, VB: -0.906 eV, Dso: 0.3288 eV, Me: 0.0784 m0, Mhh: 0.366 m0, Mlh: 0.108 m0
# layer 5 - CB: 0.947424 eV, VB: -1.012 eV, Dso: 0.3166 eV, Me: 0.0898 m0, Mhh: 0.402 m0, Mlh: 0.126 m0
# layer 6 - CB: 0.622482 eV, VB: -0.8 eV
# layer_ 1 - CB: 0.947424 eV, VB: -1.012 eV, Dso: 0.3166 eV, Me: 0.0898 m0, Mhh: 0.402 m0, Mlh: 0.126 m0
# layer_ 2 - CB: 0.810993 eV, VB: -0.906 eV, Dso: 0.3288 eV, Me: 0.0784 m0, Mhh: 0.366 m0, Mlh: 0.108 m0
# layer_ 3 - CB: 0.645469 eV, VB: -0.811 eV, Dso: 0.3404 eV, Me: 0.0676 m0, Mhh: 0.332 m0, Mlh: 0.091 m0
# layer_ 4 - CB: 0.810993 eV, VB: -0.906 eV, Dso: 0.3288 eV, Me: 0.0784 m0, Mhh: 0.366 m0, Mlh: 0.108 m0
# layer_ 5 - CB: 0.947424 eV, VB: -1.012 eV, Dso: 0.3166 eV, Me: 0.0898 m0, Mhh: 0.402 m0, Mlh: 0.126 m0
# layer_ 6 - CB: 0.622482 eV, VB: -0.8 eV


# msh = mesh.Rectangular2D.SimpleGenerator(split=True)(GEO.main)
# msh.axis1 = msh.axis1[3:-1]
# msh.axis0 = mesh.Ordered([0.5])
# plot_profile(GEO.main.get_material_field(msh).VB(hole='H'))
# plot_profile(GEO.main.get_material_field(msh).CB())
# plot_profile(GEO.main.get_material_field(msh).VB(hole='L'))
# show()


class NewGainValues(unittest.TestCase):

    def setUp(self):
        GAIN.inCarriersConcentration = 4e+18

    def testFermiLevels(self):
        #print(GAIN.get_levels())
        Fc, Fv = GAIN.get_fermi_levels(4e+18)
        self.assertAlmostEqual(Fc,  0.7811923252)
        self.assertAlmostEqual(Fv, -0.8103383518)

    def testGain(self):
        m = mesh.Rectangular2D([0.5], [0.034])
        gain = GAIN.outGain(m, 840.)[0]
        self.assertAlmostEqual(gain[0], expected_gain_te, 2)
        self.assertAlmostEqual(gain[1], expected_gain_tm, 2)

    def testBarrier(self):
        m = mesh.Rectangular2D([0.5], [0.025])
        self.assertEqual(GAIN.outGain(m, 840.)[0][0], 0.)

    def testGainSpectrum(self):
        spectrum = GAIN.spectrum(0.5, 0.034)
        gain = spectrum(840)
        self.assertAlmostEqual(gain[0], expected_gain_te, 2)
        self.assertAlmostEqual(gain[1], expected_gain_tm, 2)

    def testLuminescence(self):
        m = mesh.Rectangular2D([0.5], [0.034])
        lumin = GAIN.outLuminescence(m, 840.)[0] * 1e-27
        self.assertAlmostEqual(lumin[0], expected_luminescence_te, 2)
        self.assertAlmostEqual(lumin[1], expected_luminescence_tm, 2)

    def testLuminescenceSpectrum(self):
        spectrum = GAIN.luminescence_spectrum(0.5, 0.034)
        lumin = spectrum(840) * 1e-27
        self.assertAlmostEqual(lumin[0], expected_luminescence_te, 2)
        self.assertAlmostEqual(lumin[1], expected_luminescence_tm, 2)


if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
]]></script>

</plask>
