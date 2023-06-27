<plask loglevel="debug">

<defines>
  <define name="Eg" value="'(1.66316 - 0.5405e-3 * T**2 / (T + 204))'"/>
  <!--define name="Eg" value="1.56664"/-->
  <define name="dso" value="0.36"/>
  <define name="nr" value="3.5"/>
  <define name="broad" value="30.0"/>
  <define name="phen" value="1.278"/>
  <define name="conc" value="1.41977719686e+19"/>
  <define name="temp" value="322.975"/>
  <define name="expected_gain" value="7634.63"/>
</defines>

<materials>
  <material name="GaAsP014" base="GaAsP(0.14)">
    <nr>{nr}</nr>
    <Me>0.07358</Me>
    <Mlh>0.1054</Mlh>
    <Mhh>0.3314</Mhh>
    <CB>0.</CB>
    <VB>-{Eg}</VB>
    <Dso>{dso}</Dso>
  </material>
  <material name="In023GaAs" base="In(0.23)GaAs">
    <nr>{nr}</nr>
    <Me>0.0556932</Me>
    <Mlh>0.07551</Mlh>
    <Mhh>0.3139</Mhh>
    <Dso>{dso}</Dso>
  </material>
  <material name="well1" base="In023GaAs">
    <CB>-0.184323</CB>
    <VB>0.0993083 - {Eg} if hole == 'L' else 0.236561 - {Eg}</VB>
  </material>
  <material name="well2" base="In023GaAs">
    <CB>-0.184423</CB>
    <VB>0.0992083 - {Eg} if hole == 'L' else 0.236461 - {Eg}</VB>
  </material>
  <material name="well3" base="In023GaAs">
    <CB>-0.184523</CB>
    <VB>0.0991083 - {Eg} if hole == 'L' else 0.236361 - {Eg}</VB>
  </material>
  <material name="well4" base="In023GaAs">
    <CB>-0.184623</CB>
    <VB>0.0990083 - {Eg} if hole == 'L' else 0.236261 - {Eg}</VB>
  </material>
  <material name="well5" base="In023GaAs">
    <CB>-0.184723</CB>
    <VB>0.098983 - {Eg} if hole == 'L' else 0.236161 - {Eg}</VB>
  </material>
  <material name="well1_mod" base="well1">
    <CB>-0.184423</CB>
    <VB>0.0994083 - {Eg} if hole == 'L' else 0.236661 - {Eg}</VB>
  </material>
  <material name="well2_mod" base="well2">
    <CB>-0.184523</CB>
    <VB>0.0993083 - {Eg} if hole == 'L' else 0.236561 - {Eg}</VB>
  </material>
  <material name="well3_mod" base="well3">
    <CB>-0.184623</CB>
    <VB>0.0992083 - {Eg} if hole == 'L' else 0.236461 - {Eg}</VB>
  </material>
  <material name="well4_mod" base="well4">
    <CB>-0.184723</CB>
    <VB>0.0991083 - {Eg} if hole == 'L' else 0.236361 - {Eg}</VB>
  </material>
  <material name="well5_mod" base="well5">
    <CB>-0.184823</CB>
    <VB>0.099283 - {Eg} if hole == 'L' else 0.236261 - {Eg}</VB>
  </material>
</materials>

<geometry>
  <cartesian2d name="main" axes="x,y">
    <stack>
      <rectangle material="GaAsP014" dx="1" dy="0.100"/>
      <stack role="active">
        <rectangle role="QW" material="well1" dx="1" dy="0.004"/>
        <rectangle material="GaAsP014" dx="1" dy="0.005"/>
        <rectangle role="QW" material="well2" dx="1" dy="0.004"/>
        <rectangle material="GaAsP014" dx="1" dy="0.005"/>
        <rectangle role="QW" material="well3" dx="1" dy="0.004"/>
        <rectangle material="GaAsP014" dx="1" dy="0.005"/>
        <rectangle role="QW" material="well4" dx="1" dy="0.004"/>
        <rectangle material="GaAsP014" dx="1" dy="0.005"/>
        <rectangle role="QW" material="well5" dx="1" dy="0.004"/>
      </stack>
      <zero/>
      <rectangle material="GaAsP014" dx="1" dy="0.100"/>
    </stack>
  </cartesian2d>
  <cartesian2d name="mod" axes="x,y">
    <stack>
      <rectangle material="GaAsP014" dx="1" dy="0.100"/>
      <stack role="active">
        <rectangle role="QW" material="well1_mod" dx="1" dy="0.004"/>
        <rectangle material="GaAsP014" dx="1" dy="0.005"/>
        <rectangle role="QW" material="well2_mod" dx="1" dy="0.004"/>
        <rectangle material="GaAsP014" dx="1" dy="0.005"/>
        <rectangle role="QW" material="well3_mod" dx="1" dy="0.004"/>
        <rectangle material="GaAsP014" dx="1" dy="0.005"/>
        <rectangle role="QW" material="well4_mod" dx="1" dy="0.004"/>
        <rectangle material="GaAsP014" dx="1" dy="0.005"/>
        <rectangle role="QW" material="well5_mod" dx="1" dy="0.004"/>
      </stack>
      <zero/>
      <rectangle material="GaAsP014" dx="1" dy="0.100"/>
    </stack>
  </cartesian2d>
</geometry>

<solvers>
  <gain name="GAIN" solver="WasiakNew2D" lib="wasiak">
    <geometry ref="main" mod="mod"/>
    <config adjust-layers="no" roughness="{broad}" Tref="300"/>
  </gain>
</solvers>

<script><![CDATA[
import unittest


# msh = mesh.Rectangular2D.SimpleGenerator(split=True)(GEO.main)
# msh.axis1 = msh.axis1[1:-1]
# msh.axis0 = mesh.Ordered([0.5])
# plot_profile(GEO.main.get_material_field(msh).VB(hole='H'))
# plot_profile(GEO.main.get_material_field(msh).CB())
# plot_profile(GEO.main.get_material_field(msh).VB(hole='L'))
# show()

GAIN.inTemperature = temp


class NewGainValues(unittest.TestCase):

    def setUp(self):
        GAIN.inCarriersConcentration = conc


    def testGain(self):
        m = mesh.Rectangular2D([0.5], [0.020])
        self.assertAlmostEqual(GAIN.outGain(m, phys.eV2nm(phen))[0][0], expected_gain, 1)

    def testGainSpectrum(self):
        spectrum = GAIN.spectrum(0.5, 0.020)
        self.assertAlmostEqual(spectrum(phys.eV2nm(phen)), expected_gain, 1)

# layer   n_e (1/cm)              n_h (1/cm)
# 0       5.6253937984e+18        2.01836458502e+16
# 1       1.41028843714e+19       1.42125383124e+19
# 2       9.58195935805e+18       7.37905441174e+17
# 3       1.41897230183e+19       1.418731695e+19
# 4       9.62426424946e+18       7.3696668019e+17
# 5       1.41977719686e+19       1.41619695555e+19
# 6       9.62422036033e+18       7.35946388262e+17
# 7       1.41984484844e+19       1.41366289991e+19
# 8       9.58210178922e+18       7.34853077098e+17
# 9       1.4120767246e+19        1.41112081237e+19
# 10      5.6253937984e+18        2.01836458502e+16



if __name__ == '__main__':
    test = unittest.main(exit=False)
    sys.exit(not test.result.wasSuccessful())
]]></script>

</plask>
