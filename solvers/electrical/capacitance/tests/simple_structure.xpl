<plask loglevel="detail">

<materials>
  <material name="kontakt_Au" base="semiconductor">
    <eps>5.6</eps>
    <cond>(43878894,43878894)</cond>
    <thermk>(317.1,317.1)</thermk>
  </material>
  <material name="AlGaAs_C" base="semiconductor">
    <eps>12.9</eps>
    <cond>(4601.0,4601.0)</cond>
    <thermk>(45.0,45.0)</thermk>
  </material>
  <material name="Al99GaAS_C" base="semiconductor">
    <eps>12.9</eps>
    <cond>(1593.0,1593.0)</cond>
    <thermk>(70.1,70.1)</thermk>
  </material>
  <material name="AlGaAs_Si" base="semiconductor">
    <thermk>(45.0,45.0)</thermk>
    <eps>12.9</eps>
    <cond>(68555.0,68555.0)</cond>
  </material>
  <material name="AlOx_new" base="semiconductor">
    <eps>2.6</eps>
    <cond>(1e-7,1e-7)</cond>
    <thermk>(0.7,0.7)</thermk>
  </material>
  <material name="zlacze" base="semiconductor">
    <eps>12.9</eps>
    <cond>(1e-06,0.2)</cond>
    <thermk>(11.4,11.4)</thermk>
  </material>
</materials>

<geometry>
  <cylindrical2d name="simple_structure" axes="x,y">
    <stack>
      <shelf>
        <gap size="1"/>
        <rectangle name="p-contact" material="kontakt_Au" dx="1" dy="0.1"/>
      </shelf>
      <rectangle material="AlGaAs_C" dx="5" dy="0.2"/>
      <shelf>
        <rectangle material="Al99GaAS_C" dx="0.5" dy="0.1"/>
        <rectangle material="AlOx_new" dx="4.5" dy="0.1"/>
      </shelf>
      <rectangle material="AlGaAs_C" dx="5" dy="0.2"/>
      <rectangle name="zlacze" role="active" material="zlacze" dx="5" dy="0.1"/>
      <shelf flat="no">
        <rectangle material="AlGaAs_Si" dx="5" dy="0.2"/>
        <gap size="1"/>
        <rectangle name="n-contact" material="kontakt_Au" dx="1" dy="0.1"/>
      </shelf>
      <rectangle material="AlGaAs_Si" dx="7" dy="1"/>
    </stack>
  </cylindrical2d>
</geometry>

<grids>
  <generator name="default" type="rectangular2d" method="divide">
    <postdiv by="8"/>
  </generator>
</grids>

<solvers>
  <electrical name="ELECTRIC" solver="ShockleyCyl" lib="shockley">
    <geometry ref="simple_structure"/>
    <mesh ref="default" empty-elements="include"/>
    <voltage>
      <condition value="1.8">
        <place side="top" object="p-contact"/>
      </condition>
      <condition value="0">
        <place side="top" object="n-contact"/>
      </condition>
    </voltage>
    <junction beta0="10" js0="1"/>
  </electrical>
  <electrical name="RC" solver="CapacitanceCyl" lib="capacitance">
    <geometry ref="simple_structure"/>
    <mesh ref="default" empty-elements="include"/>
    <ac-voltage>
      <condition value="0.1">
        <place side="top" object="p-contact"/>
      </condition>
      <condition value="0">
        <place side="top" object="n-contact"/>
      </condition>
    </ac-voltage>
  </electrical>
</solvers>

<connects>
  <connect out="ELECTRIC.outDifferentialConductivity" in="RC.inDifferentialConductivity"/>
</connects>

<script><![CDATA[
import unittest

def compute(freq):
    RC.frequency = freq
    RC.compute()
    I = RC.get_ac_current(active=False)
    Iact = RC.get_ac_current(active=True)
    Z = RC.get_impedance()
    S11 = RC.get_S11()
    return I, Iact, Z, S11

class TestCapacitance(unittest.TestCase):

    reference = {
          0.01: (-0.1348-0.0001j, -0.1348+0.0001j, 741.6-  0.6j, 0.8737-0.0001j),
          0.1:  (-0.1349-0.0011j, -0.1348+0.0005j, 741.5-  5.8j, 0.8737-0.0009j),
          1.0:  (-0.1350-0.0106j, -0.1343+0.0052j, 736.3- 57.7j, 0.8735-0.0093j),
          5.0:  (-0.1377-0.0521j, -0.1247+0.0202j, 635.2-240.2j, 0.8700-0.0456j),
         10.0:  (-0.1442-0.1019j, -0.1117+0.0256j, 462.5-327.0j, 0.8613-0.0885j),
         20.0:  (-0.1643-0.1974j, -0.0995+0.0264j, 249.0-299.3j, 0.8329-0.1672j),
         50.0:  (-0.2756-0.4422j, -0.0846+0.0321j, 101.5-162.9j, 0.6938-0.3292j),
        100.0:  (-0.5131-0.6896j, -0.0657+0.0365j,  69.4- 93.3j, 0.4802-0.4062j),
    }

    places = (3, 3, 1, 3)

    def test_capacitance(self):
        ELECTRIC.compute()
        global compute

        for freq, correct in self.reference.items():
            result = compute(freq)
            for res, ref, places in zip(result, correct, self.places):
                self.assertAlmostEqual(res, ref, places=places)


if __name__ == "__main__":
    freqs = array((0.02, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                   15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0))

    ELECTRIC.compute()
    I, Iact, Z, S11 = array([compute(freq) for freq in freqs]).T

    figure()
    plot(freqs, abs(Iact))
    xlim(0., 100.)
    xlabel(r"Frequency $\nu$ (GHz)")
    ylabel(r"Active Current $I_\mathrm{act}$ (mA)")

    figure()
    plot(freqs, Z.real, label="Re($Z$)")
    plot(freqs, Z.imag, label="Im($Z$)")
    xlim(0., 100.)
    xlabel(r"Frequency $\nu$ (GHz)")
    ylabel(r"Impedance $Z$ (Ω)")
    legend()

    figure()
    plot(freqs, S11.real, label="Re($Z$)")
    plot(freqs, S11.imag, label="Im($Z$)")
    xlim(0., 100.)
    xlabel(r"Frequency $\nu$ (GHz)")
    ylabel(r"$S_{11}$")
    legend()

    show()
]]></script>

</plask>
