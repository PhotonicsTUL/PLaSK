<plask loglevel="detail">

<materials>
  <material name="zlacze" base="semiconductor">
    <eps>12.9</eps>
    <cond>(1e-6,0.2)</cond>
    <thermk>(11.3464,11.3464)</thermk>
  </material>
  <material name="Au" base="Au">
    <eps>5.6</eps>
  </material>
  <material name="AlOx" base="AlOx">
    <eps>2.6</eps>
  </material>
</materials>

<geometry>
  <cylindrical2d name="simple_structure" axes="r,z">
    <stack>
      <shelf>
        <gap size="1"/>
        <rectangle name="p-contact" material="Au" dr="1" dz="0.1"/>
      </shelf>
      <rectangle material="Al(0)GaAs:C=2e18" dr="5" dz="0.2"/>
      <shelf>
        <rectangle material="Al(0.99)GaAs:C=2e18" dr="0.5" dz="0.1"/>
        <rectangle material="AlOx" dr="4.5" dz="0.1"/>
      </shelf>
      <rectangle material="Al(0)GaAs:C=2e18" dr="5" dz="0.2"/>
      <rectangle name="zlacze" role="junction" material="zlacze" dr="5" dz="0.1"/>
      <shelf flat="no">
        <rectangle material="Al(0)GaAs:Si=2e18" dr="5" dz="0.2"/>
        <gap total="7"/>
        <rectangle material="Au" dr="1" dz="0.1"/>
      </shelf>
      <rectangle material="Al(0)GaAs:Si=2e18" dr="7" dz="1"/>
    </stack>
  </cylindrical2d>
</geometry>

<grids>
  <generator name="default" type="rectangular2d" method="divide">
    <prediv by="4"/>
  </generator>
</grids>

<solvers>
  <electrical name="ELECTRIC" solver="ShockleyCyl" lib="shockley">
    <geometry ref="simple_structure"/>
    <mesh ref="default"/>
    <voltage>
      <condition value="0">
        <place side="bottom"/>
      </condition>
      <condition value="5">
        <place side="top" object="p-contact"/>
      </condition>
    </voltage>
    <junction beta0="10" js0="1"/>
  </electrical>
  <electrical name="RC" solver="CapacitanceCyl" lib="capacitance">
    <geometry ref="simple_structure"/>
    <mesh ref="default"/>
    <ac-voltage>
      <condition value="0">
        <place side="bottom"/>
      </condition>
      <condition value="0.1">
        <place side="top" object="p-contact"/>
      </condition>
    </ac-voltage>
  </electrical>
</solvers>

<connects>
  <connect out="ELECTRIC.outDifferentialConductivity" in="RC.inDifferentialConductivity"/>
</connects>

<script><![CDATA[
ELECTRIC.compute()

def compute(freq):
    RC.frequency = freq
    RC.compute()
    Iact = RC.get_ac_current(active=True)
    Z = RC.get_impedance()
    S11 = RC.get_S11()
    return Iact, Z, S11

freqs = array((0.02, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
               15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0))

Iact, Z, S11 = array([compute(freq) for freq in freqs]).T

figure()
plot(freqs, abs(Iact))
xlim(0., 100.)
xlabel(r"Frequency $\nu$ (GHz)")
ylabel(r"Active Current $I_\mathrm{act}$ (mA)")

figure()
plot(freqs, Z.real, label="Re($Z$)")
plot(freqs, Z.imag, label="In($Z$)")
xlim(0., 100.)
xlabel(r"Frequency $\nu$ (GHz)")
ylabel(r"Impedance $Z$ (Ω)")
legend()

figure()
plot(freqs, S11.real, label="Re($Z$)")
plot(freqs, S11.imag, label="In($Z$)")
xlim(0., 100.)
xlabel(r"Frequency $\nu$ (GHz)")
ylabel(r"$S_{11}$")
legend()

show()
]]></script>

</plask>
