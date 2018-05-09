<plask loglevel="detail">

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
      <stack name="top-DBR" repeat="6">
        <rectangle material="Al(0.73)GaAs" dr="10" dz="0.05"/>
        <rectangle material="SiO2" dr="10" dz="0.112"/>
      </stack>
      <rectangle material="Al(0.73)GaAs" dr="10" dz="0.05"/>
      <stack name="bottom-DBR" repeat="6">
        <rectangle material="SiO2" dr="10" dz="0.112"/>
        <rectangle material="GaAs" dr="10" dz="0.05"/>
      </stack>
    </stack>
  </cylindrical2d>
</geometry>

<solvers>
  <optical solver="SemiVectorialCyl" name="prosty">
    <geometry ref="main"/>
  </optical>
</solvers>

<script><![CDATA[

print("Hello")

]]></script>

</plask>