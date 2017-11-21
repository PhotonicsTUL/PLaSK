<plask loglevel="detail">

<geometry>
  <cylindrical2d name="cyl">
    <stack>
      <rectangle material="AlAs" dtran="1" dvert="0.5"/>
      <rectangle material="In" dtran="1" dvert="1"/>
      <rectangle material="GaAs" dtran="1" dvert="1.5"/>
    </stack>
  </cylindrical2d>
</geometry>

<solvers>
  <optical solver="SimpleOpticalCyl" name="prosty">
    <geometry ref="cyl"/>
  </optical>
</solvers>

<script><![CDATA[
prosty.say_hello()
prosty.simpleVerticalSolver()
#print(prosty.geometry)

]]></script>

</plask>
