<plask loglevel="detail">

<geometry>
  <cartesian3d name="lattice" axes="x,y,z">
    <lattice ax="{sqrt(3)/2}" ay="0.5" az="0" bx="0" by="1" bz="0">
      <segments>-2 -2; -2 3; 2 3; 2 -2 ^ -1 -1; -1 1; 1 1; 1 -1 ^  1 4</segments>
      <cylinder material="AlN" radius="0.3" height="3"/>
    </lattice>
  </cartesian3d>
</geometry>

<script><![CDATA[
rc.figure.figsize = 6.0, 4.5
plot_geometry(GEO.lattice, margin=0.02, plane="yx")
gca().set_aspect('equal')
tight_layout(0.8)
show()
]]></script>

</plask>
