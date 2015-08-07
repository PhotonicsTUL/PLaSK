<plask loglevel="detail">

<geometry>
  <cartesian3d name="lattice" axes="xy">
    <lattice az="0.8" ax="0.2" ay="0" bz="0" bx="1" by="0">
      <segments>-2 -2; -2 3; 2 3; 2 -2 ^ -1 -1; -1 1; 1 1; 1 -1 ^  1 4</segments>
      <cylinder material="AlN" radius="0.3" height="3"/>
    </lattice>
  </cartesian3d>
</geometry>

<script><![CDATA[
plot_geometry(GEO.lattice, margin=0.02, plane="xz")
gca().set_aspect('equal')
tight_layout(0.8)
show()
]]></script>

</plask>
