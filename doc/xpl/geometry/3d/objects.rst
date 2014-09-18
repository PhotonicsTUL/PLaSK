Physical objects
^^^^^^^^^^^^^^^^

Physical objects are the leafs of the geometry tree. They represent actual objects that have some shape and defined material.

.. xml:tag:: <block3D/> (or <block/>)

   Corresponding Python class: :py:class:`plask.geometry.Block3D`.

   Rectangular block. Its origin is located in the lower back left corner.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the block material (for solid blocks).
   :attr material-bottom: Definition of the material of the bottom of the block (for blocks which material linearly change from bottom to top). You should also set ``material-top``, and these materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the block (see also ``material-bottom``).
   :attr required d{X}: where **{X}** is the longitudinal axis name: Longitudinal dimension of the cuboid. (float [µm])
   :attr required d{Y}: where **{Y}** is the transverse axis name: Transversal dimension of the cuboid. (float [µm])
   :attr required d{Z}: where **{Z}** is the vertical axis name: Vertical dimension of the cuboid. (float [µm])
   :attr depth: Alias for ``d{X}``.
   :attr width: Alias for ``d{Y}``.
   :attr height: Alias for ``d{Z}``.

   Either ``material`` or both ``material-top`` and ``material-bottom`` are required.

.. xml:tag:: <cuboid/>

   Alias for :xml:tag:`<block3d/>`.

.. xml:tag:: <cylinder/>

   Corresponding Python class: :py:class:`plask.geometry.Cylinder`.

   Cylinder with its base lying in the horizontal plane. Its origin is located at the center of the lower circular base.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the cylinder material (for solid cylinders).
   :attr material-bottom: Definition of the material of the bottom of the cylinder (for cylinders which material linearly change from bottom to top). You should also set ``material-top``, and these materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the cylinder (see also ``material-bottom``).
   :attr radius: Radius of the cylinder base.
   :attr height: Height of the cylinder.

.. xml:tag:: <sphere/> (or <circle3d/>)

   Corresponding Python class: :py:class:`plask.geometry.Sphere`.
   
   Sphere with centre at point (0, 0, 0).
   
   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the sphere material (for solid triangles).
   :attr material-bottom: Definition of the material of the bottom of the sphere (for sphere which material linearly change from bottom to top). You should also set ``material-top``, and these materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the sphere (see also ``material-bottom``).
   :attr required radius: Radius of the sphere. (float [µm])