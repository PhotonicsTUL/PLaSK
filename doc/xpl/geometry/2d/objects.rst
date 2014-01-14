Physical objects
^^^^^^^^^^^^^^^^

Physical objects are the leafs of the geometry tree. They represent actual objects that have some shape and defined material.

.. xml:tag:: <block2d/> (or <block/>)

   Corresponding Python class: :py:class:`plask.geometry.Block2D`.

   Rectangular block. Its origin is located at the lower left corner.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the block material (for solid blocks).
   :attr material-bottom: Definition of the material of the bottom of the block (for blocks which material linearly change from bottom to top). You should also set ``material-top``, and these materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the block (see also ``material-bottom``).
   :attr required d{X}: where **{X}** is the transverse axis name: Horizontal dimension of the rectangle. (float [µm])
   :attr required d{Y}: where **{Y}** is the transverse axis name: Vertical dimension of the rectangle. (float [µm])
   :attr width: Alias for ``d{X}``.
   :attr height: Alias for ``d{Y}``.

   Either ``material`` or both ``material-top`` and ``material-bottom`` are required.

.. xml:tag:: <rectangle/>

   Alias for :xml:tag:`<block2d/>`.
