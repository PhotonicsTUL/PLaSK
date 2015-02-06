Transforms
^^^^^^^^^^

Transforms always contain a single geometry object (possibly container) as their content and perform some transformation of this object.





.. xml:tag:: <clip3d> (or <clip>)

   Corresponding Python class: :py:class:`plask.geometry.Clip3D`.

   Clip child object to given cuboid.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr back: back edge of the clipping cuboid (-INF by default)
   :attr front: front edge of the clipping cuboid (+INF by default)
   :attr left: left edge of the clipping cuboid (-INF by default)
   :attr right: right edge of the clipping cuboid (+INF by default)
   :attr bottom: bottom edge of the clipping cuboid (-INF by default)
   :attr top: top edge of the clipping cuboid (+INF by default)

   .. xml:contents::

       A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`. Object to clip.


.. xml:tag:: <intersection3d> (or <intersection>)

   Corresponding Python class: :py:class:`plask.geometry.Intersection3D`.

   Clip child object to given shape.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.

   .. xml:contents::

       Two :ref:`three-dimensional geometry objects <sec-xpl-Geometry-objects-3D>`. First is a clipped object, second is a clipping envelope (materials are not required in its leafs).


.. xml:tag:: <extrusion>

   Corresponding Python class: :py:class:`plask.geometry.Extrusion`.

   Extrusion of two-dimensional object into third dimension. 2D objects are defined in the plane defined by the transverse and vertical axes. Hence, the extrusion is performed into the longitudinal direction.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr required length: Length of the extrusion.

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.





.. xml:tag:: <flip3d> (or <flip>)

   Corresponding Python class: :py:class:`plask.geometry.Flip3D`.

   Mirror reflection of the object along specified axis.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   .. xml:contents::

       A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.



.. xml:tag:: <intersection3d> (or <intersection>)

   Corresponding Python class: :py:class:`plask.geometry.Intersection3D`.

   Clip child object to given shape.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.

   .. xml:contents::

       Two :ref:`three-dimensional geometry objects <sec-xpl-Geometry-objects-3D>`. First is a child of this, second is a clipping shape (materials are not required in its leafs).



.. xml:tag:: <revolution>

   Corresponding Python class: :py:class:`plask.geometry.Revolution`.

   Revolution of the two-dimensional object around its local vertical axis. The horizontal axis of the 2D object becomes a radial axis of the resulting compound cylinder. Vertical axes of the 2D object remains the vertical axis of the resulting block.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr auto_clip: If true item will be implicitly clipped to non-negative tran. coordinates. (bool, false by default)

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`. If auto_clip is off, all the boundaries of its bounding box must have their horizontal coordinates larger or equal to zero i.e. all the object must be located at the right-hand half of the plane. If auto_clip is on, they will be implicitly clipped.



.. xml:tag:: <mirror3d> (or <mirror>)

   Corresponding Python class: :py:class:`plask.geometry.Mirror3D`.

   Object mirrored along specified axis. In other words this is transformed object together with its flipped version. The bounding box of the object cannot span at bot sides of zero along inverted axis.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   .. xml:contents::

       A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.





.. xml:tag:: <translation3D> (or <translation>)

   Corresponding Python class: :py:class:`plask.geometry.Translation3D`.

   A simple shift of the object. Note that the bounding box is shifted as well, so in containers that place their items basing on their bounding boxes, this transformation will have no effect.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr {X}: where **{X}** is the longitudinal axis name: Longitudinal position of the origin of transformed element. (float [µm])
   :attr {Y}: where **{Y}** is the transverse axis name: Transversal position of the origin of transformed element. (float [µm])
   :attr {Z}: where **{Z}** is the vertical axis name: Vertical position of the origin of transformed element. (float [µm])

   .. xml:contents::

       A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.
