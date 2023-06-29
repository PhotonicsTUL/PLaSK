.. _sec-xpl-geometry-2d-transforms:

Transforms 2D
^^^^^^^^^^^^^

Transforms always contain a single geometry object (possibly container) as their content and perform some transformation of this object.


.. xml:tag:: <arrange2d> (or <arrange>)

   Corresponding Python class: :py:class:`plask.geometry.Arrange2D`.

   Container in which replicates a single item in an equally-spaced row of its repetitions.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr d{X}: where **{X}** is the transverse axis name: Horizontal component of the spacing vector, by which the adjacent repetitions are shifted. (float, µm)
   :attr d{Y}: where **{Y}** is the vertical axis name: Vertical dimension of the spacing vector, by which the adjacent repetitions are shifted. (float, µm)
   :attr required count: Number of repetitions of the container item.

   .. xml:contents::

      A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`. Object to arrange in a row.



.. xml:tag:: <clip2d> (or <clip>)

   Corresponding Python class: :py:class:`plask.geometry.Clip2D`.

   Clip child object to given rectangle.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr left: left edge of the clipping rectangle (-INF by default)
   :attr right: right edge of the clipping rectangle (+INF by default)
   :attr bottom: bottom edge of the clipping rectangle (-INF by default)
   :attr top: top edge of the clipping rectangle (+INF by default)

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`. Object to clip.





.. xml:tag:: <flip2d> (or <flip>)

   Corresponding Python class: :py:class:`plask.geometry.Flip2D`.

   Mirror reflection of the object along specified axis.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.


.. xml:tag:: <intersection2d> (or <intersection>)

   Corresponding Python class: :py:class:`plask.geometry.Intersection2D`.

   Clip child object to given shape.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.

   .. xml:contents::

       Two :ref:`two-dimensional geometry objects <sec-xpl-Geometry-objects-2D>`. First is a clipped object, second is a clipping envelope (materials are not required in its leafs).


.. xml:tag:: <mirror2d> (or <mirror>)

   Corresponding Python class: :py:class:`plask.geometry.Mirror2D`.

   Object mirrored along specified axis. In other words this is transformed object together with its flipped version. The bounding box of the object cannot span at bot sides of zero along inverted axis.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.





.. xml:tag:: <translation2d> (or <translation>)

   Corresponding Python class: :py:class:`plask.geometry.Translation2D`.

   A simple shift of the object. Note that the bounding box is shifted as well, so in containers that place their items basing on their bounding boxes, this transformation will have no effect.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr {X}: where **{X}** is the transverse axis name: Horizontal position of the origin of transformed element. (float, µm)
   :attr {Y}: where **{Y}** is the vertical axis name: Vertical position of the origin of transformed element. (float, µm)

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.
