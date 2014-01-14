Transforms
^^^^^^^^^^

Transforms always contain a single geometry object (possibly container) as their content and perform some transformation of this object.





.. xml:tag:: <clip2D> (or <clip>)

   Corresponding Python class: :py:class:`plask.geometry.Clip2D`.

   Clip child object to given rectangle.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr required left: left edge of the clipping rectangle
   :attr required right: right edge of the clipping rectangle
   :attr required bottom: bottom edge of the clipping rectangle
   :attr required top: top edge of the clipping rectangle

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`. Object to clip.





.. xml:tag:: <flip2D> (or <flip>)

   Corresponding Python class: :py:class:`plask.geometry.Flip2D`.

   Mirror reflection of the object along specified axis.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.





.. xml:tag:: <mirror2D> (or <mirror>)

   Corresponding Python class: :py:class:`plask.geometry.Mirror2D`.

   Object mirrored along specified axis. In other words this is transformed object together with its flipped version. The bounding box of the object cannot span at bot sides of zero along inverted axis.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.





.. xml:tag:: <translation2D> (or <translation>)

   Corresponding Python class: :py:class:`plask.geometry.Translation2D`.

   A simple shift of the object. Note that the bounding box is shifted as well, so in containers that place their items basing on their bounding boxes, this transformation will have no effect.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr {X}: where **{X}** is the transverse axis name: Horizontal position of the origin of transformed element. (float [µm])
   :attr {Y}: where **{Y}** is the vertical axis name: Vertical position of the origin of transformed element. (float [µm])

   .. xml:contents::

       A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.
