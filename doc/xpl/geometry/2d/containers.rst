Containers
^^^^^^^^^^

.. xml:tag:: <align2d> (or <align>)

   Corresponding Python class: :py:class:`plask.geometry.Align2D`.

   Container in which all the items must have explicitly specified position. As the objects in this container may overlap, their order matters: latter items overwrite the former ones.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr left: Default value for horizontal alignment specification: position of the left edge of the bounding box of each element. (float [µm])
   :attr right: Default value for horizontal alignment specification: position of the right edge of the bounding box of each element. (float [µm])
   :attr trancenter: Default value for horizontal alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {X}center: (where **{X}** is the transverse axis name): Alias for ``trancenter``.
   :attr {X}: (where **{X}** is the transverse axis name): Default value for horizontal alignment specification: position of the origin of each element. (float [µm])
   :attr top: Default value for vertical alignment specification: position of the top edge of the bounding box of each element. (float [µm])
   :attr bottom: Default value for vertical alignment specification: position of the bottom edge of the bounding box of each element. (float [µm])
   :attr vertcenter: Default value for vertical alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {Y}center: (where **{Y}** is the vertical axis name): Alias for *vertcenter*.
   :attr {Y}: (where **{Y}** is the vertical axis name): Default value for vertical alignment specification: position of the origin of each element. (float [µm])

   Attributes ``left``, ``right``, ``trancenter``, **{X}**\ ``center``, **{X}**, are mutually exclusive. Attributes ``top``, ``bottom``, ``vertcenter``, **{Y}**\ ``center``, and **{Y}** are mutually exclusive. At most one alignment specification for each axis can be given. If alignment specification for some direction is not given, elements will be not alignment in this direction by default.

   .. xml:contents::

      The content of this element can be any number of other two-dimensional geometry *object* (is such case defaul alignment specifications are used) or ``<item>`` elements, which are aligned in the container according to its specification.

      *object*

         :ref:`Two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.

      .. xml:tag:: <item> [in <container2d>]

         :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
         :attr left: Horizontal alignment specification: position of the left edge of the bounding box of the element. (float [µm])
         :attr right: Horizontal alignment specification: position of the right edge of the bounding box of the element. (float [µm])
         :attr trancenter: Horizontal alignment specification: position of the center of the bounding box of the element. (float [µm])
         :attr {X}center: where **{X}** is the transverse axis name: Alias for ``trancenter``.
         :attr {X}: where **{X}** is the transverse axis name: Horizontal alignment specification: position of the origin of the element. (float [µm])
         :attr top: Vertical alignment specification: position of the top edge of the bounding box of the element. (float [µm])
         :attr bottom: Vertical alignment specification: position of the bottom edge of the bounding box of the element. (float [µm])
         :attr vertcenter: Vertical alignment specification: position of the center of the bounding box of the element. (float [µm])
         :attr {Y}center: where **{Y}** is the vertical axis name: Alias for vertcenter.
         :attr {Y}: where **{Y}** is the vertical axis name: Vertical alignment specification: position of the origin of the element. (float [µm])

         Attributes ``left``, ``right``, ``trancenter``, **{X}**\ ``center``, **{X}**, are mutually exclusive. Attributes ``top``, ``bottom``, ``vertcenter``, **{Y}**\ ``center``, and **{Y}** are mutually exclusive. At most one alignment specification for each axis can be given. If alignment specification for some direction is not given, default value is used.

         .. xml:contents::

             A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.


.. xml:tag:: <arrange2d> (or <arrange>)

   Corresponding Python class: :py:class:`plask.geometry.Arrange2D`.

   Container in which replicates a single item in an equally-spaced row of its repetitions.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr d{X}: where **{X}** is the transverse axis name: Horizontal component of the spacing vector, by which the adjacent repetitions are shifted. (float [µm])
   :attr d{Y}: where **{Y}** is the vertical axis name: Vertical dimension of the spacing vector, by which the adjacent repetitions are shifted. (float [µm])
   :attr required count: Number of repetitions of the container item.

   .. xml:contents::

      A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`. Object to arrange in a row.


.. xml:tag:: <container2d> (or <container>)

   Alias for :xml:tag:`<align2d>`.


.. xml:tag:: <shelf2d> (or shelf)

   Corresponding Python class: :py:class:`plask.geometry.Shelf2D` (if ``repeat``\ =1), :py:class:`plask.geometry.MultiShelf2D` (if ``repeat``\ >1).

   Container organizing objects side-by-side to each other, like books on a bookshelf. Items on the shelf are all bottom-aligned. Optionally it is possible to require that all the items have the same height in order to avoid the vertical gaps. However it is possible to insert intentional horizontal gaps to the shelf.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr repeat: Number of repetitive occurrences of stack content. This attribute allows to create periodic horizontal structures easily. Defaults to 1. (integer)
   :attr flat: The value of this attribute can be either ``true`` of ``false``. It specifies whether all the items in the shelf are required to have the same height (therefore the top edge of the shelf is flat). Defaults to ``true``.
   :attr shift: Horizontal position of the shelf left edge in its local coordinates. Defaults to 0. (float [µm])

   .. xml:contents::

      The content of this element can any number of other two-dimensional geometry object which are organized horizontally adjacent to each other, starting from the left.

      *object*

        :ref:`Two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.

      .. xml:tag:: <gap/> [in <shelf2d>]

         Horizontal gap between two objects. The size of the gap can be specified either as the absolute value in µm or as the total horizontal size of the shelf.

         :attr size: Size of the gap. (float [µm])
         :attr total: Total size of the shelf. The gap will adjust automatically. (float [µm])

         Exactly one of the above attributes must be specified and only one ``gap`` in the shelf can have the ``total`` attribute.

      .. xml:tag:: <zero/> [in <shelf2d>]

         This tag can appear as a shelf content only once. If present, it indicates the horizontal position of origin of the local coordinate system. Hence, it is an alternative method of specifying ``shift`` value.

.. xml:tag:: <stack2d> (or <stack>)

   Corresponding Python classes: :py:class:`plask.geometry.SingleStack2D` (if ``repeat``\ =1), :py:class:`plask.geometry.MultiStack2D` (if ``repeat``\ >1).

   Stack organizing its elements on top of the other. Horizontal alignment of the stack elements can be controlled by the alignment attributes of the whole stack or its items.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr role: Object role. Important for some solvers.
   :attr repeat: Number of repetitive occurrences of stack content. This attribute allows to create periodic vertical structures (e. g. DBRs) easily. Defaults to 1. (integer)
   :attr shift: Vertical position of the stack bottom edge in its local coordinates. Defaults to 0. (float [µm])
   :attr left: Default horizontal alignment specification: position of the left edge of the bounding box of each element. (float [µm])
   :attr right: Default horizontal alignment specification: position of the right edge of the bounding box of each element. (float [µm])
   :attr trancenter: Default horizontal alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {X}center: where **{X}** is the transverse axis name: Alias for ``trancenter``.
   :attr {X}: where **{X}** is the transverse axis name: Default horizontal alignment specification: position of the origin of the element. (float [µm])

   Attributes ``left``, ``right``, ``trancenter``, **{X}**\ ``center`` and **{X}** are mutually exclusive. Default alignment is ``left="0"``.

   .. xml:contents::

      The content of this element can any number of other two-dimensional geometry object or ``<item>`` elements which are organized in the vertical stack, ordered from top to bottom.

      *object*

         :ref:`Two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.

      .. xml:tag:: <item> [in <stack2d>]

         Tag that allows to specify additional item attributes.

         :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
         :attr {alignment}: Any of the stack alignment specification attributes (``left``, ``right``, ``trancenter``, **X**\ ``center``, **X**) that overrides the stack default for the particular item.

         .. xml:contents::

             A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.

      .. xml:tag:: <zero/> [in <stack2d>]

         This tag can appear as a stack content only once. If present, it indicates the vertical position of origin of the local coordinate system. Hence, it is an alternative method of specifying ``shift`` value.
