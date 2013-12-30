Containers
^^^^^^^^^^

.. xml:tag:: <align2D> (or <align>)

   Corresponding Python classes: :py:class:`plask.geometry.AlignContainerTran2D`, :py:class:`plask.geometry.AlignContainerVert2D`.

   Container that align its items according to specified rules specified in its attributes. The alignment for one axis only should be given. As the objects in this container usually overlap, their order matters: latter items overwrite the former ones.

   :attr name: Object name for further reference. In :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr left: Horizontal alignment specification: position of the left edge of the bounding box of each element. (float [µm])
   :attr right: Horizontal alignment specification: position of the right edge of the bounding box of each element. (float [µm])
   :attr trancenter: Horizontal alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {X}center: (where **{X}** is the transverse axis name): Alias for ``trancenter``.
   :attr {X}: (where **{X}** is the transverse axis name): Horizontal alignment specification: position of the origin of each element. (float [µm])
   :attr top: Vertical alignment specification: position of the top edge of the bounding box of each element. (float [µm])
   :attr bottom: Vertical alignment specification: position of the bottom edge of the bounding box of each element. (float [µm])
   :attr vertcenter: Vertical alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {Y}center: (where **{Y}** is the vertical axis name): Alias for *vertcenter*.
   :attr {Y}: (where **{Y}** is the vertical axis name): Vertical alignment specification: position of the origin of each element. (float [µm])

   Exactly one of the ``left``, ``right``, ``trancenter``, **{X}**\ ``center``, **{X}**, ``top``, ``bottom``, ``vertcenter``, **{Y}**\ ``center``, and **{Y}** attributes must be given.

   .. xml:contents::

      The content of this element can be any number of other two-dimensional geometry *object* or ``<item>`` elements which are organized in the vertical stack, ordered from top to bottom.

      *object*

         :ref:`Two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.

      .. xml:tag:: <item> [in <align2D>]

         Tag that allows to specify additional item attributes.

         :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
         :attr {alignment}: Any of the stack alignment specification attributes along the axis not specified in the container attributes (``left``, ``right``, ``trancenter``, **{X}**\ ``center``, **{X}**, ``top``, ``bottom``, ``vertcenter``, **{Y}**\ ``center``, **{Y}**). Specifies alignment of the item in the remaining direction. Defaults to ``left="0"`` or ``bottom="0"``.

         .. xml:contents::

            A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.


.. xml:tag:: <container2D> (or <container>)

   Corresponding Python class: :py:class:`plask.geometry.TranslationContainer2D`.

   Simple container in which all the items must have explicitly specified position. As the objects in this container may overlap, their order matters: latter items overwrite the former ones.

   :attr name: Object name for further reference. In :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.

   .. xml:contents::

      .. xml:tag:: <item> [in <container2D>]

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

         Attributes ``left``, ``right``, ``trancenter``, **{X}**\ ``center``, **{X}**, are mutually exclusive. Attributes ``top``, ``bottom``, ``vertcenter``, **{Y}**\ ``center``, and **{Y}** are mutually exclusive. At least one alignment specification for each axis must be given.

         .. xml:contents::

             A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.


.. xml:tag:: <shelf2D> (or shelf)

   Corresponding Python class: :py:class:`plask.geometry.Shelf2D`.

   Container organizing objects side-by-side to each other, like books on a bookshelf. Items on the shelf are all bottom-aligned. Optionally it is possible to require that all the items have the same height in order to avoid the vertical gaps. However it is possible to insert intentional horizontal gaps to the shelf.

   :attr name: Object name for further reference. In :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr flat: The value of this attribute can be either ``true`` of ``false``. It specifies whether all the items in the shelf are required to have the same height (therefore the top edge of the shelf is flat). Defaults to ``true``.
   :attr shift: Horizontal position of the shelf left edge in its local coordinates. Defaults to 0. (float [µm])

   .. xml:contents::

      The content of this element can any number of other two-dimensional geometry object which are organized horizontally adjacent to each other, starting from the left.

      *object*

        :ref:`Two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.

      .. xml:tag:: <gap/> [in <shelf2D>]

         Horizontal gap between two objects. The size of the gap can be specified either as the absolute value in µm or as the total horizontal size of the shelf.

        :attr size: Size of the gap. (float [µm])
        :attr total: Total size of the shelf. The gap will adjust automatically. (float [µm])

         Exactly one of the above attributes must be specified and only one ``gap`` in the shelf can have the ``total`` attribute.

      .. xml:tag:: <zero/> [in <shelf2D>]

         This tag can appear as a shelf content only once. If present, it indicates the horizontal position of origin of the local coordinate system. Hence, it is an alternative method of specifying ``shift`` value.

.. xml:tag:: <stack2D> (or <stack>)

   Corresponding Python classes: :py:class:`plask.geometry.SingleStack2D` (if ``repeat``\ =1), :py:class:`plask.geometry.MultiStack2D` (if ``repeat``\ >1).

   Stack organizing its elements on top of the other. Horizontal alignment of the stack elements can be controlled by the alignment attributes of the whole stack or its items.

   :attr name: Object name for further reference. In :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
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

      .. xml:tag:: <item> [in <stack2D>]

         Tag that allows to specify additional item attributes.

         :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
         :attr {alignment}: Any of the stack alignment specification attributes (``left``, ``right``, ``trancenter``, **X**\ ``center``, **X**) that overrides the stack default for the particular item.

         .. xml:contents::

             A single :ref:`two-dimensional geometry object <sec-xpl-Geometry-objects-2D>`.

      .. xml:tag:: <zero/> [in <stack2D>]

         This tag can appear as a stack content only once. If present, it indicates the vertical position of origin of the local coordinate system. Hence, it is an alternative method of specifying ``shift`` value.
