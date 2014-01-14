Containers
^^^^^^^^^^

Containers are objects that contain multiple other geometry objects as their items. They organize them spatially in different manners depending on the type of the container.

.. xml:tag:: <align3d> (or <align>)

   Corresponding Python classes: :py:class:`plask.geometry.AlignContainerLong3D`, :py:class:`plask.geometry.AlignContainerTran3D`, :py:class:`plask.geometry.AlignContainerVert3D`.

   Container that align its items according to specified rules specified in its attributes. The alignment for one axis only should be given. As the objects in this container usually overlap, their order matters: latter items overwrite the former ones.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr back: Longitudinal alignment specification: position of the back edge of the bounding box of each element. (float [µm])
   :attr front: Longitudinal alignment specification: position of the front edge of the bounding box of each element. (float [µm])
   :attr longcenter: Longitudinal alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {X}center: where **{X}** is the longitudinal axis name: Alias for ``longcenter``.
   :attr {X}: where **{X}** is the longitudinal axis name: Longitudinal alignment specification: position of the origin of each element. (float [µm])
   :attr left: Transversal alignment specification: position of the left edge of the bounding box of each element. (float [µm])
   :attr right: Transversal alignment specification: position of the right edge of the bounding box of each element. (float [µm])
   :attr trancenter: Transversal alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {Y}center: where **{Y}** is the transverse axis name: Alias for ``trancenter``.
   :attr {Y}: where **{Y}** is the transverse axis name: Transversal alignment specification: position of the origin of each element. (float [µm])
   :attr top: Vertical alignment specification: position of the top edge of the bounding box of each element. (float [µm])
   :attr bottom: Vertical alignment specification: position of the bottom edge of the bounding box of each element. (float [µm])
   :attr vertcenter: Vertical alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {Z}center: where **{Z}** is the vertical axis name: Alias for ``vertcenter``.
   :attr {Z}: where **{Z}** is the vertical axis name: Vertical alignment specification: position of the origin of each element. (float [µm])

   Exactly one of the ``back``, ``front``, ``longcenter``, **{X}**\ ``center``, **{X}**, ``left``, ``right``, ``trancenter``, **{Y}**\ ``center``, and **{Y}**, ``top``, ``bottom``, ``vertcenter``, **{Z}**\ ``center``, and **{Z}** attributes must be given.

   .. xml:contents::

      The content of this element can any number of other three-dimensional geometry *object* or ``<item>`` elements which are organized in the vertical stack, ordered from top to bottom.

      *object*

         :ref:`Three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.

      .. xml:tag:: <item> [in <align3d>]

         Tag that allows to specify additional item attributes.

         :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
         :attr {alignment}: Any of the stack alignment specification attributes along the axis not specified in the container attributes (``back``, ``front``, ``longcenter``, **{X}**\ ``center``, **{X}**, ``left``, ``right``, ``trancenter``, **{Y}**\ ``center``, and **{Y}**, ``top``, ``bottom``, ``vertcenter``, **{Z}**\ ``center``, **{Z}**). Specifies alignment of the item in the remaining direction. Defaults to ``back=0``, ``left="0"`` or ``bottom="0"`` (excluding the alignment of the container from the list).

         .. xml:contents::

             A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.





.. xml:tag:: <container3d> (or <container>)

   Corresponding Python class: :py:class:`plask.geometry.TranslationContainer3D`.

   Simple container in which all the items must have explicitly specified position. As the objects in this container may overlap, their order matters: latter items overwrite the former ones.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.

   .. xml:contents::

      .. xml:tag:: <item> [in <container3d>]

         :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
         :attr back: Longitudinal alignment specification: position of the back edge of the bounding box of the element. (float [µm])
         :attr front: Longitudinal alignment specification: position of the front edge of the bounding box of the element. (float [µm])
         :attr longcenter: Longitudinal alignment specification: position of the center of the bounding box of the element. (float [µm])
         :attr {X}center: where **{X}** is the longitudinal axis name: Alias for ``longcenter``.
         :attr {X}: where **{X}** is the longitudinal axis name: Longitudinal alignment specification: position of the origin of the element. (float [µm])
         :attr left: Transversal alignment specification: position of the left edge of the bounding box of the element. (float [µm])
         :attr right: Transversal alignment specification: position of the right edge of the bounding box of the element. (float [µm])
         :attr trancenter: Transversal alignment specification: position of the center of the bounding box of the element. (float [µm])
         :attr {Y}center: where **{Y}** is the transverse axis name: Alias for ``trancenter``.
         :attr {Y}: where **{Y}** is the transverse axis name: Transversal alignment specification: position of the origin of the element. (float [µm])
         :attr top: Vertical alignment specification: position of the top edge of the bounding box of the element. (float [µm])
         :attr bottom: Vertical alignment specification: position of the bottom edge of the bounding box of the element. (float [µm])
         :attr vertcenter: Vertical alignment specification: position of the center of the bounding box of each element. (float [µm])
         :attr {Z}center: where **{Z}** is the vertical axis name: Alias for ``vertcenter``.
         :attr {Z}: where **{Z}** is the vertical axis name: Vertical alignment specification: position of the origin of the element. (float [µm])

         Attributes ``back``, ``front``, ``longcenter``, **{X}**\ ``center``, **{X}**, are mutually exclusive. Attributes ``left``, ``right``, ``trancenter``, **{Y}**\ ``center``, **{Y}**, are mutually exclusive. Attributes ``top``, ``bottom``, ``vertcenter``, **{Z}**\ ``center``, and **{Z}** are mutually exclusive. At least one alignment specification for each axis must be given.

         .. xml:contents::

             A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.




.. xml:tag:: <stack3d> (or <stack>)

   Corresponding Python classes: :py:class:`plask.geometry.SingleStack3D` (if ``repeat``\ =1), :py:class:`plask.geometry.MultiStack3D` (if ``repeat``\ >1).

   Stack organizing its elements on top of the other. Horizontal alignments of the stack elements can be controlled by the alignment attributes of the whole stack or its items.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr repeat: Number of repetitive occurrences of stack content. This attribute allows to create periodic vertical structures (e. g. DBRs) easily. Defaults to 1. (integer)
   :attr shift: Vertical position of the stack bottom edge in its local coordinates. Defaults to 0. (float [µm])
   :attr back: Longitudinal alignment specification: position of the back edge of the bounding box of each element. (float [µm])
   :attr front: Longitudinal alignment specification: position of the front edge of the bounding box of each element. (float [µm])
   :attr longcenter: Longitudinal alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {X}center: where **{X}** is the longitudinal axis name: Alias for ``longcenter``.
   :attr {X}: where **{X}** is the longitudinal axis name: Longitudinal alignment specification: position of the origin of each element. (float [µm])
   :attr left: Transversal alignment specification: position of the left edge of the bounding box of each element. (float [µm])
   :attr right: Transversal alignment specification: position of the right edge of the bounding box of each element. (float [µm])
   :attr trancenter: Transversal alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {Y}center: where **{Y}** is the transverse axis name: Alias for ``trancenter``.
   :attr {Y}: where **{Y}** is the transverse axis name: Transversal alignment specification: position of the origin of each element. (float [µm])

   Attributes ``back``, ``front``, ``longcenter``, **{X}**\ ``center``, **{X}**, are mutually exclusive. Attributes ``left``, ``right``, ``trancenter``, **{Y}**\ ``center``, **{Y}**, are mutually exclusive. Default alignment is ``left="0"`` and ``back="0"``.

   .. xml:contents::

      The content of this element can any number of other three-dimensional geometry object or ``<item>`` elements which are organized in the vertical stack, ordered from top to bottom.

      *object*

         :ref:`Three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.

      .. xml:tag:: <item> [in <stack3d>]

         Tag that allows to specify additional item attributes.

         :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
         :attr {alignment}: Any of the stack alignment specification attributes (``back``, ``front``, ``longcenter``, **{X}**\ ``center``, **{X}**, ``left``, ``right``, ``trancenter``, **{Y}**\ ``center``, **{Y}**) that overrides the stack default for the particular item.

         .. xml:contents::

             A single :ref:`three-dimensional geometry object <sec-xpl-Geometry-objects-3D>`.

      .. xml:tag:: <zero/> [in <stack3d>]

         This tag can appear as a stack content only once. If present, it indicates the vertical position of origin of the local coordinate system. Hence, it is an alternative method of specifying ``shift`` value.
