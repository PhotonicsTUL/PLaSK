.. _sec-XPL-geometry:

Section <geometry>
==================

In this section geometries of the analyze structures are defined. More than one geometry can be specified.

.. xml:tag:: <geometry>

   Inside each geometry tag there must be a single geometry object: usually it is some container.

   :attr axes: Default value of axes attribute for all geometries defined in this section.

Available elements
------------------

.. xml:tag:: <cartesian2d>

   Corresponding Python class: :py:class:`plask.geometry.Cartesian2D`.

   Two-dimensional Cartesian geometry.

   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr bottom: Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr left: Specification of the left border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr name: Geometry name for further reference. (unique identifier string)
   :attr right: Specification of the right border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr top: Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

   :Contents: Any object from section :ref:`sec-XPL-Geometry-objects-2D`.


.. xml:tag:: <cylindrical2d>

   Corresponding Python class: :py:class:`plask.geometry.Cylindrical2D`.

   Two-dimensional cylindrical geometry.

   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).wszystkie możliwości
   :attr bottom: Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr inner: Specification of the inner radical border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr length: Longitudinal dimension of the geometry (float [µm]).: Default value is: *+\infty*.
   :attr outer: Specification of the outer radical border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr name: Geometry name for further reference. (unique identifier string)
   :attr top: Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

   :Contents: Any object from section :ref:`sec-XPL-Geometry-objects-2D`. If ``length`` was not given, xml:tag:`extrusion` is also accepted.



.. xml:tag:: <cartesian3d>

   Corresponding Python class: :py:class:`plask.geometry.Cartesian3D`.

   Three-dimensional Cartesian geometry.

   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr back: Specification of the back border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr bottom: Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr front: Specification of the front border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr left: Specification of the left border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr name: Geometry name for further reference. (unique identifier string)
   :attr right: Specification of the right border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr top: Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

   :Contents: Any object from section :ref:`sec-XPL-Geometry-objects-3D`.


.. _sec-XPL-Geometry-objects-2D:

Geometry objects 2D
-------------------

The following elements are specifying two-dimensional geometry objects for use with 2D geometries. Each of them can have an optional attribute name, which allows to give the name to the object for further reference (either in the geometry specification or in the computational script). The correct value of the object name is unique identifier string.

Containers
^^^^^^^^^^

.. xml:tag:: <align2D> (or <align>)

   Corresponding Python classes: :py:class:`plask.geometry.AlignContainerTran2D`, :py:class:`plask.geometry.AlignContainerVert2D`.

   Container that align its items according to specified rules specified in its attributes. The alignment for one axis only should be given. As the objects in this container usually overlap, their order matters: latter items overwrite the former ones.

   :attr name: Object name for further reference.
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

   :Contents:

   The content of this element can any number of other two-dimensional geometry *object* or ``<item>`` elements which are organized in the vertical stack, ordered from top to bottom.

   *object*

      :ref:`Two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.

   .. xml:tag:: <item> [in <align2D>]

      Tag that allows to specify additional item attributes.

      :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
      :attr {alignment}: Any of the stack alignment specification attributes along the axis not specified in the container attributes (``left``, ``right``, ``trancenter``, **X**\ ``center``, **X**, ``top``, ``bottom``, ``vertcenter``, **Y**\ ``center``, **Y**). Specifies alignment of the item in the remaining direction. Defaults to ``left="0"`` or ``bottom="0"``.

      :Contents: A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.


.. xml:tag:: <container2D> (or <container>)

   Corresponding Python class: :py:class:`plask.geometry.TranslationContainer2D`.

   Simple container in which all the items must have explicitly specified position. As the objects in this container may overlap, their order matters: latter items overwrite the former ones.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.

   :Contents:

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

      :Contents: A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.


.. xml:tag:: <shelf2D> (or shelf)

   Corresponding Python class: :py:class:`plask.geometry.Shelf2D`.

   Container organizing objects side-by-side to each other, like books on a bookshelf. Items on the shelf are all bottom-aligned. Optionally it is possible to require that all the items have the same height in order to avoid the vertical gaps. However it is possible to insert intentional horizontal gaps to the shelf.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr flat: The value of this attribute can be either ``true`` of ``false``. It specifies whether all the items in the shelf are required to have the same height (therefore the top edge of the shelf is flat). Defaults to ``true``.

   :Contents:

   The content of this element can any number of other two-dimensional geometry object which are organized horizontally adjacent to each other, starting from the left.

   *object*

     :ref:`Two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.

   .. xml:tag:: <gap/> [in <shelf2D>]

      Horizontal gap between two objects. The size of the gap can be specified either as the absolute value in µm or as the total horizontal size of the shelf.

     :attr size: Size of the gap. (float [µm])
     :attr total: Total size of the shelf. The gap will adjust automatically. (float [µm])

      Exactly one of the above attributes must be specified and only one ``gap`` in the shelf can have the ``total`` attribute.

   .. xml:tag:: <zero/> [in <shelf2D>]

      This tag can appear as stack content only once. If present, it indicates the horizontal position of origin of the local coordinate system. Hence, it is an alternative method of specifying ``shift`` value.

.. xml:tag:: <stack2D> (or <stack>)

   Corresponding Python classes: :py:class:`plask.geometry.SingleStack2D` (if ``repeat``\ =1), :py:class:`plask.geometry.MultiStack2D` (if ``repeat``\ >1).

   Stack organizing its elements on top of the other. Horizontal alignment of the stack elements can be controlled by the alignment attributes of the whole stack or its items.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr repeat: Number of repetitive occurrences of stack content. This attribute allows to create periodic vertical structures (e. g. DBRs) easily. Defaults to 1. (integer)
   :attr shift: Vertical position of the stack bottom edge in its local coordinates. This attribute really makes sense only if the stack is the main element of the geometry, as in such case its local coordinates define global geometry coordinate system. Defaults to 0. (float [µm])
   :attr left: Default horizontal alignment specification: position of the left edge of the bounding box of each element. (float [µm])
   :attr right: Default horizontal alignment specification: position of the right edge of the bounding box of each element. (float [µm])
   :attr trancenter: Default horizontal alignment specification: position of the center of the bounding box of each element. (float [µm])
   :attr {X}center: where **{X}** is the transverse axis name: Alias for ``trancenter``.
   :attr {X}: where **{X}** is the transverse axis name: Default horizontal alignment specification: position of the origin of the element. (float [µm])

   Attributes ``left``, ``right``, ``trancenter``, **{X}**\ ``center`` and **{X}** are mutually exclusive. Default alignment is ``left="0"``.

   :Contents:

   The content of this element can any number of other two-dimensional geometry object or ``<item>`` elements which are organized in the vertical stack, ordered from top to bottom.

   *object*

      :ref:`Two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.

   .. xml:tag:: <item> [in <stack2D>]

      Tag that allows to specify additional item attributes.

      :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
      :attr {alignment}: Any of the stack alignment specification attributes (``left``, ``right``, ``trancenter``, **X**\ ``center``, **X**) that overrides the stack default for the particular item.

      :Contents: A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.

   .. xml:tag:: <zero/> [in <stack2D>]

      This tag can appear as stack content only once. If present, it indicates the vertical position of origin of the local coordinate system. Hence, it is an alternative method of specifying ``shift`` value.


Transforms
^^^^^^^^^^

Transforms always contain a single geometry object (possibly container) as their content and perform some transformation of this object.

.. xml:tag:: <flip2D> (or <flip>)

   Corresponding Python class: :py:class:`plask.geometry.Flip2D`.

   Mirror reflection of the object along specified axis.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   :Contents: A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.

.. xml:tag:: <mirror2D> (or <mirror>)

   Corresponding Python class: :py:class:`plask.geometry.Mirror2D`.

   Object mirrored along specified axis. In other words this is transformed object together with its flipped version. The bounding box of the object cannot span at bot sides of zero along inverted axis.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   :Contents: A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.

.. xml:tag:: <translation2D> (or <translation>)

   Corresponding Python class: :py:class:`plask.geometry.Translation2D`.

   A simple shift of the object. Note that the bounding box is shifted as well, so in containers that place their items basing on their bounding boxes, this transformation will have no effect.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr {X}: where **{X}** is the transverse axis name: Horizontal position of the origin of transformed element. (float [µm])
   :attr {Y}: where **{Y}** is the vertical axis name: Vertical position of the origin of transformed element. (float [µm])

   :Contents: A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.

Physical objects
^^^^^^^^^^^^^^^^

Physical objects are the leafs of the geometry tree. They represent actual objects that have some shape and defined material.

.. xml:tag:: <block2D/> (or <block/>)

   Corresponding Python class: :py:class:`plask.geometry.Block2D`.

   Rectangular block. Its origin is located at the lower left corner.

   :attr name: Object name for further reference.
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

   Alias for :xml:tag:`<block2D/>`.

Other
^^^^^

2D geometry object can be also obtained by refer to previously defined 2D object (see :xml:tag:`<again/>`) or copy of previously defined 2D object (see :xml:tag:`<copy>`). See section :ref:`sec-XPL-Geometry-objects-copy-ref` for more details.


.. _sec-XPL-Geometry-objects-3D:

Geometry objects 3D
-------------------

Containers
^^^^^^^^^^

Containers are objects that contain multiple other geometry objects as their items. They organize them spatially in different manners depending on the type of the container.

.. xml:tag:: <align3D> (or <align>)

   Corresponding Python classes: :py:class:`plask.geometry.AlignContainerLong3D`, :py:class:`plask.geometry.AlignContainerTran3D`, :py:class:`plask.geometry.AlignContainerVert3D`.

   Container that align its items according to specified rules specified in its attributes. The alignment for one axis only should be given. As the objects in this container usually overlap, their order matters: latter items overwrite the former ones.

   :attr name: Object name for further reference.
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

   :Contents:

   The content of this element can any number of other three-dimensional geometry *object* or ``<item>`` elements which are organized in the vertical stack, ordered from top to bottom.

   *object*

      :ref:`Three-dimensional geometry object <sec-XPL-Geometry-objects-3D>`.

   .. xml:tag:: <item> [in <align3D>]

      Tag that allows to specify additional item attributes.

      :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
      :attr {alignment}: Any of the stack alignment specification attributes along the axis not specified in the container attributes (``back``, ``front``, ``longcenter``, **{X}**\ ``center``, **{X}**, ``left``, ``right``, ``trancenter``, **{Y}**\ ``center``, and **{Y}**, ``top``, ``bottom``, ``vertcenter``, **{Z}**\ ``center``, **{Z}**). Specifies alignment of the item in the remaining direction. Defaults to ``back=0``, ``left="0"`` or ``bottom="0"`` (excluding the alignment of the container from the list).

      :Contents: A single :ref:`three-dimensional geometry object <sec-XPL-Geometry-objects-3D>`.

.. xml:tag:: <container3D> (or <container>)

   Corresponding Python class: :py:class:`plask.geometry.TranslationContainer3D`.

   Simple container in which all the items must have explicitly specified position. As the objects in this container may overlap, their order matters: latter items overwrite the former ones.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.

   :Contents:

   .. xml:tag:: <item> [in <container3D>]

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

      :Contents: A single :ref:`three-dimensional geometry object <sec-XPL-Geometry-objects-3D>`.

.. xml:tag:: <stack3D> (or <stack>)

   Corresponding Python classes: :py:class:`plask.geometry.SingleStack3D` (if ``repeat``\ =1), :py:class:`plask.geometry.MultiStack3D` (if ``repeat``\ >1).

   Stack organizing its elements on top of the other. Horizontal alignments of the stack elements can be controlled by the alignment attributes of the whole stack or its items.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr repeat: Number of repetitive occurrences of stack content. This attribute allows to create periodic vertical structures (e. g. DBRs) easily. Defaults to 1. (integer)
   :attr shift: Vertical position of the stack bottom edge in its local coordinates. This attribute really makes sense only if the stack is the main element of the geometry, as in such case its local coordinates define global geometry coordinate system. Defaults to 0. (float [µm])
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

   :Contents:

   The content of this element can any number of other three-dimensional geometry object or ``<item>`` elements which are organized in the vertical stack, ordered from top to bottom.

   *object*

      :ref:`Three-dimensional geometry object <sec-XPL-Geometry-objects-3D>`.

   .. xml:tag:: <item> [in <stack3D>]

      Tag that allows to specify additional item attributes.

      :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
      :attr {alignment}: Any of the stack alignment specification attributes (``back``, ``front``, ``longcenter``, **{X}**\ ``center``, **{X}**, ``left``, ``right``, ``trancenter``, **{Y}**\ ``center``, **{Y}**) that overrides the stack default for the particular item.

      :Contents: A single :ref:`three-dimensional geometry object <sec-XPL-Geometry-objects-3D>`.

   .. xml:tag:: <zero/> [in <stack3D>]

      This tag can appear as stack content only once. If present, it indicates the vertical position of origin of the local coordinate system. Hence, it is an alternative method of specifying ``shift`` value.

Transforms
^^^^^^^^^^

Transforms always contain a single geometry object (possibly container) as their content and perform some transformation of this object.

.. xml:tag:: <extrusion>

   Corresponding Python class: :py:class:`plask.geometry.Extrusion`.

   Extrusion of two-dimensional object into third dimension. 2D objects are defined in the plane defined by the transverse and vertical axes. Hence, the extrusion is performed into the longitudinal direction.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr required length: Length of the extrusion.

   :Contents: A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.

.. xml:tag:: <flip3D> (or <flip>)

   Corresponding Python class: :py:class:`plask.geometry.Flip3D`.

   Mirror reflection of the object along specified axis.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   :Contents: A single :ref:`three-dimensional geometry object <sec-XPL-Geometry-objects-3D>`.

.. xml:tag:: <revolution>

   Corresponding Python class: :py:class:`plask.geometry.Revolution`.

   Revolution of the two-dimensional object around its local vertical axis. The horizontal axis of the 2D object becomes a radial axis of the resulting compound cylinder. Vertical axes of the 2D object remains the vertical axis of the resulting block.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.

   :Contents: A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`. All the boundaries of its bounding box must have their horizontal coordinates larger or equal to zero i.e. all the object must be located at the right-hand half of the plane.

.. xml:tag:: <mirror3D> (or <mirror>)

   Corresponding Python class: :py:class:`plask.geometry.Mirror3D`.

   Object mirrored along specified axis. In other words this is transformed object together with its flipped version. The bounding box of the object cannot span at bot sides of zero along inverted axis.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr required axis: Name of the inverted axis (i.e. perpendicular to the reflection plane).

   :Contents: A single :ref:`three-dimensional geometry object <sec-XPL-Geometry-objects-3D>`.

.. xml:tag:: <translation3D> (or <translation>)

   Corresponding Python class: :py:class:`plask.geometry.Translation3D`.

   A simple shift of the object. Note that the bounding box is shifted as well, so in containers that place their items basing on their bounding boxes, this transformation will have no effect.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr {X}: where **{X}** is the longitudinal axis name: Longitudinal position of the origin of transformed element. (float [µm])
   :attr {Y}: where **{Y}** is the transverse axis name: Transversal position of the origin of transformed element. (float [µm])
   :attr {Z}: where **{Z}** is the vertical axis name: Vertical position of the origin of transformed element. (float [µm])

   :Contents: A single :ref:`three-dimensional geometry object <sec-XPL-Geometry-objects-3D>`.

Physical objects
^^^^^^^^^^^^^^^^

Physical objects are the leafs of the geometry tree. They represent actual objects that have some shape and defined material.

.. xml:tag:: <block3D/> (or <block/>)

   Corresponding Python class: :py:class:`plask.geometry.Block3D`.

   Rectangular block. Its origin is located in the lower back left corner.

   :attr name: Object name for further reference.
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

   Alias for :xml:tag:`<block3D/>`.

.. xml:tag:: <cylinder/>

   Corresponding Python class: :py:class:`plask.geometry.Cylinder`.

   Cylinder with its base lying in the horizontal plane. Its origin is located at the center of the lower circular base.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr material: Definition of the cylinder material (for solid cylinders).
   :attr material-bottom: Definition of the material of the bottom of the cylinder (for cylinders which material linearly change from bottom to top). You should also set ``material-top``, and these materials can differs only in composition or amount of dopant.
   :attr material-top: Definition of the material of top of the cylinder (see also ``material-bottom``).
   :attr radius: Radius of the cylinder base.
   :attr height: Height of the cylinder.

Other
^^^^^

3D geometry object can be also obtained by refer to previously defined 3D object (see :xml:tag:`<again/>`) or copy of previously defined 3D object (see :xml:tag:`<copy>`). See section :ref:`sec-XPL-Geometry-objects-copy-ref` for more details.


.. _sec-XPL-Geometry-objects-copy-ref:

Copies and references to geometry objects
-----------------------------------------

.. xml:tag:: <again/>

   This tag can be used to insert any previously defined and named (with the name attribute) two or three dimensional object again in the geometry tree.

   :attr required ref: Name of the referenced object.

.. xml:tag:: <copy>

   Modified copy of any previously defined and named (with the name attribute) two or three dimensional object.

   :attr name: Object name for further reference.
   :attr role: Object role. Important for some solvers.
   :attr required from: Name of the source two or three dimensional object to make modified copy of. Usually it is some container that has some other named its items or sub-items.

   :Contents:

   The content of this element contains the tags specifying desired modifications of the source object. The source object remains unchanged, but its copy has alternations described by the following tags:

   .. xml:tag:: <delete/>

      Delete some item or sub-item of the copied object.

      :attr required object: Name of the object to delete.

   .. xml:tag:: <replace/>

      Replace some item or sub-item of the copied object with some other named object specified anywhere earlier in the geometry.

      :attr required object: Name of the object to delete.
      :attr with: Name of the object to replace with. This object does not need to be located in the subtree of the copied object.
      :contents: A new geometry object to replace the original one. Must be specified if and only if the with attribute is not provided.

   .. xml:tag:: <toblock/>

      Replace some item or sub-item of the copied object with uniform block that has dimensions exactly equal to the bounding box of the original element.

      :attr required object: Name of the object to replace with the the solid block.
      :attr required material: Material of the solid block.
