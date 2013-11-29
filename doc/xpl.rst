.. _sec-XPL:

******************
XPL File Reference
******************

:term:`XPL` files follow :term:`XML` specification. Thus all the general rules of creating correct XML files apply top XPL ones as well. Please refer to the external documentation for information on XML syntax and grammar [#XML-tutoruals]_. Details specific to XPL are covered in this chapter.

First of all each XML document must have a parent tag. In XPL files such tag is named:

.. xml:tag:: <plask>

Thus, the all information in the data file are content of this tag and have to be located between ``<plask>`` and ``</plask>`` tags. Inside there are several sections that can be included in the XPL file: :xml:tag:`<defines>`, :xml:tag:`<materials>`, :xml:tag:`<geometry>`, :xml:tag:`<grids>`, :xml:tag:`<solvers>`, :xml:tag:`<connects>`, and :xml:tag:`<script>`. Each of them is optional, however, if present, they must be specified in the order shown in the previous sentence. Formal specification of each section is presented below.


``<plask>`` section can take an optional attribute ``loglevel``, which value is the name of any valid log level. Any log message with its priority lower than the specified value will not be printed by the logging system. The default value is ``"detail"``.
Instead of the section content, it is possible to use it as a single tag with the attribute ``external``, which has a name of some XPL file as its value. In such a case, the content of the relevant section is read from this external file.


.. _sec-XPL-defines:

Section <defines>
=================

This section allows to define some constant parameters (that can be later overridden either in the command line or while reading the XPL file from Python). Each parameter is defined with the only tag allowed in this section:

.. xml:tag:: <defines>

   Definition of a parameter for use in the rest of the file.

   :attr required name: Name of the parameter (each name must be unique).
   :attr required value: Value of the parameter. Any valid Python function can be used here, as well as any previously defined parameter.


.. _sec-XPL-materials:

Section <materials>
===================

.. xml:tag:: <materials>

This section contains specification of custom materials that can be used together with the library materials in the structure geometry. The only allowed tag in this section — that can, however, appear more than once — is the tag ``<material>``:

.. xml:tag:: <material>

   Corresponding Python class: :py:class:`Material`.

   Definition of a custom material.

   :attr required name: Name of the material. As all custom materials are simple materials, it can be an arbitrary identifier string. However, it may also contain a doping specification without the doping amount.
   :attr kind: (either **kind** or **base** is required) Kind of the new material. Any of: *semiconductor*, *dielectric*, *oxide*, *metal*, *liquid crystal*.
   :attr base: (either **kind** or **base** is required) Textual specification of the base material. The doping amount information can be skipped from it, in which case the doping amount will have to be specified when the custom material is used.

   :Contents:

   The content of this element is the list of user-defined material properties. Each element of such list is a tag specifying the particular property which content is a mathematical expression computing this property. Each such expression can use several variables: the ones specified below next to each tag and ``dc`` or ``cc`` that will contain the user specified doping amounts: dopant or carriers concentration, respectively (at most one of ``cc`` or ``dc`` is defined, never both).

   Some properties are anisotropic and can have different values for lateral and vertical components. In such case, two separate values may (but do not have to) be defined in the contents of the material property tag and they should be separated with a comma.

   The accepted material properties are as follows:

   .. xml:tag:: <A>

      Monomolecular recombination coefficient coefficient *A* [1/s].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <absb>

      Absorption coefficient *α* [cm\ :sup:`-1`].

      Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

   .. xml:tag:: <ac>

      Hydrostatic deformation potential for the conduction band *a*\ :sub:`c` [eV].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <av>

      Hydrostatic deformation potential for the valence band *a*\ :sub:`v` [eV].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <B>

      Radiative recombination coefficient *B* [m\ :sup:`3`/s].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <b>

      Shear deformation potential *b* [eV].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <C>

      Auger recombination coefficient *C* [m\ :sup:`6`/s].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <c11>

      Elastic constant *c*\ :sub:`11` [GPa].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <c12>

      Elastic constant *c*\ :sub:`12` [GPa].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <CB>

      Conduction band level *CB* [eV].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
      ``point`` — point in the Brillouin zone [-].

   .. xml:tag:: <chi>

      Electron affinity *χ* [eV].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
      ``point`` — point in the Brillouin zone [-].

   .. xml:tag:: <cond>

      Electrical conductivity sigma in-plane (lateral) and cross-plane (vertical) direction [S/m].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <condtype>

      Electrical conductivity type. In semiconductors this indicates what type of carriers :xml:tag:`<Nf>` refers to.

   .. xml:tag:: <cp>

      Specific heat heat at constant pressure [J/(kg K)].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <D>

      Ambipolar diffusion coefficient *D* [m\ :sup:`2`/s].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <dens>

      Density [kg/m\ :sup:`3`].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <Dso>

      Split-off energy *D*\ :sub:`so` [eV].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

   .. xml:tag:: <EactA>

      Acceptor ionization energy *E*\ :sub:`actA` [eV].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <EactD>

      Donor ionization energy *E*\ :sub:`actD` [eV].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <Eg>

      Energy gap *E*\ :sub:`g` [eV].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
      ``point`` — point in the Brillouin zone [-].

   .. xml:tag:: <eps>

      Donor ionization energy *ε*\ :sub:`R` [-].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <lattC>

      Lattice constant [Å].

      Variables: ``T`` — temperature [K], ``x`` — lattice parameter [-].

   .. xml:tag:: <Me>

      Electron effective mass *M*\ :sub:`e` in in-plane (lateral)
      and cross-plane (vertical) direction [*m*\ :sub:`0`].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
      ``point`` — point in the irreducible Brillouin zone [-].

   .. xml:tag:: <Mh>

      Hole effective mass *M*\ :sub:`h` in in-plane (lateral)
      and cross-plane (vertical) direction [*m*\ :sub:`0`].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

   .. xml:tag:: <Mhh>

      Heavy hole effective mass *M*\ :sub:`hh` in in-plane (lateral)
      and cross-plane (vertical) direction [*m*\ :sub:`0`].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

   .. xml:tag:: <Mlh>

      Light hole effective mass *M*\ :sub:`lh` in in-plane (lateral)
      and cross-plane (vertical) direction [*m*\ :sub:`0`].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

   .. xml:tag:: <mob>

      Majority carriers mobility in-plane (lateral) and cross-plane (vertical) direction
      [m\ :sup:`2`/(V s)].

      Variables: T — temperature [K].

   .. xml:tag:: <Mso>

      Split-off mass *M*\ :sub:`so`` [*m*\ :sub:`0`].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

   .. xml:tag:: <Nc>

      Effective density of states in the conduction band *N*\ :sub:`c` [cm\ :sup:`-3`].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
      ``point`` — point in the Brillouin zone [-].

   .. xml:tag:: <Nf>
   
      Free carrier concentration *N* [cm\ :sup:`-3`].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <Ni>

      Intrinsic carrier concentration *N*\ :sub:`i` [cm\ :sup:`-3`].

      Variables: ``T`` — temperature [K].

   .. xml:tag:: <Nr>

      Complex refractive index *n*\ :sub:`R` [-].

      Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

   .. xml:tag:: <nr>

      Real refractive index *n*\ :sub:`R` [-].

      Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

   .. xml:tag:: <Nr-tensor>

      Anisotropic complex refractive index tensor *n*\ :sub:`R` [-].
      Tensor must have the form [ *n*\ :sub:`00`, *n*\ :sub:`11`, *n*\ :sub:`22`, *n*\ :sub:`01`, *n*\ :sub:`10` ].

      Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

   .. xml:tag:: <Nv>

      Effective density of states in the valance band *N*\ :sub:`v` [cm\ :sup:`-3`].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
      ``point`` — point in the Brillouin zone [-].

   .. xml:tag:: <thermk>

      Thermal conductivity in in-plane (lateral) and cross-plane (vertical) direction *k* [W/(m K)].

      Variables: ``T`` — temperature [K], ``h`` — layer thickness [µm].

   .. xml:tag:: <VB>

      Valance band level offset *VB* [eV].

      Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
      ``hole`` — hole type (``'H'`` or ``'L'``) [-].


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


.. _sec-XPL-grids:

Section <grids>
===============

.. xml:tag:: <grids>

In this section one can define computational meshes for use by solvers. It can be done by one of the two ways: either by specifying the mesh directly or, by creating a generator that will automatically construct the required mesh basing on the structure geometry when the calculations in the solver using particular generator are about to begin. Hence the two allowed tags in this section are ``<mesh>`` and ``<generator>``. The contents of these tags are determined by the particular mesh or generator type, while their attributes are always the same:

.. xml:tag:: <generator>

   Specification of the mesh generator.

   :attr required name: Object name for further reference.
   :attr required type: Type of the mesh to generate.
   :attr required method: Generation method i.e. the type of the generator.

   :Contents: The content of this element depends on the values of the type and method tag. It specifies generator configuration (if any). See below for details.

.. xml:tag:: <mesh>

   Specification of the mesh.

   :attr required name: Name of the mesh for reference in configuration of the solvers.
   :attr required type: Type of the mesh.

   :Contents: The content of this element depends on the value of the type tag. See below for details.

Possible <mesh> contents for different types
--------------------------------------------

.. xml:tag:: <mesh type="rectilinear1d"> [rectilinear1d]

   One-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis> [in rectilinear1d mesh]

      Specification of the horizontal axis.

      If any of the following attributes are specified, the points along this axis are equally distributed like in regular meshes. In such a case the contents must be empty.

      :attr start: Position of the first point on the axis. (float [µm])
      :attr stop: Position of the last point on the axis. (float [µm])
      :attr num: Number of the equally distributed points along the axis. (integer)

      :Contents: Comma-separated list of the mesh points along this axis.

.. xml:tag:: <mesh type="rectilinear2d"> [rectilinear2d]

   Two-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis0> [in rectilinear2d mesh]

      Specification of the horizontal axis.

      If any of the following attributes are specified, the points along this axis are equally distributed like in regular meshes. In such a case the contents must be empty.

      :attr start: Position of the first point on the axis. (float [µm])
      :attr stop: Position of the last point on the axis. (float [µm])
      :attr num: Number of the equally distributed points along the axis. (integer)

      :Contents: Comma-separated list of the mesh points along this axis.

   .. xml:tag:: <axis1> [in rectilinear2d mesh]

      Specification of the vertical axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in rectilinear2d mesh]`.

.. xml:tag:: <mesh type="rectilinear3d"> [rectilinear3d]

   Three-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis0> [in rectilinear3d mesh]

      Specification of the longitudinal axis.

      If any of the following attributes are specified, the points along this axis are equally distributed like in regular meshes. In such a case the contents must be empty.

      :attr start: Position of the first point on the axis. (float [µm])
      :attr stop: Position of the last point on the axis. (float [µm])
      :attr num: Number of the equally distributed points along the axis. (integer)

      :Contents: Comma-separated list of the mesh points along this axis.

   .. xml:tag:: <axis1> [in rectilinear3d mesh]

      Specification of the transversal axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in rectilinear3d mesh]`.

   .. xml:tag:: <axis2> [in rectilinear3d mesh]

      Specification of the vertical axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in rectilinear3d mesh]`.

.. xml:tag:: <mesh type="regular1d"> [regular1d]

   One-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis> [in regular1d mesh]

      Specification of the horizontal axis.

      :attr required start: Position of the first point on the axis. (float [µm])
      :attr required stop: Position of the last point on the axis. (float [µm])
      :attr required num: Number of the equally distributed points along the axis. (integer)

.. xml:tag:: <mesh type="regular2d"> [regular2d]

   Two-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis0> [in regular2d mesh]

      Specification of the horizontal axis.

      :attr required start: Position of the first point on the axis. (float [µm])
      :attr required stop: Position of the last point on the axis. (float [µm])
      :attr required num: Number of the equally distributed points along the axis. (integer)

   .. xml:tag:: <axis1> [in regular2d mesh]

      Specification of the vertical axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in regular2d mesh]`.

.. xml:tag:: <mesh type="regular3d">

   Three-dimensional rectangular mesh with regular intervals.

   :Contents:

   .. xml:tag:: <axis0> [in regular3d mesh]

      Specification of the longitudinal axis.

      :attr required start: Position of the first point on the axis. (float [µm])
      :attr required stop: Position of the last point on the axis. (float [µm])
      :attr required num: Number of the equally distributed points along the axis. (integer)

   .. xml:tag:: <axis1> [in regular3d mesh]

      Specification of the transversal axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in regular3d mesh]`.

   .. xml:tag:: <axis2> [in regular3d mesh]

      Specification of the vertical axis.

      Attributes and contents are in the same format as in :xml:tag:`<axis0> [in regular3d mesh]`.


Possible <generator> contents for different types and methods
-------------------------------------------------------------

.. xml:tag:: <generator type="rectilinear1d" method="divide"> [rectilinear1d, divide]

   Generator that divides each geometry object along both axes into a specified number of elements, ensuring that two adjacent do not differ in size more than twice.

   :Contents:

   .. xml:tag:: <no-gradual/> [in rectilinear1d, divide generator]

      Turn off smooth mesh step (i.e. the adjacent elements of the generated mesh may differ more than by the factor of two).

   .. xml:tag:: <prediv/> [in rectilinear1d, divide generator]

      Set number of the initial divisions of each geometry object.

      :attr by: Number of parts each object is divided into along horizontal axis.

   .. xml:tag:: <postdiv/> [in rectilinear1d, divide generator]

      Set number of the final divisions of each geometry object.

      :attr by: Number of parts each object is divided into along horizontal axis.

   .. xml:tag:: <refinements> [in rectilinear1d, divide generator]

      Specify list of additional refinements of the generated mesh.

      :Contents:

      .. xml:tag:: <axis0/> [in rectilinear1d, divide generator]

         Add refinement to the horizontal axis.

         :attr required object: Name of the geometry object to add additional division to.
         :attr path: Path name, specifying particular instance of the object given in the object attribute.
         :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
         :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
         :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

         Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

   .. xml:tag:: <warnings/>

      Control printing of the warnings.

      :attr missing: Warn if any refinement references to non-existing object. Defaults to true. (boolean)
      :attr multiple: Warn if any refinement references to multiple objects. Defaults to true. (boolean)
      :attr outside: Warn if refining line lies outside of the specified object. Defaults to true. (boolean)

.. xml:tag:: <generator type="rectilinear1d" method="simple"> [rectilinear1d, simple]

   Simple generator creating the rectangular rectilinear mesh lines at the edges of bounding box of each object of the geometry. This generator has no configuration.

.. xml:tag:: <generator type="rectilinear2d" method="divide"> [rectilinear1d, divide]

   Generator that divides each geometry object along both axes into a specified number of elements, ensuring that two adjacent do not differ in size more than twice.

   :Contents:

   .. xml:tag:: <no-gradual/> [in rectilinear2d, divide generator]

      Turn off smooth mesh step (i.e. the adjacent elements of the generated mesh may differ more than by the factor of two).

   .. xml:tag:: <prediv/> [in rectilinear2d, divide generator]

      Set number of the initial divisions of each geometry object.

      :attr by0: Number of parts each object is divided into along horizontal axis.
      :attr by1: Number of parts each object is divided into along vertical axis.
      :attr by: Set values of ``by0`` and ``by1`` both at once. It this attribute is specified, no other ones are allowed.

   .. xml:tag:: <postdiv/> [in rectilinear2d, divide generator]

      Set number of the final divisions of each geometry object.

      It has same attributes as :xml:tag:`<prediv/> [in rectilinear2d, divide generator]`.

   .. xml:tag:: <refinements> [in rectilinear2d, divide generator]

      Specify list of additional refinements of the generated mesh.

      :Contents:

      .. xml:tag:: <axis0/> [in rectilinear2d, divide generator]

         Add refinement to the horizontal axis.

         :attr required object: Name of the geometry object to add additional division to.
         :attr path: Path name, specifying particular instance of the object given in the object attribute.
         :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
         :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
         :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

         Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

      .. xml:tag:: <axis1/> [in rectilinear2d, divide generator]

         Add refinement to the vertical axis.

         It has same attributes as :xml:tag:`<axis0/> [in rectilinear2d, divide generator]`.

   .. xml:tag:: <warnings/>

      Control printing of the warnings.

      :attr missing: Warn if any refinement references to non-existing object. Defaults to true. (boolean)
      :attr multiple: Warn if any refinement references to multiple objects. Defaults to true. (boolean)
      :attr outside: Warn if refining line lies outside of the specified object. Defaults to true. (boolean)

.. xml:tag:: <generator type="rectilinear2d" method="simple"> [rectilinear2d, simple]

   Simple generator creating the rectangular rectilinear mesh lines at the edges of bounding box of each object of the geometry. This generator has no configuration.

.. xml:tag:: <generator type=”rectilinear3d” method=”divide”> [rectilinear3d, divide]

   Generator that divides each geometry object along both axes into a specified number of elements, ensuring that two adjacent do not differ in size more than twice.

   :Contents:

   .. xml:tag:: <no-gradual/> [in rectilinear3d, divide generator]

      Turn off smooth mesh step (i.e. the adjacent elements of the generated mesh may differ more than by the factor of two).

   .. xml:tag:: <prediv/> [in rectilinear3d, divide generator]

      Set number of the initial divisions of each geometry object.

      :attr by0: Number of parts each object is divided into along longitudinal axis.
      :attr by1: Number of parts each object is divided into along trnasverse axis.
      :attr by2: Number of parts each object is divided into along vertical axis.
      :attr by: Set values of ``by0``, ``by1`` and ``by2`` at once. It this attribute is specified, no other ones are allowed.

   .. xml:tag:: <postdiv/> [in rectilinear3d, divide generator]

      Set number of the final divisions of each geometry object.

      It has same attributes as :xml:tag:`<prediv/> [in rectilinear3d, divide generator]`.

   .. xml:tag:: <refinements> [in rectilinear3d, divide generator]

      Specify list of additional refinements of the generated mesh.

      :Contents:

      .. xml:tag:: <axis0/> [in rectilinear3d, divide generator]

         Add refinement to the longitudinal axis.

         :attr required object: Name of the geometry object to add additional division to.
         :attr path: Path name, specifying particular instance of the object given in the object attribute.
         :attr at: If this attribute is present, a single refinement line is placed at the position specified in it (in the local object coordinates).
         :attr by: If this attribute is present, multiple refinement lines are placed dividing the object into a specified number of equal parts.
         :attr every: If this attribute is present, multiple refinement lines are places at distance from each other specified in the attribute value.

         Exactly one of ``at``, ``by``, or ``every`` attribute must be present.

      .. xml:tag:: <axis1/> [in rectilinear3d, divide generator]

         Add refinement to the transverse axis.

         It has same attributes as :xml:tag:`<axis0/> [in rectilinear3d, divide generator]`.

      .. xml:tag:: <axis2/> [in rectilinear3d, divide generator]

         Add refinement to the vertical axis.

         It has same attributes as :xml:tag:`<axis0/> [in rectilinear3d, divide generator]`.

   .. xml:tag:: <warnings/>

      Control printing of the warnings.

      :attr missing: Warn if any refinement references to non-existing object. Defaults to true. (boolean)
      :attr multiple: Warn if any refinement references to multiple objects. Defaults to true. (boolean)
      :attr outside: Warn if refining line lies outside of the specified object. Defaults to true. (boolean)

.. xml:tag:: <generator type="rectilinear3d" method="simple"> [rectilinear3d, simple]

   Simple generator creating the rectangular rectilinear mesh lines at the edges of bounding box of each object of the geometry. This generator has no configuration.


.. _sec-XPL-solvers:

Section <solvers>
=================

.. xml:tag:: <solvers>

In this section used computational solvers are defined and configured. Also here, the data filters are set-up, as in general, they are only special kinds of solvers. Each XML element in this section correspond to separate solver. The content of such element depends strongly on particular solver, while its name and attributes are standard (although there are differences in attributes of strict computational solvers and data filters). The details of XML content of this section is presented below.

The computational solvers are declared with an XML tag, which name is the category of the solver, e.g. *thermal*, *electrical*, *gain*, or *optical* and that has the standard set of attributes:

.. xml:tag:: <category> []

   Definition of computational solver.

   :attr required name: Solver name. In Python script there is a automatically created solver object with such name. (identifier string)
   :attr required solver: Actual solver type. In Python script this defines the class of the solver object.
   :attr lib: Library in which this solver is implemented. For most standard solvers, PLaSK can automatically determine its proper value. For other solver types this attribute is required.

   :contents: The contents of each solver depends on the category and the solver type (i.e. the tag name and the value of the solver attribute). It is specified in the following subsections.


.. _sec-XPL-Boundary-conditions:

Boundary conditions
-------------------

Most of the solvers have some boundary conditions. They differ by name or type, but all share the same structure: some value is set at mesh points in some region (usually the edge of the whole geometry, or the edge of some geometrical object. Hence, the structure of boundary conditions is usually the same and looks as follows [#different-boundary-conditions]_:

.. xml:tag:: <boundary_conditions> []

   Some boundary conditions specifications.

   .. xml:tag:: <condition>

      Specification of one boundary condition.

      :attr required value: Value of the boundary condition. In some boundary condition value is given in different attributes. In such case, this attribute should be replaced with the specific ones.
      :attr placename: Name of the boundary condition location for further reference.
      :attr placeref: Set location of boundary conditions to some location previously named with ``placename``.
      :attr place: Set one of standard location of boundary condition. The value of this attribute depends on the mesh.

      :contents:

      .. xml:tag:: <place>

         Set location of boundary condition. This tag can be used instead of the ``place`` attribute if more detailed description of the boundary condition location is required. Its attributes are mesh-specific. Below there are most common examples of attribute sets for rectangular meshes:

         **Boundary conditions at the side of some object:**

         :attr required object: Name of the geometry object to set boundary conditions at.
         :attr path: Path name, specifying particular instance of the object given in the object attribute.
         :attr required side: Side of the object to set boundary conditions at. (``left``, ``right``, ``top``, ``bottom``, ``back``, ``front``)

         **Boundary conditions at some line (2D meshes):**

         :attr required line: Direction of the line. (``vertical`` or ``horizontal``)
         :attr required at: Location of the line i.e. its position on the perpendicular axis.
         :attr required start: Position of the start of the line on the parallel axis.
         :attr required stop: Position of the end of the line on the parallel axis.


Thermal solvers
---------------

.. xml:tag:: <thermal solver="Static2D"> [Static2D]

   Two-dimensional static thermal solver in Cartesian geometry, based on finite-element method.

   :contents:

   .. xml:tag:: <geometry/> [in Static2D thermal solver]

      Geometry for use by this solver.

      :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

   .. xml:tag:: <mesh/> [in Static2D thermal solver]

      Mesh used by this solver.

      :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

   .. xml:tag:: <loop/> [in Static2D thermal solver]

      Configuration of the self-consistent loop.

      :attr inittemp: Initial temperature. (float [K])
      :attr maxerr: Maximum allowed error. (float [K])

   .. xml:tag:: <matrix/> [in Static2D thermal solver]

      Configuration of the matrix solver.

      :attr algorithm: Solution algorithm. Defaults to ``cholesky``. (``cholesky``, ``gauss``, or ``iterative``)
      :attr itererr: Allowed residual error for the iterative algorithm.
      :attr iterlim: Maximum number of iterations for the iterative algorithm.
      :attr logfreq: Frequency of logging iterative solver progress.
.. :attr preconditioner: Preconditioner for the iterative (conjugate gradient) algorithm. (``jacobi`` or ``factor``)

   .. xml:tag:: <temperature> [in Static2D thermal solver]

      Boundary conditions: constant temperature. See subsection :ref:`sec-XPL-Boundary-conditions`.

   .. xml:tag:: <heatflux> [in Static2D thermal solver]

      Boundary conditions: constant heat flux. See subsection :ref:`sec-XPL-Boundary-conditions`.

   .. xml:tag:: <convection> [in Static2D thermal solver]

      Boundary conditions: convection. See subsection :ref:`sec-XPL-Boundary-conditions`.

      This boundary condition does not have ``value`` attribute. Use ``coeff`` for convection coefficient and ``ambient`` for ambient temperature instead.

   .. xml:tag:: <radiation> [in Static2D thermal solver]

      Boundary conditions: radiation. See subsection :ref:`sec-XPL-Boundary-conditions`.

      This boundary condition does not have ``value`` attribute. Use ``emissivity`` for surface emissivity and ``ambient`` for ambient temperature instead.

.. xml:tag:: <thermal solver="StaticCyl"> [StaticCyl]

      Two-dimensional static thermal solver in cylindrical geometry, based on finite-element method.

      :contents: See :xml:tag:`<thermal solver="Static2D"> [Static2D]`.

.. xml:tag:: <thermal solver="Static3D"> [Static3D]

      Three-dimensional static thermal solver, based on finite-element method.

      :contents: See :xml:tag:`<thermal solver="Static2D"> [Static2D]`.


Electrical solvers
------------------

.. xml:tag:: <electrical solver="Shockley2D"> [Shockley2D]

   Two-dimensional phenomenological solver in Cartesian geometry, based on finite-element method.

   :contents:

   .. xml:tag:: <geometry/> [in Shockley2D electrical solver]

      Geometry for use by this solver.

      :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

   .. xml:tag:: <mesh/> [in Shockley2D electrical solver]

      Mesh used by this solver.

      :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

   .. xml:tag:: <loop/> [in Shockley2D electrical solver]

      Configuration of the self-consistent loop.

      :attr maxerr: Maximum allowed error. (float [%])

   .. xml:tag:: <matrix/> [in Shockley2D electrical solver]

      Configuration of the matrix solver.

      :attr algorithm: Solution algorithm. Defaults to ``cholesky``. (``cholesky``, ``gauss``, or ``iterative``)
      :attr itererr: Allowed residual error for the iterative algorithm.
      :attr iterlim: Maximum number of iterations for the iterative algorithm.
      :attr logfreq: Frequency of logging iterative solver progress.
.. :attr preconditioner: Preconditioner for the iterative (conjugate gradient) algorithm. (``jacobi`` or ``factor``)

   .. xml:tag:: <junction/> [in Shockley2D electrical solver]

      Configuration of the effective model of p-n junction.

      :attr js: Reverse bias current density. (float :math:`[A/m^{2}]`)
      :attr Shockley: Junction coefficient.
      :attr pnjcond: Initial vertical conductivity of the junction. (float [S/m])
      :attr heat: Method of determination of the heat generated in the junction. (``joules`` or ``wavelength``)
      :attr wavelength: Emitted wavelength if ``heat`` is set to ``wavelength``.

   .. xml:tag:: <contacts/> [in Shockley2D electrical solver]

      Properties of the contacts.

      :attr pcond: p-contact conductivity. (float [S/m])
      :attr ncond: n-contact conductivity. (float [S/m])

   .. xml:tag:: <voltage> [in Shockley2D electrical solver]

      Boundary conditions: electric potential. See subsection :ref:`sec-XPL-Boundary-conditions`.

.. xml:tag:: <electrical solver="ShockleyCyl"> [ShockleyCyl]

      Two-dimensional phenomenological solver in cylindrical geometry, based on finite-element method.

      :contents: See :xml:tag:`<electrical solver="Shockley2D"> [Shockley2D]`.

.. xml:tag:: <electrical solver="Shockley3D"> [Shockley3D]

      Three-dimensional phenomenological solver in Cartesian geometry, based on finite-element method.

      :contents: See :xml:tag:`<electrical solver="Shockley2D"> [Shockley2D]`.

.. xml:tag:: <electrical solver="Diffusion2D"> [Diffusion2D]

   Two-dimensional diffusion solver in Cartesian geometry.

   :contents:

   .. xml:tag:: <geometry/> [in Diffusion2D electrical solver]

      Geometry for use by this solver.

      :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

   .. xml:tag:: <mesh/> [in Diffusion2D electrical solver]

      One-dimensional horizontal initial mesh used by this solver.

      :attr required start: Position of the first mesh point. (float [µm])
      :attr required stop: Position of the last mesh point. (float [µm])
      :attr required num: Number of the mesh points. (integer)

   .. xml:tag:: <config/> [in Diffusion2D electrical solver]

      :attr fem-method: Order of the finite-element method. (``linear`` or ``parabolic``)
      :attr accuracy: Required relative accuracy. (float [%])
      :attr abs-accuracy: Required absolute minimal concentration accuracy. (float :math:`[cm^{-3}]`)
      :attr interpolation: Current density interpolation method name.
      :attr maxiters: Maximum number of allowed iterations before attempting to refine mesh. (integer)
      :attr maxrefines: Maximum number of allowed mesh refinements. (integer)

.. xml:tag:: <electrical solver="DiffusionCyl"> [DiffusionCyl]

      Two-dimensional diffusion solver in cylindrical geometry.

      :contents: See :xml:tag:`<electrical solver="Diffusion2D"> [Diffusion2D]`.


Gain solvers
------------

.. xml:tag:: <gain solver="Fermi2D"> [Fermi2D]

   Simple gain solver based on Fermi Golden Rule for two-dimensional Cartesian geometry.

   :contents:

   .. xml:tag:: <geometry/> [in Fermi2D gain solver]

      Geometry for use by this solver.

      :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

   .. xml:tag:: <mesh/> [in Fermi2D gain solver]

      Optional mesh used by this solver. If it is set then the gain is computed only in the mesh points and interpolated in-between. Otherwise, the full gain calculation is performed in each requested point.

      :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

   .. xml:tag:: <config/> [in Fermi2D gain solver]

      Configuration of the self-consistent loop.

      :attr lifetime: Carriers lifetime.
      :attr matrix-elem: Value of the matrix element in gain computations (if not set it is estimated automatically).

   .. xml:tag:: <levels/> [in Fermi2D gain solver]

      Custom energy levels in quantum wells. If this tag is used all levels must be set.

      :attr required le: Comma-separated list of electron levels.
      :attr required hh: Comma-separated list of heavy hole levels.
      :attr required lh: Comma-separated list of light hole levels.

.. xml:tag:: <gain solver="FermiCyl"> [FermiCyl]

   Simple gain solver based on Fermi Golden Rule for two-dimensional cylindrical geometry.

   :contents: See :xml:tag:`<gain solver="Fermi2D"> [Fermi2D]`.


Optical solvers
---------------

.. xml:tag:: <optical solver=”EffectiveIndex2D”> [EffectiveIndex2D]

   Scalar optical solver based on effective index method.

   .. xml:tag:: <geometry/> [in EffectiveIndex2D optical solver]

      Geometry for use by this solver.

      :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

   .. xml:tag:: <mesh/> [in EffectiveIndex2D optical solver]

      Mesh used by this solver.

      :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

   .. xml:tag:: <mode> [in EffectiveIndex2D optical solver]

      Mode properties.

      :attr polarization: Light polatization. (``TE`` or ``TM``)
      :attr symmetry: Mode symmetry with respect to vertical symmetry axis (if present). (``none``, ``positive``, or ``negative``)
      :attr wavelength: Light wavelength. (float [nm])

   .. xml:tag:: <root> [in EffectiveIndex2D optical solver]

      Parameters of the global root-finding algorithm.

      :attr tolx: Tolerance on effective index. (float [-])
      :attr tolf-min: Minimum value of the determinant sufficient to assume convergence. (float [a.u.])
      :attr tolf-max: Maximum value of the determinant required to assume convergence. (float [a.u.])
      :attr maxstep: Maximum step in one iteration of root finding. (float [-])
      :attr maxiter: Maximum number of root finding iterations. (integer)

   .. xml:tag:: <stripe-root> [in EffectiveIndex2D optical solver]

      Parameters of root-finding algorithm for one stripe.

      It has same attributes as :xml:tag:`<root> [in EffectiveIndex2D optical solver]`.

   .. xml:tag:: <mirrors> [in EffectiveIndex2D optical solver]

      Mirror losses.

      :attr required R1: Reflectivity of the first mirror. (float [-])
      :attr required R2: Reflectivity of the second mirror. (float [-])

   .. xml:tag:: <outer> [in EffectiveIndex2D optical solver]

      Configuration of handling area outside of the computational domain.

      :attr required distance: Distance from the computational domain boundaries where material for the outermost layer is sampled. (float [µm])

.. xml:tag:: <optical solver=”EffectiveFrequencyCyl”> [EffectiveFrequencyCyl]

   Scalar optical solver based on effective index method.

   .. xml:tag:: <geometry/> [in EffectiveFrequencyCyl optical solver]

      Geometry for use by this solver.

      :attr required ref: Name of the geometry defined in the :xml:tag:`<geometry>` section.

   .. xml:tag:: <mesh/> [in EffectiveFrequencyCyl optical solver]

      Mesh used by this solver.

      :attr required ref: Name of the mesh defined in the :xml:tag:`<grids>` section.

   .. xml:tag:: <mode> [in EffectiveFrequencyCyl optical solver]

      Mode properties.

      :attr lam0: Approximate wavelength. (float [nm])
      :attr k0: Approximate normalized frequency. (float [1/µm])
      :attr emission: Direction of emission, necessary for over-threshold power computations (``top`` or ``bottom``)
      :attr vlam: "vertical wavelength" i.e. the wavelength what would be in the absence of lateral confinement; setting this value helps to find models in very long resonators (float [nm])

      Attributes ``lam0`` and ``k0`` are mutually exclusive.

   .. xml:tag:: <root> [in EffectiveFrequencyCyl optical solver]

      Parameters of the global root-finding algorithm.

      :attr tolx: Tolerance on effective index. (float [-])
      :attr tolf-min: Minimum value of the determinant sufficient to assume convergence. (float [a.u.])
      :attr tolf-max: Maximum value of the determinant required to assume convergence. (float [a.u.])
      :attr maxstep: Maximum step in one iteration of root finding. (float [-])
      :attr maxiter: Maximum number of root finding iterations. (integer)

   .. xml:tag:: <stripe-root> [in EffectiveFrequencyCyl optical solver]

      Parameters of root-finding algorithm for one stripe.

      It has same attributes as :xml:tag:`<root> [in EffectiveFrequencyCyl optical solver]`.

   .. xml:tag:: <outer> [in EffectiveFrequencyCyl optical solver]

      Configuration of handling area outside of the computational domain.

      :attr required distance: Distance from the computational domain boundaries where material for the outermost layer is sampled. (float [µm])


.. _sec-data-filters:

Data filters
------------


.. _sec-XPL-connects:

Section <connects>
==================

.. xml:tag:: <connects>

The purpose of this section is to define the relations between solvers i.e. the connections of providers and receivers. There is only one type of tags allowed in this section:

.. xml:tag:: <connect>

   Connect provider to receiver.

   :attr required out: Provider to connect in the format "solver_name.outProviderName".
   :attr required in: Receiver to connect in the format "solver_name.inReceiverName". If *solver_name* is a :ref:`filter <sec-data-filters>`, this attribute should have form "solver_name[object]" or "solver_name[geometry@path]", where object (optionally specified by *path*) is the geometry in which the provider specified in ``out`` attribute provides data.



.. _sec-XPL-script:

Section <script>
================

.. xml:tag:: <script>

This section contains only Python script that is run to do the computations. No attributes nor other XML tags inside this section are allowed, just the script. You must remember that, as in Python the text indentation matters, the first line of the script must begin in the first column (i. e. it cannot be indented in any way).

In order to be able to easily use ``<`` and ``&`` characters in the script, it is recommended to put its content as XML CDATA element as in the following example:

.. code-block:: xml

   <script><![CDATA[

   if 42 <= 6*9:
       print_log(LOG_RESULT, "Life, the Universe and Everything!")

   ]]></script>


.. rubric:: Footnotes
.. [#XML-tutoruals] Good resources are http://www.w3.org/TR/REC-xml/, http://en.wikipedia.org/wiki/XML, and http://www.w3schools.com/xml/.
.. [#different-boundary-conditions] In some cases where structure of boundary conditions description is different, it is shown in the reference of particular solver.
