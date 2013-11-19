******************
XPL File Reference
******************

:term:`XPL` files follow :term:`XML` specification. Thus all the general rules of creating correct XML files apply top XPL ones as well. Please refer to the external documentation for information on XML syntax and grammar [#XML-tutoruals]_. Details specific to XPL are covered in this chapter.

First of all each XML document must have a parent tag. In XPL files such tag is named ``<plask>``. Thus, the all information in the data file are content of this tag and have to be located between ``<plask>`` and ``</plask>`` tags. Inside there are several sections that can be included in the XPL file: ``<defines>``, ``<materials>``, ``<geometry>``, ``<grids>``, ``<solvers>``, ``<connects>``, and ``<script>``. Each of them is optional, however, if present, they must be specified in the order shown in the previous sentence. Formal specification of each section is presented below.

``<plask>`` section can take an optional attribute ``loglevel``, which value is the name of any valid log level. Any log message with its priority lower than the specified value will not be printed by the logging system. The default value is ``"detail"``.

Instead of the section content, it is possible to use it as a single tag with the attribute ``external``, which has a name of some XPL file as its value. In such a case, the content of the relevant section is read from this external file.


Section <defines>
=================

This section allows to define some constant parameters (that can be later overridden either in the command line or while reading the XPL file from Python). Each parameter is defined with the only tag allowed in this section:

.. xml:tag:: define

    Definition of a parameter for use in the rest of the file.

    :attr required name: Name of the parameter (each name must be unique).
    :attr required value: Value of the parameter. Any valid Python function can be used here, as well as any previously defined parameter.


Section <materials>
===================

This section contains specification of custom materials that can be used together with the library materials in the structure geometry. The only allowed tag in this section — that can, however, appear more than once — is the tag ``<material>``:

.. xml:tag:: material

    Definition of a custom material.

    :attr required name: Name of the material. As all custom materials are simple materials, it can be an arbitrary identifier string. However, it may also contain a doping specification without the doping amount.
    :attr kind: (either **kind** or **base** is required) Kind of the new material. Any of: *semiconductor*, *dielectric*, *oxide*, *metal*, *liquid crystal*.
    :attr base: (either **kind** or **base** is required) Textual specification of the base material. The doping amount information can be skipped from it, in which case the doping amount will have to be specified when the custom material is used.

    :Contents:

    The content of this element is the list of user-defined material properties. Each element of such list is a tag specifying the particular property which content is a mathematical expression computing this property. Each such expression can use several variables: the ones specified below next to each tag and ``dc`` or ``cc`` that will contain the user specified doping amounts: dopant or carriers concentration, respectively (at most one of ``cc`` or ``dc`` is defined, never both).

    Some properties are anisotropic and can have different values for lateral and vertical components. In such case, two separate values may (but do not have to) be defined in the contents of the material property tag and they should be separated with a comma.

    The accepted material properties are as follows:

    .. xml:tag:: A

            Monomolecular recombination coefficient coefficient *A* [1/s].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: absb

            Absorption coefficient *α* [cm\ :sup:`-1`].

            Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

    .. xml:tag:: ac

            Hydrostatic deformation potential for the conduction band *a*\ :sub:`c` [eV].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: av

            Hydrostatic deformation potential for the valence band *a*\ :sub:`v` [eV].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: B

            Radiative recombination coefficient *B* [m\ :sup:`3`/s].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: b

            Radiative recombination coefficient *b* [m\ :sup:`3`/s].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: C

            Auger recombination coefficient *C* [m\ :sup:`6`/s].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: c11

            Elastic constant *c*\ :sub:`11` [GPa].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: c12

            Elastic constant *c*\ :sub:`12` [GPa].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: CB

            Conduction band level *CB* [eV].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the Brillouin zone [-].

    .. xml:tag:: chi

            Electron affinity *χ* [eV].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the Brillouin zone [-].

    .. xml:tag:: cond

            Electrical conductivity sigma in-plane (lateral) and cross-plane (vertical) direction [S/m].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: condtype

            Electrical conductivity type. In semiconductors this indicates what type of carriers Nf refers to.

    .. xml:tag:: cp

            Specific heat heat at constant pressure [J/(kg K)].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: D

            Ambipolar diffusion coefficient *D* [m\ :sup:`2`/s].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: dens

            Density [kg/m\ :sup:`3`].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: Dso

            Split-off energy *D*\ :sub:`so` [eV].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

    .. xml:tag:: EactA

            Acceptor ionization energy *E*\ :sub:`actA` [eV].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: EactD

            Donor ionization energy *E*\ :sub:`actD` [eV].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: Eg

            Energy gap *E*\ :sub:`g` [eV].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the Brillouin zone [-].

    .. xml:tag:: eps

            Donor ionization energy *ε*\ :sub:`R` [-].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: lattC

            Lattice constant [Å].

            Variables: ``T`` — temperature [K], ``x`` — lattice parameter [-].

    .. xml:tag:: Me

            Electron effective mass *M*\ :sub:`e` in in-plane (lateral)
            and cross-plane (vertical) direction [*m*\ :sub:`0`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the irreducible Brillouin zone [-].

    .. xml:tag:: Mh

            Hole effective mass *M*\ :sub:`h` in in-plane (lateral)
            and cross-plane (vertical) direction [*m*\ :sub:`0`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

    .. xml:tag:: Mhh

            Heavy hole effective mass *M*\ :sub:`hh` in in-plane (lateral)
            and cross-plane (vertical) direction [*m*\ :sub:`0`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

    .. xml:tag:: Mlh

            Light hole effective mass *M*\ :sub:`lh` in in-plane (lateral)
            and cross-plane (vertical) direction [*m*\ :sub:`0`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

    .. xml:tag:: mob

            Majority carriers mobility in-plane (lateral) and cross-plane (vertical) direction
            [m\ :sup:`2`/(V s)].

            Variables: T — temperature [K].

    .. xml:tag:: Mso

            Split-off mass *M*\ :sub:`so`` [*m*\ :sub:`0`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

    .. xml:tag:: Nc

            Effective density of states in the conduction band *N*\ :sub:`c` [cm\ :sup:`-3`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the Brillouin zone [-].

    .. xml:tag:: Nf

            Free carrier concentration *N* [cm\ :sup:`-3`].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: Ni

            Intrinsic carrier concentration *N*\ :sub:`i` [cm\ :sup:`-3`].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: Nr

            Complex refractive index *n*\ :sub:`R` [-].

            Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

    .. xml:tag:: nr

            Real refractive index *n*\ :sub:`R` [-].

            Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

    .. xml:tag:: Nr-tensor

            Anisotropic complex refractive index tensor *n*\ :sub:`R` [-].
            Tensor must have the form [ *n*\ :sub:`00`, *n*\ :sub:`11`, *n*\ :sub:`22`, *n*\ :sub:`01`, *n*\ :sub:`10` ].

            Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

    .. xml:tag:: Nv

            Effective density of states in the valance band *N*\ :sub:`v` [cm\ :sup:`-3`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the Brillouin zone [-].

    .. xml:tag:: thermk

            Thermal conductivity in in-plane (lateral) and cross-plane (vertical) direction *k* [W/(m K)].

            Variables: ``T`` — temperature [K], ``h`` — layer thickness [µm].

    .. xml:tag:: VB

            Valance band level offset *VB* [eV].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``hole`` — hole type (``'H'`` or ``'L'``) [-].



Section <geometry>
==================

In this section geometries of the analyze structures are defined. More than one geometry can be specified.

.. xml:tag:: geometry

    Inside each geometry tag there must be a single geometry object: usually it is some container.

    :attr: axes Default value of axes attribute for all geometries defined in this section.

Available elements
^^^^^^^^^^^^^^^^^^

.. xml:tag:: cartesian2d

    Two-dimensional Cartesian geometry.

    :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
    :attr bottom: Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)
    :attr left: Specification of the left border. (any material name, ``mirror``, ``periodic``, or ``extend``)
    :attr name: Geometry name for further reference. (unique identifier string)
    :attr right: Specification of the right border. (any material name, ``mirror``, ``periodic``, or ``extend``)
    :attr top: Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    :Contents: Any object from section :ref:`sec-XPL-Geometry-objects-2D`.


.. xml:tag:: cylindrical2d

    Two-dimensional cylindrical geometry.

    :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).wszystkie możliwości
    :attr bottom: Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)
    :attr inner: Specification of the inner radical border. (any material name, ``mirror``, ``periodic``, or ``extend``)
    :attr length: Longitudinal dimension of the geometry (float [µm]).: Default value is: *+\infty*.
    :attr outer: Specification of the outer radical border. (any material name, ``mirror``, ``periodic``, or ``extend``)
    :attr name: Geometry name for further reference. (unique identifier string)
    :attr top: Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    :Contents: Any object from section :ref:`sec-XPL-Geometry-objects-2D`. If ``length`` was not given, xml:tag:`extrusion` is also accepted.



.. xml:tag:: cartesian3d

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
===================

The following elements are specifying two-dimensional geometry objects for use with 2D geometries. Each of them can have an optional attribute name, which allows to give the name to the object for further reference (either in the geometry specification or in the computational script). The correct value of the object name is unique identifier string.

Containers
^^^^^^^^^^

.. xml:tag:: align

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

    .. xml:tag:: item

        Tag that allows to specify additional item attributes.

        :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
        :attr {alignment}: Any of the stack alignment specification attributes along the axis not specified in the container attributes (``left``, ``right``, ``trancenter``, **X**\ ``center``, **X**, ``top``, ``bottom``, ``vertcenter``, **Y**\ ``center``, **Y**). Specifies alignment of the item in the remaining direction. Defaults to ``left="0"`` or ``bottom="0"``.

        :Contents: A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.


.. xml:tag:: container

    Simple container in which all the items must have explicitly specified position. As the objects in this container may overlap, their order matters: latter items overwrite the former ones.

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


.. xml:tag:: shelf

    Container organizing objects side-by-side to each other, like books on a bookshelf. Items on the shelf are all bottom-aligned. Optionally it is possible to require that all the items have the same height in order to avoid the vertical gaps. However it is possible to insert intentional horizontal gaps to the shelf.

    :attr name: Object name for further reference.
    :attr role: Object role. Important for some solvers.
    :attr flat: The value of this attribute can be either ``true`` of ``false``. It specifies whether all the items in the shelf are required to have the same height (therefore the top edge of the shelf is flat). Defaults to ``true``.

    :Contents:
    
    The content of this element can any number of other two-dimensional geometry object which are organized horizontally adjacent to each other, starting from the left.

    *object*

        :ref:`Two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.
        
    .. xml:tag:: gap
    
        Horizontal gap between two objects. The size of the gap can be specified either as the absolute value in µm or as the total horizontal size of the shelf.
    
        :attr size: Size of the gap. (float [µm])
        :attr total: Total size of the shelf. The gap will adjust automatically. (float [µm])

        Exactly one of the above attributes must be specified and only one ``gap`` in the shelf can have the ``total`` attribute.
        
        
.. xml:tag:: stack

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

    .. xml:tag:: item

        Tag that allows to specify additional item attributes.

        :attr path: Name of a path that can be later on used to distinguish between multiple occurrences of the same object.
        :attr {alignment}: Any of the stack alignment specification attributes (``left``, ``right``, ``trancenter``, **X**\ ``center``, **X**) that overrides the stack default for the particular item.

        :Contents: A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.

    .. xml:tag:: zero

        This tag can appear as stack content only once. If present, it indicates the vertical position of origin of the local coordinate system. Hence, it is an alternative method of specifying ``shift`` value.

.. rubric:: Footnotes
.. [#XML-tutoruals] Good resources are http://www.w3.org/TR/REC-xml/, http://en.wikipedia.org/wiki/XML, and http://www.w3schools.com/xml/.
