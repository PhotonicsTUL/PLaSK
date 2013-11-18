******************
XPL File Reference
******************

:term:`XPL` files follow :term:`XML` specification. Thus all the general rules of creating correct XML files apply top XPL ones as well. Please refer to the external documentation for information on XML syntax and grammar [#XML-tutoruals]_. Details specific to XPL are covered in this chapter.

First of all each XML document must have a parent tag. In XPL files such tag is named ``<plask>``. Thus, the all information in the data file are content of this tag and have to be located between ``<plask>`` and ``</plask>`` tags. Inside there are several sections that can be included in the XPL file: ``<defines>``, ``<materials>``, ``<geometry>``, ``<grids>``, ``<solvers>``, ``<connects>``, and ``<script>``. Each of them is optional, however, if present, they must be specified in the order shown in the previous sentence. Formal specification of each section is presented below.

``<plask>`` section can take an optional attribute ``loglevel``, which value is the name of any valid log level. Any log message with its priority lower than the specified value will not be printed by the logging system. The default value is ``"detail"``.

Instead of the section content, it is possible to use it as a single tag with the attribute ``external``, which has a name of some XPL file as its value. In such a case, the content of the relevant section is read from this external file.


Section <defines>
=================

This section allows to define some constant parameters (that can be later overridden either in the command line or while reading the XPL file from Python). Each parameter is defined with the only tag allowed in this section:

.. xml:tag:: define

    Definition of a parameter for use in the rest of the file.

    .. rubric:: Attributes:

    .. object:: name (required)

        Name of the parameter (each name must be unique).

    .. object:: value (required)

        Value of the parameter. Any valid Python function can be used here, as well as any previously defined parameter.


Section <materials>
===================

This section contains specification of custom materials that can be used together with the library materials in the structure geometry. The only allowed tag in this section — that can, however, appear more than once — is the tag ``<material>``:

.. xml:tag:: material

    Definition of a custom material.

    .. rubric:: Attributes:

    .. object:: name (required)

        Name of the material. As all custom materials are simple materials, it can be an arbitrary identifier string. However, it may also contain a doping specification without the doping amount.

    .. object:: kind (either kind or base is required)

        Kind of the new material. Any of: *semiconductor*, *dielectric*, *oxide*, *metal*, *liquid crystal*.

    .. object:: base (either kind or base is required)

        Textual specification of the base material. The doping amount information can be skipped from it, in which case the doping amount will have to be specified when the custom material is used.

    .. rubric:: Content:

    The content of this element is the list of user-defined material properties. Each element of such list is a tag specifying the particular property which content is a mathematical expression computing this property. Each such expression can use several variables: the ones specified below next to each tag and ``dc`` or ``cc`` that will contain the user specified doping amounts: dopant or carriers concentration, respectively (at most one of ``cc`` or ``dc`` is defined, never both).

    Some properties are anisotropic and can have different values for lateral and vertical components. In such case, two separate values may (but do not have to) be defined in the contents of the material property tag and they should be separated with a comma.

    The accepted material properties are as follows:

    .. xml:tag:: A

            Monomolecular recombination coefficient coefficient *A* [1/s].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: absb

            Absorption coefficient *α* [cm\ :sup:`-1`].

            Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

    .. xml:tag:: ac

            Hydrostatic deformation potential for the conduction band *a*\ :sub:`c` [eV].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: av

            Hydrostatic deformation potential for the valence band *a*\ :sub:`v` [eV].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: B

            Radiative recombination coefficient *B* [m\ :sup:`3`/s].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: b

            Radiative recombination coefficient *b* [m\ :sup:`3`/s].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: C

            Auger recombination coefficient *C* [m\ :sup:`6`/s].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: c11

            Elastic constant *c*\ :sub:`11` [GPa].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: c12

            Elastic constant *c*\ :sub:`12` [GPa].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: CB

            Conduction band level *CB* [eV].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the Brillouin zone [-].

    .. xml:tag:: chi

            Electron affinity *χ* [eV].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the Brillouin zone [-].

    .. xml:tag:: cond

            Electrical conductivity sigma in-plane (lateral) and cross-plane (vertical) direction [S/m].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: condtype

            Electrical conductivity type. In semiconductors this indicates what type of carriers Nf refers to.

    .. xml:tag:: cp

            Specific heat heat at constant pressure [J/(kg K)].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: D

            Ambipolar diffusion coefficient *D* [m\ :sup:`2`/s].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: dens

            Density [kg/m\ :sup:`3`].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: Dso

            Split-off energy *D*\ :sub:`so` [eV].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

    .. xml:tag:: EactA

            Acceptor ionization energy *E*\ :sub:`actA` [eV].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: EactD

            Donor ionization energy *E*\ :sub:`actD` [eV].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: Eg

            Energy gap *E*\ :sub:`g` [eV].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the Brillouin zone [-].

    .. xml:tag:: eps

            Donor ionization energy *ε*\ :sub:`R` [-].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: lattC

            Lattice constant [Å].

            Variables: ``T`` — temperature [K], ``x`` — lattice parameter [-].

    .. xml:tag:: Me

            Electron effective mass *M*\ :sub:`e` in in-plane (lateral)
            and cross-plane (vertical) direction [*m*\ :sub:`0`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the irreducible Brillouin zone [-].

    .. xml:tag:: Mh

            Hole effective mass *M*\ :sub:`h` in in-plane (lateral)
            and cross-plane (vertical) direction [*m*\ :sub:`0`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

    .. xml:tag:: Mhh

            Heavy hole effective mass *M*\ :sub:`hh` in in-plane (lateral)
            and cross-plane (vertical) direction [*m*\ :sub:`0`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

    .. xml:tag:: Mlh

            Light hole effective mass *M*\ :sub:`lh` in in-plane (lateral)
            and cross-plane (vertical) direction [*m*\ :sub:`0`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

    .. xml:tag:: mob

            Majority carriers mobility in-plane (lateral) and cross-plane (vertical) direction
            [m\ :sup:`2`/(V s)].

            Variables: T — temperature [K].

    .. xml:tag:: Mso

            Split-off mass *M*\ :sub:`so`` [*m*\ :sub:`0`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-].

    .. xml:tag:: Nc

            Effective density of states in the conduction band *N*\ :sub:`c` [cm\ :sup:`-3`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the Brillouin zone [-].

    .. xml:tag:: Nf

            Free carrier concentration *N* [cm\ :sup:`-3`].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: Ni

            Intrinsic carrier concentration *N*\ :sub:`i` [cm\ :sup:`-3`].

            Variables: ``T`` — temperature [K].

    .. xml:tag:: Nr

            Complex refractive index *n*\ :sub:`R` [-].

            Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

    .. xml:tag:: nr

            Real refractive index *n*\ :sub:`R` [-].

            Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

    .. xml:tag:: Nr-tensor

            Anisotropic complex refractive index tensor *n*\ :sub:`R` [-].
            Tensor must have the form [ *n*\ :sub:`00`, *n*\ :sub:`11`, *n*\ :sub:`22`, *n*\ :sub:`01`, *n*\ :sub:`10` ].

            Variables: ``wl`` — wavelength [nm], ``T`` — temperature [K].

    .. xml:tag:: Nv

            Effective density of states in the valance band *N*\ :sub:`v` [cm\ :sup:`-3`].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``point`` — point in the Brillouin zone [-].

    .. xml:tag:: thermk

            Thermal conductivity in in-plane (lateral) and cross-plane (vertical) direction *k* [W/(m K)].

            Variables: ``T`` — temperature [K], ``h`` — layer thickness [µm].

    .. xml:tag:: VB

            Valance band level offset *VB* [eV].

            Variables: ``T`` — temperature [K], ``e`` — lateral strain [-],
            ``hole`` — hole type (``'H'`` or ``'L'``) [-].



Section <geometry>
==================

In this section geometries of the analyze structures are defined. More than one geometry can be specified. Inside each geometry tag there must be a single geometry object: usually it is some container.

.. rubric:: Attributes:

.. object:: axes

    Default value of axes attribute for all geometries defined in this section.

Available elements
^^^^^^^^^^^^^^^^^^

.. xml:tag:: cartesian2d

    Two-dimensional Cartesian geometry.

    .. rubric:: Attributes:

    .. object:: axes

        Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).

.. TODO wszystkie możliwości

    .. object:: bottom

        Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: left

        Specification of the left border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: name

        Geometry name for further reference. (unique identifier string)

    .. object:: right

        Specification of the right border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: top

        Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. rubric:: Contents:

    Any object from section :ref:`sec-XPL-Geometry-objects-2D`.


.. xml:tag:: cylindrical2d

    Two-dimensional cylindrical geometry.

    .. rubric:: Attributes:

    .. object:: axes

        Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).wszystkie możliwości

    .. object:: bottom

        Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: inner

        Specification of the inner radical border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: length

        Longitudinal dimension of the geometry (float [µm]).

        Default value is: *+\infty*.

    .. object:: outer

        Specification of the outer radical border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: name

        Geometry name for further reference. (unique identifier string)

    .. object:: top

        Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. rubric:: Contents:

    Any object from section :ref:`sec-XPL-Geometry-objects-2D`. If ``length`` was not given, ref:`tag-extrusion` is also accepted.



.. xml:tag:: cartesian3d

    Three-dimensional Cartesian geometry.

    .. rubric:: Attributes:

    .. object:: axes

        Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).

    .. object:: back

        Specification of the back border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: bottom

        Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: front

        Specification of the front border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: left

        Specification of the left border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: name

        Geometry name for further reference. (unique identifier string)

    .. object:: right

        Specification of the right border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. object:: top

        Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

    .. rubric:: Contents:
    Any object from section :ref:`sec-XPL-Geometry-objects-3D`.



.. _sec-XPL-Geometry-objects-2D:

Geometry objects 2D
===================

The following elements are specifying two-dimensional geometry objects for use with 2D geometries. Each of them can have an optional attribute name, which allows to give the name to the object for further reference (either in the geometry specification or in the computational script). The correct value of the object name is unique identifier string.

Containers
^^^^^^^^^^

.. object:: <align>

    Container that align its items according to specified rules specified in its attributes. The alignment for one axis only should be given. As the objects in this container usually overlap, their order matters: latter items overwrite the former ones.

    .. rubric:: Attributes:

    .. object:: name

        Object name for further reference.

    .. object:: role

        Object role. Important for some solvers.

    .. object:: left

        Horizontal alignment specification: position of the left edge of the bounding box of each element. (float [µm])

    .. object:: right

        Horizontal alignment specification: position of the right edge of the bounding box of each element. (float [µm])

    .. object:: trancenter

        Horizontal alignment specification: position of the center of the bounding box of each element. (float [µm])

    .. object:: Xcenter

        (where *X* is the transverse axis name)

        Alias for ``trancenter``.

    .. object:: X

        (where *X* is the transverse axis name)

        Horizontal alignment specification: position of the origin of each element. (float [µm])

    .. object:: top

        Vertical alignment specification: position of the top edge of the bounding box of each element. (float [µm])

    .. object:: bottom

        Vertical alignment specification: position of the bottom edge of the bounding box of each element. (float [µm])

    .. object:: vertcenter

        Vertical alignment specification: position of the center of the bounding box of each element. (float [µm])

    .. object:: Ycenter

        (where *Y* is the vertical axis name)

        Alias for *vertcenter*.

    .. object:: Y

        (where *Y* is the vertical axis name)

        Vertical alignment specification: position of the origin of each element. (float [µm])

    Exactly one of the ``left``, ``right``, ``trancenter``, ``Xcenter``, ``X``, ``top``, ``bottom``, ``vertcenter``, ``Ycenter``, and ``Y`` attributes must be given.

    .. rubric:: Contents:
    The content of this element can any number of other two-dimensional geometry object or ``<item>`` elements which are organized in the vertical stack, ordered from top to bottom. 

    *object*

        :ref:`Two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.

    .. object:: <item>

        Tag that allows to specify additional item attributes.

        .. rubric:: Attributes:

        .. object:: path

            Name of a path that can be later on used to distinguish between multiple occurrences of the same object.

        *alignment*

            Any of the stack alignment specification attributes along the axis not specified in the container attributes (``left``, ``right``, ``trancenter``, *X*\ ``center``, *X*, ``top``, ``bottom``, ``vertcenter``, *Y*\ ``center``, *Y*). Specifies alignment of the item in the remaining direction. Defaults to ``left="0"`` or ``bottom="0"``.

        .. rubric:: Contents:
        A single :ref:`two-dimensional geometry object <sec-XPL-Geometry-objects-2D>`.


.. rubric:: Footnotes
.. [#XML-tutoruals] Good resources are http://www.w3.org/TR/REC-xml/, http://en.wikipedia.org/wiki/XML, and http://www.w3schools.com/xml/.
