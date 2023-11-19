************************
Using Materials in PLaSK
************************

Physical properties of geometrical objects are defined by materials. During geometry definition, you assign a particular material or each object. In general, it should be a real material used in the modelled structure, however, it is not necessary. You can create your own material with any properties you want (see :ref:`sec-custom-materials`).


Getting Materials
=================

PLaSK provides a number of predefined materials. They are stored in the default material database. In GUI, you can simple select them from a list. In Python, you can create them using the :func:`plask.material.get` function. The function takes the name of the material as an argument and returns the material object. For example, to create a material object representing gallium arsenide, you can use the following code::

    geometry_object.material = plask.material.get('GaAs')

Materials can be simple ones or can be alloys with a variable composition. They may also have doping specified.

Naming Conventions
------------------

In PLaSK, names of simple materials are allowed to be arbitrary strings. However, when it comes to alloy names, specific rules apply. They must consist of element names, each accompanied by a composition fraction. The composition fractions for elements within a single group of the periodic table must add up to one. Alternatively, the composition of one element from a group can be omitted, and it will be calculated from the rest of the elements in the group. Valid examples of alloy names include ``Al(0.2)Ga(0.8)As``, ``Al(0.35)GaAs``, ``Al(0.3)In(0.1)GaAs``, and ``Ga(0.8)InN(0.1)As``.

For custom materials, as explained in the sections on custom materials exploration and custom materials in Python, alloy names can include an optional label tag. This tag is placed after the components and separated by an underscore (``_``). For instance, ``Al(0.2)GaAs``, ``Al(0.2)Ga(0.80)As_custom``, and ``Al(0.2)Ga(0.80)As_modified`` represent three distinct materials with the same composition.

Doping in materials can be specified by adding a doping specification after a colon (``:``). The specification is a string containing the doping element and doping concentration, separated by an equal sign. For example, ``GaAs:Si=1e18`` denotes a GaAs material doped with silicon to a concentration of 1×10\ :sup:`18` cm\ :sup:`-3`. It is worth noting that the doping specification can be omitted; in such cases, the material is considered undoped. For example, ``GaAs`` represents an undoped GaAs material.

In any case, the used material name, including doping must either exist in PLasK database or must be defined as a custom material. Otherwise, an error will be raised if you try to use it. For example ``GaAs:Si=1e18`` will be recognized as gallium arsenide doped with silicon to a concentration of 1×10\ :sup:`18` cm\ :sup:`-3`, but ``GaAs:N=1e18`` will raise an error, because GaAs doped with nitride is not defined in PLaSK database. However, you may define it on your own (see :ref:`sec-custom-materials`) and then use it.

Materials with modified properties
----------------------------------

If you need to define a material with one or more properties changed and consider defining a new material an overkill, you may add ``[property1=value1 property2=value2]`` after a material name (with optional space between the material name and ``[``). For example, ``GaAs [Eg=1.5]`` will create a material with the same properties as gallium arsenide, but with the band gap set to 1.5 eV.


Material properties
===================
.. _sec-materials-properties:

Materials are Python objects of class :class:`plask.material.Material`. They have a number of methods that return the material properties. These properties may depend on a number of arguments (e.g. most of them depend on temperature). The following table lists all properties and their arguments.

+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|Parameter         |Arguments                        |Unit             |Description                                                |
+==================+=================================+=================+===========================================================+
|``A``             |``T``                            |1/s              |Monomolecular recombination coefficient                    |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``absp``          |``lam``, ``T``                   |cm\ :sup:`-1`    |Absorption coefficient                                     |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``ac``            |``T``                            |eV               |Hydrostatic deformation potential for the                  |
|                  |                                 |                 |conduction band                                            |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``av``            |``T``                            |eV               |Hydrostatic deformation potential for the                  |
|                  |                                 |                 |valence band                                               |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``B``             |``T``                            |cm\ :sup:`3`/s   |Radiative recombination coefficient                        |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``b``             |``T``                            |eV               |Shear deformation potential                                |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``C``             |``T``                            |cm\ :sup:`6`/s   |Auger recombination coefficient                            |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Ce``            |``T``                            |cm\ :sup:`6`/s   |Auger recombination coefficient for electrons              |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Ch``            |``T``                            |cm\ :sup:`6`/s   |Auger recombination coefficient for holes                  |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``c11``           |``T``                            |GPa              |Elastic constant                                           |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``c12``           |``T``                            |GPa              |Elastic constant                                           |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``c13``           |``T``                            |GPa              |Elastic constant                                           |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``c33``           |``T``                            |GPa              |Elastic constant                                           |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``c44``           |``T``                            |GPa              |Elastic constant                                           |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``CB``            |``T``, ``e``, ``point``          |eV               |Conduction band level                                      |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``chi``           |``T``, ``e``, ``point``          |eV               |Electron affinity χ                                        |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``cond``          |``T``                            |S/m              |Electrical conductivity                                    |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``cp``            |``T``                            |J/(kg K)         |Specific heat at constant pressure                         |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``D``             |``T``                            |cm\ :sup:`2`/s   |Ambipolar diffusion coefficient                            |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``dens``          |``T``                            |kg/m\ :sup:`3`   |Density                                                    |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Dso``           |``T``, ``e``                     |eV               |Split-off energy                                           |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``e13``           |``T``                            |C/m\ :sup:`2`    |Piezoelectric constant                                     |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``e15``           |``T``                            |C/m\ :sup:`2`    |Piezoelectric constant                                     |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``e33``           |``T``                            |C/m\ :sup:`2`    |Piezoelectric constant                                     |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``EactA``         |``T``                            |eV               |Acceptor ionization energy                                 |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``EactD``         |``T``                            |eV               |Donor ionization energy                                    |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Eg``            |``T``, ``e``, ``point``          |eV               |Energy band gap                                            |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``eps``           |``T``                            |                 |Dielectric constant                                        |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``lattC``         |``T``, ``x``                     |Å                |Lattice constant                                           |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Me``            |``T``, ``e``, ``point``          |*m*\ :sub:`0`    |Electron effective mass                                    |
|                  |                                 |                 |in in-plane (lateral) and cross-plane (vertical) direction |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Mh``            |``T``, ``e``                     |*m*\ :sub:`0`    |Hole effective mass                                        |
|                  |                                 |                 |in in-plane (lateral) and cross-plane (vertical) direction |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Mhh``           |``T``, ``e``                     |*m*\ :sub:`0`    |Heavy hole effective mass                                  |
|                  |                                 |                 |in in-plane (lateral) and cross-plane (vertical) direction |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Mhl``           |``T``, ``e``                     |*m*\ :sub:`0`    |Light hole effective mass                                  |
|                  |                                 |                 |in in-plane (lateral) and cross-plane (vertical) direction |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``mob``           |``T``                            |cm\ :sup:`2`/(Vs)|Majority carriers mobilit                                  |
|                  |                                 |                 |in in-plane (lateral) and cross-plane (vertical) direction |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``mobe``          |``T``                            |cm\ :sup:`2`/(Vs)|Electron mobility                                          |
|                  |                                 |                 |in in-plane (lateral) and cross-plane (vertical) direction |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``mobh``          |``T``                            |cm\ :sup:`2`/(Vs)|Hole mobility                                              |
|                  |                                 |                 |in in-plane (lateral) and cross-plane (vertical) direction |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Mso``           |``T``, ``e``                     |*m*\ :sub:`0`    |Split-off mass                                             |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Na``            |                                 |cm\ :sup:`-3`    |Acceptor concentration                                     |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Nc``            |``T``, ``e``, ``point            |cm\ :sup:`-3`    |Effective density of states in the conduction band         |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Nd``            |                                 |cm\ :sup:`-3`    |Donor concentration                                        |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Nf``            |``T``                            |cm\ :sup:`-3`    |Free carrier concentration                                 |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Ni``            |``T``                            |cm\ :sup:`-3`    |Intrinsic carrier concentration                            |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Nr``            |``T``, ``lam``, ``n``            |                 |Complex refractive index                                   |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``nr``            |``T``, ``lam``, ``n``            |                 |Real refractive index                                      |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``NR``            |``T``, ``lam``, ``n``            |                 |Anisotropic complex refractive index tensor                |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Nv``            |``T``, ``e``, ``point``          |cm\ :sup:`-3`    |Effective density of states in the valance band            |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``Psp``           |``T``                            |C/m\ :sup:`2`    |Spontaneous polarization                                   |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``taue``          |``T``                            |ns               |Monomolecular electrons lifetime                           |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``tauh``          |``T``                            |ns               |Monomolecular holes lifetime                               |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``thermk``        |``T``, ``h``                     |W/(m K)          |Thermal conductivity                                       |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``VB``            |``T``, ``e``, ``point``, ``hole``|eV               |Valance band level offset                                  |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``y1``            |                                 |                 |Luttinger parameter                                        |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``y2``            |                                 |                 |Luttinger parameter                                        |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+
|``y3``            |                                 |                 |Luttinger parameter                                        |
+------------------+---------------------------------+-----------------+-----------------------------------------------------------+

The meaning of parameters arguments is as follows:

+----------+-----------------------------------------------------------------------------------------------------+-------------+
|Symbol    |Description                                                                                          |Unit         |
+==========+=====================================================================================================+=============+
|``T``     |Temperature                                                                                          |K            |
+----------+-----------------------------------------------------------------------------------------------------+-------------+
|``lam``   |Wavelength                                                                                           |nm           |
+----------+-----------------------------------------------------------------------------------------------------+-------------+
|``e``     |Lateral strain                                                                                       |             |
+----------+-----------------------------------------------------------------------------------------------------+-------------+
|``x``     |Lattice parameter                                                                                    |             |
+----------+-----------------------------------------------------------------------------------------------------+-------------+
|``point`` |Point in the Brillouin zone. ``'*'`` means minimum bandgap.                                          |             |
+----------+-----------------------------------------------------------------------------------------------------+-------------+
|``hole``  |Hole type. ``'H'`` means heavy hole, ``'L'`` means light hole.                                       |             |
+----------+-----------------------------------------------------------------------------------------------------+-------------+
|``h``     |Layer thickness                                                                                      |µm           |
+----------+-----------------------------------------------------------------------------------------------------+-------------+
|``n``     |Injected carriers concentration                                                                      |cm\ :sup:`-3`|
+----------+-----------------------------------------------------------------------------------------------------+-------------+


Material Parameters Preview in GUI
==================================
