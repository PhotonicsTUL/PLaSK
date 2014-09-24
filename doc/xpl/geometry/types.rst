.. _sec-xpl-geometry-types:

Geometry types
--------------

These elements are are placed directly in the :xml:tag`geometry` section. All other elements must be put inside one of these geometries.

.. xml:tag:: <cartesian2d>

   Corresponding Python class: :py:class:`plask.geometry.Cartesian2D`.

   Two-dimensional Cartesian geometry.

   :attr name: Geometry name for further reference. In the :xml:tag:`script` section, the geometry is available by ``GEO`` table, which is indexed by names of geometry objects. (unique identifier string)
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr length: Longitudinal dimension of the geometry (float [µm]).: Default value is: *+\infty*.
   :attr bottom: Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr left: Specification of the left border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr right: Specification of the right border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr top: Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

   .. xml:contents::

       Any object from section :ref:`sec-xpl-Geometry-objects-2D`. If ``length`` was not given :xml:tag:`extrusion` is also accepted.


.. xml:tag:: <cylindrical2d>

   Corresponding Python class: :py:class:`plask.geometry.Cylindrical2D`.

   Two-dimensional cylindrical geometry.

   :attr name: Geometry name for further reference. In the :xml:tag:`script` section, the geometry is available by ``GEO`` table, which is indexed by names of geometry objects. (unique identifier string)
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr bottom: Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr inner: Specification of the inner radical border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr outer: Specification of the outer radical border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr top: Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

   .. xml:contents::

       Any object from section :ref:`sec-xpl-Geometry-objects-2D`.



.. xml:tag:: <cartesian3d>

   Corresponding Python class: :py:class:`plask.geometry.Cartesian3D`.

   Three-dimensional Cartesian geometry.

   :attr name: Geometry name for further reference. In the :xml:tag:`script` section, the geometry is available by ``GEO`` table, which is indexed by names of geometry objects. (unique identifier string)
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr back: Specification of the back border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr bottom: Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr front: Specification of the front border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr left: Specification of the left border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr right: Specification of the right border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr top: Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)

   .. xml:contents::

       Any object from section :ref:`sec-xpl-Geometry-objects-3D`.
