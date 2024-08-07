.. _sec-xpl-geometry-types:

Geometry types
--------------

These elements are are placed directly in the :xml:tag`geometry` section. All other elements must be put inside one of these geometries.

.. xml:tag:: <cartesian2d>

   Corresponding Python class: :py:class:`plask.geometry.Cartesian2D`.

   Two-dimensional Cartesian geometry.

   :attr name: Geometry name for further reference. In the :xml:tag:`script` section, the geometry is available by ``GEO`` table, which is indexed by names of geometry objects. (unique identifier string)
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr length: Longitudinal dimension of the geometry (float, µm).: Default value is: *+\inf*.
   :attr bottom: Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr left: Specification of the left border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr right: Specification of the right border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr top: Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr borders: Specification of all borders. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr planar: Specification of all planar borders, i.e. all borders except top-bottom. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr {axis}-lo: where **{axis}** is the name of axis: Specification of lower border in **{axis}** direction. Alias to ``bottom`` or ``left``. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr {axis}-hi: where **{axis}** is the name of axis: Specification of higher border in **{axis}** direction. Alias to ``top`` or ``right``. (any material name, ``mirror``, ``periodic``, or ``extend``)

   .. xml:contents::

       Any object from section :ref:`sec-xpl-Geometry-objects-2D`. If ``length`` was not given :xml:tag:`extrusion` is also accepted.


.. xml:tag:: <cylindrical2d> (or <cylindrical>)

   Corresponding Python class: :py:class:`plask.geometry.Cylindrical`.

   Two-dimensional cylindrical geometry.

   :attr name: Geometry name for further reference. In the :xml:tag:`script` section, the geometry is available by ``GEO`` table, which is indexed by names of geometry objects. (unique identifier string)
   :attr axes: Specification of the axes. Most popular values are ``xy``, ``yz``, ``rz`` (letters are names of the horizontal and vertical axis, respectively).
   :attr bottom: Specification of the bottom border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr inner: Specification of the inner radical border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr outer: Specification of the outer radical border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr top: Specification of the top border. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr borders: Specification of all borders. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr planar: Specification of all planar borders, i.e. all borders except top-bottom. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr {axis}-lo: where **{axis}** is the name of axis: Specification of lower border in **{axis}** direction. Alias to ``bottom`` or ``inner``. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr {axis}-hi: where **{axis}** is the name of axis: Specification of higher border in **{axis}** direction. Alias to ``top`` or ``outer``. (any material name, ``mirror``, ``periodic``, or ``extend``)

   .. xml:contents::

       Any object from section :ref:`sec-xpl-Geometry-objects-2D` or :xml:tag:`revolution`.



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
   :attr borders: Specification of all borders. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr planar: Specification of all planar borders, i.e. all borders except top-bottom. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr {axis}-lo: where **{axis}** is the name of axis: Specification of lower border in **{axis}** direction. Alias to ``back``, ``bottom`` or ``left``. (any material name, ``mirror``, ``periodic``, or ``extend``)
   :attr {axis}-hi: where **{axis}** is the name of axis: Specification of higher border in **{axis}** direction. Alias to ``front``, ``top`` or ``right``. (any material name, ``mirror``, ``periodic``, or ``extend``)

   .. xml:contents::

       Any object from section :ref:`sec-xpl-Geometry-objects-3D`.
