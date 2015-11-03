.. _sec-geometry:

**********************
Geometry Specification
**********************

Specifying geometry in XPL file
===============================

See :ref:`sec-xpl-geometry`.

  .. _sec-geometry-python:
	
The whole definition of the structure's geometry have to be located between tags ``<geometry>`` and ``</geometry>``. More than one geometry can be specified, however inside each geometry tag there must be a single geometry object. 
There are three different types of geometries:

.. xml:tag:: <cartesian2d>

   It is a two-dimensional Cartesian geometry.

.. xml:tag:: <cylindrical2d> (or <cylindrical>)

   It is a two-dimensional cylindrical geometry.
   
.. xml:tag:: <cartesian3d>

   It is a three-dimensional Cartesian geometry.

For attributes of these geometries see section :ref:`sec-xpl-geometry-types`.

  .. _sec-xpl-geometry-types:

Geometry objects 2D
===================

Possible two-dimensional objects are:

.. toctree::
   :maxdepth: 2

   xpl/geometry/2d/containers
   xpl/geometry/2d/transforms
   xpl/geometry/2d/objects

Geometry objects 3D
===================

Possible three-dimensional objects are:

.. toctree::
   :maxdepth: 2

   xpl/geometry/3d/containers
   xpl/geometry/3d/transforms
   xpl/geometry/3d/objects

Each geometry object can have an optional name for further reference in computational script or further in geometry specification. Each geometry object can therefore be copied - see section :ref:`sec-xpl-geometry-copies`. 

  .. _sec-xpl-Geometry-objects-copy-ref:
  
Physical geometry objects are representing actual objects having defined shape, dimensions and material. One can define following physical objects for **two-dimensional** geometry:

.. xml:tag:: <block2d/> (or <block/>)
  
  Rectangular block. Its origin is located at the lower left corner.
  
.. xml:tag:: <triangle/> 
  
  Triangle with one vertex at point (0, 0).
  
.. xml:tag:: <circle2d/> (or <circle/>)

   Circle with centre at point (0, 0).
 
or these for **three-dimensional** geometry:

.. xml:tag:: <block3D/> (or <block/>)

   Rectangular block. Its origin is located in the lower back left corner. 
   
.. xml:tag:: <cylinder/>

   Cylinder with its base lying in the horizontal plane. Its origin is located at the center of the lower circular base.  
   
.. xml:tag:: <sphere/> (or <circle3d/>)

   Sphere with centre at point (0, 0, 0).
   
Possible attributes for these objects are defined in sections:

.. toctree::
   xpl/geometry/2d/objects
   xpl/geometry/3d/objects

   
Creating Geometry from Python
=============================

See section :mod:`plask.geometry`.


Reference of Geometry Objects
=============================

See sections :ref:`sec-xpl-geometry` and :mod:`plask.geometry`.
