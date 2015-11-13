.. _sec-geometry:

**********************
Geometry Specification
**********************

.. _sec-geometry-xpl:

The whole definition of the structure's geometry is based on the geometry tree. Every physical object is defined as a leaf of this tree.  

First one have to define the type of the geometry. There are three possibilities: a two-dimensional Cartesian geometry, a two-dimensional cylindrical geometry or a three-dimensional Cartesian geometry. For attributes of these geometries see section :ref:`sec-xpl-geometry-types`. **Name** is obligatory for further reference. More than one geometry can be specified.

Into newly defined geometry various items can be included:


1. **containers** are branches of the tree which include leafs. I.e. containers are boxes in which physical objects are placed. Containers are:

	* *align*
	
	* *shelf* (only in 2d geometry) 
	
	* *stack*
	
For their attributes see section :ref:`sec-xpl-geometry-2d-containers` or :ref:`sec-xpl-geometry-3d-containers`.)

2. **transforms** - transforms always contain a single geometry object (possibly a container) as their content and perform some transformation of this object. Transforms are: 
	
	* *arrange*
	
	* *clip*
	
	* *flip*
	
	* *intersection*
	
	* *mirror*
	
	* *translation*
	
	* *extrusion* (only in 3d geometry)
	
	* *lattice* (only in 3d geometry)
	
	* *revolution* (only in 3d geometry)
	
For their attributes see section :ref:`sec-xpl-geometry-2d-transforms` or :ref:`sec-xpl-geometry-3d-transforms`.)

3. **physical objects** which are the leafs of the whole geometry tree. Physical geometry objects are representing actual objects having defined shape, dimensions and material. One can define following physical objects for **two-dimensional** geometry:

..

	* Rectangular **block**. Its origin is located at the lower left corner.

	* **Triangle** with one vertex at point (0, 0).

	* **Circle** with centre at point (0, 0).
	
  or these for **three-dimensional** geometry:

	* Rectangular **block**. Its origin is located at the lower back left corner. 

	* **Cylinder** with its base lying in the horizontal plane. Its origin is located at the center of the lower circular base.  

	* **Sphere** with centre at point (0, 0, 0).

For their attributes see section :ref:`sec-xpl-geometry-2d-objects` or :ref:`sec-xpl-geometry-3d-objects`.

Each geometry object can have an optional name for further reference in computational script or further in geometry specification. Each geometry object can therefore be copied - see section :ref:`sec-xpl-geometry-copies`. 

Specifying geometry in XPL file
===============================

See :ref:`sec-xpl-geometry`.

Creating Geometry from Python
=============================

.. _sec-geometry-python:

See section :mod:`plask.geometry`.


Reference of Geometry Objects
=============================

See sections :ref:`sec-xpl-geometry` and :mod:`plask.geometry`.
