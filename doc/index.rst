.. PLaSK documentation master file, created by
   sphinx-quickstart on Tue Oct  8 15:58:59 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************************
Welcome to PLaSK's documentation!
*********************************

.. toctree::
   :maxdepth: 2

.. contents:: Table of Contents
   :depth: 3

.. highlight:: python



****************
PLaSK User Guide
****************

.. include:: preface.rst

.. include:: introduction.rst



Tutorials and Examples
======================

.. include:: tutorial1.rst

.. include:: tutorial2.rst


Finding threshold current of gallium-arsenide VCSEL
---------------------------------------------------

putting all the above together

manual refinements of divide mesh generator

diffusion, gain

real-time structure modifications


Searching threshold of a gallium-nitride array
----------------------------------------------

multiple geometries

repeating objects

paths

filters


Optical analysis of photonic crystal slab waveguide
---------------------------------------------------

3D

variables in XPL file

lattice


Finding threshold current of photonic crystal VCSEL
---------------------------------------------------

combining 2D and 3D geometries with filters


Using high-level algorithms
---------------------------




*************************
General Concepts of PLaSK
*************************

Definitions
===========

.. _sec-geometry:
**********************
Geometry Specification
**********************

Specifying geometry in XPL file
===============================

Creating geometry from Python
=============================

Reference of geometry objects
=============================
See :ref:`Python API for Geometry`


.. _sec-materials:
******************
Managing Materials
******************

Materials in PLaSK
==================

Compiled material libraries
===========================

Custom material class in Python
===============================

If custom __init__ ALWAYS call superclass __init__.


Custom materials in XPL
=======================

.. _sec-meshes:
*****************
Specifying Meshes
*****************

Concept of meshes and generators
================================

Rectangular meshes
==================

Triangular meshes
=================

Other meshes
============


.. _sec-solvers:
*********************
Computational Solvers
*********************

Solvers overview
================

Specifying solvers in XPL file
==============================

Creating and managing solvers from Python
=========================================

Running computations
====================


*********************
High-level algorithms
*********************


**********************
Other Useful Utilities
**********************

Creating plots of the results
=============================

Storing and retrieving results
==============================


************************
Graphical User Interface
************************

Thermal model
=============

Electrical model
================

.. _sec-Solver-electrical-beta:
Phenomenological junction :math:`\beta` model
=============================================

Gain model
==========

Optical model
=============

Strain model
============

Dynamic models
==============


.. _sec-units-in-plask:
**************
Units in PLaSK
**************

.. include:: XPL.rst



****************
PLaSK Python API
****************

.. _sec-API-geometry
geometry
========
.. automodule:: plask.geometry
   :members:

mesh
====
.. automodule:: plask.mesh
   :members:

pylab
=====
.. automodule:: plask.pylab
   :members: plot_field, plot_vectors, plot_stream, plot_geometry, plot_mesh, plot_boundary, plot_material_param

hdf5
====
.. automodule:: plask.hdf5
   :members:

********
Glossary
********
.. glossary::

    Python
        	Python is a remarkably powerful dynamic programming language that is used in a wide variety of application domains. See: http://python.org/

    shell
        See: http://en.wikipedia.org/wiki/Command-line_interface

    XPL
        Standard extension of files that are used by PLaSK, a shortcut from "**X**\ ML in **PL**\ aSK"

    XML
        Extensible Markup Language (XML) is a markup language that defines a set of rules for encoding documents in a format that is both human-readable and machine-readable.

        In XML every element is described by *tags*, which are denoted by ``<…>`` brackets. Tags always have some name and may optionally contain some attributes and content. Attributes are always put together with the tag name inside the ``<…>`` brackets, i.e. ``<tag attr1="value1" attr2="value2">``. On the other hand, the tag content is always put after the brackets and must end with the ``</tag>`` element. Inside the tag content you may put text or some other tags, depending on the kind of input you want (and more importantly may) to enter. If a tag does not have any content, this must be indicated by putting the ``/`` character before the closing bracket (i.e. ``<tag/>``). Tag attributes are still allowed in such a case.

        See: http://en.wikipedia.org/wiki/XML
.. :ref:`short description of XML in Tutorial <desc-XML>` and

    HDF5
        HDF5 is a data model, library, and file format for storing and managing data. See: http://www.hdfgroup.org/HDF5/

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

