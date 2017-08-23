.. _sec-xpl-solvers:

Section <solvers>
=================

.. xml:tag:: <solvers>

This section contains definitions and configuation of the computational solvers and :ref:`data filters <sec-solvers-filters>` used in the computations. Each such a solver is declared in the separate XML element of this section. The content of such element depends strongly on the particular solver, while its name and attributes are standarized (although there are differences in attributes of strict computational solvers and :ref:`data filters <sec-solvers-filters>`). The details of this standard representation of XML elements are presented below.

The computational solvers are declared with an XML tag, which name is the category of the solver, e.g. *thermal*, *electrical*, *gain*, or *optical* and that has the standard set of attributes:

.. xml:tag:: <category> []

   Definition of computational solver.

   :attr required name: Solver name. In Python script there is a automatically created solver object with such name. (identifier string)
   :attr required solver: Actual solver type. In Python script this defines the class of the solver object.
   :attr lib: Library in which this solver is implemented. For most standard solvers, PLaSK can automatically determine its proper value. For other solver types this attribute is required.

   .. xml.. xml:contents::

       The contents of each solver depends on the category and the solver type (i.e. the tag name and the value of the solver attribute). It is specified in the following subsections.

Most solvers need to have boundary conditions specified in a some way. This is always done in a consistent method described below:

.. toctree::

   solvers/boundary_conditions

The available solvers are as follows:

.. toctree::

   solvers/thermal
   solvers/electrical
   solvers/gain
   solvers/optical
   solvers/meta
   solvers/filters
