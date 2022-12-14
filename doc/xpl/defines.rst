.. _sec-xpl-defines:

Section <defines>
=================

.. xml:tag:: <defines>

This section allows to define some constant parameters (that can be later overridden either in the command line or while reading the XPL file from Python). Each parameter is defined with the only tag allowed in this section:

.. xml:tag:: <define>

   Definition of a parameter for use in the rest of the file. In the :xml:tag:`script` section, the parameter is available by ``DEF`` table, which is indexed by names of parameters.

   :attr required name: Name of the parameter (each name must be unique).
   :attr required value: Value of the parameter. Any valid Python function can be used here, as well as any previously defined parameter.
