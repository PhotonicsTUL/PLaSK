.. _sec-XPL-defines:

Section <defines>
=================

This section allows to define some constant parameters (that can be later overridden either in the command line or while reading the XPL file from Python). Each parameter is defined with the only tag allowed in this section:

.. xml:tag:: <defines>

   Definition of a parameter for use in the rest of the file.

   :attr required name: Name of the parameter (each name must be unique).
   :attr required value: Value of the parameter. Any valid Python function can be used here, as well as any previously defined parameter.
