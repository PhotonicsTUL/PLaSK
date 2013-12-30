.. _sec-xpl:

******************
XPL File Reference
******************

:term:`XPL` files follow :term:`XML` specification. Thus all the general rules of creating correct XML files apply top XPL ones as well. Please refer to the external documentation for information on XML syntax and grammar [#XML-tutorials]_. Details specific to XPL are covered in this chapter.

First of all each XML document must have a parent tag. In XPL files such tag is named:

.. xml:tag:: <plask>

Thus, the all information in the data file are content of this tag and have to be located between ``<plask>`` and ``</plask>`` tags. Inside there are several sections that can be included in the XPL file: :xml:tag:`<defines>`, :xml:tag:`<materials>`, :xml:tag:`<geometry>`, :xml:tag:`<grids>`, :xml:tag:`<solvers>`, :xml:tag:`<connects>`, and :xml:tag:`<script>`. Each of them is optional, however, if present, they must be specified in the order shown in the previous sentence. Formal specification of each section is presented below.

Instead of the section content, it is possible to use it as a single tag with the attribute ``external``, which has a name of some XPL file as its value. In such a case, the content of the relevant section is read from this external file.

``<plask>`` section can take an optional attribute ``loglevel``, which value is the name of any valid log level. Any log message with its priority lower than the specified value will not be printed by the logging system. The default value is ``"detail"``.

.. toctree::

   xpl_defines
   xpl_materials
   xpl_geometry
   xpl_grids
   xpl_solvers
   xpl_connects
   xpl_script

.. rubric:: Footnotes

.. [#XML-tutorials] Good resources are http://www.w3.org/TR/REC-xml/, http://en.wikipedia.org/wiki/XML, and http://www.w3schools.com/xml/.
