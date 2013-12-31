.. _sec-xpl-script:

Section <script>
================

.. xml:tag:: <script>

This section contains only Python script that is run to do the computations. No attributes nor other XML tags inside this section are allowed, just the script. You must remember that, as in Python the text indentation matters, the first line of the script must begin in the first column (i. e. it cannot be indented in any way).

In order to be able to easily use ``<`` and ``&`` characters in the script, it is recommended to put its content as XML CDATA element as in the following example:

.. code-block:: xml

   <script><![CDATA[

   if 42 <= 6*9:
       print_log(LOG_RESULT, "Life, the Universe and Everything!")

   ]]></script>
