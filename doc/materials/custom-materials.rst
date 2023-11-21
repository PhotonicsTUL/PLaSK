.. _sec-custom-materials:

****************
Custom Materials
****************

Built-in materials database contains some of the most common materials with estimates of their properties. However, for your simulation you should use the real material parameters. If you do not have them, you can use the built-in materials, but you should be aware that the results may be inaccurate. If you have the parameters, you can define your own material. You can do it in two ways: either directly in the XPL file or in Python code.

General rules
=============

When creating a new material, you must first decide whether it is a simple material or an alloy and if it contains a dopant. Then you must name the material accordingly. The name of the material must be unique. If you create a material with the same name as an already existing material, the latter one will be overwritten (however, you may still use it as a base for your custom material). Simple materials can have an arbitrary name. If the material is an alloy, its name must then consist of element names with no composition and an optional custom label after the "``_``" character. For example: ``GaInN``, ``AlGaAs_custom``. For both simple and alloy materials, you may add a dopant name without the doping amount after a colon, e.g. ``myGaAs:Si``.

When you create an alloy or a material with doping, you will be able to specify the composition or doping amount when using your material in the geometry. This way, you can reuse your custom definition many times with different parameters. For example, if you have defined an alloy ``AlGaAs_custom:Si``, you retrieve should apply it to your geometry as e.g. ``Al(0.2)GaAs_custom:Si=5e18``.

Every custom material has a *base*. This is an already existing material (or its template) that you want to modify. Every property that you do not overwrite, will be taken form the base. As a base you can always put a complete material name, the same you cat get e.g. with ``plask.material.get(...)`` function. If your material is an alloy, its base may by an alloy name with the same elements and without composition. In this case, the composition of the base will be taken as the composition of your material. For example, if you create an alloy material ``AlGaAs_custom`` with base ``AlGaAs``, the composition of a particular instance of your material and of the base will match. Similarly, if your material and its base are both doped with the same dopant, you may skip the doping amount in the base definition and it will be taken from the particular instance of your material.


Materials in the XPL File
=========================

The first way is to define a custom material is directly in the XPL file. You can do it by selecting a *Materials* tab in GUI or by adding a ``material`` section to the XPL file. There, you may create either a simple material or an alloy with optional dopant. Please refer to sec-xpl-materials_ for details how to specify custom materials manually in the XPL file.

In GUI, you can simple add new materials to the left-hand-size table in the *Materials* tab, specify its name, base and decide whether the material is an alloy (by selecting the checkbox in the "``(…)``" column). Then in the righ-hand-side table, you can add your custom properties. For each od them, you need to write a proper Python expression that evaluates to the value of the property. Each such expression can use several parameters: the ones shown in the last column of the table and ``self``, which refers to the material itself and allows to access its doping (``self.doping``) and composition in case of alloys (e.g. ``self.Ga`` for the amount of gallium). You may also access the parameters of base materials using ``super()`` function (e.g. ``super().thermk(T)``).


Custom Material Class in Python
===============================
.. _sec-custom-materials-python:

The second way is to define a custom material class in Python. This is more flexible and allows you to use all the features of Python language. You can do it by creating a Python module with definitions of your materials. Then, you may load this material in your XPL file by adding

.. code-block:: xml

    <module name="module_name"/>

to the ``<materials>`` section or by selectinf *Add Module* in GUI.



If custom __init__ ALWAYS call superclass __init__.
