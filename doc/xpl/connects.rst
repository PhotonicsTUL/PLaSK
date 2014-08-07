.. _sec-xpl-connects:

Section <connects>
==================

.. xml:tag:: <connects>

The purpose of this section is to define the relations between solvers i.e. the connections of providers and receivers. There is only one type of tags allowed in this section:

.. xml:tag:: <connect>

   Connect provider to receiver.

   :attr required out: Provider to connect in the format "solver_name.outProviderName" (or "filter_name.out").
   :attr required in: Receiver to connect in the format "solver_name.inReceiverName". If *solver_name* is a :ref:`filter <sec-solvers-filters>`, this attribute should have form ``"solver_name[object]"`` or ``"solver_name[object@path]"``, where object (optionally specified by *path*) is the geometry in which the provider specified in ``out`` attribute provides data.

   .. rubric:: Example

   .. code-block:: xml

      <connect out="therm.outTemperature" in="electr.inTemperature"/>
      <connect out="therm.outTemperature" in="filter[geometry_object]"/>
      <connect out="filter.out" in="other.inTemperature"/>
