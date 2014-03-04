Data filters
------------

.. xml:tag:: <filter/>

   Filter is a special kind of solver which "solves" the problem using another solvers.

   It calculates its output using input of similar type and changing it in some way,
   for example translating it from one space to another (2D -> 3D, 3D -> 2D, etc.).

   Typically filter has one or more inputs and one output (output provider, named ``out``).

   :attr required name: Solver (filter) name. In Python script there is a automatically created solver object with such name. (identifier string)
   :attr required for: name of property type that this filter will pass (``out`` will provide data of this type), e.g.: ``Temperature``.
   :attr required geometry: Name of the geometry defined in the :xml:tag:`<geometry>` section. Filter will provide data in coordinates of given geometry.

   Some information about connecting filters with solvers are in the :xml:tag:`connects` sections.
