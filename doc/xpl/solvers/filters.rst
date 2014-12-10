Data filters
------------

.. xml:tag:: <filter/>

   Filter is a special kind of solver which "solves" the problem using another solvers.

   It calculates its output using input of similar type and changing it in some way,
   for example translating it from one space to another (2D -> 3D, 3D -> 2D, etc.).

   Filter has one or more inputs and one output (output provider, named ``out``).

   :attr required name: Solver (filter) name. In Python script there is a automatically created solver object with such name. (identifier string)
   :attr required for: name of property type that this filter will pass (``out`` will provide data of this type), e.g.: ``Temperature``.
   :attr required geometry: Name of the geometry defined in the :xml:tag:`<geometry>` section. Filter will provide data in coordinates of given geometry.

   Filter can transfer data between two geometry objects only if the first is successor of the second in the geometry graph (first is in subtree with rooted with second). In case of geometries, filter can transfer data between two geometries only if main object of the first geometry is successor of the second geometry. By a main object of a geometry we mean:

   - in case of cartesian 3D geometry: a child of the geometry (3D object);
   - in case of cartesian 2D geometry, depending on context: a child of the geometry (extrusion) or a child of this child (2D object);
   - in case of cylindrical geometry, depending on context: a child of the geometry (revolution) or a child of this child (2D object).

   Filter supposes that each input solver provides data in whole bounding box of geometry which is used by this solver (we mean 2D box in case of 2D geometry).
   If bounding boxes of two input geometries overlap after transformation to output geometry, then source of filter data is undetermined in points belonged to overlapped fragment.

   Some information about connecting filters with solvers are in the :xml:tag:`connects` sections.
