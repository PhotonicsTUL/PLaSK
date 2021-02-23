.. _sec-xpl-Geometry-objects-copy-ref:

Copies and references to geometry objects
-----------------------------------------


.. xml:tag:: <again/>

   This tag can be used to insert any previously defined and named (with the name attribute) two or three dimensional object again in the geometry tree.

   :attr required ref: Name of the referenced object.



.. xml:tag:: <copy>

   Modified copy of any previously defined and named (with the name attribute) two or three dimensional object.

   :attr name: Object name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
   :attr role: Object role. Important for some solvers.
   :attr required from: Name of the source two or three dimensional object to make modified copy of. Usually it is some container that has some other named its items or sub-items.
   :attr steps-num: Maximum number of the mesh steps in each direction the object is divided into if it is non-uniform. Allowed only if ``from`` points to physical object.
   :attr steps-dist: Minimum step size if the object is non-uniform. Allowed only if ``from`` points to physical object.

   .. xml:contents::

      The content of this element contains the tags specifying desired modifications of the source object. The source object remains unchanged, but its copy has alternations described by the following tags:

      .. xml:tag:: <delete/>

         Delete some item or sub-item of the copied object.

         :attr required object: Name of the object to delete.

      .. xml:tag:: <replace/>

         Replace some item or sub-item of the copied object with some other named object specified anywhere earlier in the geometry.

         :attr required object: Name of the object to delete.
         :attr with: Name of the object to replace with. This object does not need to be located in the subtree of the copied object.
         :contents: A new geometry object to replace the original one. Must be specified if and only if the with attribute is not provided.

      .. xml:tag:: <toblock/>

         Replace some item or sub-item of the copied object with uniform block that has dimensions exactly equal to the bounding box of the original element.

         :attr required object: Name of the object to replace with the the solid block.
         :attr required material: Material of the solid block.
         :attr name: Replacing block name for further reference. In the :xml:tag:`script` section, the object is available by ``GEO`` table, which is indexed by names of geometry objects.
         :attr role: Replacing block role. Important for some solvers.

      .. xml:tag:: <simplify-gradients/>

         Simplify all the gradient layers for optical calculations. Each such layer is replaced with two uniform layers providing the same transfer matrix. Using such simplified gradients for optical computations strongly improves the accuracy and speed of optical computations.

         :attr required lam: Wavelength for which simplified gradient is determined.
         :attr linear: Specifies which parameter is linear: refractive index (`nr`) or permittivity (`eps`).
         :attr temp: Temperature at which the refractive indices are retrieved.
         :attr dtemp: Temperature step for approximating temperature dependence of the simplified refractive indices.
         :attr only-role: Only gradients with this role are simplified. If empty, all gradients are simplified.
