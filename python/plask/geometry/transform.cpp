#include <plask/geometry/transform.h>

#include "geometry.h"

namespace plask { namespace python {

DECLARE_GEOMETRY_ELEMENT_23D(Translation, "Translation", "Holds a translated geometry element together with translation vector ("," version)")
{
    typedef py::init< shared_ptr<GeometryElementD<dim>>, const Vec<dim,double>& > init;
    GEOMETRY_ELEMENT_23D(Translation, GeometryElementTransform<dim>, init())
        .def_readwrite("translation", &Translation<dim>::translation, "Translation vector")
    ;
}

void register_geometry_transform()
{
    init_Translation<2>();
    init_Translation<3>();
}

}} // namespace plask::python
