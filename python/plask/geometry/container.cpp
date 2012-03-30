#include "geometry.h"
#include "../../util/py_set.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <plask/geometry/container.h>
#include <plask/geometry/stack.h>


namespace plask { namespace python {

DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementContainer, "GeometryElementContainer", "Base class for all "," containers") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementContainer, GeometryElementD<dim>)
        //.def("__getitem__" TODO
        //.def("__len__" TODO
        //.def("__iter__" TODO
        //.add_property("leafs" TODO getLeafsWithTranslations
        //.add_property("leafsBoundingBoxes" TODO
    ;
}




DECLARE_GEOMETRY_ELEMENT_23D(TranslationContainer, "TranslationContainer",
                             "Geometry elements container in which every child has an associated translation vector ("," version)")
{
    GEOMETRY_ELEMENT_23D_DEFAULT(TranslationContainer, GeometryElementContainer<dim>)
        .def("append", &TranslationContainer<dim>::add, (py::arg("child"), py::arg("trans")=Vec<2,double>(0.,0.)),
             "Add new element to the container")
        .def("__contains__", &GeometryElementContainer<dim>::isInSubtree, (py::arg("item")))
        //.def("__deltiem__" TODO
    ;
}


void register_geometry_container_stack();

void register_geometry_container()
{
    init_GeometryElementContainer<2>();
    init_GeometryElementContainer<3>();

    init_TranslationContainer<2>();
    init_TranslationContainer<3>();

    register_geometry_container_stack();
}



}} // namespace plask::python
