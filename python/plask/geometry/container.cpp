#include "geometry.h"
#include "../../util/py_set.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <plask/geometry/container.h>


namespace plask { namespace python {

template <int dim>
static bool Container__contains__(const GeometryElementContainer<dim>& self, shared_ptr<typename GeometryElementContainer<dim>::ChildType> child) {
    for (auto trans: self.getChildrenVector()) {
        if (trans->getChild() == child) return true;
    }
    return false;
}

DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementContainer, "GeometryElementContainer", "Base class for all "," containers") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementContainer, GeometryElementD<dim>)
        .def("__contains__", &Container__contains__<dim>)
        .def("__len__", &GeometryElementD<dim>::getChildCount)
        //.def("__getitem__" TODO
        //.def("__iter__" TODO
        //.def("__delitem__" TODO
    ;
}




DECLARE_GEOMETRY_ELEMENT_23D(TranslationContainer, "TranslationContainer",
                             "Geometry elements container in which every child has an associated translation vector ("," version)")
{
    GEOMETRY_ELEMENT_23D_DEFAULT(TranslationContainer, GeometryElementContainer<dim>)
        .def("append", &TranslationContainer<dim>::add, (py::arg("child"), py::arg("trans")=Vec<2,double>(0.,0.)),
             "Add new element to the container")
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
