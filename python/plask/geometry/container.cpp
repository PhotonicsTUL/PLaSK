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

template <int dim>
static auto Container__begin(const GeometryElementContainer<dim>& self) -> decltype(self.getChildrenVector().begin()) {
    return self.getChildrenVector().begin();
}

template <int dim>
static auto Container__end(const GeometryElementContainer<dim>& self) -> decltype(self.getChildrenVector().end()) {
    return self.getChildrenVector().end();
}

template <int dim>
static shared_ptr<GeometryElement> Container__getitem__int(py::object oself, int i) {
    GeometryElementContainer<dim>* self = py::extract<GeometryElementContainer<dim>*>(oself);
    int n = self->getChildrenCount();
    if (i < 0) i = n - i;
    if (i < 0 || i >= n) {
        PyErr_SetString(PyExc_IndexError, format("%1% index %2% out of range (0 <= index < %3%)",
            std::string(py::extract<std::string>(oself.attr("__class__").attr("__name__"))), i, n).c_str());
        throw py::error_already_set();
    }
    return self->getChildAt(i);
}

template <int dim>
static std::set<shared_ptr<GeometryElement>> Container__getitem__hints(const GeometryElementContainer<dim>& self, const PathHints& hints) {
    std::set<shared_ptr<GeometryElement>> result = hints.getChildren(self);
    return result;
}


DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementContainer, "GeometryElementContainer", "Base class for all "," containers") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementContainer, GeometryElementD<dim>)
        .def("__contains__", &Container__contains__<dim>)
        .def("__len__", &GeometryElementD<dim>::getChildrenCount)
        .def("__getitem__", &Container__getitem__int<dim>)
        .def("__getitem__", &Container__getitem__hints<dim>)
        .def("__iter__", py::range(Container__begin<dim>, Container__end<dim>))
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
