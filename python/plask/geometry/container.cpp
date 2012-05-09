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
    if (i < 0) i = n + i;
    if (i < 0 || i >= n) {
        throw IndexError("%1% index %2% out of range (0 <= index < %3%)",
            std::string(py::extract<std::string>(oself.attr("__class__").attr("__name__"))), i, n);
    }
    return self->getChildAt(i);
}

template <int dim>
static std::set<shared_ptr<GeometryElement>> Container__getitem__hints(const GeometryElementContainer<dim>& self, const PathHints& hints) {
    std::set<shared_ptr<GeometryElement>> result = hints.getChildren(self);
    return result;
}

template <int dim>
static void Container__delitem__(GeometryElementContainer<dim>& self, py::object item) {
    try {
        int i = py::extract<int>(item);
        if (i < 0) i = self.getRealChildrenCount() + i;
        self.removeAt(i);
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    try {
        PathHints::Hint* hint = py::extract<PathHints::Hint*>(item);
        self.remove(PathHints(*hint));
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    try {
        PathHints* hints = py::extract<PathHints*>(item);
        self.remove(*hints);
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    try {
        shared_ptr<typename GeometryElementContainer<dim>::TranslationT> child = py::extract<shared_ptr<typename GeometryElementContainer<dim>::TranslationT>>(item);
        self.removeT(child);
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    try {
        shared_ptr<typename GeometryElementContainer<dim>::ChildType> child = py::extract<shared_ptr<typename GeometryElementContainer<dim>::ChildType>>(item);
        self.remove(child);
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    throw TypeError("unrecognized element %s delete from container", std::string(py::extract<std::string>(py::str(item))));
}


DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementContainer, "GeometryElementContainer", "Base class for all "," containers") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementContainer, GeometryElementD<dim>)
        .def("__contains__", &Container__contains__<dim>)
        .def("__getitem__", &Container__getitem__int<dim>)
        .def("__getitem__", &Container__getitem__hints<dim>)
        .def("__len__", &GeometryElementD<dim>::getChildrenCount)
        .def("__iter__", py::range(Container__begin<dim>, Container__end<dim>))
        .def("__delitem__", &Container__delitem__<dim>)
    ;
}

static PathHints::Hint TranslationContainer2_add
    (TranslationContainer<2>& self, shared_ptr<typename TranslationContainer<2>::ChildType> el, double c0, double c1) {
    return self.add(el, Vec<2>(c0, c1));
}

static PathHints::Hint TranslationContainer3_add
    (TranslationContainer<3>& self, shared_ptr<typename TranslationContainer<3>::ChildType> el, double c0, double c1, double c2) {
    return self.add(el, Vec<3>(c0, c1, c2));
}



void register_geometry_container_stack();

void register_geometry_container()
{
    init_GeometryElementContainer<2>();
    init_GeometryElementContainer<3>();

    py::class_<TranslationContainer<2>, shared_ptr<TranslationContainer<2>>, py::bases<GeometryElementContainer<2>>, boost::noncopyable>
    ("TranslationContainer2D",
     "Container in which every child has an associated translation vector\n\n"
     "TranslationContainer2D()\n    Create a new container"
    )
        .def("append", &TranslationContainer<2>::add, (py::arg("child"), py::arg("translation")=Vec<2,double>(0.,0.)),
             "Add new element to the container with provided translation vector")
        .def("append", &TranslationContainer2_add, (py::arg("child"), "c0", "c1"),
             "Add new element to the container with tranlastion [c0,c1]")
    ;

    py::class_<TranslationContainer<3>, shared_ptr<TranslationContainer<3>>, py::bases<GeometryElementContainer<3>>, boost::noncopyable>
    ("TranslationContainer3D",
     "Container in which every child has an associated translation vector\n\n"
     "TranslationContainer3D()\n    Create a new container"
    )
        .def("append", &TranslationContainer<3>::add, (py::arg("child"), py::arg("translation")=Vec<3,double>(0.,0.,0.)),
             "Add new element to the container with provided translation vector")
        .def("append", &TranslationContainer3_add, (py::arg("child"), "c0", "c1", "c2"),
             "Add new element to the container with translation [c0,c1,c2]")
    ;

    register_geometry_container_stack();
}



}} // namespace plask::python
