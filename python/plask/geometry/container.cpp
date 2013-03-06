#include "geometry.h"
#include "../../util/py_set.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/raw_function.hpp>

#include <plask/geometry/container.h>
#include <plask/geometry/translation_container.h>


namespace plask { namespace python {

template <int dim>
static bool Container__contains__(const GeometryObjectContainer<dim>& self, shared_ptr<typename GeometryObjectContainer<dim>::ChildType> child) {
    for (auto trans: self.getChildrenVector()) {
        if (trans->getChild() == child) return true;
    }
    return false;
}

template <int dim>
static auto Container__begin(const GeometryObjectContainer<dim>& self) -> decltype(self.getChildrenVector().begin()) {
    return self.getChildrenVector().begin();
}

template <int dim>
static auto Container__end(const GeometryObjectContainer<dim>& self) -> decltype(self.getChildrenVector().end()) {
    return self.getChildrenVector().end();
}

template <int dim>
static shared_ptr<GeometryObject> Container__getitem__int(py::object oself, int i) {
    GeometryObjectContainer<dim>* self = py::extract<GeometryObjectContainer<dim>*>(oself);
    int n = self->getChildrenCount();
    if (i < 0) i = n + i;
    if (i < 0 || i >= n) {
        throw IndexError("%1% index %2% out of range (0 <= index < %3%)",
            std::string(py::extract<std::string>(oself.attr("__class__").attr("__name__"))), i, n);
    }
    return self->getChildNo(i);
}

template <int dim>
static std::set<shared_ptr<GeometryObject>> Container__getitem__hints(const GeometryObjectContainer<dim>& self, const PathHints& hints) {
    std::set<shared_ptr<GeometryObject>> result = hints.getChildren(self);
    return result;
}

template <int dim>
static void Container__delitem__(GeometryObjectContainer<dim>& self, py::object item) {
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
        shared_ptr<typename GeometryObjectContainer<dim>::TranslationT> child = py::extract<shared_ptr<typename GeometryObjectContainer<dim>::TranslationT>>(item);
        self.removeT(child);
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    try {
        shared_ptr<typename GeometryObjectContainer<dim>::ChildType> child = py::extract<shared_ptr<typename GeometryObjectContainer<dim>::ChildType>>(item);
        self.remove(child);
        return;
    } catch (py::error_already_set) { PyErr_Clear(); }
    throw TypeError("unrecognized object %s delete from container", std::string(py::extract<std::string>(py::str(item))));
}


DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectContainer, "GeometryObjectContainer", "Base class for all "," containers") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectContainer, GeometryObjectD<dim>)
        .def("__contains__", &Container__contains__<dim>)
        .def("__getitem__", &Container__getitem__int<dim>)
        .def("__getitem__", &Container__getitem__hints<dim>)
        .def("__len__", &GeometryObjectD<dim>::getChildrenCount)
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


template <typename ContainerT>
PathHints::Hint TranslationContainer_add(py::tuple args, py::dict kwargs) {
    parseKwargs("append", args, kwargs, "self", "item");
    ContainerT* self = py::extract<ContainerT*>(args[0]);
    shared_ptr<typename ContainerT::ChildType> child = py::extract<shared_ptr<typename ContainerT::ChildType>>(args[1]);
    if (py::len(kwargs) == 0)
        return self->add(child);
    else
        return self->add(child, py::extract<typename ContainerT::ChildAligner>(kwargs));
}


void register_geometry_container_stack();

void register_geometry_container()
{
    init_GeometryObjectContainer<2>();
    init_GeometryObjectContainer<3>();

    py::class_<TranslationContainer<2>, shared_ptr<TranslationContainer<2>>, py::bases<GeometryObjectContainer<2>>, boost::noncopyable>
    ("TranslationContainer2D",
     "Container in which every child has an associated translation vector\n\n"
     "TranslationContainer2D()\n    Create a new container"
    )
        .def("append", py::raw_function(&TranslationContainer_add<TranslationContainer<2>>),
             "Add new object to the container with provided alignment")
        .def("append", (PathHints::Hint(TranslationContainer<2>::*)(shared_ptr<TranslationContainer<2>::ChildType>,const Vec<2>&))&TranslationContainer<2>::add,
             (py::arg("item"), py::arg("translation")=Vec<2>(0.,0.)),
             "Add new object to the container with provided translation vector")
        .def("append", &TranslationContainer2_add, (py::arg("item"), "c0", "c1"),
             "Add new object to the container with tranlastion [c0,c1]")
       ;

    py::class_<TranslationContainer<3>, shared_ptr<TranslationContainer<3>>, py::bases<GeometryObjectContainer<3>>, boost::noncopyable>
    ("TranslationContainer3D",
     "Container in which every child has an associated translation vector\n\n"
     "TranslationContainer3D()\n    Create a new container"
    )
        .def("append", py::raw_function(&TranslationContainer_add<TranslationContainer<2>>),
             "Add new object to the container with provided alignment")
        .def("append", (PathHints::Hint(TranslationContainer<3>::*)(shared_ptr<TranslationContainer<3>::ChildType>,const Vec<3>&))&TranslationContainer<3>::add,
             (py::arg("item"), py::arg("translation")=Vec<3>(0.,0.,0.)),
             "Add new object to the container with provided translation vector")
        .def("append", &TranslationContainer3_add, (py::arg("item"), "c0", "c1", "c2"),
             "Add new object to the container with translation [c0,c1,c2]")
    ;

    register_geometry_container_stack();
}



}} // namespace plask::python
