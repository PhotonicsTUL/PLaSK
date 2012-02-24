#include "geometry.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <plask/geometry/container.h>


namespace plask { namespace python {

// struct PersistentHint {
//     const shared_ptr<GeometryElement> first;
//     const shared_ptr<GeometryElement> second;
//
//     PersistentHint(const shared_ptr<GeometryElement>& container, const shared_ptr<GeometryElement>& child) :
//         first(container), second(child) {}
//
//     PersistentHint(const PathHints::Hint& hint) :
//         first(hint.first.lock()), second(hint.second.lock()) {}
//
//     shared_ptr<GeometryElement> getChild() {
//         shared_ptr<Translation<2>> T2 = dynamic_pointer_cast<Translation<2>>(second);
//         if (T2) return T2->getChild();
//         shared_ptr<Translation<3>> T3 = dynamic_pointer_cast<Translation<3>>(second);
//         if (T3) return T3->getChild();
//         return second;
//     }
//
//     py::object translation() {
//         shared_ptr<Translation<2>> T2 = dynamic_pointer_cast<Translation<2>>(second);
//         if (T2) { return py::object(T2->translation); }
//         shared_ptr<Translation<3>> T3 = dynamic_pointer_cast<Translation<3>>(second);
//         if (T3) return py::object(T3->translation);
//         PyErr_SetString(PyExc_TypeError, "child object does not have a translation");
//         throw py::error_already_set();
//         assert(0);
//     }
//
//     operator PathHints::Hint() { return PathHints::Hint(first, second); }
// };


DECLARE_GEOMETRY_ELEMENT_23D(TranslationContainer, "TranslationContainer",
                             "Geometry elements container in which every child has an associated translation vector ("," version)")
{
    GEOMETRY_ELEMENT_23D_DEFAULT(TranslationContainer, GeometryElementContainer<dim>)
        .def("add", &TranslationContainer<dim>::add, (py::arg("child"), py::arg("trans")=Vec<2,double>(0.,0.)),
             "Add new element to the container")
//         .def("remove", (void(TranslationContainer<dim>::*)(const PathHints&))&TranslationContainer<dim>::remove,
//              "Remove element from container")
        .def("__contains__", &GeometryElementContainerImpl<dim>::isInSubtree, (py::arg("item")))
        //.def("__getitem__" TODO
        //.def("__deltiem__" TODO
        //.def("__iter__" TODO
    ;
}

// template <int dim, typename S>
// inline static PersistentHint Stack__getitem__(shared_ptr<S> self, int i) {
//     if (i < 0) i = self->children.size() - i;
//     if (i < 0 || i >= self->children.size()) {
//         PyErr_SetString(PyExc_IndexError, "index out of range");
//         throw py::error_already_set();
//     }
//     shared_ptr<Translation<dim>> tchild = self->children[i];
//     return PersistentHint(self, tchild);
// }

// template <int dim> inline static Vec<dim,double> vvec(double v);
// template <> inline Vec<2,double> vvec<2>(double v) { return Vec<2,double>(0,v); }
// template <> inline Vec<3,double> vvec<3>(double v) { return Vec<3,double>(0,0,v); }
// template <int dim>
// inline static PersistentHint MultiStack_repeatedItem(shared_ptr<MultiStackContainer<dim>> self, int i) {
//     int n = self->children.size();
//     int s = self->repeat_count * n;
//     if (i < 0) i =  s - i;
//     if (i < 0 || i >=  s) {
//         PyErr_SetString(PyExc_IndexError, "index out of range");
//         throw py::error_already_set();
//     }
//     int j = i % n, I = i / n;
//     Vec<dim,double> shift = vvec<dim>(I * (self->stackHeights.back()-self->stackHeights.front()));
//     shared_ptr<Translation<dim>> tchild = self->children[j];
//     shared_ptr<Translation<dim>> trans { new Translation<dim>(tchild->getChild(), tchild->translation + shift) };
//     return PersistentHint(self, trans);
// }

// static shared_ptr<GeometryElement> Hint_child(const PathHints::Hint& hint) {
//     shared_ptr<Translation<2>> T2 = PathHints::getTranslationChild<2>(hint);
//     if (T2) return T2->getChild();
//     shared_ptr<Translation<3>> T3 = PathHints::getTranslationChild<3>(hint);
//     if (T3) return T3->getChild();
//     return PathHints::getChild(hint);
// }
//
// static py::object Hint_translation(const PathHints::Hint& hint) {
//     shared_ptr<Translation<2>> T2 = PathHints::getTranslationChild<2>(hint);
//     if (T2) return py::object(T2->translation);
//     shared_ptr<Translation<3>> T3 = PathHints::getTranslationChild<3>(hint);
//     if (T3) return py::object(T3->translation);
//     PyErr_SetString(PyExc_TypeError, "child object does not have a translation");
//     throw py::error_already_set();
//     assert(0);
// }

void register_geometry_container()
{
//     py::class_<PersistentHint>("ContainerChild",
//                                "ContainerChild stores references to container and its child with translation.\n\n"
//                                "It should be used as an intermediate object to either add it to Path, PathHints, or\n"
//                                "to retrieve the container, child, or translation elements.",
//                                py::no_init)
//         .def_readonly("container", &PersistentHint::first)
//         .add_property("child", &PersistentHint::getChild)
//         .add_property("translation", &PersistentHint::translation)
    ;

    py::class_<PathHints::Hint>("Hint",
                                "Hint stores weak references to container and its child with translation.\n\n"
                                "It may only be used as an intermediate object to either add it to Path, PathHints, or\n"
                                "to retrieve the container, child, or translation elements.",
                                py::no_init)
//         .add_property("container", &PathHints::getContainer)
//         .add_property("child", &Hint_child)
//         .add_property("translation", &Hint_translation)
    ;

//     py::implicitly_convertible<PersistentHint, PathHints::Hint>();

    py::class_<PathHints>("PathHints",
                          "PathHints is used to resolve ambiguities if any element is present in the geometry\n"
                          "tree more than once. It contains a set of ElementHint objects holding weak references\n"
                          "to containers and their childred.")
        .def("add", (void (PathHints::*)(const PathHints::Hint&)) &PathHints::addHint, "Add hint to the path.", (py::arg("container_child")))
        .def(py::self += py::other<PathHints::Hint>())
    ;

//                           "Path is used to specify unique instance of every element in the geometry,\n"
//                           "even if this element is inserted to the geometry tree in more than one place.\n\n"
//                           "It contains a set of ContainerChild objects holding weak references to containers\n"
//                           "and their childred.")


    // Translation container
    init_TranslationContainer<2>();
    init_TranslationContainer<3>();

    // Stack container

    py::class_<StackContainer2d, shared_ptr<StackContainer2d>, py::bases<GeometryElementContainer<2>>, boost::noncopyable>("Stack2D",
        "Container that organizes its childern in vertical stack (2D version)\n\n"
        "StackContainer2D(base_level=0) -> Create the stack with the bottom side of the first element at the base_level (in container local coordinates)",
         py::init<double>((py::arg("base_level")=0.)))
        .def("add", &StackContainer2d::add, (py::arg("child"), py::arg("shift")=0.), "Add new element to the container")
        .def("append", &StackContainer2d::add, (py::arg("child"), py::arg("shift")=0.), "Add new element to the container")
        .def("__contains__", &GeometryElementContainerImpl<2>::isInSubtree, (py::arg("item")))
//         .def("__getitem__", &Stack__getitem__<2, StackContainer2d>, (py::arg("item")))
        //.def("__iter__" TODO
        //.def("__deltiem__" TODO
    ;

    py::class_<StackContainer3d, shared_ptr<StackContainer3d>, py::bases<GeometryElementContainer<3>>, boost::noncopyable>("Stack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "Stack3D(base_level=0) -> Create the stack with the bottom side of the first element at the base_level (in container local coordinates)",
        py::init<double>((py::arg("base_level")=0.)))
        .def("append", &StackContainer3d::add, (py::arg("child"), py::arg("shift0")=0., py::arg("shift1")=0.),
             "Add new element to the container")
        .def("__contains__", &GeometryElementContainerImpl<3>::isInSubtree, (py::arg("item")))
//         .def("__getitem__", &Stack__getitem__<3, StackContainer3d>, (py::arg("item")))
        //.def("__iter__" TODO
        //.def("__deltiem__" TODO
    ;

    // Multi-stack constainer

    py::class_<MultiStackContainer<2>, shared_ptr<MultiStackContainer<2>>, py::bases<StackContainer2d>, boost::noncopyable>("MultiStack2D",
        "Stack container which repeats its contents (2D version)\n\n"
        "MultiStack2D(repeat_count=1, base_level=0) -> Create new multi-stack with repeatCount repetitions",
         py::init<int, double>((py::arg("repeat_count")=1, py::arg("base_level")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<2>::repeat_count, "Number of repeats of the stack content")
//         .def("repeatedItem", &MultiStack_repeatedItem<2>, (py::arg("index")),
//              "Return new hint for a repeated item in te stack as if all repetitions were added separately")
    ;

    py::class_<MultiStackContainer<3>, shared_ptr<MultiStackContainer<3>>, py::bases<StackContainer3d>, boost::noncopyable>("MultiStack3D",
        "Container that organizes its childern in vertical stack (3D version)\n\n"
        "MultiStack3D(repeat_count=1, base_level=0) -> Create new multi-stack with repeatCount repetitions",
        py::init<int, double>((py::arg("repeat_count")=1, py::arg("base_level")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<3>::repeat_count, "Number of repeats of the stack content")
//         .def("repeatedItem", &MultiStack_repeatedItem<3>, (py::arg("index")),
//              "Return new hint for a repeated item in te stack as if all repetitions were added separately")
    ;
}



}} // namespace plask::python
