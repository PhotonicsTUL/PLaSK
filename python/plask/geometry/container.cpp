#include "geometry.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <plask/geometry/container.h>


namespace plask { namespace python {

DECLARE_GEOMETRY_ELEMENT_23D(TranslationContainer, "TranslationContainer",
                             "Geometry elements container in which every child has an associated translation vector ("," version)")
{
    GEOMETRY_ELEMENT_23D_DEFAULT(TranslationContainer, GeometryElementContainer<dim>)
        .def("add", &TranslationContainer<dim>::add, "Add new element to the container")
    ;
}

// Stack wrappers
template <typename S>
inline static PathHints::Hint Stack_add(S& self, const shared_ptr<typename S::ChildType>& el) {
    return self.add(el);
}

template <int dim, typename S>
inline static py::tuple Stack__getitem__(const S& self, int i) {
    if (i < 0) i = self.children.size() - i;
    if (i < 0 || i >= self.children.size()) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw py::error_already_set();
    }
    shared_ptr<Translation<dim>> t = self.children[i];
    return py::make_tuple(t->getChild(), t->translation);
}

template <int dim> inline static Vec<dim,double> vvec(double v);
template <> inline Vec<2,double> vvec<2>(double v) { return Vec<2,double>(0,v); }
template <> inline Vec<3,double> vvec<3>(double v) { return Vec<3,double>(0,0,v); }
template <int dim>
inline static py::tuple MultiStack__getitem__(const MultiStackContainer<dim>& self, int i) {
    int n = self.children.size();
    int s = self.repeat_count * n;
    if (i < 0) i =  s - i;
    if (i < 0 || i >=  s) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw py::error_already_set();
    }
    int j = i % n, I = i / n;
    shared_ptr<Translation<dim>> t = self.children[j];
    return py::make_tuple( t->getChild(), t->translation + vvec<dim>(I * (self.stackHeights.back()-self.stackHeights.front())) );
}



void register_geometry_container()
{
    // Path hints (jest for future use)

    py::class_<PathHints::Hint>("PathHint",
                                "Objects of this class are returned by methods which add new elements to containers and can be added to path Hints",
                                py::no_init);

    py::class_<PathHints>("PathHints", "Hints are used to to find unique path for every element in the geometry tree, "
                                       "even if this element is inserted to geometry graph in more than one place.")
        .def("add", (void (PathHints::*)(const PathHints::Hint&)) &PathHints::addHint, "Add hint to hints map.")
        .def(py::self += py::other<PathHints::Hint>())
    ;

    // Translation container
    init_TranslationContainer<2>();
    init_TranslationContainer<3>();

    // Stack container

    py::class_<StackContainer2d, shared_ptr<StackContainer2d>, py::bases<GeometryElementContainer<2>>>("Stack2D",
        "Container that organizes its childern in vertical stack (2D version)\n\n"
        "StackContainer2D(baseLevel = 0) -> Create the stack with the bottom side of the first element at the baseLevel (in container local coordinates)",
         py::init<double>())
        .def("append", &StackContainer2d::add, "Add new element to the container")
        .def("append", &Stack_add<StackContainer2d>, "Add new element to the container")
        .def("__getitem__", &Stack__getitem__<2, StackContainer2d>)
        //.def("__iter__"
    ;

    py::class_<StackContainer3d, shared_ptr<StackContainer3d>, py::bases<GeometryElementContainer<3>>>("Stack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "Stack3D(baseLevel = 0) -> Create the stack with the bottom side of the first element at the baseLevel (in container local coordinates)",
        py::init<double>())
        .def("append", &StackContainer3d::add, "Add new element to the container")
        .def("append", &Stack_add<StackContainer3d>, "Add new element to the container")
        .def("__getitem__", &Stack__getitem__<3, StackContainer3d>)
        //.def("__iter__"
    ;

    // Multi-stack constainer

    py::class_<MultiStackContainer<2>, shared_ptr<MultiStackContainer<2>>, py::bases<StackContainer2d>>("MultiStack2D",
        "Stack container which repeats its contents (2D version)\n\n"
        "MultiStack2D(repeatCount = 1, baseLevel = 0) -> Create new multi-stack with repeatCount repetitions",
         py::init<int, double>())
        .def_readwrite("repeats", &MultiStackContainer<2>::repeat_count, "Number of repeats of the stack content")
        .def("__getitem__", &MultiStack__getitem__<2>)
    ;

    py::class_<MultiStackContainer<3>, shared_ptr<MultiStackContainer<3>>, py::bases<StackContainer3d>>("MultiStack3D",
        "Container that organizes its childern in vertical stack (3D version)\n\n"
        "MultiStack3D(repeatCount = 1, baseLevel = 0) -> Create new multi-stack with repeatCount repetitions",
        py::init<int, double>())
        .def_readwrite("repeats", &MultiStackContainer<3>::repeat_count, "Number of repeats of the stack content")
        .def("__getitem__", &MultiStack__getitem__<3>)
    ;

}



}} // namespace plask::python
