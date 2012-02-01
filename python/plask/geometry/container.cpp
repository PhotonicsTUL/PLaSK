#include "geometry.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <plask/geometry/container.h>


namespace plask { namespace python {

/**
 * Class returned by containers __getitem__ methods
 */
template <int dim>
struct ContainerElement {
    const shared_ptr<GeometryElementContainer<dim>> container;
    const shared_ptr<Translation<dim>> trans_geom;
    const Vec<dim,double> translation;

    ContainerElement(const shared_ptr<GeometryElementContainer<dim>>& container,
                     const shared_ptr<Translation<dim>>& child,
                     const Vec<dim,double> trans) :
        container(container), trans_geom(child), translation(trans) {}

    shared_ptr<GeometryElementD<dim>> child() { return trans_geom->getChild(); }

    operator PathHints::Hint() { return PathHints::Hint(container, trans_geom); }
};

DECLARE_GEOMETRY_ELEMENT_23D(ContainerElement, "ContainerElement",
                             "Object holding cointainer element, its translation and hint ("," version)")
{
    py::class_<ContainerElement<dim>> (ContainerElement_pyname<dim>(), ContainerElement_pydoc<dim>(), py::no_init)
        .add_property("element", &ContainerElement<dim>::child, "Hold element")
        .def_readonly("translation", &ContainerElement<dim>::translation, "Translation vector of the element in the container")
    ;
}



DECLARE_GEOMETRY_ELEMENT_23D(TranslationContainer, "TranslationContainer",
                             "Geometry elements container in which every child has an associated translation vector ("," version)")
{
    GEOMETRY_ELEMENT_23D_DEFAULT(TranslationContainer, GeometryElementContainer<dim>)
        .def("add", &TranslationContainer<dim>::add, (py::arg("child"), py::arg("trans")=Vec<2,double>(0.,0.)),
             "Add new element to the container")
//         .def("remove", (void(TranslationContainer<dim>::*)(const PathHints&))&TranslationContainer<dim>::remove,
//              "Remove element from container")
        .def("__contains__", &GeometryElementContainerImpl<dim>::isInSubtree)
        //.def("__getitem__" TODO
        //.def("__deltiem__" TODO
        //.def("__iter__" TODO
    ;
}

template <int dim, typename S>
inline static ContainerElement<dim> Stack__getitem__(shared_ptr<S>& self, int i) {
    if (i < 0) i = self->children.size() - i;
    if (i < 0 || i >= self->children.size()) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw py::error_already_set();
    }
    shared_ptr<Translation<dim>> t = self->children[i];
    return ContainerElement<dim>( self, t, t->translation );
}

template <int dim> inline static Vec<dim,double> vvec(double v);
template <> inline Vec<2,double> vvec<2>(double v) { return Vec<2,double>(0,v); }
template <> inline Vec<3,double> vvec<3>(double v) { return Vec<3,double>(0,0,v); }
template <int dim>
inline static ContainerElement<dim> MultiStack__getitem__(shared_ptr<MultiStackContainer<dim>>& self, int i) {
    int n = self->children.size();
    int s = self->repeat_count * n;
    if (i < 0) i =  s - i;
    if (i < 0 || i >=  s) {
        PyErr_SetString(PyExc_IndexError, "index out of range");
        throw py::error_already_set();
    }
    int j = i % n, I = i / n;
    shared_ptr<Translation<dim>> t = self->children[j];
    return ContainerElement<dim>( self, t, t->translation + vvec<dim>(I * (self->stackHeights.back()-self->stackHeights.front())) );
}



void register_geometry_container()
{
    py::class_<PathHints::Hint>("Hint",
                                "Hints are returned by methods which add new elements to containers and can be added"
                                "to the geometry.Path to specify unique instance of any object in the geometry tree.",
                                py::no_init);

    py::class_<PathHints>("Path", "Path is used to specify unique instance of every element in the geometry tree, "
                                  "even if this element is inserted to the geometry graph in more than one place.")
        .def("add", (void (PathHints::*)(const PathHints::Hint&)) &PathHints::addHint, "Add hint to the path.")
        .def(py::self += py::other<PathHints::Hint>())
    ;

    init_ContainerElement<2>();
    init_ContainerElement<3>();
    py::implicitly_convertible<ContainerElement<2>, PathHints::Hint>();
    py::implicitly_convertible<ContainerElement<3>, PathHints::Hint>();

    // Translation container
    init_TranslationContainer<2>();
    init_TranslationContainer<3>();

    // Stack container

    py::class_<StackContainer2d, shared_ptr<StackContainer2d>, py::bases<GeometryElementContainer<2>>>("Stack2D",
        "Container that organizes its childern in vertical stack (2D version)\n\n"
        "StackContainer2D(baseLevel = 0) -> Create the stack with the bottom side of the first element at the baseLevel (in container local coordinates)",
         py::init<double>())
        .def("add", &StackContainer2d::add, (py::arg("child"), py::arg("shift")=0.), "Add new element to the container")
        .def("append", &StackContainer2d::add, (py::arg("child"), py::arg("shift")=0.), "Add new element to the container")
        .def("__contains__", &GeometryElementContainerImpl<2>::isInSubtree)
        .def("__getitem__", &Stack__getitem__<2, StackContainer2d>)
        //.def("__iter__" TODO
        //.def("__deltiem__" TODO
    ;

    py::class_<StackContainer3d, shared_ptr<StackContainer3d>, py::bases<GeometryElementContainer<3>>>("Stack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "Stack3D(baseLevel = 0) -> Create the stack with the bottom side of the first element at the baseLevel (in container local coordinates)",
        py::init<double>())
        .def("append", &StackContainer3d::add, (py::arg("child"), py::arg("shift0")=0., py::arg("shift1")=0.),
             "Add new element to the container")
        .def("__contains__", &GeometryElementContainerImpl<3>::isInSubtree)
        .def("__getitem__", &Stack__getitem__<3, StackContainer3d>)
        //.def("__iter__" TODO
        //.def("__deltiem__" TODO
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
