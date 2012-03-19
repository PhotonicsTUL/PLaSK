#include "geometry.h"
#include "../../util/py_set.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <plask/geometry/container.h>
#include <plask/geometry/stack.h>


namespace plask { namespace python {

DECLARE_GEOMETRY_ELEMENT_23D(TranslationContainer, "TranslationContainer",
                             "Geometry elements container in which every child has an associated translation vector ("," version)")
{
    GEOMETRY_ELEMENT_23D_DEFAULT(TranslationContainer, GeometryElementContainer<dim>)
        .def("append", &TranslationContainer<dim>::add, (py::arg("child"), py::arg("trans")=Vec<2,double>(0.,0.)),
             "Add new element to the container")
//         .def("remove", (void(TranslationContainer<dim>::*)(const PathHints&))&TranslationContainer<dim>::remove,
//              "Remove element from container")
        .def("__contains__", &GeometryElementContainer<dim>::isInSubtree, (py::arg("item")))
        //.def("__getitem__" TODO
        //.def("__deltiem__" TODO
        //.def("__iter__" TODO
    ;
}


void register_geometry_container()
{
    // Translation container
    init_TranslationContainer<2>();
    init_TranslationContainer<3>();

    // Stack container

    py::class_<StackContainer<2>, shared_ptr<StackContainer<2>>, py::bases<GeometryElementContainer<2>>, boost::noncopyable>("Stack2D",
        "Container that organizes its childern in vertical stack (2D version)\n\n"
        "Stack2D(base_level=0)\n    Create the stack with the bottom side of the first element at the base_level (in container local coordinates)\n",
         py::init<double>((py::arg("base_level")=0.)))
        .def("append", &StackContainer<2>::add, (py::arg("child"), py::arg("align")=StackContainer<2>::CenterAligner()), "Append new element to the container")
        .def("__contains__", &GeometryElementContainer<2>::isInSubtree, (py::arg("item")))
//         .def("__getitem__", &Stack__getitem__<2, StackContainer2d>, (py::arg("item")))
        //.def("__iter__" TODO
        //.def("__deltiem__" TODO
    ;

    py::class_<StackContainer<3>, shared_ptr<StackContainer<3>>, py::bases<GeometryElementContainer<3>>, boost::noncopyable>("Stack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "Stack3D(base_level=0)\n    Create the stack with the bottom side of the first element at the base_level (in container local coordinates)\n",
        py::init<double>((py::arg("base_level")=0.)))
        .def("append", &StackContainer<3>::add, (py::arg("child"), py::arg("align")=StackContainer<3>::CenterAligner()), "Append new element to the container")
        .def("__contains__", &GeometryElementContainer<3>::isInSubtree, (py::arg("item")))
//         .def("__getitem__", &Stack__getitem__<3, StackContainer3d>, (py::arg("item")))
        //.def("__iter__" TODO
        //.def("__deltiem__" TODO
    ;

    // Multi-stack constainer

    //TODO aligners
    py::class_<MultiStackContainer<2>, shared_ptr<MultiStackContainer<2>>, py::bases<StackContainer<2>>, boost::noncopyable>("MultiStack2D",
        "Stack container which repeats its contents (2D version)\n\n"
        "MultiStack2D(repeat_count=1, base_level=0)\n    Create new multi-stack with repeatCount repetitions\n",
         py::init<int, double>((py::arg("repeat_count")=1, py::arg("base_level")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<2>::repeat_count, "Number of repeats of the stack content")
//         .def("repeatedItem", &MultiStack_repeatedItem<2>, (py::arg("index")),
//              "Return new hint for a repeated item in te stack as if all repetitions were added separately")
    ;

    //TODO aligners
    py::class_<MultiStackContainer<3>, shared_ptr<MultiStackContainer<3>>, py::bases<StackContainer<3>>, boost::noncopyable>("MultiStack3D",
        "Container that organizes its childern in vertical stack (3D version)\n\n"
        "MultiStack3D(repeat_count=1, base_level=0)\n    Create new multi-stack with repeatCount repetitions\n",
        py::init<int, double>((py::arg("repeat_count")=1, py::arg("base_level")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<3>::repeat_count, "Number of repeats of the stack content")
//         .def("repeatedItem", &MultiStack_repeatedItem<3>, (py::arg("index")),
//              "Return new hint for a repeated item in te stack as if all repetitions were added separately")
    ;
}



}} // namespace plask::python
