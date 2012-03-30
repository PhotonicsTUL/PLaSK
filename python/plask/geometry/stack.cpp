#include "geometry.h"

#include <plask/geometry/stack.h>


namespace plask { namespace python {

template <int dim>
shared_ptr<StackContainer<dim>> StackContainer__init__(double base_level, py::object extend_obj) {
    typename StackContainer<dim>::StackExtension extend;
    std::string extend_str = "";
    if (extend_obj != py::object()) extend_str = py::extract<std::string>(extend_obj);
    for (auto s: extend_str) s = std::tolower(s);
    if (extend_str == "" || extend_str == "none")
        extend = StackContainer<dim>::EXTEND_NONE;
    else if (extend_str == "vert" || extend_str == "vertical" || extend_str == "vertically")
        extend = StackContainer<dim>::EXTEND_VERTICALLY;
    else if (extend_str == "hor" || extend_str == "horizontal" || extend_str == "horizontally")
        extend = StackContainer<dim>::EXTEND_HORIZONTALLY;
    else if (extend_str == "all" || extend_str == "both")
        extend = StackContainer<dim>::EXTEND_ALL;
    else {
        PyErr_SetString(PyExc_ValueError, format("unexpected value '%1%' for 'extend'", extend_str).c_str());
        throw py::error_already_set();
    }
    return make_shared<StackContainer<dim>>(base_level, extend);
}


void register_geometry_container_stack()
{
    // Stack container

    py::class_<StackContainer<2>, shared_ptr<StackContainer<2>>, py::bases<GeometryElementContainer<2>>, boost::noncopyable>("Stack2D",
        "Container that organizes its childern in vertical stack (2D version)\n\n"
        "Stack2D(base_level=0, extend=None)\n"
        "    Create the stack with the bottom side of the first element at the base_level (in container local coordinates).\n"
        "    Extend can be a string with value 'vert', 'hor' or 'all'. If set it indicates that the elements in the stack\n"
        "    are extended infinitely in specified direction.",
        py::no_init)
        .def("__init__", py::make_constructor(StackContainer__init__<2>, py::default_call_policies(), (py::arg("base_level")=0., py::arg("extend")=py::object())))
        .def("append", &StackContainer<2>::add, (py::arg("child"), py::arg("align")=StackContainer<2>::CenterAligner()), "Append new element to the container")
        .def("__contains__", &GeometryElementContainer<2>::isInSubtree, (py::arg("item")))
//         .def("__getitem__", &Stack__getitem__<2, StackContainer2d>, (py::arg("item")))
        //.def("__iter__" TODO
        //.def("__deltiem__" TODO
    ;

    py::class_<StackContainer<3>, shared_ptr<StackContainer<3>>, py::bases<GeometryElementContainer<3>>, boost::noncopyable>("Stack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "Stack3D(base_level=0, extend=None)\n"
        "    Create the stack with the bottom side of the first element at the base_level (in container local coordinates).\n"
        "    Extend can be a string with value 'vert', 'hor' or 'all'. If set it indicates that the elements in the stack\n"
        "    are extended infinitely in specified direction.",
        py::no_init)
        .def("__init__", py::make_constructor(StackContainer__init__<3>, py::default_call_policies(), (py::arg("base_level")=0., py::arg("extend")=py::object())))
        .def("append", &StackContainer<3>::add, (py::arg("child"), py::arg("align")=StackContainer<3>::CenterAligner()), "Append new element to the container")
        .def("__contains__", &GeometryElementContainer<3>::isInSubtree, (py::arg("item")))
//         .def("__getitem__", &Stack__getitem__<3, StackContainer3d>, (py::arg("item")))
        //.def("__iter__" TODO
        //.def("__deltiem__" TODO
    ;

    // Multi-stack constainer

    py::class_<MultiStackContainer<2>, shared_ptr<MultiStackContainer<2>>, py::bases<StackContainer<2>>, boost::noncopyable>("MultiStack2D",
        "Stack container which repeats its contents (2D version)\n\n"
        "MultiStack2D(repeat_count=1, base_level=0)\n    Create new multi-stack with repeatCount repetitions\n",
         py::init<int, double>((py::arg("repeat_count")=1, py::arg("base_level")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<2>::repeat_count, "Number of repeats of the stack content")
//         .def("repeatedItem", &MultiStack_repeatedItem<2>, (py::arg("index")),
//              "Return new hint for a repeated item in te stack as if all repetitions were added separately")
    ;

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
