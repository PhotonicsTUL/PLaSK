#include "geometry.h"

#include <plask/geometry/stack.h>


namespace plask { namespace python {

void register_geometry_container_stack()
{
    // Stack container

    py::class_<StackContainer<2>, shared_ptr<StackContainer<2>>, py::bases<GeometryObjectContainer<2>>, boost::noncopyable>("Stack2D",
        "Container that organizes its children in vertical stack (2D version)\n\n"
        "Stack2D(shift=0)\n"
        "    Create the stack with the bottom side of the first object at the shift position (in container local coordinates).",
        py::init<double>((py::arg("shift")=0.)))
        .def("append", &StackContainer<2>::push_back, (py::arg("child"), py::arg("align")=StackContainer<2>::DefaultAligner()), "Append new object to the container")
        .def("prepend", &StackContainer<2>::push_front, (py::arg("child"), py::arg("align")=StackContainer<2>::DefaultAligner()), "Prepend new object to the container")
        .def("insert", &StackContainer<2>::insert, (py::arg("child"), "pos", py::arg("align")=StackContainer<2>::DefaultAligner()), "Insert new object to the container")
        .def("set_zero_below", &StackContainer<2>::setZeroHeightBefore, py::arg("index"), "Set zero below item with index 'index'")
    ;

    py::class_<StackContainer<3>, shared_ptr<StackContainer<3>>, py::bases<GeometryObjectContainer<3>>, boost::noncopyable>("Stack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "Stack3D(shift=0)\n"
        "    Create the stack with the bottom side of the first object at the shift position (in container local coordinates).",
        py::init<double>((py::arg("shift")=0.)))
        .def("append", &StackContainer<3>::push_back, (py::arg("child"), py::arg("align")=StackContainer<3>::DefaultAligner()), "Append new object to the container")
        .def("prepend", &StackContainer<3>::push_front, (py::arg("child"), py::arg("align")=StackContainer<3>::DefaultAligner()), "Prepend new object to the container")
        .def("insert", &StackContainer<3>::insert, (py::arg("child"), "pos", py::arg("align")=StackContainer<3>::DefaultAligner()), "Insert new object to the container")
        .def("set_zero_below", &StackContainer<3>::setZeroHeightBefore, py::arg("index"), "Set zero below item with index 'index'")
    ;

    // Multi-stack constainer

    py::class_<MultiStackContainer<2>, shared_ptr<MultiStackContainer<2>>, py::bases<StackContainer<2>>, boost::noncopyable>("MultiStack2D",
        "Stack container which repeats its contents (2D version)\n\n"
        "MultiStack2D(repeat_count=1, shift=0)\n    Create new multi-stack with repeatCount repetitions\n",
         py::init<int, double>((py::arg("repeat_count")=1, py::arg("shift")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<2>::repeat_count, "Number of repeats of the stack content")
    ;

    py::class_<MultiStackContainer<3>, shared_ptr<MultiStackContainer<3>>, py::bases<StackContainer<3>>, boost::noncopyable>("MultiStack3D",
        "Container that organizes its children in vertical stack (3D version)\n\n"
        "MultiStack3D(repeat_count=1, shift=0)\n    Create new multi-stack with repeatCount repetitions\n",
         py::init<int, double>((py::arg("repeat_count")=1, py::arg("shift")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<3>::repeat_count, "Number of repeats of the stack content")
    ;

    // Shelf (horizontal stack)
    py::class_<ShelfContainer2D, shared_ptr<ShelfContainer2D>, py::bases<GeometryObjectContainer<2>>, boost::noncopyable>("Shelf2D",
        "Container that organizes its children one next to another (like books on a bookshelf)\n\n"
        "Shelf2D(shift=0)\n"
        "    Create the shelf with the left side of the first object at the shift position (in container local coordinates).",
        py::init<double>((py::arg("shift")=0.)))
        .def("append", &ShelfContainer2D::push_back, (py::arg("child")), "Append new object to the container")
        .def("prepend", &ShelfContainer2D::push_front, (py::arg("child")), "Prepend new object to the container")
        .def("insert", &ShelfContainer2D::insert, (py::arg("child"), "pos"), "Insert new object to the container")
        .def("set_zero_before", &StackContainer<3>::setZeroHeightBefore, py::arg("index"), "Set zero left of item with index 'index'")
        .add_property("isflat", &ShelfContainer2D::isFlat, "True if all children has the same height (the top line is flat)")
    ;
    py::scope().attr("Shelf") = py::scope().attr("Shelf2D");
}



}} // namespace plask::python
