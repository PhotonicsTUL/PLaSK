#include "geometry.h"

#include <plask/geometry/stack.h>


namespace plask { namespace python {

void register_geometry_container_stack()
{
    // Stack container

    py::class_<StackContainer<2>, shared_ptr<StackContainer<2>>, py::bases<GeometryElementContainer<2>>, boost::noncopyable>("Stack2D",
        "Container that organizes its children in vertical stack (2D version)\n\n"
        "Stack2D(base=0)\n"
        "    Create the stack with the bottom side of the first element at the base (in container local coordinates).",
        py::init<double>((py::arg("base")=0.)))
        .def("append", &StackContainer<2>::push_back, (py::arg("child"), py::arg("align")=StackContainer<2>::DefaultAligner()), "Append new element to the container")
        .def("prepend", &StackContainer<2>::push_front, (py::arg("child"), py::arg("align")=StackContainer<2>::DefaultAligner()), "Prepend new element to the container")
        .def("insert", &StackContainer<2>::insert, (py::arg("child"), "pos", py::arg("align")=StackContainer<2>::DefaultAligner()), "Insert new element to the container")
    ;

    py::class_<StackContainer<3>, shared_ptr<StackContainer<3>>, py::bases<GeometryElementContainer<3>>, boost::noncopyable>("Stack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "Stack3D(base=0)\n"
        "    Create the stack with the bottom side of the first element at the base (in container local coordinates).",
        py::init<double>((py::arg("base")=0.)))
        .def("append", &StackContainer<3>::push_back, (py::arg("child"), py::arg("align")=StackContainer<3>::DefaultAligner()), "Append new element to the container")
        .def("prepend", &StackContainer<3>::push_front, (py::arg("child"), py::arg("align")=StackContainer<3>::DefaultAligner()), "Prepend new element to the container")
    ;

    // Multi-stack constainer

    py::class_<MultiStackContainer<2>, shared_ptr<MultiStackContainer<2>>, py::bases<StackContainer<2>>, boost::noncopyable>("MultiStack2D",
        "Stack container which repeats its contents (2D version)\n\n"
        "MultiStack2D(repeat_count=1, base=0)\n    Create new multi-stack with repeatCount repetitions\n",
         py::init<int, double>((py::arg("repeat_count")=1, py::arg("base")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<2>::repeat_count, "Number of repeats of the stack content")
    ;

    py::class_<MultiStackContainer<3>, shared_ptr<MultiStackContainer<3>>, py::bases<StackContainer<3>>, boost::noncopyable>("MultiStack3D",
        "Container that organizes its children in vertical stack (3D version)\n\n"
        "MultiStack3D(repeat_count=1, base=0)\n    Create new multi-stack with repeatCount repetitions\n",
         py::init<int, double>((py::arg("repeat_count")=1, py::arg("base")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<3>::repeat_count, "Number of repeats of the stack content")
    ;

    // Shelf (horizontal stack)
    py::class_<ShelfContainer2d, shared_ptr<ShelfContainer2d>, py::bases<GeometryElementContainer<2>>, boost::noncopyable>("Shelf2D",
        "Container that organizes its children one next to another (like books on a bookshelf)\n\n"
        "Shelf2D(base=0)\n"
        "    Create the shelf with the left side of the first element at the base (in container local coordinates).",
        py::init<double>((py::arg("base")=0.)))
        .def("append", &ShelfContainer2d::push_back, (py::arg("child")), "Append new element to the container")
        .def("prepend", &ShelfContainer2d::push_front, (py::arg("child")), "Prepend new element to the container")
        .def("insert", &ShelfContainer2d::insert, (py::arg("child"), "pos"), "Insert new element to the container")
        .add_property("flat", &ShelfContainer2d::isFlat, "Return true if all children has the same height (the top line is flat)")
    ;
    py::scope().attr("Shelf") = py::scope().attr("Shelf2D");
}



}} // namespace plask::python
