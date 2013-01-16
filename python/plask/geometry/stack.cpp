#include "geometry.h"
#include <boost/python/raw_function.hpp>

#include <plask/geometry/stack.h>


namespace plask { namespace python {

template <typename StackT>
PathHints::Hint Stack_push_back(py::tuple args, py::dict kwargs) {
    parseKwargs("append", args, kwargs, "self", "child");
    StackT* self = py::extract<StackT*>(args[0]);
    shared_ptr<typename StackT::ChildType> child = py::extract<shared_ptr<typename StackT::ChildType>>(args[1]);
    if (py::len(kwargs) == 0)
        return self->push_back(child);
    else
        return self->push_back(child, py::extract<typename StackT::Aligner>(kwargs));
}

template <typename StackT>
PathHints::Hint Stack_push_front(py::tuple args, py::dict kwargs) {
    parseKwargs("prepend", args, kwargs, "self", "child");
    StackT* self = py::extract<StackT*>(args[0]);
    shared_ptr<typename StackT::ChildType> child = py::extract<shared_ptr<typename StackT::ChildType>>(args[1]);
    if (py::len(kwargs) == 0)
        return self->push_front(child);
    else
        return self->push_front(child, py::extract<typename StackT::Aligner>(kwargs));
}

template <typename StackT>
PathHints::Hint Stack_insert(py::tuple args, py::dict kwargs) {
    parseKwargs("insert", args, kwargs, "self", "child", "pos");
    StackT* self = py::extract<StackT*>(args[0]);
    shared_ptr<typename StackT::ChildType> child = py::extract<shared_ptr<typename StackT::ChildType>>(args[1]);
    size_t pos = py::extract<size_t>(args[2]);
    if (py::len(kwargs) == 0)
        return self->insert(child, pos);
    else
        return self->insert(child, pos, py::extract<typename StackT::Aligner>(kwargs));
}

void register_geometry_container_stack()
{
    // Stack container

    py::class_<StackContainer<2>, shared_ptr<StackContainer<2>>, py::bases<GeometryObjectContainer<2>>, boost::noncopyable>("SingleStack2D",
        "Container that organizes its children in vertical stack (2D version)\n\n"
        "SingleStack2D(shift=0)\n"
        "    Create the stack with the bottom side of the first object at the 'shift' position (in container local coordinates).\n\n"
        "See geometry.Stack2D().\n",
        py::init<double>((py::arg("shift")=0.)))
        .def("append", py::raw_function(&Stack_push_back<StackContainer<2>>), "Append new object to the container")
        .def("prepend", py::raw_function(&Stack_push_front<StackContainer<2>>), "Prepend new object to the container")
        .def("insert", py::raw_function(&Stack_insert<StackContainer<2>>), "Insert new object to the container")
        .def("set_zero_below", &StackContainer<2>::setZeroHeightBefore, py::arg("index"), "Set zero below item with index 'index'")
    ;

    py::class_<StackContainer<3>, shared_ptr<StackContainer<3>>, py::bases<GeometryObjectContainer<3>>, boost::noncopyable>("SingleStack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "SingleStack3D(shift=0)\n"
        "    Create the stack with the bottom side of the first object at the 'shift' position (in container local coordinates).\n\n"
        "See geometry.Stack3D().\n",
        py::init<double>((py::arg("shift")=0.)))
        .def("append", py::raw_function(&Stack_push_back<StackContainer<3>>), "Append new object to the container")
        .def("prepend", py::raw_function(&Stack_push_front<StackContainer<3>>), "Prepend new object to the container")
        .def("insert", py::raw_function(&Stack_insert<StackContainer<3>>), "Insert new object to the container")
        .def("set_zero_below", &StackContainer<3>::setZeroHeightBefore, py::arg("index"), "Set zero below item with index 'index'")
    ;

    // Multi-stack constainer

    py::class_<MultiStackContainer<2>, shared_ptr<MultiStackContainer<2>>, py::bases<StackContainer<2>>, boost::noncopyable>("MultiStack2D",
        "Stack container which repeats its contents (2D version)\n\n"
        "MultiStack2D(repeat=1, shift=0)\n"
        "    Create new multi-stack with 'repeat' repetitions \n    and the first object at the 'shift' position (in container local coordinates)..\n\n"
        "See geometry.Stack2D().\n",
         py::init<int, double>((py::arg("repeat")=1, py::arg("shift")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<2>::repeat_count, "Number of repeats of the stack content")
    ;

    py::class_<MultiStackContainer<3>, shared_ptr<MultiStackContainer<3>>, py::bases<StackContainer<3>>, boost::noncopyable>("MultiStack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "MultiStack3D(repeat=1, shift=0)\n"
        "    Create new multi-stack with 'repeat' repetitions \n    and the first object at the 'shift' position (in container local coordinates)..\n\n"
        "See geometry.Stack3D().\n",
         py::init<int, double>((py::arg("repeat")=1, py::arg("shift")=0.)))
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
        .def("append_gap", &ShelfContainer2D::addGap, py::arg("size"), "Add gap of the size 'size' to the end of the shelf")
        .add_property("isflat", &ShelfContainer2D::isFlat, "True if all children has the same height (the top line is flat)")
    ;
    py::scope().attr("Shelf") = py::scope().attr("Shelf2D");
}



}} // namespace plask::python
