#include "geometry.h"
#include <boost/python/raw_function.hpp>
#include "../../util/raw_constructor.h"

#include <plask/geometry/stack.h>


namespace plask { namespace python {

template <typename StackT>
PathHints::Hint Stack_push_back(py::tuple args, py::dict kwargs) {
    parseKwargs("append", args, kwargs, "self", "item");
    StackT* self = py::extract<StackT*>(args[0]);
    shared_ptr<typename StackT::ChildType> child = py::extract<shared_ptr<typename StackT::ChildType>>(args[1]);
    if (py::len(kwargs) == 0)
        return self->push_back(child);
    else
        return self->push_back(child, py::extract<typename StackT::ChildAligner>(kwargs));
}

template <typename StackT>
PathHints::Hint Stack_push_front(py::tuple args, py::dict kwargs) {
    parseKwargs("prepend", args, kwargs, "self", "item");
    StackT* self = py::extract<StackT*>(args[0]);
    shared_ptr<typename StackT::ChildType> child = py::extract<shared_ptr<typename StackT::ChildType>>(args[1]);
    if (py::len(kwargs) == 0)
        return self->push_front(child);
    else
        return self->push_front(child, py::extract<typename StackT::ChildAligner>(kwargs));
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
        return self->insert(child, pos, py::extract<typename StackT::ChildAligner>(kwargs));
}

template <int dim>
shared_ptr<StackContainer<dim>> Stack__init__(const py::tuple& args, py::dict kwargs) {
    kwargs = kwargs.copy();
    double shift = 0.;
    if (py::len(args) > 1) {
        if (kwargs.has_key("shift"))
            throw TypeError("__init__() got multiple values for keyword argument 'shift'");
        shift = py::extract<double>(args[1]);
        if (py::len(args) > 2)
            throw TypeError("__init__() takes at most 2 non-keyword arguments (%1% given)", py::len(args));
    } else if (kwargs.has_key("shift")) {
        shift = py::extract<double>(kwargs["shift"]);
        py::delitem(kwargs, py::str("shift"));
    }
    if (py::len(kwargs) == 0)
        return make_shared<StackContainer<dim>>(shift);
    else
        return make_shared<StackContainer<dim>>(shift, py::extract<typename StackContainer<dim>::ChildAligner>(kwargs));
}

template <int dim>
shared_ptr<MultiStackContainer<dim>> MultiStack__init__(const py::tuple& args, py::dict kwargs) {
    kwargs = kwargs.copy();
    double shift;
    size_t repeat;
    if (py::len(args) > 1) {
        if (kwargs.has_key("repeat"))
            throw TypeError("__init__() got multiple values for keyword argument 'shift'");
        repeat = py::extract<size_t>(args[1]);
    } else if (kwargs.has_key("repeat")) {
        repeat = py::extract<size_t>(kwargs["repeat"]);
        py::delitem(kwargs, py::str("repeat"));
    } else
        throw TypeError("__init__() takes at least 2 arguments (%1% given)", py::len(args));
    if (py::len(args) > 2) {
        if (kwargs.has_key("shift"))
            throw TypeError("__init__() got multiple values for keyword argument 'shift'");
        shift = py::extract<double>(args[2]);
        if (py::len(args) > 3)
            throw TypeError("__init__() takes at most 3 non-keyword arguments (%1% given)", py::len(args));
    } else if (kwargs.has_key("shift")) {
        shift = py::extract<double>(kwargs["shift"]);
        py::delitem(kwargs, py::str("shift"));
    }
    if (py::len(kwargs) == 0)
        return make_shared<MultiStackContainer<dim>>(repeat, shift);
    else
        return make_shared<MultiStackContainer<dim>>(repeat, shift, py::extract<typename StackContainer<dim>::ChildAligner>(kwargs));
}

void register_geometry_container_stack()
{
    // Stack container

    py::class_<StackContainer<2>, shared_ptr<StackContainer<2>>, py::bases<GeometryObjectContainer<2>>, boost::noncopyable>("SingleStack2D",
        "Container that organizes its children in vertical stack (2D version)\n\n"
        "SingleStack2D(shift=0, **kwargs)\n"
        "    Create the stack with the bottom side of the first object at the 'shift' position (in container local coordinates).\n"
        "    'kwargs' may contain default aligner specification.\n\n"
        "See geometry.Stack2D(...).\n", py::no_init)
        .def("__init__", raw_constructor(Stack__init__<2>))
        .def("append", py::raw_function(&Stack_push_back<StackContainer<2>>), "Append new object to the container")
        .def("prepend", py::raw_function(&Stack_push_front<StackContainer<2>>), "Prepend new object to the container")
        .def("insert", py::raw_function(&Stack_insert<StackContainer<2>>), "Insert new object to the container")
        .def("set_zero_below", &StackContainer<2>::setZeroHeightBefore, py::arg("index"), "Set zero below item with index 'index'")
        .def("move", py::raw_function(&Container_move<StackContainer<2>>), "Move item in container")
        .add_property("default_aligner", py::make_getter(&StackContainer<2>::default_aligner, py::return_value_policy<py::return_by_value>()),
                      py::make_setter(&StackContainer<2>::default_aligner), "Default alignment for new stack item")
    ;

    py::class_<StackContainer<3>, shared_ptr<StackContainer<3>>, py::bases<GeometryObjectContainer<3>>, boost::noncopyable>("SingleStack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "SingleStack3D(shift=0, **kwargs)\n"
        "    Create the stack with the bottom side of the first object at the 'shift' position (in container local coordinates).\n\n"
        "    'kwargs' may contain default aligner specification.\n\n"
        "See geometry.Stack3D(...).\n", py::no_init)
        .def("__init__", raw_constructor(Stack__init__<3>))
        .def("append", py::raw_function(&Stack_push_back<StackContainer<3>>), "Append new object to the container")
        .def("prepend", py::raw_function(&Stack_push_front<StackContainer<3>>), "Prepend new object to the container")
        .def("insert", py::raw_function(&Stack_insert<StackContainer<3>>), "Insert new object to the container")
        .def("set_zero_below", &StackContainer<3>::setZeroHeightBefore, py::arg("index"), "Set zero below item with index 'index'")
        .def("move", py::raw_function(&Container_move<StackContainer<3>>), "Move item in container")
        .add_property("default_aligner", py::make_getter(&StackContainer<3>::default_aligner, py::return_value_policy<py::return_by_value>()),
                      py::make_setter(&StackContainer<3>::default_aligner), "Default alignment for new stack items")
    ;

    // Multi-stack constainer

    py::class_<MultiStackContainer<2>, shared_ptr<MultiStackContainer<2>>, py::bases<StackContainer<2>>, boost::noncopyable>("MultiStack2D",
        "Stack container which repeats its contents (2D version)\n\n"
        "MultiStack2D(repeat=1, shift=0, **kwargs)\n"
        "    Create new multi-stack with 'repeat' repetitions \n    and the first object at the 'shift' position (in container local coordinates)..\n\n"
        "    'kwargs' may contain default aligner specification.\n\n"
        "See geometry.Stack2D(...).\n", py::no_init)
        .def("__init__", raw_constructor(MultiStack__init__<2>))
        .def_readwrite("repeats", &MultiStackContainer<2>::repeat_count, "Number of repeats of the stack content")
    ;

    py::class_<MultiStackContainer<3>, shared_ptr<MultiStackContainer<3>>, py::bases<StackContainer<3>>, boost::noncopyable>("MultiStack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "MultiStack3D(repeat=1, shift=0, **kwargs)\n"
        "    Create new multi-stack with 'repeat' repetitions \n    and the first object at the 'shift' position (in container local coordinates)..\n\n"
        "    'kwargs' may contain default aligner specification.\n\n"
        "See geometry.Stack3D(...).\n", py::no_init)
        .def("__init__", raw_constructor(MultiStack__init__<3>))
        .def_readwrite("repeats", &MultiStackContainer<3>::repeat_count, "Number of repeats of the stack content")
    ;

    // Shelf (horizontal stack)
    py::class_<ShelfContainer2D, shared_ptr<ShelfContainer2D>, py::bases<GeometryObjectContainer<2>>, boost::noncopyable>("Shelf2D",
        "Container that organizes its children one next to another (like books on a bookshelf)\n\n"
        "Shelf2D(shift=0)\n"
        "    Create the shelf with the left side of the first object at the shift position (in container local coordinates).",
        py::init<double>((py::arg("shift")=0.)))
        .def("append", &ShelfContainer2D::push_back, (py::arg("item")), "Append new object to the container")
        .def("prepend", &ShelfContainer2D::push_front, (py::arg("item")), "Prepend new object to the container")
        .def("insert", &ShelfContainer2D::insert, (py::arg("item"), "pos"), "Insert new object to the container")
        .def("set_zero_before", &StackContainer<3>::setZeroHeightBefore, py::arg("index"), "Set zero left of item with index 'index'")
        .def("append_gap", &ShelfContainer2D::addGap, py::arg("size"), "Add gap of the size 'size' to the end of the shelf")
        .add_property("flat", &ShelfContainer2D::isFlat, "True if all children has the same height (the top line is flat)")
    ;
    py::scope().attr("Shelf") = py::scope().attr("Shelf2D");
}



}} // namespace plask::python
