#include "geometry.h"

#include <plask/geometry/stack.h>


namespace plask { namespace python {

template <int dim>
static void Stack__delitem__int(py::object oself, int i) {
    StackContainer<dim>* self = py::extract<StackContainer<dim>*>(oself);
    int n = self->getRealChildrenCount();
    if (i < 0) i = n + i;
    if (i < 0 || i >= n) {
        throw IndexError("%1% index %2% out of range (0 <= index < %3%)",
            std::string(py::extract<std::string>(oself.attr("__class__").attr("__name__"))), i, n);
    }
    self->remove_if([&](const shared_ptr<Translation<dim>>& c){ return c == self->getRealChildAt(i); });
}

template <int dim>
static void Stack__delitem__hints(StackContainer<dim>& self, const PathHints& hints) {
    self.remove(hints);
}

template <int dim>
static void Stack__delitem__object(py::object oself, shared_ptr<typename StackContainer<dim>::ChildType> elem) {
    StackContainer<dim>* self = py::extract<StackContainer<dim>*>(oself);
    if (!self->remove(elem)) {
        throw KeyError("no such child");
    }
}


void register_geometry_container_stack()
{
    // Stack container

    py::class_<StackContainer<2>, shared_ptr<StackContainer<2>>, py::bases<GeometryElementContainer<2>>, boost::noncopyable>("Stack2D",
        "Container that organizes its childern in vertical stack (2D version)\n\n"
        "Stack2D(base_level=0)\n"
        "    Create the stack with the bottom side of the first element at the base_level (in container local coordinates).",
        py::init<double>((py::arg("base_level")=0.)))
        .def("append", &StackContainer<2>::push_back, (py::arg("child"), py::arg("align")=StackContainer<2>::DefaultAligner()), "Append new element to the container")
        .def("prepend", &StackContainer<2>::push_front, (py::arg("child"), py::arg("align")=StackContainer<2>::DefaultAligner()), "Prepend new element to the container")
        .def("__delitem__", &Stack__delitem__int<2>)
        .def("__delitem__", &Stack__delitem__hints<2>)
        .def("__delitem__", &Stack__delitem__object<2>)
    ;

    py::class_<StackContainer<3>, shared_ptr<StackContainer<3>>, py::bases<GeometryElementContainer<3>>, boost::noncopyable>("Stack3D",
        "Stack container which repeats its contents (3D version)\n\n"
        "Stack3D(base_level=0)\n"
        "    Create the stack with the bottom side of the first element at the base_level (in container local coordinates).",
        py::init<double>((py::arg("base_level")=0.)))
        .def("append", &StackContainer<3>::push_back, (py::arg("child"), py::arg("align")=StackContainer<3>::DefaultAligner()), "Append new element to the container")
        .def("prepend", &StackContainer<3>::push_front, (py::arg("child"), py::arg("align")=StackContainer<3>::DefaultAligner()), "Prepend new element to the container")
        .def("__delitem__", &Stack__delitem__int<3>)
        .def("__delitem__", &Stack__delitem__hints<3>)
        .def("__delitem__", &Stack__delitem__object<3>)
    ;

    // Multi-stack constainer

    py::class_<MultiStackContainer<2>, shared_ptr<MultiStackContainer<2>>, py::bases<StackContainer<2>>, boost::noncopyable>("MultiStack2D",
        "Stack container which repeats its contents (2D version)\n\n"
        "MultiStack2D(repeat_count=1, base_level=0)\n    Create new multi-stack with repeatCount repetitions\n",
         py::init<int, double>((py::arg("repeat_count")=1, py::arg("base_level")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<2>::repeat_count, "Number of repeats of the stack content")
    ;

    py::class_<MultiStackContainer<3>, shared_ptr<MultiStackContainer<3>>, py::bases<StackContainer<3>>, boost::noncopyable>("MultiStack3D",
        "Container that organizes its childern in vertical stack (3D version)\n\n"
        "MultiStack3D(repeat_count=1, base_level=0)\n    Create new multi-stack with repeatCount repetitions\n",
         py::init<int, double>((py::arg("repeat_count")=1, py::arg("base_level")=0.)))
        .def_readwrite("repeats", &MultiStackContainer<3>::repeat_count, "Number of repeats of the stack content")
    ;
}



}} // namespace plask::python
