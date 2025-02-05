/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "geometry.hpp"
#include <boost/python/raw_function.hpp>
#include "../python/util/raw_constructor.hpp"

#include <plask/geometry/stack.hpp>

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
    parseKwargs("insert", args, kwargs, "self", "index", "item");
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
            throw TypeError(u8"__init__() got multiple values for keyword argument 'shift'");
        shift = py::extract<double>(args[1]);
        if (py::len(args) > 2)
            throw TypeError(u8"__init__() takes at most 2 non-keyword arguments ({0} given)", py::len(args));
    } else if (kwargs.has_key("shift")) {
        shift = py::extract<double>(kwargs["shift"]);
        py::delitem(kwargs, py::str("shift"));
    }
    if (py::len(kwargs) == 0)
        return plask::make_shared<StackContainer<dim>>(shift);
    else
        return plask::make_shared<StackContainer<dim>>(shift, py::extract<typename StackContainer<dim>::ChildAligner>(kwargs));
}

template <int dim>
shared_ptr<MultiStackContainer<plask::StackContainer<dim>>> MultiStack__init__(const py::tuple& args, py::dict kwargs) {
    kwargs = kwargs.copy();
    double shift;
    size_t repeat;
    if (py::len(args) > 1) {
        if (kwargs.has_key("repeat"))
            throw TypeError(u8"__init__() got multiple values for keyword argument 'repeat'");
        repeat = py::extract<size_t>(args[1]);
    } else if (kwargs.has_key("repeat")) {
        repeat = py::extract<size_t>(kwargs["repeat"]);
        py::delitem(kwargs, py::str("repeat"));
    } else
        throw TypeError(u8"__init__() takes at least 2 arguments ({0} given)", py::len(args));
    if (py::len(args) > 2) {
        if (kwargs.has_key("shift"))
            throw TypeError(u8"__init__() got multiple values for keyword argument 'shift'");
        shift = py::extract<double>(args[2]);
        if (py::len(args) > 3)
            throw TypeError(u8"__init__() takes at most 3 non-keyword arguments ({0} given)", py::len(args));
    } else if (kwargs.has_key("shift")) {
        shift = py::extract<double>(kwargs["shift"]);
        py::delitem(kwargs, py::str("shift"));
    }
    if (py::len(kwargs) == 0)
        return plask::make_shared<MultiStackContainer<plask::StackContainer<dim>>>(repeat, shift);
    else
        return plask::make_shared<MultiStackContainer<plask::StackContainer<dim>>>(repeat, shift, py::extract<typename StackContainer<dim>::ChildAligner>(kwargs));
}

template <typename T>
static void align_zero_on_1(T& self, const shared_ptr<typename T::ChildType>& item) {
    self.alignZeroOn(item);
}

template <typename T>
static void align_zero_on_2(T& self, const shared_ptr<typename T::ChildType>& item, const PathHints& path) {
    self.alignZeroOn(item, path);
}

template <typename T>
static void align_zero_on_3(T& self, const shared_ptr<typename T::ChildType>& item, double pos) {
    self.alignZeroOn(item, nullptr, pos);
}

template <typename T>
static void align_zero_on_4(T& self, const shared_ptr<typename T::ChildType>& item, const PathHints& path, double pos) {
    self.alignZeroOn(item, path, pos);
}

void register_geometry_container_stack()
{
    // Stack container

    py::class_<StackContainer<2>, shared_ptr<StackContainer<2>>, py::bases<GeometryObjectContainer<2>>, boost::noncopyable>("SingleStack2D",
        u8"SingleStack2D(shift=0, **alignment)\n\n"
        u8"Container that organizes its items in a vertical stack (2D version).\n\n"
        u8"The bottom side of the first object is located at the `shift` position in\n"
        u8"container local coordinates. Each consecutive object is placed on the top of\n"
        u8"the previous one.\n\n"
        u8"Args:\n"
        u8"    shift (float): Position in the local coordinates of the bottom of the stack.\n"
        u8"    alignment (dict): Horizontal alignment specifications. This dictionary can\n"
        u8"                      contain only one item. Its key can be ``left``, ``right``,\n"
        u8"                      ``#center``, and ``#`` where `#` is the horizontal axis\n"
        u8"                      name. The corresponding value is the position of the given\n"
        u8"                      edge/center/origin of the item. This alignment can be\n"
        u8"                      overridden while adding the objects to the stack.\n"
        u8"                      By default the alignment is ``{'left': 0}``.\n"
        u8"See also:\n"
        u8"    Function :func:`plask.geometry.Stack2D`.\n", py::no_init)
        .def("__init__", raw_constructor(Stack__init__<2>))
        .def("append", py::raw_function(&Stack_push_back<StackContainer<2>>),
             u8"append(item, **alignment)\n\n"
             u8"Append a new object to the stack.\n\n"
             u8"This method adds a new item to the stack and places it at its top.\n\n"
             u8"Args:\n"
             u8"    item (GeometryObject2D): Object to append to the stack.\n"
             u8"    alignment (dict): Horizontal alignment specifications. This dictionary can\n"
             u8"                      contain only one item. Its key can be ``left``, ``right``,\n"
             u8"                      ``#center``, and ``#`` where `#` is the horizontal axis\n"
             u8"                      name. By default the object is aligned according to the\n"
             u8"                      specification in the stack constructor.\n")
        .def("prepend", py::raw_function(&Stack_push_front<StackContainer<2>>),
             u8"prepend(item, **alignment)\n\n"
             u8"Prepend a new object to the stack.\n\n"
             u8"This method adds a new item to the stack and places it at its bottom.\n\n"
             u8"Args:\n"
             u8"    item (GeometryObject2D): Object to prepend to the stack.\n"
             u8"    alignment (dict): Horizontal alignment specifications. This dictionary can\n"
             u8"                      contain only one item. Its key can be ``left``, ``right``,\n"
             u8"                      ``#center``, and ``#`` where `#` is the horizontal axis\n"
             u8"                      name. By default the object is aligned according to the\n"
             u8"                      specification in the stack constructor.\n")
        .def("insert", py::raw_function(&Stack_insert<StackContainer<2>>),
             u8"insert(index, item, **alignment)\n\n"
             u8"Insert a new object to the stack.\n\n"
             u8"This method adds a new item to the stack and places it at the position\n"
             u8"specified by `index`.\n\n"
             u8"Args:\n"
             u8"    index (int): Index of the inserted item in the stack.\n"
             u8"    item (GeometryObject2D): Object to insert to the stack.\n"
             u8"    alignment (dict): Horizontal alignment specifications. This dictionary can\n"
             u8"                      contain only one item. Its key can be ``left``, ``right``,\n"
             u8"                      ``#center``, and ``#`` where `#` is the horizontal axis\n"
             u8"                      name. By default the object is aligned according to the\n"
             u8"                      specification in the stack constructor.\n")
        .def<void (StackContainer<2>::*)(const shared_ptr<StackContainer<2>::ChildType>&, const PathHints&)>(
            "set_zero_below", &StackContainer<2>::setZeroBefore, (py::arg("item"), py::arg("path")=py::object()),
             u8"Set zero below the specified item.\n\n"
             u8"This method shifts the local coordinates of the stack vertically. The vertical\n"
             u8"coordinate of the stack origin is placed at the bottom edge of the specified\n"
             u8"item.\n\n"
             u8"Args:\n"
             u8"    item: Stack item which lower edge should lie at height 0.\n"
             u8"    path: Path specifying a particular item instance.\n")
        .def("align_zero_on", &align_zero_on_1<StackContainer<2>>, py::arg("item"))
        .def("align_zero_on", &align_zero_on_2<StackContainer<2>>, (py::arg("item"), "path"))
        .def("align_zero_on", &align_zero_on_3<StackContainer<2>>, (py::arg("item"), "pos"))
        .def("align_zero_on", &align_zero_on_4<StackContainer<2>>, (py::arg("item"), "path", "pos"),
             u8"Align zero with the vertical position ``pos`` of the specified item.\n\n"
             u8"This method shifts the local coordinates of the stack vertically. The vertical\n"
             u8"coordinate of the stack origin is placed at the local position ``pos``\n"
             u8"of the specified item (0 by default).\n\n"
             u8"Args:\n"
             u8"    item: Stack item which lower edge should lie at height 0.\n"
             u8"    path: Path specifying a particular item instance.\n"
             u8"    pos: Local position of the specified object to place at stack zero.\n")
        .def("move_item", py::raw_function(&Container_move<StackContainer<2>>),
             u8"move_item(path, **alignment)\n\n"
             u8"Move horizontally item existing in the stack, setting its position according\n"
             u8"to the new aligner.\n\n"
             u8"Args:\n"
             u8"    path (Path): Path returned by :meth:`~plask.geometry.Align2D.append`\n"
             u8"                 specifying the object to move.\n"
             u8"    alignment (dict): Alignment specifications. The only key in this dictionary\n"
             u8"                      are can be ``left``, ``right``, ``#center``, and `#``\n"
             u8"                      where `#` is an axis name. The corresponding values is\n"
             u8"                      the positions of a given edge/center/origin of the item.\n"
             u8"                      Exactly one alignment for horizontal axis must be given.\n")
        .add_property("default_aligner", py::make_getter(&StackContainer<2>::default_aligner, py::return_value_policy<py::return_by_value>()),
                      py::make_setter(&StackContainer<2>::default_aligner, py::return_value_policy<py::return_by_value>()),
                      u8"Default alignment for new stack items.")
    ;

    py::class_<StackContainer<3>, shared_ptr<StackContainer<3>>, py::bases<GeometryObjectContainer<3>>, boost::noncopyable>("SingleStack3D",
        u8"SingleStack3D(shift=0, **alignments)\n\n"
        u8"Container that organizes its items in a vertical stack (3D version).\n\n"
        u8"The bottom side of the first object is located at the `shift` position in\n"
        u8"container local coordinates. Each consecutive object is placed on the top of\n"
        u8"the previous one.\n\n"
        u8"Args:\n"
        u8"    shift (float): Position in the local coordinates of the bottom of the stack.\n"
        u8"    alignments (dict): Horizontal alignments specifications. Keys in this dict\n"
        u8"                       can be ``left``, ``right``, ``back``, ``front``,\n"
        u8"                       ``#center``, and ``#`` where `#` are the horizontal axis\n"
        u8"                       names. The corresponding value is the position of the\n"
        u8"                       given edge/center/origin of the item. This alignment can\n"
        u8"                       be overridden while adding the objects to the stack.\n"
        u8"                       By default the alignment is ``{'left': 0, 'back': 0}``.\n"
        u8"See also:\n"
        u8"    Function :func:`plask.geometry.Stack3D`.\n", py::no_init)
        .def("__init__", raw_constructor(Stack__init__<3>))
        .def("append", py::raw_function(&Stack_push_back<StackContainer<3>>),
             u8"append(item, **alignments)\n\n"
             u8"Append a new object to the stack.\n\n"
             u8"This method adds a new item to the stack and places it at its top.\n\n"
             u8"Args:\n"
             u8"    item (GeometryObject3D): Object to append to the stack.\n"
             u8"    alignments (dict): Horizontal alignments specifications. Keys in this dict\n"
             u8"                       can be ``left``, ``right``, ``back``, ``front``,\n"
             u8"                       ``#center``, and ``#`` where `#` are the horizontal axis\n"
             u8"                       names. The corresponding value is the position of the\n"
             u8"                       given edge/center/origin of the item. By default the\n"
             u8"                       object is aligned according to the specification in the\n"
             u8"                       stack constructor.\n")
        .def("prepend", py::raw_function(&Stack_push_front<StackContainer<3>>),
             u8"prepend(item, **alignments)\n\n"
             u8"Prepend a new object to the stack.\n\n"
             u8"This method adds a new item to the stack and places it at its bottom.\n\n"
             u8"Args:\n"
             u8"    item (GeometryObject3D): Object to prepend to the stack.\n"
             u8"    alignments (dict): Horizontal alignments specifications. Keys in this dict\n"
             u8"                       can be ``left``, ``right``, ``back``, ``front``,\n"
             u8"                       ``#center``, and ``#`` where `#` are the horizontal axis\n"
             u8"                       names. The corresponding value is the position of the\n"
             u8"                       given edge/center/origin of the item. By default the\n"
             u8"                       object is aligned according to the specification in the\n"
             u8"                       stack constructor.\n")
        .def("insert", py::raw_function(&Stack_insert<StackContainer<3>>),
             u8"insert(index, item, **alignments)\n\n"
             u8"Insert a new object to the stack.\n\n"
             u8"This method adds a new item to the stack and places it at the position\n"
             u8"specified by `index`.\n\n"
             u8"Args:\n"
             u8"    index (int): Index of the inserted item in the stack.\n"
             u8"    item (GeometryObject3D): Object to insert to the stack.\n"
             u8"    alignments (dict): Horizontal alignments specifications. Keys in this dict\n"
             u8"                       can be ``left``, ``right``, ``back``, ``front``,\n"
             u8"                       ``#center``, and ``#`` where `#` are the horizontal axis\n"
             u8"                       names. The corresponding value is the position of the\n"
             u8"                       given edge/center/origin of the item. By default the\n"
             u8"                       object is aligned according to the specification in the\n"
             u8"                       stack constructor.\n")
        .def<void (StackContainer<3>::*)(const shared_ptr<StackContainer<3>::ChildType>&, const PathHints&)>(
            "set_zero_below", &StackContainer<3>::setZeroBefore, (py::arg("item"), py::arg("path")=py::object()),
             u8"Set zero below the specified item.\n\n"
             u8"This method shifts the local coordinates of the stack vertically. The vertical\n"
             u8"coordinate of the stack origin is placed at the bottom edge of the specified\n"
             u8"item.\n\n"
             u8"Args:\n"
             u8"    item: Stack item which lower edge should lie at height 0.\n"
             u8"    path: Path specifying a particular item instance.\n")
        .def("align_zero_on", &align_zero_on_1<StackContainer<3>>, py::arg("item"))
        .def("align_zero_on", &align_zero_on_2<StackContainer<3>>, (py::arg("item"), "path"))
        .def("align_zero_on", &align_zero_on_3<StackContainer<3>>, (py::arg("item"), "pos"))
        .def("align_zero_on", &align_zero_on_4<StackContainer<3>>, (py::arg("item"), "path", "pos"),
             u8"Align zero with the vertical position ``pos`` of the specified item.\n\n"
             u8"This method shifts the local coordinates of the stack vertically. The vertical\n"
             u8"coordinate of the stack origin is placed at the local position ``pos``\n"
             u8"of the specified item (0 by default).\n\n"
             u8"Args:\n"
             u8"    item: Stack item which lower edge should lie at height 0.\n"
             u8"    path: Path specifying a particular item instance.\n"
             u8"    pos: Local position of the specified object to place at stack zero.\n")
        .def("move_item", py::raw_function(&Container_move<StackContainer<3>>),
             u8"move_item(path, **alignments)\n\n"
             u8"Move horizontally item existing in the stack, setting its position according\n"
             u8"to the new aligner.\n\n"
             u8"Args:\n"
             u8"    path (Path): Path returned by :meth:`~plask.geometry.Align2D.append`\n"
             u8"                 specifying the object to move.\n"
             u8"    alignments (dict): Horizontal alignments specifications. Keys in this dict\n"
             u8"                       can be ``left``, ``right``, ``back``, ``front``,\n"
             u8"                       ``#center``, and ``#`` where `#` are the horizontal axis\n"
             u8"                       names. The corresponding value is the position of the\n"
             u8"                       given edge/center/origin of the item. Exactly one\n"
             u8"                       alignment for each horizontal axis must be given.\n")
        .add_property("default_aligner", py::make_getter(&StackContainer<3>::default_aligner, py::return_value_policy<py::return_by_value>()),
                      py::make_setter(&StackContainer<3>::default_aligner, py::return_value_policy<py::return_by_value>()), u8"Default alignment for new stack items")
    ;

    // Multi-stack container

    py::class_<MultiStackContainer<plask::StackContainer<2>>, shared_ptr<MultiStackContainer<plask::StackContainer<2>>>, py::bases<StackContainer<2>>, boost::noncopyable>("MultiStack2D",
        u8"MultiStack2D(repeat=1, shift=0, **alignment)\n\n"
        u8"Stack container that repeats it contents (2D version).\n\n"
        u8"The bottom side of the first object is located at the `shift` position in\n"
        u8"container local coordinates. Each consecutive object is placed on the top of\n"
        u8"the previous one. Then the whole stack is repeated *repeat* times.\n\n"
        u8"Args:\n"
        u8"    repeat (int): Number of the stack contents repetitions.\n"
        u8"    shift (float): Position in the local coordinates of the bottom of the stack.\n"
        u8"    alignment (dict): Horizontal alignment specifications. This dictionary can\n"
        u8"                      contain only one item. Its key can be ``left``, ``right``,\n"
        u8"                      ``#center``, and ``#`` where `#` is the horizontal axis\n"
        u8"                      name. The corresponding value is the position of the given\n"
        u8"                      edge/center/origin of the item. This alignment can be\n"
        u8"                      overridden while adding the objects to the stack.\n"
        u8"                      By default the alignment is ``{'left': 0}``.\n"
        u8"See also:\n"
        u8"   Function :func:`plask.geometry.Stack2D`.\n", py::no_init)
        .def("__init__", raw_constructor(MultiStack__init__<2>))
        .add_property("repeat", &MultiStackContainer<plask::StackContainer<2>>::getRepeatCount, &MultiStackContainer<plask::StackContainer<2>>::setRepeatCount,
                      u8"Number of repeats of the stack contents.")
    ;

    py::class_<MultiStackContainer<plask::StackContainer<3>>, shared_ptr<MultiStackContainer<StackContainer<3>>>, py::bases<StackContainer<3>>, boost::noncopyable>("MultiStack3D",
        u8"MultiStack3D(repeat=1, shift=0, **alignments)\n\n"
        u8"Stack container that repeats it contents (3D version).\n\n"
        u8"The bottom side of the first object is located at the `shift` position in\n"
        u8"container local coordinates. Each consecutive object is placed on the top of\n"
        u8"the previous one. Then the whole stack is repeated *repeat* times.\n\n"
        u8"Args:\n"
        u8"    repeat (int): Number of the stack contents repetitions.\n"
        u8"    shift (float): Position in the local coordinates of the bottom of the stack.\n"
        u8"    alignments (dict): Horizontal alignments specifications. Keys in this dict\n"
        u8"                       can be ``left``, ``right``, ``back``, ``front``,\n"
        u8"                       ``#center``, and ``#`` where `#` are the horizontal axis\n"
        u8"                       names. The corresponding value is the position of the\n"
        u8"                       given edge/center/origin of the item. This alignment can\n"
        u8"                       be overridden while adding the objects to the stack.\n"
        u8"                       By default the alignment is ``{'left': 0, 'back': 0}``.\n"
        u8"See also:\n"
        u8"   Function :func:`plask.geometry.Stack3D`.\n", py::no_init)
        .def("__init__", raw_constructor(MultiStack__init__<3>))
        .add_property("repeat", &MultiStackContainer<plask::StackContainer<3>>::getRepeatCount, &MultiStackContainer<StackContainer<3>>::setRepeatCount,
                      u8"Number of repeats of the stack contents.")
    ;

    // Shelf (horizontal stack)

    py::class_<ShelfContainer2D, shared_ptr<ShelfContainer2D>, py::bases<GeometryObjectContainer<2>>, boost::noncopyable>("Shelf2D",
        u8"Shelf2D(shift=0)\n\n"
        u8"2D container that organizes its items one next to another.\n\n"
        u8"The objects are placed in this container like books on a bookshelf.\n"
        u8"The left side of the first object is located at the `shift` position in\n"
        u8"container local coordinates. Each consecutive object is placed to the right of\n"
        u8"the previous one. All the items are vertically aligned according to its bottom\n"
        u8"edge.\n\n"
        u8"Args:\n"
        u8"    shift (float): Position in the local coordinates of the left side of the\n"
        u8"                   shelf.\n\n"
        u8"See also:\n"
        u8"   Function :func:`plask.geometry.Shelf`.\n",
        py::init<double>((py::arg("shift")=0.)))
        .def("append", &ShelfContainer2D::push_back, (py::arg("item")),
             u8"Append a new object to the shelf.\n\n"
             u8"This method adds a new item to the shelf and places it at its right.\n\n"
             u8"Args:\n"
             u8"    item (GeometryObject2D): Object to append to the stack.\n")
        .def("prepend", &ShelfContainer2D::push_front, (py::arg("item")),
             u8"Prepend a new object to the shelf.\n\n"
             u8"This method adds a new item to the shelf and places it at its left.\n\n"
             u8"Args:\n"
             u8"    item (GeometryObject2D): Object to append to the stack.\n")
        .def("insert", &ShelfContainer2D::insert, (py::arg("item"), "pos"),
             u8"Insert a new object to the shelf.\n\n"
             u8"This method adds a new item to the shelf and places it at the position\n"
             u8"specified by `index`.\n\n"
             u8"Args:\n"
             u8"    item (GeometryObject2D): Object to insert to the shelf.\n"
             u8"    index (int): Index of the inserted item in the stack.\n")
        .def<void (ShelfContainer2D::*)(const shared_ptr<ShelfContainer2D::ChildType>&, const PathHints&)>(
            "set_zero_before", &ShelfContainer2D::setZeroBefore, (py::arg("item"), py::arg("path")=py::object()),
             u8"Set zero to the left the specified item.\n\n"
             u8"This method shifts the local coordinates of the stack horizontally.\n"
             u8"The horizontal coordinate of the shelf origin is placed at the left edge\n"
             u8"of the specified item.\n\n"
             u8"Args:\n"
             u8"    item: Shelf item which left edge should lie at 0.\n"
             u8"    path: Path specifying a particular item instance.\n")
        .def("align_zero_on", &align_zero_on_1<ShelfContainer2D>, py::arg("item"))
        .def("align_zero_on", &align_zero_on_2<ShelfContainer2D>, (py::arg("item"), "path"))
        .def("align_zero_on", &align_zero_on_3<ShelfContainer2D>, (py::arg("item"), "pos"))
        .def("align_zero_on", &align_zero_on_4<ShelfContainer2D>, (py::arg("item"), "path", "pos"),
             u8"Align zero with the horizontal position ``pos`` of the specified item.\n\n"
             u8"This method shifts the local coordinates of the stack horizontally.\n"
             u8"The horizontal coordinate of the shelf origin is placed at the local\n"
             u8"position ``pos`` of the specified item (0 by default).\n\n"
             u8"Args:\n"
             u8"    item: Shelf item which lower edge should lie at height 0.\n"
             u8"    path: Path specifying a particular item instance.\n"
             u8"    pos: Local position of the specified object to place at shelf zero.\n")
        .def("append_gap", &ShelfContainer2D::addGap, py::arg("size"),
             u8"Add a gap to the end of the shelf.\n\n"
             u8"This method adds a gap to the end of the shelf. All consecutive items will be\n"
             u8"separated by the specified width from the previous ones.\n\n"
             u8"Args:\n"
             u8"    size (float): Size of the gap (µm).\n")
        .add_property("flat", &ShelfContainer2D::isFlat,
            u8"True if all items has the same height (the shelf top edge is flat).")
    ;
    py::scope().attr("Shelf") = py::scope().attr("Shelf2D");

    py::class_<MultiStackContainer<plask::ShelfContainer2D>, shared_ptr<MultiStackContainer<ShelfContainer2D>>, py::bases<ShelfContainer2D>, boost::noncopyable>("MultiShelf2D",
        u8"MultiShelf2D(repeat=1, shift=0)\n\n"
        u8"Shelf container that repeats its contents.\n\n"
        u8"The objects are placed in this container like books on a bookshelf.\n"
        u8"The left side of the first object is located at the `shift` position in\n"
        u8"container local coordinates. Each consecutive object is placed to the right\n"
        u8"of the previous one. Then the whole shelf is repeated *repeat* times. All the"
        u8"items are vertically aligned according to its bottom edge.\n\n"
        u8"Args:\n"
        u8"    repeat (int): Number of the shelf contents repetitions.\n"
        u8"    shift (float): Position in the local coordinates of the left side of the\n"
        u8"                   shelf.\n\n"
        u8"See also:\n"
        u8"   Function :func:`plask.geometry.Shelf`.\n",
        py::init<size_t, double>((py::arg("repeat")=1, py::arg("shift")=0.)))
        .add_property("repeat", &MultiStackContainer<plask::ShelfContainer2D>::getRepeatCount, &MultiStackContainer<ShelfContainer2D>::setRepeatCount,
                      u8"Number of repeats of the shelf contents.")
    ;


}



}} // namespace plask::python
