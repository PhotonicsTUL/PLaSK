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
#define PY_ARRAY_UNIQUE_SYMBOL PLASK_ARRAY_API
#define NO_IMPORT_ARRAY

// clang-format off
#include "../python_globals.hpp"
#include "../python_numpy.hpp"
#include "../python_mesh.hpp"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>

#include "plask/mesh/mesh.hpp"
#include "plask/mesh/interpolation.hpp"
#include "plask/mesh/generator_rectangular.hpp"
// clang-format on

#define DIM RectangularMeshRefinedGenerator<dim>::DIM

namespace plask { namespace python {

extern AxisNames current_axes;

template <typename T> shared_ptr<T> __init__empty() { return plask::make_shared<T>(); }

template <typename T> static std::string __str__(const T& self) {
    std::stringstream out;
    out << self;
    return out.str();
}

template <typename To, typename From = To> static shared_ptr<To> Mesh__init__(const From& from) {
    return plask::make_shared<To>(from);
}

namespace detail {

struct OrderedAxis_from_Sequence {
    OrderedAxis_from_Sequence() {
        boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<OrderedAxis>());
    }

    static void* convertible(PyObject* obj) {
        if (!PySequence_Check(obj)) return NULL;
        return obj;
    }

    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((boost::python::converter::rvalue_from_python_storage<OrderedAxis>*)data)->storage.bytes;
        py::stl_input_iterator<double> begin(py::object(py::handle<>(py::borrowed(obj)))), end;
        new (storage) OrderedAxis(std::vector<double>(begin, end));
        data->convertible = storage;
    }
};

struct OrderedAxis_from_SingleNumber {
    OrderedAxis_from_SingleNumber() {
        boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<OrderedAxis>());
    }

    static void* convertible(PyObject* obj) {
        if (PyFloat_Check(obj) || PyLong_Check(obj)) return obj;
        return NULL;
    }

    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((boost::python::converter::rvalue_from_python_storage<OrderedAxis>*)data)->storage.bytes;
        new (storage) OrderedAxis({py::extract<double>(obj)()});
        data->convertible = storage;
    }
};

}  // namespace detail

static py::object OrderedAxis__array__(py::object self, py::object dtype) {
    OrderedAxis* axis = py::extract<OrderedAxis*>(self);
    npy_intp dims[] = {npy_intp(axis->size())};
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)&(*axis->begin()));
    if (arr == nullptr) throw TypeError("cannot create array");
    confirm_array<double>(arr, self, dtype);
    return py::object(py::handle<>(arr));
}

template <typename RectilinearT> shared_ptr<RectilinearT> Rectilinear__init__seq(py::object seq) {
    py::stl_input_iterator<double> begin(seq), end;
    return plask::make_shared<RectilinearT>(std::vector<double>(begin, end));
}

static std::string OrderedAxis__repr__(const OrderedAxis& self) { return "Rectilinear(" + __str__(self) + ")"; }

static py::object OrderedAxis__getitem__(const OrderedAxis& self, const py::object& slice) {
    py::extract<int> index(slice);
    if (index.check()) {
        int i = index();
        if (i < 0) i += int(self.size());
        if (i < 0 || size_t(i) >= self.size()) throw IndexError("axis/mesh index out of range");
        return py::object(self[i]);
    } else {
        if (!PySlice_Check(slice.ptr())) throw TypeError("axis indices must be integers or slices");
        Py_ssize_t start, stop, stride, length;
#       if PY_VERSION_HEX < 0x03060100
            if (PySlice_GetIndicesEx(slice.ptr(), self.size(), &start, &stop, &stride, &length) < 0)
                throw py::error_already_set();
#       else
            if (PySlice_Unpack(slice.ptr(), &start, &stop, &stride) < 0) throw py::error_already_set();
               length = PySlice_AdjustIndices(self.size(), &start, &stop, stride);
#       endif
        std::vector<double> points;
        points.reserve(length);
        for (int i = start; i < stop; i += stride) points.push_back(self[i]);
        return py::object(plask::make_shared<OrderedAxis>(std::move(points)));
    }
}

static void OrderedAxis__delitem__(OrderedAxis& self, const py::object& slice) {
    int size = int(self.size());
    py::extract<int> index(slice);
    if (index.check()) {
        int i = index();
        if (i < 0) i += size;
        if (i < 0 || i >= size) throw IndexError("axis/mesh index out of range");
        self.removePoint(i);
    } else {
        if (!PySlice_Check(slice.ptr())) throw TypeError("axis indices must be integers or slices");
        Py_ssize_t start, stop, step, length;
#       if PY_VERSION_HEX < 0x03060100
            if (PySlice_GetIndicesEx(py::object(slice).ptr(), size, &start, &stop, &step, &length) < 0)
                throw py::error_already_set();
#       else
            if (PySlice_Unpack(py::object(slice).ptr(), &start, &stop, &step) < 0)
                throw py::error_already_set();
            length = PySlice_AdjustIndices(size, &start, &stop, step);
#       endif
        self.removePoints(start, stop, step);
    }
}

static void OrderedAxis_extend(OrderedAxis& self, py::object sequence) {
    py::stl_input_iterator<double> begin(sequence), end;
    std::vector<double> points(begin, end);
    std::sort(points.begin(), points.end());
    self.addOrderedPoints(points.begin(), points.end());
}

/*namespace detail {
    struct RegularAxisFromTupleOrFloat
    {
        RegularAxisFromTupleOrFloat() {
            boost::python::converter::registry::push_back(&convertible, &construct,
boost::python::type_id<RegularAxis>());
        }

        static void* convertible(PyObject* obj) {
            if (PyTuple_Check(obj) || PyFloat_Check(obj) || PyLong_Check(obj)) return obj;
            if (PySequence_Check(obj) && PySequence_Length(obj) == 1) return obj;
            return NULL;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<RegularAxis>*)data)->storage.bytes;
            auto tuple = py::object(py::handle<>(py::borrowed(obj)));
            try {
                if (PyFloat_Check(obj) || PyLong_Check(obj)) {
                    double val = py::extract<double>(tuple);
                    new(storage) RegularAxis(val, val, 1);
                } else if (py::len(tuple) == 1) {
                    double val = py::extract<double>(tuple[0]);
                    new(storage) RegularAxis(val, val, 1);
                } else if (py::len(tuple) == 3) {
                    new(storage) RegularAxis(py::extract<double>(tuple[0]), py::extract<double>(tuple[1]),
py::extract<unsigned>(tuple[2])); } else throw py::error_already_set(); data->convertible = storage; } catch
(py::error_already_set) { throw TypeError("Must provide either mesh.Regular or a tuple (first[, last=first, count=1])");
            }
        }
    };
}*/

template <typename RegularT> shared_ptr<RegularT> Regular__init__one_param(double val) {
    return plask::make_shared<RegularT>(val, val, 1);
}

template <typename RegularT> shared_ptr<RegularT> Regular__init__params(double first, double last, int count) {
    return plask::make_shared<RegularT>(first, last, count);
}

static std::string RegularAxis__repr__(const RegularAxis& self) {
    return format("Regular({0}, {1}, {2})", self.first(), self.last(), self.size());
}

static py::object RegularAxis__getitem__(const RegularAxis& self, const py::object& slice) {
    py::extract<int> index(slice);
    if (index.check()) {
        int i = index();
        if (i < 0) i += int(self.size());
        if (i < 0 || size_t(i) >= self.size()) throw IndexError("axis/mesh index out of range");
        return py::object(self[i]);
    } else {
        if (!PySlice_Check(slice.ptr())) throw TypeError("axis indices must be integers or slices");
        Py_ssize_t start, stop, stride, length;
#       if PY_VERSION_HEX < 0x03060100
            if (PySlice_GetIndicesEx(slice.ptr(), self.size(), &start, &stop, &stride, &length) < 0)
                throw py::error_already_set();
#       else
            if (PySlice_Unpack(slice.ptr(), &start, &stop, &stride) < 0) throw py::error_already_set();
               length = PySlice_AdjustIndices(self.size(), &start, &stop, stride);
#       endif
        double step = self.step() * stride;
        double first = self.first() + self.step() * start;
        return py::object(make_shared<RegularAxis>(first, first + step * (length - 1), length));
    }
}

static void RegularAxis_resize(RegularAxis& self, int count) { self.reset(self.first(), self.last(), count); }

static void RegularAxis_setFirst(RegularAxis& self, double first) { self.reset(first, self.last(), self.size()); }

static void RegularAxis_setLast(RegularAxis& self, double last) { self.reset(self.first(), last, self.size()); }

template <typename MeshT, typename AxesT> static shared_ptr<MeshT> RectangularMesh1D__init__axis(const AxesT& axis) {
    return plask::make_shared<MeshT>(axis);
}

static void RectangularMesh2D__setOrdering(RectangularMesh<2>& self, std::string order) {
    if (order == "best" || order == "optimal")
        self.setOptimalIterationOrder();
    else if (order == "10")
        self.setIterationOrder(RectangularMesh<2>::ORDER_10);
    else if (order == "01")
        self.setIterationOrder(RectangularMesh<2>::ORDER_01);
    else {
        throw ValueError("order must be '01', '10' or 'best'");
    }
}

template <typename MeshT> static shared_ptr<MeshT> RectangularMesh2D__init__empty(std::string order) {
    auto mesh = plask::make_shared<MeshT>();
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<MeshAxis> extract_axis(const py::object& axis) {
    py::extract<shared_ptr<MeshAxis>> convert(axis);
    if (convert.check()) {
        return convert;
    } else if (PySequence_Check(axis.ptr())) {
        py::stl_input_iterator<double> begin(axis), end;
        return plask::make_shared<OrderedAxis>(std::vector<double>(begin, end));
    } else if (PyFloat_Check(axis.ptr()) || PyLong_Check(axis.ptr())) {
        return plask::make_shared<OrderedAxis>(std::initializer_list<double>({py::extract<double>(axis)()}));
    } else {
        throw TypeError("Wrong type of axis, it must derive from Rectangular1D or be a sequence.");
    }
}

static shared_ptr<RectangularMesh<2>> RectangularMesh2D__init__axes(py::object axis0,
                                                                    py::object axis1,
                                                                    std::string order) {
    auto mesh = plask::make_shared<RectangularMesh<2>>(extract_axis(axis0), extract_axis(axis1));
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

static Vec<2, double> RectangularMesh2D__getitem__(const RectangularMesh<2>& self, py::object index) {
    try {
        int indx = py::extract<int>(index);
        if (indx < 0) indx += int(self.size());
        if (indx < 0 || size_t(indx) >= self.size()) throw IndexError("mesh index out of range");
        return self[indx];
    } catch (py::error_already_set&) {
        PyErr_Clear();
    }
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 += int(self.axis[0]->size());
    if (index0 < 0 || index0 >= int(self.axis[0]->size())) {
        throw IndexError("first mesh index ({0}) out of range (0<=index<{1})", index0, self.axis[0]->size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 += int(self.axis[1]->size());
    if (index1 < 0 || index1 >= int(self.axis[1]->size())) {
        throw IndexError("second mesh index ({0}) out of range (0<=index<{1})", index1, self.axis[1]->size());
    }
    return self(index0, index1);
}

static std::string RectangularMesh2D__getOrdering(RectangularMesh<2>& self) {
    return (self.getIterationOrder() == RectangularMesh<2>::ORDER_10) ? "10" : "01";
}

void RectangularMesh3D__setOrdering(RectangularMesh<3>& self, std::string order) {
    if (order == "best" || order == "optimal")
        self.setOptimalIterationOrder();
    else if (order == "012")
        self.setIterationOrder(RectangularMesh<3>::ORDER_012);
    else if (order == "021")
        self.setIterationOrder(RectangularMesh<3>::ORDER_021);
    else if (order == "102")
        self.setIterationOrder(RectangularMesh<3>::ORDER_102);
    else if (order == "120")
        self.setIterationOrder(RectangularMesh<3>::ORDER_120);
    else if (order == "201")
        self.setIterationOrder(RectangularMesh<3>::ORDER_201);
    else if (order == "210")
        self.setIterationOrder(RectangularMesh<3>::ORDER_210);
    else {
        throw ValueError("order must be any permutation of '012' or 'best'");
    }
}

std::string RectangularMesh3D__getOrdering(RectangularMesh<3>& self) {
    switch (self.getIterationOrder()) {
        case RectangularMesh<3>::ORDER_012: return "012";
        case RectangularMesh<3>::ORDER_021: return "021";
        case RectangularMesh<3>::ORDER_102: return "102";
        case RectangularMesh<3>::ORDER_120: return "120";
        case RectangularMesh<3>::ORDER_201: return "201";
        case RectangularMesh<3>::ORDER_210: return "210";
    }
    return "unknown";
}

template <typename MeshT> shared_ptr<MeshT> RectangularMesh3D__init__empty(std::string order) {
    auto mesh = plask::make_shared<MeshT>();
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectangularMesh<3>> RectangularMesh3D__init__axes(py::object axis0,
                                                             py::object axis1,
                                                             py::object axis2,
                                                             std::string order) {
    auto mesh = plask::make_shared<RectangularMesh<3>>(extract_axis(axis0), extract_axis(axis1), extract_axis(axis2));
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}

template <typename MeshT> Vec<3, double> RectangularMesh3D__getitem__(const MeshT& self, py::object index) {
    try {
        int indx = py::extract<int>(index);
        if (indx < 0) indx += int(self.size());
        if (indx < 0 || size_t(indx) >= self.size()) throw IndexError("mesh index out of range");
        return self[indx];
    } catch (py::error_already_set&) {
        PyErr_Clear();
    }
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 += int(self.axis[0]->size());
    if (index0 < 0 || index0 >= int(self.axis[0]->size())) {
        throw IndexError("first mesh index ({0}) out of range (0<=index<{1})", index0, self.axis[0]->size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 += int(self.axis[1]->size());
    if (index1 < 0 || index1 >= int(self.axis[1]->size())) {
        throw IndexError("second mesh index ({0}) out of range (0<=index<{1})", index1, self.axis[1]->size());
    }
    int index2 = py::extract<int>(index[2]);
    if (index2 < 0) index2 = int(self.axis[2]->size());
    if (index2 < 0 || index2 >= int(self.axis[2]->size())) {
        throw IndexError("third mesh index ({0}) out of range (0<=index<{1})", index2, self.axis[2]->size());
    }
    return self(index0, index1, index2);
}

shared_ptr<RectangularMesh<2>> RectilinearMesh2D__init__geometry(const shared_ptr<GeometryObjectD<2>>& geometry,
                                                                 std::string order) {
    auto mesh = RectangularMesh2DSimpleGenerator().generate_t<RectangularMesh<2>>(geometry);
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectangularMesh<3>> RectilinearMesh3D__init__geometry(const shared_ptr<GeometryObjectD<3>>& geometry,
                                                                 std::string order) {
    auto mesh = RectangularMesh3DSimpleGenerator().generate_t<RectangularMesh<3>>(geometry);
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}

namespace detail {

template <typename T, int dim, typename GT> struct AxisParamProxy {
    typedef AxisParamProxy<T, dim, GT> ThisT;

    typedef T (GT::*GetF)(typename Primitive<DIM>::Direction) const;
    typedef void (GT::*SetF)(typename Primitive<DIM>::Direction, T);

    GT& obj;
    GetF getter;
    SetF setter;

    AxisParamProxy(GT& obj, GetF get, SetF set) : obj(obj), getter(get), setter(set) {}

    T get(int i) const { return (obj.*getter)(typename Primitive<DIM>::Direction(i)); }

    T __getitem__(int i) const {
        if (i < 0) i += dim;
        if (i > dim || i < 0) throw IndexError("tuple index out of range");
        return get(i);
    }

    void set(int i, T v) { (obj.*setter)(typename Primitive<DIM>::Direction(i), v); }

    void __setitem__(int i, T v) {
        if (i < 0) i += dim;
        if (i > dim || i < 0) throw IndexError("tuple index out of range");
        set(i, v);
    }

    py::tuple __mul__(T f) const;

    py::tuple __div__(T f) const;

    std::string __str__() const;

    struct Iter {
        const ThisT& obj;
        int i;
        Iter(const ThisT& obj) : obj(obj), i(-1) {}
        T next() {
            ++i;
            if (i == dim) throw StopIteration("");
            return obj.get(i);
        }
    };

    shared_ptr<Iter> __iter__() { return plask::make_shared<Iter>(*this); }

    // even if unused, scope argument is important as it sets python scope
    static void register_proxy(py::scope /*scope*/) {
        py::class_<ThisT, shared_ptr<ThisT>, boost::noncopyable> cls("_Proxy", py::no_init);
        cls.def("__getitem__", &ThisT::__getitem__)
            .def("__setitem__", &ThisT::__setitem__)
            .def("__mul__", &ThisT::__mul__)
            .def("__div__", &ThisT::__div__)
            .def("__truediv__", &ThisT::__div__)
            .def("__floordiv__", &ThisT::__div__)
            .def("__iter__", &ThisT::__iter__, py::with_custodian_and_ward_postcall<0, 1>())
            .def("__str__", &ThisT::__str__);
        py::delattr(py::scope(), "_Proxy");

        py::scope scope2 = cls;
        (void)scope2;  // don't warn about unused variable scope2
        py::class_<Iter, shared_ptr<Iter>, boost::noncopyable>("_Iterator", py::no_init)
            .def("__next__", &Iter::next)
            .def("__iter__", pass_through);
    }
};

template <> py::tuple AxisParamProxy<size_t, 2, RectangularMeshDivideGenerator<2>>::__mul__(size_t f) const {
    return py::make_tuple(get(0) * f, get(1) * f);
}
template <> py::tuple AxisParamProxy<size_t, 3, RectangularMeshDivideGenerator<3>>::__mul__(size_t f) const {
    return py::make_tuple(get(0) * f, get(1) * f, get(2) * f);
}

template <> py::tuple AxisParamProxy<size_t, 2, RectangularMeshDivideGenerator<2>>::__div__(size_t f) const {
    if (get(0) < f || get(1) < f) throw ValueError("Refinement already too small.");
    return py::make_tuple(get(0) / f, get(1) / f);
}
template <> py::tuple AxisParamProxy<size_t, 3, RectangularMeshDivideGenerator<3>>::__div__(size_t f) const {
    if (get(0) < f || get(1) < f || get(2) < f) throw ValueError("Refinement already too small.");
    return py::make_tuple(get(0) / f, get(1) / f, get(2) / f);
}

template <> std::string AxisParamProxy<size_t, 2, RectangularMeshDivideGenerator<2>>::__str__() const {
    return format("({0}, {1})", get(0), get(1));
}
template <> std::string AxisParamProxy<size_t, 3, RectangularMeshDivideGenerator<3>>::__str__() const {
    return format("({0}, {1}, {2})", get(0), get(1), get(2));
}

template <> py::tuple AxisParamProxy<double, 2, RectangularMeshSmoothGenerator<2>>::__mul__(double f) const {
    return py::make_tuple(get(0) * f, get(1) * f);
}
template <> py::tuple AxisParamProxy<double, 3, RectangularMeshSmoothGenerator<3>>::__mul__(double f) const {
    return py::make_tuple(get(0) * f, get(1) * f, get(2) * f);
}

template <> py::tuple AxisParamProxy<double, 2, RectangularMeshSmoothGenerator<2>>::__div__(double f) const {
    return py::make_tuple(get(0) / f, get(1) / f);
}
template <> py::tuple AxisParamProxy<double, 3, RectangularMeshSmoothGenerator<3>>::__div__(double f) const {
    return py::make_tuple(get(0) / f, get(1) / f, get(2) / f);
}

template <> std::string AxisParamProxy<double, 2, RectangularMeshSmoothGenerator<2>>::__str__() const {
    return format("({0}, {1})", get(0), get(1));
}
template <> std::string AxisParamProxy<double, 3, RectangularMeshSmoothGenerator<3>>::__str__() const {
    return format("({0}, {1}, {2})", get(0), get(1), get(2));
}

template <int dim> struct DivideGeneratorDivMethods {
    typedef AxisParamProxy<size_t, dim, RectangularMeshDivideGenerator<dim>> ProxyT;

    static shared_ptr<ProxyT> getPre(RectangularMeshDivideGenerator<dim>& self) {
        return plask::make_shared<ProxyT>(self, &RectangularMeshDivideGenerator<dim>::getPreDivision,
                                          &RectangularMeshDivideGenerator<dim>::setPreDivision);
    }

    static shared_ptr<ProxyT> getPost(RectangularMeshDivideGenerator<dim>& self) {
        return plask::make_shared<ProxyT>(self, &RectangularMeshDivideGenerator<dim>::getPostDivision,
                                          &RectangularMeshDivideGenerator<dim>::setPostDivision);
    }

    static void setPre(RectangularMeshDivideGenerator<dim>& self, py::object val) {
        // try {
        //     size_t v = py::extract<size_t>(val);
        //     for (int i = 0; i < dim; ++i) self.pre_divisions[i] = v;
        // } catch (py::error_already_set) {
        //     PyErr_Clear();
        if (py::len(val) != dim)
            throw ValueError("Wrong size of 'prediv' ({0} items provided and {1} required)", py::len(val), dim);
        for (int i = 0; i < dim; ++i) self.pre_divisions[i] = py::extract<size_t>(val[i]);
        // }
        self.fireChanged();
    }

    static void setPost(RectangularMeshDivideGenerator<dim>& self, py::object val) {
        // try {
        //     size_t v = py::extract<size_t>(val);
        //     for (int i = 0; i < dim; ++i) self.post_divisions[i] = v;
        // } catch (py::error_already_set) {
        //     PyErr_Clear();
        if (py::len(val) != dim)
            throw ValueError("Wrong size of 'postdiv' ({0} items provided and {1} required)", py::len(val), dim);
        for (int i = 0; i < dim; ++i) self.post_divisions[i] = py::extract<size_t>(val[i]);
        // }
        self.fireChanged();
    }

    static void register_proxy(py::object scope) {
        AxisParamProxy<size_t, dim, RectangularMeshDivideGenerator<dim>>::register_proxy(scope);
    }
};

template <> struct DivideGeneratorDivMethods<1> {
    static size_t getPre(RectangularMeshDivideGenerator<1>& self) {
        return self.getPreDivision(Primitive<2>::DIRECTION_TRAN);
    }

    static size_t getPost(RectangularMeshDivideGenerator<1>& self) {
        return self.getPostDivision(Primitive<2>::DIRECTION_TRAN);
    }

    static void setPre(RectangularMeshDivideGenerator<1>& self, py::object val) {
        self.setPreDivision(Primitive<2>::DIRECTION_TRAN, py::extract<size_t>(val));
    }

    static void setPost(RectangularMeshDivideGenerator<1>& self, py::object val) {
        self.setPostDivision(Primitive<2>::DIRECTION_TRAN, py::extract<size_t>(val));
    }

    static void register_proxy(py::object) {}
};

template <int dim> struct SmoothGeneratorParamMethods {
    typedef AxisParamProxy<double, dim, RectangularMeshSmoothGenerator<dim>> ProxyT;

    static shared_ptr<ProxyT> getSmall(RectangularMeshSmoothGenerator<dim>& self) {
        return plask::make_shared<ProxyT>(self, &RectangularMeshSmoothGenerator<dim>::getFineStep,
                                          &RectangularMeshSmoothGenerator<dim>::setFineStep);
    }

    static shared_ptr<ProxyT> getLarge(RectangularMeshSmoothGenerator<dim>& self) {
        return plask::make_shared<ProxyT>(self, &RectangularMeshSmoothGenerator<dim>::getMaxStep,
                                          &RectangularMeshSmoothGenerator<dim>::setMaxStep);
    }

    static shared_ptr<ProxyT> getFactor(RectangularMeshSmoothGenerator<dim>& self) {
        return plask::make_shared<ProxyT>(self, &RectangularMeshSmoothGenerator<dim>::getFactor,
                                          &RectangularMeshSmoothGenerator<dim>::setFactor);
    }

    static void setSmall(RectangularMeshSmoothGenerator<dim>& self, py::object val) {
        // try {
        //     double v = py::extract<double>(val);
        //     for (int i = 0; i < dim; ++i) self.finestep[i] = v;
        // } catch (py::error_already_set) {
        //     PyErr_Clear();
        if (py::len(val) != dim)
            throw ValueError("Wrong size of 'small' ({0} items provided and {1} required)", py::len(val), dim);
        for (int i = 0; i < dim; ++i) self.finestep[i] = py::extract<double>(val[i]);
        // }
        self.fireChanged();
    }

    static void setLarge(RectangularMeshSmoothGenerator<dim>& self, py::object val) {
        // try {
        //     double v = py::extract<double>(val);
        //     for (int i = 0; i < dim; ++i) self.finestep[i] = v;
        // } catch (py::error_already_set) {
        //     PyErr_Clear();
        if (py::len(val) != dim)
            throw ValueError("Wrong size of 'large' ({0} items provided and {1} required)", py::len(val), dim);
        for (int i = 0; i < dim; ++i) self.maxstep[i] = py::extract<double>(val[i]);
        // }
        self.fireChanged();
    }

    static void setFactor(RectangularMeshSmoothGenerator<dim>& self, py::object val) {
        // try {
        //     double v = py::extract<double>(val);
        //     for (int i = 0; i < dim; ++i) self.factor[i] = v;
        // } catch (py::error_already_set) {
        //     PyErr_Clear();
        if (py::len(val) != dim)
            throw ValueError("Wrong size of 'factor' ({0} items provided and {1} required)", py::len(val), dim);
        for (int i = 0; i < dim; ++i) self.factor[i] = py::extract<double>(val[i]);
        // }
        self.fireChanged();
    }

    static void register_proxy(py::object scope) {
        AxisParamProxy<double, dim, RectangularMeshSmoothGenerator<dim>>::register_proxy(scope);
    }
};

template <> struct SmoothGeneratorParamMethods<1> {
    static double getSmall(RectangularMeshSmoothGenerator<1>& self) {
        return self.getFineStep(Primitive<2>::DIRECTION_TRAN);
    }

    static double getLarge(RectangularMeshSmoothGenerator<1>& self) {
        return self.getMaxStep(Primitive<2>::DIRECTION_TRAN);
    }

    static double getFactor(RectangularMeshSmoothGenerator<1>& self) {
        return self.getFactor(Primitive<2>::DIRECTION_TRAN);
    }

    static void setSmall(RectangularMeshSmoothGenerator<1>& self, py::object val) {
        self.setFineStep(Primitive<2>::DIRECTION_TRAN, py::extract<double>(val));
    }

    static void setLarge(RectangularMeshSmoothGenerator<1>& self, py::object val) {
        self.setMaxStep(Primitive<2>::DIRECTION_TRAN, py::extract<double>(val));
    }

    static void setFactor(RectangularMeshSmoothGenerator<1>& self, py::object val) {
        self.setFactor(Primitive<2>::DIRECTION_TRAN, py::extract<double>(val));
    }

    static void register_proxy(py::object) {}
};
}  // namespace detail

template <int dim>
void RectangularMeshRefinedGenerator_addRefinement1(RectangularMeshDivideGenerator<dim>& self,
                                                    const std::string& axis,
                                                    GeometryObjectD<DIM>& object,
                                                    const PathHints& path,
                                                    double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i),
                       dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), path, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_addRefinement2(RectangularMeshDivideGenerator<dim>& self,
                                                    const std::string& axis,
                                                    GeometryObjectD<DIM>& object,
                                                    double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i),
                       dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), position);
}

template <int dim>
void RectangularMeshRefinedGenerator_addRefinement3(RectangularMeshDivideGenerator<dim>& self,
                                                    const std::string& axis,
                                                    GeometryObject::Subtree subtree,
                                                    double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i), subtree, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_addRefinement4(RectangularMeshDivideGenerator<dim>& self,
                                                    const std::string& axis,
                                                    Path path,
                                                    double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i), path, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinement1(RectangularMeshDivideGenerator<dim>& self,
                                                       const std::string& axis,
                                                       GeometryObjectD<DIM>& object,
                                                       const PathHints& path,
                                                       double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i),
                          dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), path, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinement2(RectangularMeshDivideGenerator<dim>& self,
                                                       const std::string& axis,
                                                       GeometryObjectD<DIM>& object,
                                                       double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i),
                          dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), position);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinement3(RectangularMeshDivideGenerator<dim>& self,
                                                       const std::string& axis,
                                                       GeometryObject::Subtree subtree,
                                                       double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i), subtree, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinement4(RectangularMeshDivideGenerator<dim>& self,
                                                       const std::string& axis,
                                                       Path path,
                                                       double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i), path, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinements1(RectangularMeshDivideGenerator<dim>& self,
                                                        GeometryObjectD<DIM>& object,
                                                        const PathHints& path) {
    self.removeRefinements(dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), path);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinements2(RectangularMeshDivideGenerator<dim>& self, const Path& path) {
    self.removeRefinements(path);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinements3(RectangularMeshDivideGenerator<dim>& self,
                                                        const GeometryObject::Subtree& subtree) {
    self.removeRefinements(subtree);
}

template <int dim>
py::dict RectangularMeshRefinedGenerator_listRefinements(const RectangularMeshDivideGenerator<dim>& self,
                                                         const std::string& axis) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    py::dict refinements;
    for (auto refinement : self.getRefinements(typename Primitive<DIM>::Direction(i))) {
        py::object object{const_pointer_cast<GeometryObjectD<DIM>>(refinement.first.first.lock())};
        auto pth = refinement.first.second;
        py::object path;
        if (pth.hintFor.size() != 0) path = py::object(pth);
        py::list refs;
        for (auto x : refinement.second) {
            refs.append(x);
        }
        refinements[py::make_tuple(object, path)] = refs;
    }
    return refinements;
}

template <int dim, typename RegisterT> static void register_refined_generator_base(RegisterT& cls) {
    cls.add_property("aspect", &RectangularMeshDivideGenerator<dim>::getAspect,
                     &RectangularMeshDivideGenerator<dim>::setAspect,
                     u8"Maximum aspect ratio for the elements generated by this generator.")
        .def("add_refinement", &RectangularMeshRefinedGenerator_addRefinement1<dim>,
             u8"Add a refining line inside the object", (py::arg("axis"), "object", "path", "at"))
        .def("add_refinement", &RectangularMeshRefinedGenerator_addRefinement2<dim>,
             u8"Add a refining line inside the object", (py::arg("axis"), "object", "at"))
        .def("add_refinement", &RectangularMeshRefinedGenerator_addRefinement3<dim>,
             u8"Add a refining line inside the object", (py::arg("axis"), "subtree", "at"))
        .def("add_refinement", &RectangularMeshRefinedGenerator_addRefinement4<dim>,
             u8"Add a refining line inside the object", (py::arg("axis"), "path", "at"))
        .def("remove_refinement", &RectangularMeshRefinedGenerator_removeRefinement1<dim>,
             u8"Remove the refining line from the object", (py::arg("axis"), "object", "path", "at"))
        .def("remove_refinement", &RectangularMeshRefinedGenerator_removeRefinement2<dim>,
             u8"Remove the refining line from the object", (py::arg("axis"), "object", "at"))
        .def("remove_refinement", &RectangularMeshRefinedGenerator_removeRefinement3<dim>,
             u8"Remove the refining line from the object", (py::arg("axis"), "subtree", "at"))
        .def("remove_refinement", &RectangularMeshRefinedGenerator_removeRefinement4<dim>,
             u8"Remove the refining line from the object", (py::arg("axis"), "path", "at"))
        .def("remove_refinements", &RectangularMeshRefinedGenerator_removeRefinements1<dim>,
             u8"Remove the all refining lines from the object", (py::arg("object"), py::arg("path") = py::object()))
        .def("remove_refinements", &RectangularMeshRefinedGenerator_removeRefinements2<dim>,
             u8"Remove the all refining lines from the object", py::arg("path"))
        .def("remove_refinements", &RectangularMeshRefinedGenerator_removeRefinements3<dim>,
             u8"Remove the all refining lines from the object", py::arg("subtree"))
        .def("clear_refinements", &RectangularMeshDivideGenerator<dim>::clearRefinements, u8"Clear all refining lines",
             py::arg("subtree"))
        .def("get_refinements", &RectangularMeshRefinedGenerator_listRefinements<dim>, py::arg("axis"),
             u8"Get list of all the refinements defined for this generator for specified axis");
}

template <int dim>
shared_ptr<RectangularMeshDivideGenerator<dim>> RectangularMeshDivideGenerator__init__(py::object prediv,
                                                                                       py::object postdiv,
                                                                                       double aspect,
                                                                                       py::object gradual) {
    auto result = plask::make_shared<RectangularMeshDivideGenerator<dim>>();
    if (!prediv.is_none()) detail::DivideGeneratorDivMethods<dim>::setPre(*result, prediv);
    if (!postdiv.is_none()) detail::DivideGeneratorDivMethods<dim>::setPost(*result, postdiv);
    if (gradual.ptr() == Py_True)
        result->gradual = 255;
    else if (gradual.ptr() == Py_False)
        result->gradual = 0;
    else {
        result->gradual = 0;
        for (int i = 0; i != dim; ++i)
            result->setGradual(i, py::extract<bool>(gradual[i]));
    }
    result->aspect = aspect;
    return result;
};

template <int dim>
py::object RectangularMeshDivideGenerator_getGradual(const RectangularMeshDivideGenerator<dim>& self);

template <>
py::object RectangularMeshDivideGenerator_getGradual<1>(const RectangularMeshDivideGenerator<1>& self) {
    return py::object(bool(self.gradual));
}

template <>
py::object RectangularMeshDivideGenerator_getGradual<2>(const RectangularMeshDivideGenerator<2>& self) {
    return py::make_tuple(self.getGradual(0), self.getGradual(1));
}

template <>
py::object RectangularMeshDivideGenerator_getGradual<3>(const RectangularMeshDivideGenerator<3>& self) {
    return py::make_tuple(self.getGradual(0), self.getGradual(1), self.getGradual(2));
}

template <int dim>
void RectangularMeshDivideGenerator_setGradual(RectangularMeshDivideGenerator<dim>& self, py::object value) {
    if (value.ptr() == Py_True) {
        self.gradual = 7;
        self.fireChanged();
    } else if (value.ptr() == Py_False) {
        self.gradual = 0;
        self.fireChanged();
    } else {
        self.gradual = 0;
        for (int i = 0; i != dim; ++i)
            self.setGradual(i, py::extract<bool>(value[i]));
    }
}



template <int dim> void register_divide_generator() {
    py::class_<RectangularMeshDivideGenerator<dim>, shared_ptr<RectangularMeshDivideGenerator<dim>>,
               py::bases<MeshGeneratorD<dim>>, boost::noncopyable>
        dividecls("DivideGenerator",
                  format(u8"Generator of Rectilinear{0}D mesh by simple division of the geometry.\n\n"
                         u8"DivideGenerator()\n"
                         u8"    create generator without initial division of geometry objects",
                         dim)
                      .c_str(),
                  py::no_init);
    register_refined_generator_base<dim>(dividecls);
    dividecls
        .def("__init__",
             py::make_constructor(&RectangularMeshDivideGenerator__init__<dim>, py::default_call_policies(),
                                  (py::arg("prediv") = py::object(), py::arg("postdiv") = py::object(),
                                   py::arg("aspect") = 0, py::arg("gradual") = true)))
        .add_property("gradual", &RectangularMeshDivideGenerator_getGradual<dim>,
                      &RectangularMeshDivideGenerator_setGradual<dim>,
                      "Limit maximum adjacent objects size change to the factor of two.");
    py::implicitly_convertible<shared_ptr<RectangularMeshDivideGenerator<dim>>,
                               shared_ptr<const RectangularMeshDivideGenerator<dim>>>();

    if (dim != 1)
        dividecls
            .add_property("prediv",
                          py::make_function(&detail::DivideGeneratorDivMethods<dim>::getPre,
                                            py::with_custodian_and_ward_postcall<0, 1>()),
                          &detail::DivideGeneratorDivMethods<dim>::setPre, u8"initial division of all geometry objects")
            .add_property("postdiv",
                          py::make_function(&detail::DivideGeneratorDivMethods<dim>::getPost,
                                            py::with_custodian_and_ward_postcall<0, 1>()),
                          &detail::DivideGeneratorDivMethods<dim>::setPost, u8"final division of all geometry objects");
    else
        dividecls
            .add_property("prediv", &detail::DivideGeneratorDivMethods<dim>::getPre,
                          &detail::DivideGeneratorDivMethods<dim>::setPre, u8"initial division of all geometry objects")
            .add_property("postdiv", &detail::DivideGeneratorDivMethods<dim>::getPost,
                          &detail::DivideGeneratorDivMethods<dim>::setPost, u8"final division of all geometry objects");

    detail::DivideGeneratorDivMethods<dim>::register_proxy(dividecls);
}

template <int dim>
shared_ptr<RectangularMeshSmoothGenerator<dim>> RectangularMeshSmoothGenerator__init__(py::object small_,
                                                                                       py::object large,
                                                                                       py::object factor,
                                                                                       double aspect) {
    auto result = plask::make_shared<RectangularMeshSmoothGenerator<dim>>();
    if (!small_.is_none()) detail::SmoothGeneratorParamMethods<dim>::setSmall(*result, small_);
    if (!large.is_none()) detail::SmoothGeneratorParamMethods<dim>::setLarge(*result, large);
    if (!factor.is_none()) detail::SmoothGeneratorParamMethods<dim>::setFactor(*result, factor);
    result->aspect = aspect;
    return result;
}

static shared_ptr<RectangularMesh2DRegularGenerator> RectangularMesh2DRegularGenerator__init__1(double spacing,
                                                                                                bool split) {
    return make_shared<RectangularMesh2DRegularGenerator>(spacing, split);
}

static shared_ptr<RectangularMesh2DRegularGenerator> RectangularMesh2DRegularGenerator__init__2(
    double spacing0,
    const py::object& spacing1) {
    if (PyBool_Check(spacing1.ptr()))
        return make_shared<RectangularMesh2DRegularGenerator>(spacing0, py::extract<bool>(spacing1)());
    else
        return make_shared<RectangularMesh2DRegularGenerator>(spacing0, py::extract<double>(spacing1)());
}

static shared_ptr<RectangularMesh2DRegularGenerator> RectangularMesh2DRegularGenerator__init__3(double spacing0,
                                                                                                double spacing1,
                                                                                                bool split) {
    return make_shared<RectangularMesh2DRegularGenerator>(spacing0, spacing1, split);
}

static shared_ptr<RectangularMesh3DRegularGenerator> RectangularMesh3DRegularGenerator__init__1(double spacing,
                                                                                                bool split) {
    return make_shared<RectangularMesh3DRegularGenerator>(spacing, split);
}

static shared_ptr<RectangularMesh3DRegularGenerator> RectangularMesh3DRegularGenerator__init__3(double spacing0,
                                                                                                double spacing1,
                                                                                                double spacing2,
                                                                                                bool split) {
    return make_shared<RectangularMesh3DRegularGenerator>(spacing0, spacing1, spacing2, split);
}

template <int dim> void register_smooth_generator() {
    py::class_<RectangularMeshSmoothGenerator<dim>, shared_ptr<RectangularMeshSmoothGenerator<dim>>,
               py::bases<MeshGeneratorD<dim>>, boost::noncopyable>
        dividecls("SmoothGenerator",
                  format(u8"Generator of Rectilinear{0}D mesh with dense sampling at edges and smooth change of "
                         u8"element size.\n\n"
                         u8"SmoothGenerator()\n"
                         u8"    create generator without initial division of geometry objects",
                         dim)
                      .c_str(),
                  py::no_init);
    register_refined_generator_base<dim>(dividecls);
    dividecls.def(
        "__init__",
        py::make_constructor(&RectangularMeshSmoothGenerator__init__<dim>, py::default_call_policies(),
                             (py::arg("small") = py::object(), py::arg("large") = py::object(),
                              py::arg("factor") = py::object(), py::arg("aspect") = 0)));
    py::implicitly_convertible<shared_ptr<RectangularMeshSmoothGenerator<dim>>,
                               shared_ptr<const RectangularMeshSmoothGenerator<dim>>>();

    if (dim != 1)
        dividecls
            .add_property("small",
                          py::make_function(&detail::SmoothGeneratorParamMethods<dim>::getSmall,
                                            py::with_custodian_and_ward_postcall<0, 1>()),
                          &detail::SmoothGeneratorParamMethods<dim>::setSmall,
                          u8"small size of mesh elements near object edges along each axis")
            .add_property("large",
                          py::make_function(&detail::SmoothGeneratorParamMethods<dim>::getLarge,
                                            py::with_custodian_and_ward_postcall<0, 1>()),
                          &detail::SmoothGeneratorParamMethods<dim>::setLarge,
                          u8"maximum size of mesh elements along each axis")
            .add_property("factor",
                          py::make_function(&detail::SmoothGeneratorParamMethods<dim>::getFactor,
                                            py::with_custodian_and_ward_postcall<0, 1>()),
                          &detail::SmoothGeneratorParamMethods<dim>::setFactor,
                          u8"factor by which element sizes increase along each axis");
    else
        dividecls
            .add_property("small", &detail::SmoothGeneratorParamMethods<dim>::getSmall,
                          &detail::SmoothGeneratorParamMethods<dim>::setSmall,
                          u8"small size of mesh elements near object edges along each axis")
            .add_property("large", &detail::SmoothGeneratorParamMethods<dim>::getLarge,
                          &detail::SmoothGeneratorParamMethods<dim>::setLarge,
                          u8"maximum size of mesh elements along each axis")
            .add_property("factor", &detail::SmoothGeneratorParamMethods<dim>::getFactor,
                          &detail::SmoothGeneratorParamMethods<dim>::setFactor,
                          u8"factor by which element sizes increase along each axis");

    detail::SmoothGeneratorParamMethods<dim>::register_proxy(dividecls);
}

template <int dim> shared_ptr<RectangularMesh<dim>> RectangularMesh_getMidpoints(const RectangularMesh<dim>& src) {
    writelog(LOG_WARNING, "RectangularMesh{0}D.get_midpoints() is obsolete: use RectangularMesh{0}D.elements.mesh",
             dim);
    return src.getElementMesh();
}

shared_ptr<MeshAxis> MeshAxis_getMidpoints(const MeshAxis& src) {
    writelog(LOG_WARNING, "Axis.get_midpoints() is obsolete: use Axis.midpoints");
    return src.getMidpointAxis();
}

template <typename MeshT> shared_ptr<MeshT> RectangularMesh_ElementMesh(const typename MeshT::Elements& self) {
    return static_cast<const MeshT*>(self.mesh)->getElementMesh();
}

static py::tuple RectangularMesh2D_Element_nodes(const RectangularMesh2D::Element& self) {
    return py::make_tuple(self.getLoLoIndex(), self.getLoUpIndex(), self.getUpLoIndex(), self.getUpUpIndex());
}

static py::tuple RectangularMesh3D_Element_nodes(const RectilinearMesh3D::Element& self) {
    return py::make_tuple(self.getLoLoLoIndex(), self.getLoLoUpIndex(), self.getLoUpLoIndex(), self.getLoUpUpIndex(),
                          self.getUpLoLoIndex(), self.getUpLoUpIndex(), self.getUpUpLoIndex(), self.getUpUpUpIndex());
}

static RectilinearMesh3D::Elements RectangularMesh3D_elements(const RectangularMesh3D& self) { return self.elements(); }

static Box3D RectangularMesh3D_Element_box(const RectilinearMesh3D::Element& self) {
    return Box3D(self.getLoLoLo(), self.getUpUpUp());
}

static double RectangularMesh3D_Element_volume(const RectilinearMesh3D::Element& self) {
    Vec<3, double> span = self.getSize();
    return span.c0 * span.c1 * span.c2;
}

void register_mesh_rectangular() {
    py::class_<MeshAxis, shared_ptr<MeshAxis>, py::bases<MeshD<1>>, boost::noncopyable>(
        "Axis", u8"Base class for all 1D meshes (used as axes by 2D and 3D rectangular meshes).", py::no_init)
        .add_property("midpoints", &MeshAxis::getMidpointAxis,
                      u8"Mesh with points in the middles of elements of this mesh")
        .def("get_midpoints", &MeshAxis_getMidpoints)

        ;

    py::class_<OrderedAxis, shared_ptr<OrderedAxis>, py::bases<MeshAxis>> rectilinear1d(
        "Ordered",
        u8"One-dimesnional rectilinear mesh, used also as rectangular mesh axis\n\n"
        u8"Ordered()\n    create empty mesh\n\n"
        u8"Ordered(points)\n    create mesh filled with points provides in sequence type");
    rectilinear1d.def("__init__", py::make_constructor(&__init__empty<OrderedAxis>))
        .def("__init__", py::make_constructor(&Rectilinear__init__seq<OrderedAxis>, py::default_call_policies(),
                                              (py::arg("points"))))
        .def("__getitem__", &OrderedAxis__getitem__)
        .def("__delitem__", &OrderedAxis__delitem__)
        .def("__str__", &__str__<OrderedAxis>)
        .def("__repr__", &OrderedAxis__repr__)
        .def("__array__", &OrderedAxis__array__, py::arg("dtype") = py::object())
        .def("insert", (bool (OrderedAxis::*)(double)) & OrderedAxis::addPoint, "Insert point into the mesh",
             (py::arg("point")))
        .def("extend", &OrderedAxis_extend, "Insert points from the sequence to the mesh", (py::arg("points")))
        .def(py::self == py::self)
        .def("__iter__", py::range(&OrderedAxis::begin, &OrderedAxis::end));
    detail::OrderedAxis_from_Sequence();
    detail::OrderedAxis_from_SingleNumber();
    py::implicitly_convertible<shared_ptr<OrderedAxis>, shared_ptr<const OrderedAxis>>();

    {
        py::scope scope = rectilinear1d;
        (void)scope;  // don't warn about unused variable scope

        py::class_<OrderedMesh1DSimpleGenerator, shared_ptr<OrderedMesh1DSimpleGenerator>, py::bases<MeshGeneratorD<1>>,
                   boost::noncopyable>(
            "SimpleGenerator",
            u8"Generator of ordered 1D mesh with lines at transverse edges of all objects.\n\n"
            u8"SimpleGenerator()\n    create generator");
        py::implicitly_convertible<shared_ptr<OrderedMesh1DSimpleGenerator>,
                                   shared_ptr<const OrderedMesh1DSimpleGenerator>>();

        py::class_<OrderedMesh1DRegularGenerator, shared_ptr<OrderedMesh1DRegularGenerator>,
                   py::bases<MeshGeneratorD<1>>, boost::noncopyable>(
            "RegularGenerator",
            u8"Generator of ordered 1D mesh with lines at transverse edges of all objects\n"
            u8"and fine regular division of each object with spacing approximately equal to\n"
            u8"specified spacing\n\n"
            u8"RegularGenerator(spacing)\n    create generator",
            py::init<double>(py::arg("spacing")));
        py::implicitly_convertible<shared_ptr<OrderedMesh1DRegularGenerator>,
                                   shared_ptr<const OrderedMesh1DRegularGenerator>>();

        register_divide_generator<1>();
        register_smooth_generator<1>();
    }

    py::class_<RegularAxis, shared_ptr<RegularAxis>, py::bases<MeshAxis>>(
        "Regular",
        u8"One-dimesnional regular mesh, used also as rectangular mesh axis\n\n"
        u8"Regular()\n    create empty mesh\n\n"
        u8"Regular(start, stop, num)\n    create mesh of count points equally distributed between start and stop")
        .def("__init__", py::make_constructor(&__init__empty<RegularAxis>))
        .def("__init__", py::make_constructor(&Regular__init__one_param<RegularAxis>, py::default_call_policies(),
                                              (py::arg("value"))))
        .def("__init__", py::make_constructor(&Regular__init__params<RegularAxis>, py::default_call_policies(),
                                              (py::arg("start"), "stop", "num")))
        .add_property("start", &RegularAxis::first, &RegularAxis_setFirst, u8"Position of the beginning of the mesh")
        .add_property("stop", &RegularAxis::last, &RegularAxis_setLast, u8"Position of the end of the mesh")
        .add_property("step", &RegularAxis::step)
        .def("__getitem__", &RegularAxis__getitem__)
        .def("__str__", &__str__<RegularAxis>)
        .def("__repr__", &RegularAxis__repr__)
        .def("resize", &RegularAxis_resize, u8"Change number of points in this mesh", (py::arg("num")))
        .def(py::self == py::self)
        .def("__iter__", py::range(&RegularAxis::begin, &RegularAxis::end));
    // detail::RegularAxisFromTupleOrFloat();
    py::implicitly_convertible<RegularAxis, OrderedAxis>();
    py::implicitly_convertible<shared_ptr<RegularAxis>, shared_ptr<const RegularAxis>>();

    py::class_<RectangularMeshBase2D, shared_ptr<RectangularMeshBase2D>, py::bases<MeshD<2>>, boost::noncopyable>
        rectangularBase2D("RectangularBase2D",
                          u8"Base class for 2D rectangular meshes."
                          u8"Do not use it directly.",
                          py::no_init);
    rectangularBase2D
        .def("Left", &RectangularMeshBase2D::getLeftBoundary, u8"Left edge of the mesh for setting boundary conditions")
        .staticmethod("Left")
        .def("Right", &RectangularMeshBase2D::getRightBoundary,
             u8"Right edge of the mesh for setting boundary conditions")
        .staticmethod("Right")
        .def("Top", &RectangularMeshBase2D::getTopBoundary, u8"Top edge of the mesh for setting boundary conditions")
        .staticmethod("Top")
        .def("Bottom", &RectangularMeshBase2D::getBottomBoundary,
             u8"Bottom edge of the mesh for setting boundary conditions")
        .staticmethod("Bottom")
        .def("LeftOf",
             (RectangularMeshBase2D::Boundary(*)(shared_ptr<const GeometryObject>, const PathHints&)) &
                 RectangularMeshBase2D::getLeftOfBoundary,
             u8"Boundary left of specified object", (py::arg("object"), py::arg("path") = py::object()))
        .staticmethod("LeftOf")
        .def("RightOf",
             (RectangularMeshBase2D::Boundary(*)(shared_ptr<const GeometryObject>, const PathHints&)) &
                 RectangularMeshBase2D::getRightOfBoundary,
             u8"Boundary right of specified object", (py::arg("object"), py::arg("path") = py::object()))
        .staticmethod("RightOf")
        .def("TopOf",
             (RectangularMeshBase2D::Boundary(*)(shared_ptr<const GeometryObject>, const PathHints&)) &
                 RectangularMeshBase2D::getTopOfBoundary,
             u8"Boundary top of specified object", (py::arg("object"), py::arg("path") = py::object()))
        .staticmethod("TopOf")
        .def("BottomOf",
             (RectangularMeshBase2D::Boundary(*)(shared_ptr<const GeometryObject>, const PathHints&)) &
                 RectangularMeshBase2D::getBottomOfBoundary,
             u8"Boundary bottom of specified object", (py::arg("object"), py::arg("path") = py::object()))
        .staticmethod("BottomOf")
        .def("Horizontal",
             (RectangularMeshBase2D::Boundary(*)(double, double, double)) &
                 RectangularMeshBase2D::getHorizontalBoundaryNear,
             u8"Boundary at horizontal line", (py::arg("at"), "start", "stop"))
        .def("Horizontal",
             (RectangularMeshBase2D::Boundary(*)(double)) & RectangularMeshBase2D::getHorizontalBoundaryNear,
             u8"Boundary at horizontal line", py::arg("at"))
        .staticmethod("Horizontal")
        .def("Vertical",
             (RectangularMeshBase2D::Boundary(*)(double, double, double)) &
                 RectangularMeshBase2D::getVerticalBoundaryNear,
             u8"Boundary at vertical line", (py::arg("at"), "start", "stop"))
        .def("Vertical", (RectangularMeshBase2D::Boundary(*)(double)) & RectangularMeshBase2D::getVerticalBoundaryNear,
             u8"Boundary at vertical line", py::arg("at"))
        .staticmethod("Vertical");
    ExportBoundary<RectangularMeshBase2D::Boundary>{rectangularBase2D};
    py::implicitly_convertible<shared_ptr<RectangularMeshBase2D>, shared_ptr<const RectangularMeshBase2D>>();

    py::class_<RectangularMesh<2>, shared_ptr<RectangularMesh<2>>, py::bases<RectangularMeshBase2D>> rectangular2D(
        "Rectangular2D",
        u8"Two-dimensional mesh\n\n"
        u8"Rectangular2D(ordering='01')\n    create empty mesh\n\n"
        u8"Rectangular2D(axis0, axis1, ordering='01')\n    create mesh with axes supplied as sequences of numbers\n\n"
        u8"Rectangular2D(geometry, ordering='01')\n    create coarse mesh based on bounding boxes of geometry "
        u8"objects\n\n"
        u8"ordering can be either '01', '10' and specifies ordering of the mesh points (last index changing fastest).",
        py::no_init);
    rectangular2D
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__empty<RectangularMesh<2>>,
                                              py::default_call_policies(), (py::arg("ordering") = "01")))
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__axes, py::default_call_policies(),
                                              (py::arg("axis0"), py::arg("axis1"), py::arg("ordering") = "01")))
        .def("__init__", py::make_constructor(&RectilinearMesh2D__init__geometry, py::default_call_policies(),
                                              (py::arg("geometry"), py::arg("ordering") = "01")))
        //.def("__init__", py::make_constructor(&Mesh__init__<RectilinearMesh2D,RegularMesh2D>,
        // py::default_call_policies(), py::arg("src")))
        .def("copy", &Mesh__init__<RectangularMesh<2>, RectangularMesh<2>>,
             u8"Make a copy of this mesh")  // TODO should this be a deep copy?
        .add_property("axis0", &RectangularMesh<2>::getAxis0, &RectangularMesh<2>::setAxis0,
                      u8"The first (transverse) axis of the mesh")
        .add_property("axis1", &RectangularMesh<2>::getAxis1, &RectangularMesh<2>::setAxis1,
                      u8"The second (vertical) axis of the mesh")
        .add_property("axis_tran", &RectangularMesh<2>::getAxis0, &RectangularMesh<2>::setAxis0,
                      u8"The first (transverse) axis of the mesh, alias for :attr:`axis0`")
        .add_property("axis_vert", &RectangularMesh<2>::getAxis1, &RectangularMesh<2>::setAxis1,
                      u8"The second (vertical) axis of the mesh, alias for :attr:`axis1`")
        .add_property("major_axis", &RectangularMesh<2>::majorAxis, u8"The slower changing axis")
        .add_property("minor_axis", &RectangularMesh<2>::minorAxis, u8"The quicker changing axis")
        //.def("clear", &RectangularMesh<2>::clear, u8"Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh2D__getitem__)
        .def("index", (size_t(RectangularMesh<2>::*)(size_t, size_t) const) & RectangularMesh<2>::index,
             u8"Return single index of the point indexed with index0 and index1",
             (py::arg("index0"), py::arg("index1")))
        .def("index0", &RectangularMesh<2>::index0, u8"Return index in the first axis of the point with given index",
             (py::arg("index")))
        .def("index1", &RectangularMesh<2>::index1, u8"Return index in the second axis of the point with given index",
             (py::arg("index")))
        .def_readwrite("index_tran", &RectangularMesh<2>::index0, u8"Alias for :attr:`index0`")
        .def_readwrite("index_vert", &RectangularMesh<2>::index1, u8"Alias for :attr:`index1`")
        .def("major_index", &RectangularMesh<2>::majorIndex,
             u8"Return index in the major axis of the point with given index", (py::arg("index")))
        .def("minor_index", &RectangularMesh<2>::minorIndex,
             u8"Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("set_optimal_ordering", &RectangularMesh<2>::setOptimalIterationOrder,
             u8"Set the optimal ordering of the points in this mesh")
        .add_property("ordering", &RectangularMesh2D__getOrdering, &RectangularMesh2D__setOrdering,
                      u8"Ordering of the points in this mesh")
        .add_property("elements",
                      py::make_function(&RectangularMesh2D::elements, py::with_custodian_and_ward_postcall<0, 1>()),
                      u8"Element list in the mesh")
        .def("get_midpoints", &RectangularMesh_getMidpoints<2>, py::with_custodian_and_ward_postcall<0, 1>(),
             u8"Get new mesh with points in the middles of elements of this mesh")
        .def(py::self == py::self);
    py::implicitly_convertible<shared_ptr<RectangularMesh<2>>, shared_ptr<const RectangularMesh<2>>>();

    {
        py::scope scope = rectangular2D;
        (void)scope;  // don't warn about unused variable scope

        py::class_<RectangularMesh2D::Element>(
            "Element", u8"Element (FEM-like, rectangular) of the :py:class:`~plask.mesh.Rectangular2D` mesh", py::no_init)
            .add_property("index0", /*size_t*/ &RectangularMesh2D::Element::getIndex0,
                          u8"Element index in the first axis")
            .add_property("index1", /*size_t*/ &RectangularMesh2D::Element::getIndex1,
                          u8"Element index in the second axis")
            .add_property("left", /*double*/ &RectangularMesh2D::Element::getLower0,
                          u8"Position of the left edge of the element")
            .add_property("right", /*double*/ &RectangularMesh2D::Element::getUpper0,
                          u8"Position of the right edge of the element")
            .add_property("top", /*double*/ &RectangularMesh2D::Element::getUpper1,
                          u8"Position of the top edge of the element")
            .add_property("bottom", /*double*/ &RectangularMesh2D::Element::getLower1,
                          u8"Position of the bottom edge of the element")
            .add_property("width", /*double*/ &RectangularMesh2D::Element::getSize0, u8"Width of the element")
            .add_property("height", /*double*/ &RectangularMesh2D::Element::getSize1, u8"Height of the element")
            .add_property("center", /*Vec<2,double>*/ &RectangularMesh2D::Element::getMidpoint,
                          u8"Position of the element center")
            .add_property("index", /*size_t*/ &RectangularMesh2D::Element::getIndex, u8"Element index")
            .add_property("box", /*Box2D*/ &RectangularMesh2D::Element::toBox, u8"Bounding box of the element")
            .add_property("area", /*double*/ &RectangularMesh2D::Element::getArea, u8"Area of the element")
            .add_property("volume", /*double*/ &RectangularMesh2D::Element::getVolume, u8"Alias for :attr:`area`")
            .add_property("nodes", &RectangularMesh2D_Element_nodes,
                          u8"Indices of the element vertices on the orignal mesh\n\n"
                          u8"Order of the vertices is bottom left, bottom right, top left, and top right.")
            .def("__contains__", &RectangularMesh2D::Element::contains,
                 "check if given point is included in this element")
            // .add_property("bottom_left", /*Vec<2,double>*/ &RectangularMesh2D::Element::getLoLo, u8"Position of the
            // bottom left vertex of the elemnent") .add_property("top_left", /*Vec<2,double>*/
            // &RectangularMesh2D::Element::getLoUp, u8"Position of the top left vertex of the elemnent")
            // .add_property("bottom_right", /*Vec<2,double>*/ &RectangularMesh2D::Element::getUpLo, u8"Position of the
            // bottom right vertex of the elemnent") .add_property("top_right", /*Vec<2,double>*/
            // &RectangularMesh2D::Element::getUpUp, u8"Position of the top left right vertex of the elemnent")
            ;

        py::class_<RectangularMesh2D::Elements>("Elements", u8"Element list in the :py:class:`~plask.mesh.Rectangular2D` mesh",
                                                py::no_init)
            .def("__len__", &RectangularMesh2D::Elements::size)
            .def("__getitem__", &RectangularMesh2D::Elements::operator[], py::with_custodian_and_ward_postcall<0, 1>())
            .def("__getitem__", &RectangularMesh2D::Elements::operator(), py::with_custodian_and_ward_postcall<0, 1>())
            .def("__iter__", py::range<py::with_custodian_and_ward_postcall<0, 1>>(&RectangularMesh2D::Elements::begin,
                                                                                   &RectangularMesh2D::Elements::end))
            .add_property("mesh", &RectangularMesh_ElementMesh<RectangularMesh2D>, "Mesh with element centers");

        py::class_<RectangularMesh2DSimpleGenerator, shared_ptr<RectangularMesh2DSimpleGenerator>,
                   py::bases<MeshGeneratorD<2>>, boost::noncopyable>(
            "SimpleGenerator",
            u8"Generator of Rectangular2D mesh with lines at edges of all objects.\n\n"
            u8"SimpleGenerator(split=False)\n    create generator\n\n"
            u8"Args:\n"
            u8"   split (bool): If ``True``, the mesh lines are split into two at each object\n"
            u8"                 boundary.\n",
            py::init<bool>(py::arg("split") = false));
        py::implicitly_convertible<shared_ptr<RectangularMesh2DSimpleGenerator>,
                                   shared_ptr<const RectangularMesh2DSimpleGenerator>>();

        py::class_<RectangularMesh2DRegularGenerator, shared_ptr<RectangularMesh2DRegularGenerator>,
                   py::bases<MeshGeneratorD<2>>, boost::noncopyable>(
            "RegularGenerator",
            u8"Generator of Rectilinear2D mesh with lines at transverse edges of all objects\n"
            u8"and fine regular division of each object with spacing approximately equal to\n"
            u8"specified spacing.\n\n"
            u8"RegularGenerator(spacing, split=False)\n"
            u8"    create generator with equal spacing in all directions\n\n"
            u8"RegularGenerator(spacing0, spacing1, split=False)\n"
            u8"    create generator with equal spacing\n\n"
            u8"Args:\n"
            u8"   spacing (float): Approximate spacing between mesh lines in all directions.\n"
            u8"   spacing0 (float): Approximate spacing between mesh lines in transverse\n"
            u8"                     direction.\n"
            u8"   spacing1 (float): Approximate spacing between mesh lines in vertical\n"
            u8"                     direction.\n"
            u8"   split (bool): If ``True``, the mesh lines are split into two at each object\n"
            u8"                 boundary.\n",
            py::no_init)
            .def("__init__",
                 py::make_constructor(RectangularMesh2DRegularGenerator__init__1, py::default_call_policies(),
                                      (py::arg("spacing"), py::arg("split") = false)))
            .def("__init__",
                 py::make_constructor(RectangularMesh2DRegularGenerator__init__2, py::default_call_policies(),
                                      (py::arg("spacing0"), py::arg("spacing1"))))
            .def("__init__",
                 py::make_constructor(RectangularMesh2DRegularGenerator__init__3, py::default_call_policies(),
                                      (py::arg("spacing0"), py::arg("spacing1"), py::arg("split"))));
        py::implicitly_convertible<shared_ptr<RectangularMesh2DRegularGenerator>,
                                   shared_ptr<const RectangularMesh2DRegularGenerator>>();

        register_divide_generator<2>();
        register_smooth_generator<2>();
    }

    py::class_<RectangularMeshBase3D, shared_ptr<RectangularMeshBase3D>, py::bases<MeshD<3>>, boost::noncopyable>
        rectangularBase3D("RectangularBase3D",
                          u8"Base class for 3D rectangular meshes."
                          u8"Do not use it directly.",
                          py::no_init);
    rectangularBase3D
        .def("Front", &RectangularMeshBase3D::getFrontBoundary,
             u8"Front side of the mesh for setting boundary conditions")
        .staticmethod("Front")
        .def("Back", &RectangularMeshBase3D::getBackBoundary, u8"Back side of the mesh for setting boundary conditions")
        .staticmethod("Back")
        .def("Left", &RectangularMeshBase3D::getLeftBoundary, u8"Left side of the mesh for setting boundary conditions")
        .staticmethod("Left")
        .def("Right", &RectangularMeshBase3D::getRightBoundary,
             u8"Right side of the mesh for setting boundary conditions")
        .staticmethod("Right")
        .def("Top", &RectangularMeshBase3D::getTopBoundary, u8"Top side of the mesh for setting boundary conditions")
        .staticmethod("Top")
        .def("Bottom", &RectangularMeshBase3D::getBottomBoundary,
             u8"Bottom side of the mesh for setting boundary conditions")
        .staticmethod("Bottom")
        .def("FrontOf",
             (RectangularMeshBase3D::Boundary(*)(shared_ptr<const GeometryObject>, const PathHints&)) &
                 RectangularMesh<3>::getFrontOfBoundary,
             u8"Boundary in front of specified object", (py::arg("object"), py::arg("path") = py::object()))
        .staticmethod("FrontOf")
        .def("BackOf",
             (RectangularMeshBase3D::Boundary(*)(shared_ptr<const GeometryObject>, const PathHints&)) &
                 RectangularMesh<3>::getBackOfBoundary,
             u8"Boundary back of specified object", (py::arg("object"), py::arg("path") = py::object()))
        .staticmethod("BackOf")
        .def("LeftOf",
             (RectangularMeshBase3D::Boundary(*)(shared_ptr<const GeometryObject>, const PathHints&)) &
                 RectangularMesh<3>::getLeftOfBoundary,
             u8"Boundary left of specified object", (py::arg("object"), py::arg("path") = py::object()))
        .staticmethod("LeftOf")
        .def("RightOf",
             (RectangularMeshBase3D::Boundary(*)(shared_ptr<const GeometryObject>, const PathHints&)) &
                 RectangularMesh<3>::getRightOfBoundary,
             u8"Boundary right of specified object", (py::arg("object"), py::arg("path") = py::object()))
        .staticmethod("RightOf")
        .def("TopOf",
             (RectangularMeshBase3D::Boundary(*)(shared_ptr<const GeometryObject>, const PathHints&)) &
                 RectangularMesh<3>::getTopOfBoundary,
             u8"Boundary top of specified object", (py::arg("object"), py::arg("path") = py::object()))
        .staticmethod("TopOf")
        .def("BottomOf",
             (RectangularMeshBase3D::Boundary(*)(shared_ptr<const GeometryObject>, const PathHints&)) &
                 RectangularMesh<3>::getBottomOfBoundary,
             u8"Boundary bottom of specified object", (py::arg("object"), py::arg("path") = py::object()))
        .staticmethod("BottomOf");
    ExportBoundary<RectangularMeshBase3D::Boundary>{rectangularBase3D};
    py::implicitly_convertible<shared_ptr<RectangularMeshBase3D>, shared_ptr<const RectangularMeshBase3D>>();

    py::class_<RectangularMesh<3>, shared_ptr<RectangularMesh<3>>, py::bases<RectangularMeshBase3D>> rectangular3D(
        "Rectangular3D",
        u8"Three-dimensional mesh\n\n"
        u8"Rectangular3D(ordering='012')\n    create empty mesh\n\n"
        u8"Rectangular3D(axis0, axis1, axis2, ordering='012')\n    create mesh with axes supplied as "
        u8"mesh.OrderedAxis\n\n"
        u8"Rectangular3D(geometry, ordering='012')\n    create coarse mesh based on bounding boxes of geometry "
        u8"objects\n\n"
        u8"ordering can be any a string containing any permutation of and specifies ordering of the\n"
        u8"mesh points (last index changing fastest).",
        py::no_init);
    rectangular3D
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__empty<RectangularMesh<3>>,
                                              py::default_call_policies(), (py::arg("ordering") = "012")))
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__axes, py::default_call_policies(),
                                              (py::arg("axis0"), "axis1", "axis2", py::arg("ordering") = "012")))
        .def("__init__", py::make_constructor(&RectilinearMesh3D__init__geometry, py::default_call_policies(),
                                              (py::arg("geometry"), py::arg("ordering") = "012")))
        //.def("__init__", py::make_constructor(&Mesh__init__<RectilinearMesh3D, RegularMesh3D>,
        // py::default_call_policies(), py::arg("src")))
        .def("copy", &Mesh__init__<RectangularMesh<3>, RectangularMesh<3>>, "Make a copy of this mesh")
        .add_property("axis0", &RectangularMesh<3>::getAxis0, &RectangularMesh<3>::setAxis0,
                      u8"The first (longitudinal) axis of the mesh")
        .add_property("axis1", &RectangularMesh<3>::getAxis1, &RectangularMesh<3>::setAxis1,
                      u8"The second (transverse) axis of the mesh")
        .add_property("axis2", &RectangularMesh<3>::getAxis2, &RectangularMesh<3>::setAxis2,
                      u8"The third (vertical) axis of the mesh")
        .add_property("axis_long", &RectangularMesh<3>::getAxis0, &RectangularMesh<3>::setAxis0,
                      u8"The first (longitudinal) axis of the mesh, alias for :attr:`axis0`")
        .add_property("axis_tran", &RectangularMesh<3>::getAxis1, &RectangularMesh<3>::setAxis1,
                      u8"The second (transverse) axis of the mesh, alias for :attr:`axis1`")
        .add_property("axis_vert", &RectangularMesh<3>::getAxis2, &RectangularMesh<3>::setAxis2,
                      u8"The third (vertical) axis of the mesh, alias for :attr:`axis2`")
        .add_property("major_axis", &RectangularMesh<3>::majorAxis, u8"The slowest changing axis")
        .add_property("medium_axis", &RectangularMesh<3>::mediumAxis, u8"The middle changing axis")
        .add_property("minor_axis", &RectangularMesh<3>::minorAxis, u8"The quickest changing axis")
        //.def("clear", &RectangularMesh<3>::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh3D__getitem__<RectangularMesh<3>>)
        .def("index", (size_t(RectangularMesh<3>::*)(size_t, size_t, size_t) const) & RectangularMesh<3>::index,
             (py::arg("index0"), py::arg("index1"), py::arg("index2")),
             "Return single index of the point indexed with index0, index1, and index2")
        .def("index0", &RectangularMesh<3>::index0, u8"Return index in the first axis of the point with given index",
             (py::arg("index")))
        .def("index1", &RectangularMesh<3>::index1, u8"Return index in the second axis of the point with given index",
             (py::arg("index")))
        .def("index2", &RectangularMesh<3>::index2, u8"Return index in the third axis of the point with given index",
             (py::arg("index")))
        .def_readwrite("index_long", &RectangularMesh<3>::index0, "Alias for :attr:`index0`")
        .def_readwrite("index_tran", &RectangularMesh<3>::index1, "Alias for :attr:`index1`")
        .def_readwrite("index_vert", &RectangularMesh<3>::index2, "Alias for :attr:`index2`")
        .def("major_index", &RectangularMesh<3>::majorIndex,
             u8"Return index in the major axis of the point with given index", (py::arg("index")))
        .def("middle_index", &RectangularMesh<3>::middleIndex,
             u8"Return index in the middle axis of the point with given index", (py::arg("index")))
        .def("minor_index", &RectangularMesh<3>::minorIndex,
             u8"Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("set_optimal_ordering", &RectangularMesh<3>::setOptimalIterationOrder,
             u8"Set the optimal ordering of the points in this mesh")
        .add_property("ordering", &RectangularMesh3D__getOrdering, &RectangularMesh3D__setOrdering,
                      u8"Ordering of the points in this mesh")
        .add_property("elements",
                      py::make_function(&RectangularMesh3D_elements, py::with_custodian_and_ward_postcall<0, 1>()),
                      u8"Element list in the mesh")
        .def("get_midpoints", &RectangularMesh_getMidpoints<3>, py::with_custodian_and_ward_postcall<0, 1>(),
             u8"Get new mesh with points in the middles of of elements of this mesh")
        .def(py::self == py::self);
    py::implicitly_convertible<shared_ptr<RectangularMesh<3>>, shared_ptr<const RectangularMesh<3>>>();

    {
        py::scope scope = rectangular3D;
        (void)scope;  // don't warn about unused variable scope

        py::class_<RectilinearMesh3D::Element>(
            "Element", u8"Element (FEM-like, rectangular) of the :py:class:`~plask.mesh.Rectangular3D` mesh", py::no_init)
            .add_property("index0", /*size_t*/ &RectilinearMesh3D::Element::getIndex0,
                          u8"Element index in the first axis")
            .add_property("index1", /*size_t*/ &RectilinearMesh3D::Element::getIndex1,
                          u8"Element index in the second axis")
            .add_property("index2", /*size_t*/ &RectilinearMesh3D::Element::getIndex2,
                          u8"Element index in the third axis")
            .add_property("back", /*double*/ &RectilinearMesh3D::Element::getLower0,
                          u8"Position of the back edge of the element")
            .add_property("front", /*double*/ &RectilinearMesh3D::Element::getUpper0,
                          u8"Position of the front edge of the element")
            .add_property("left", /*double*/ &RectilinearMesh3D::Element::getLower1,
                          u8"Position of the left edge of the element")
            .add_property("right", /*double*/ &RectilinearMesh3D::Element::getUpper1,
                          u8"Position of the right edge of the element")
            .add_property("top", /*double*/ &RectilinearMesh3D::Element::getUpper2,
                          u8"Position of the top edge of the element")
            .add_property("bottom", /*double*/ &RectilinearMesh3D::Element::getLower2,
                          u8"Position of the bottom edge of the element")
            .add_property("depth", /*double*/ &RectilinearMesh3D::Element::getSize0, u8"Depth of the element")
            .add_property("width", /*double*/ &RectilinearMesh3D::Element::getSize1, u8"Width of the element")
            .add_property("height", /*double*/ &RectilinearMesh3D::Element::getSize2, u8"Height of the element")
            .add_property("center", /*Vec<3,double>*/ &RectilinearMesh3D::Element::getMidpoint,
                          u8"Position of the element center")
            .add_property("index", /*size_t*/ &RectilinearMesh3D::Element::getIndex, u8"Element index")
            .add_property("box", /*Box3D*/ &RectangularMesh3D_Element_box, u8"Bounding box of the element")
            .add_property("volume", /*double*/ &RectangularMesh3D_Element_volume, u8"Volume of the element")
            .add_property("nodes", &RectangularMesh3D_Element_nodes,
                          u8"Indices of the element vertices on the orignal mesh\n\n"
                          u8"Order of the vertices is back bottom left, back  bottom right, top left,\n"
                          u8"back top right, front bottom left, front bottom right, front top left,\n"
                          u8"and front top right.")
            .def("__contains__", &RectilinearMesh3D::Element::contains,
                 "check if given point is included in this element");

        py::class_<RectilinearMesh3D::Elements>("Elements", u8"Element list in the :py:class:`~plask.mesh.Rectangular3D` mesh",
                                                py::no_init)
            .def("__len__", &RectilinearMesh3D::Elements::size)
            .def("__getitem__", &RectilinearMesh3D::Elements::operator[], py::with_custodian_and_ward_postcall<0, 1>())
            .def("__getitem__", &RectilinearMesh3D::Elements::operator(), py::with_custodian_and_ward_postcall<0, 1>())
            .def("__iter__", py::range<py::with_custodian_and_ward_postcall<0, 1>>(&RectilinearMesh3D::Elements::begin,
                                                                                   &RectilinearMesh3D::Elements::end))
            .add_property("mesh", &RectangularMesh_ElementMesh<RectangularMesh3D>, "Mesh with element centers");

        py::class_<RectangularMesh3DSimpleGenerator, shared_ptr<RectangularMesh3DSimpleGenerator>,
                   py::bases<MeshGeneratorD<3>>, boost::noncopyable>(
            "SimpleGenerator",
            u8"Generator of Rectangular3D mesh with lines at edges of all objects.\n\n"
            u8"SimpleGenerator(split=False)\n    create generator\n\n"
            u8"Args:\n"
            u8"   split (bool): If ``True``, the mesh lines are split into two at each object\n"
            u8"                 boundary.\n",
            py::init<bool>(py::arg("split") = false));
        py::implicitly_convertible<shared_ptr<RectangularMesh3DSimpleGenerator>,
                                   shared_ptr<const RectangularMesh3DSimpleGenerator>>();

        py::class_<RectangularMesh3DRegularGenerator, shared_ptr<RectangularMesh3DRegularGenerator>,
                   py::bases<MeshGeneratorD<3>>, boost::noncopyable>(
            "RegularGenerator",
            u8"Generator of Rectilinear3D mesh with lines at transverse edges of all objects\n"
            u8"and fine regular division of each object with spacing approximately equal to\n"
            u8"specified spacing\n\n"
            u8"RegularGenerator(spacing, split=False)\n"
            u8"    create generator with equal spacing in all directions\n\n"
            u8"RegularGenerator(spacing0, spacing1, spacing2, split=False)\n"
            u8"    create generator with equal spacing\n\n"
            u8"Args:\n"
            u8"   spacing (float): Approximate spacing between mesh lines in all directions.\n"
            u8"   spacing0 (float): Approximate spacing between mesh lines in longitudinal\n"
            u8"                     direction.\n"
            u8"   spacing1 (float): Approximate spacing between mesh lines in transverse\n"
            u8"                     direction.\n"
            u8"   spacing2 (float): Approximate spacing between mesh lines in vertical\n"
            u8"                     direction.\n"
            u8"   split (bool): If ``True``, the mesh lines are split into two at each object\n"
            u8"                 boundary.\n",
            py::no_init)
            .def("__init__",
                 py::make_constructor(RectangularMesh3DRegularGenerator__init__1, py::default_call_policies(),
                                      (py::arg("spacing"), py::arg("split") = false)))
            .def("__init__",
                 py::make_constructor(
                     RectangularMesh3DRegularGenerator__init__3, py::default_call_policies(),
                     (py::arg("spacing0"), py::arg("spacing1"), py::arg("spacing2"), py::arg("split") = false)));
        py::implicitly_convertible<shared_ptr<RectangularMesh3DRegularGenerator>,
                                   shared_ptr<const RectangularMesh3DRegularGenerator>>();

        register_divide_generator<3>();
        register_smooth_generator<3>();
    }
}

}}  // namespace plask::python
