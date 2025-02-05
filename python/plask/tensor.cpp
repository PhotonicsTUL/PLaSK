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
#include <sstream>

#define PY_ARRAY_UNIQUE_SYMBOL PLASK_ARRAY_API
#define NO_IMPORT_ARRAY

#include <plask/config.hpp>
#include <plask/exceptions.hpp>
#include <plask/vector/tensor2.hpp>
#include <plask/vector/tensor3.hpp>

#include "python/globals.hpp"
#include "python/numpy.hpp"
#include "python/util/raw_constructor.hpp"

#include <boost/concept_check.hpp>
#include <boost/python/stl_iterator.hpp>

#define A(o) (PyArrayObject*)(o)

namespace plask {

// clang-format off
namespace detail {
    template <int dim, typename T> struct TensorSelector;
    template <typename T> struct TensorSelector<2, T> { typedef Tensor2<T> type; };
    template <typename T> struct TensorSelector<3, T> { typedef Tensor3<T> type; };
}
// clang-format on

template <int dim, typename T> using Tensor = typename detail::TensorSelector<dim, T>::type;

namespace python {

// clang-format off
namespace detail {

    template <int dim, typename T> struct TensorFromPython;

    template <typename T> struct TensorFromPython<2, T> {
        TensorFromPython() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<Tensor2<T>>());
        }

        static void* convertible(PyObject* obj) {
            if (!PySequence_Check(obj) && !py::extract<T>(obj).check()) return NULL;
            return obj;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<Tensor2<double>>*)data)->storage.bytes;
            T first, second;
            if (py::extract<T>(obj).check()) {
                first = second = py::extract<T>(obj);
            } else if (PySequence_Length(obj) == 2) {
                auto src = py::object(py::handle<>(py::borrowed(obj)));
                auto ofirst = src[0];
                auto osecond = src[1];
                first = py::extract<T>(ofirst);
                second = py::extract<T>(osecond);
            } else {
                throw TypeError(u8"float or sequence of exactly two floats required");
            }
            new (storage) Tensor2<T>(first, second);
            data->convertible = storage;
        }
    };

    template <typename T> struct TensorFromPython<3, T> {
        TensorFromPython() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<Tensor3<T>>());
        }

        static void* convertible(PyObject* obj) {
            if (PyArray_Check(obj)) {
                if (PyArray_NDIM(A(obj)) == 2 && PyArray_DIM(A(obj), 0) == 3 && PyArray_DIM(A(obj), 1) == 3) return obj;
                return NULL;
            }
            if (PySequence_Check(obj)) {
                auto len = PySequence_Length(obj);
                if (len == 1 || len == 3 || len == 4 || len == 6 || len == 9) {
                    for (int i = 0; i < len; ++i) {
                        if (!py::extract<T>(PySequence_GetItem(obj, i)).check()) return NULL;
                    }
                    return obj;
                }
                return NULL;
            }
            if (py::extract<T>(obj).check()) return obj;
            return NULL;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<Tensor3<dcomplex>>*)data)->storage.bytes;
            py::extract<T> val(obj);
            if (val.check()) {
                new (storage) Tensor3<T>(val());
                data->convertible = storage;
                return;
            } else if (PyArray_Check(obj)) {
                auto arr = py::object(py::handle<>(py::borrowed(obj)));
                PyObject* array = PyArray_ContiguousFromObject(obj, typenum<T>(), 2, 2);
                if (array == nullptr) throw plask::CriticalException(u8"Cannot create tensor from array");
                T* data = reinterpret_cast<T*>(PyArray_DATA(A(array)));
                new (storage) Tensor3<T>(data);
                Py_DECREF(obj);
            } else {
                auto seq = py::object(py::handle<>(py::borrowed(obj)));
                auto len = py::len(seq);
                if (len >= 1) {}
                    const T c0 = py::extract<T>(seq[0]);
                    if (len >= 3) {
                        const T c1 = py::extract<T>(seq[1]);
                        const T c2 = py::extract<T>(seq[2]);
                        if (len >= 4) {
                            const T c3 = py::extract<T>(seq[3]);
                            if (len >= 6) {
                                const T c4 = py::extract<T>(seq[4]);
                                const T c5 = py::extract<T>(seq[5]);
                                if (len >= 9) {
                                    const T c6 = py::extract<T>(seq[6]);
                                    const T c7 = py::extract<T>(seq[7]);
                                    const T c8 = py::extract<T>(seq[8]);
                                    new (storage) Tensor3<T>(c0, c1, c2, c3, c4, c5, c6, c7, c8);
                                } else
                                    new (storage) Tensor3<T>(c0, c1, c2, c3, c4, c5);
                            } else
                                new (storage) Tensor3<T>(c0, c1, c2, c3);
                        } else
                            new (storage) Tensor3<T>(c0, c1, c2);
                    } else
                        new (storage) Tensor3<T>(c0);
            }
            data->convertible = storage;
        }
    };

    template <int dim, typename T> struct TensorMethods;

    template <typename T> struct TensorMethods<2, T> {
        static std::string __str__(const Tensor<2, T>& to_print) {
            std::stringstream out;
            out << "[[" << pyformat(to_print.c00) << ", " << pyformat(to_print.c11) << "]]";
            return out.str();
        }
        static std::string __repr__(const Tensor<2, T>& to_print) {
            std::stringstream out;
            out << "plask.tensor(" << pyformat(to_print.c00) << ", " << pyformat(to_print.c11) << ")";
            return out.str();
        }
        static py::object __array__(py::object self, py::object dtype, py::object copy) {
            Tensor<2, T>* tensor = py::extract<Tensor<2, T>*>(self);
            npy_intp dims[] = {2};
            PyObject* arr = PyArray_SimpleNewFromData(1, dims, detail::typenum<T>(), (void*)(&tensor->c00));
            if (arr == nullptr) throw plask::CriticalException(u8"cannot create array from tensor");
            confirm_array<T>(arr, self, dtype, copy);
            return py::object(py::handle<>(arr));
        }
        py::object list__array__(py::object self, py::object dtype, py::object copy) {
            std::vector<Tensor<2, T>>* list = py::extract<std::vector<Tensor<2, T>>*>(self);
            npy_intp dims[] = {(npy_int)list->size(), 2};
            PyObject* arr = PyArray_SimpleNewFromData(2, dims, detail::typenum<T>(), (void*)(&(*list)[0].c00));
            if (arr == nullptr) throw plask::CriticalException(u8"cannot create array from tensor list");
            confirm_array<T>(arr, self, dtype, copy);
            return py::object(py::handle<>(arr));
        }
    };

    template <typename T> struct TensorMethods<3, T> {
        static std::string __str__(const Tensor<3, T>& to_print) {
            std::stringstream out;
            out << "[[" << pyformat(to_print.c00) << ", " << pyformat(to_print.c01) << ", " << pyformat(to_print.c02) << "], "
                << "[" << pyformat(to_print.c10) << ", " << pyformat(to_print.c11) << ", " << pyformat(to_print.c12) << "], "
                << "[" << pyformat(to_print.c20) << ", " << pyformat(to_print.c21) << ", " << pyformat(to_print.c22) << "]]";
            return out.str();
        }
        static std::string __repr__(const Tensor<3, T>& to_print) {
            std::stringstream out;
            out << "plask.tensor("
                << pyformat(to_print.c00) << ", "
                << pyformat(to_print.c11) << ", "
                << pyformat(to_print.c22) << ", "
                << pyformat(to_print.c01) << ", "
                << pyformat(to_print.c10) << ", "
                << pyformat(to_print.c02) << ", "
                << pyformat(to_print.c20) << ", "
                << pyformat(to_print.c12) << ", "
                << pyformat(to_print.c21) << ")";
            return out.str();
        }
        static py::object __array__(py::object self, py::object dtype, py::object copy) {
            Tensor<3, T>* tensor = py::extract<Tensor<3, T>*>(self);
            npy_intp dims[] = {3, 3};
            PyObject* arr = PyArray_SimpleNewFromData(2, dims, detail::typenum<T>(), (void*)(&tensor->c00));
            if (arr == nullptr) throw plask::CriticalException(u8"cannot create array from tensor");
            confirm_array<T>(arr, self, dtype, copy);
            return py::object(py::handle<>(arr));
        }
        py::object list__array__(py::object self, py::object dtype, py::object copy) {
            std::vector<Tensor<2, T>>* list = py::extract<std::vector<Tensor<2, T>>*>(self);
            npy_intp dims[] = {(npy_int)list->size(), 3, 3};
            PyObject* arr = PyArray_SimpleNewFromData(3, dims, detail::typenum<T>(), (void*)(&(*list)[0].c00));
            if (arr == nullptr) throw plask::CriticalException(u8"cannot create array from tensor list");
            confirm_array<T>(arr, self, dtype, copy);
            return py::object(py::handle<>(arr));
        }
    };

}  // namespace detail
// clang-format on

// len(v)
template <int dim> static constexpr int tensor_size();
template <> constexpr int tensor_size<2>() { return 2; }
template <> constexpr int tensor_size<3>() { return 9; }
template <int dim> static int tensor__len__(const py::object&) { return tensor_size<dim>(); }

// clang-format off
namespace detail {
    template <int dim> struct tensor_indices { static const size_t idx[]; };
    template <> const size_t tensor_indices<2>::idx[] = {0, 1};
    template <> const size_t tensor_indices<3>::idx[] = {0, 4, 8, 1, 3, 2, 6, 5, 7};
}
// clang-format on

// v = tensor[i]
template <int dim, typename T> static T tensor__getitem__(Tensor<dim, T>& self, int i) {
    if (i < 0) i = tensor_size<dim>() + i;
    if (i >= tensor_size<dim>() || i < 0) throw IndexError("tensor index out of range");
    return self[detail::tensor_indices<dim>::idx[i]];
}

// v.__iter__
template <int dim, typename T> struct Tensor_iterator {
    Tensor<dim, T>* tensor;
    int i;
    static Tensor_iterator<dim, T> new_iterator(Tensor<dim, T>& v) { return Tensor_iterator<dim, T>(v); }

    Tensor_iterator(Tensor<dim, T>& v) : tensor(&v), i(0) {}

    Tensor_iterator<dim, T>* __iter__() { return this; }

    T next() {
        if (i >= tensor_size<dim>()) {
            PyErr_SetString(PyExc_StopIteration, u8"No more components.");
            boost::python::throw_error_already_set();
        }
        return (*tensor)[detail::tensor_indices<dim>::idx[i++]];
    }
};

// copy v
template <int dim, typename T> static Tensor<dim, T> copy_tensor(const Tensor<dim, T>& v) { return v; }

// dtype
template <int dim, typename T> inline static py::handle<> tensor_dtype() { return detail::dtype<T>(); }

// Access components by name
template <int dim> inline static int tensor_attr_indx(const std::string& attr);

template <> inline int tensor_attr_indx<2>(const std::string& attr) {
    int i0 = int(current_axes[attr.substr(0, 1)]) - 1;
    if (i0 == 3) i0 = 4;
    int i1 = int(current_axes[attr.substr(1, 1)]) - 1;
    if (i1 == 3) i1 = 4;
    if (i0 < 0 || i0 >= 2 || i0 != i1) {
        if (attr == "xx" || attr == "yy" || attr == "zz" || attr == "rr" || attr == "pp" || attr == "ll" || attr == "tt")
            throw AttributeError("tensor attribute '{}' has no sense for 2D tensor if config.axes = '{}'", attr,
                                 current_axes.str());
        else
            throw AttributeError("'tensor' object has no attribute '{}'", attr);
    }
    return i0;
}

template <> inline int tensor_attr_indx<3>(const std::string& attr) {
    int i0 = int(current_axes[attr.substr(0, 1)]);
    int i1 = int(current_axes[attr.substr(1, 1)]);
    if (i0 < 0 || i0 >= 3 || i1 < 0 || i1 >= 3) {
        if (attr == "xx" || attr == "yy" || attr == "zz" || attr == "rr" || attr == "pp" || attr == "ll" || attr == "tt" ||
            attr == "xy" || attr == "yz" || attr == "xz" || attr == "pr" || attr == "pz" || attr == "rz" || attr == "lt" ||
            attr == "lv" || attr == "tv" || attr == "yx" || attr == "zy" || attr == "zx" || attr == "rp" || attr == "zp" ||
            attr == "zr" || attr == "tl" || attr == "vl" || attr == "vt")
            throw AttributeError(u8"tensor attribute '{}' has no sense for 3D tensor if config.axes = '{}'", attr,
                                 current_axes.str());
        else
            throw AttributeError(u8"'tensor' object has no attribute '{}'", attr);
    }
    return 3 * i0 + i1;
}

template <int dim, typename T> struct TensorAttr {
    typedef Tensor<dim, T> V;
    static T get(const V& self, const std::string& attr) { return self[tensor_attr_indx<dim>(attr)]; }
    static void set(V& /*self*/, const std::string& attr, T /*val*/) {
        throw TypeError("'tuple' object does not support attribute assignment", attr);
    }
};

template <int dim, typename T> static Tensor<dim, T> tensor__div__float(const Tensor<dim, T>& self, double f) { return self / f; }

template <int dim, typename T> static Tensor<dim, dcomplex> tensor__div__complex(const Tensor<dim, T>& self, dcomplex f) {
    return self * (1. / f);
}

// Register tensor class to python
template <int dim, typename T> inline static py::class_<Tensor<dim, T>> register_tensor_class(std::string name = "tensor") {
    typedef Tensor<dim, T> V;
    typedef Tensor<dim, double> VR;
    typedef Tensor<dim, dcomplex> VC;

    V (*c)(const V&) = &plask::conj<T>;

    py::class_<V> tensor_class = py::class_<V>(name.c_str(),
                                               "PLaSK tensor.\n\n"
                                               "See Also:\n"
                                               "    tensor: create a new tensor.\n",
                                               py::no_init);
    tensor_class.def("__getattr__", &TensorAttr<dim, T>::get)
        .def("__setattr__", &TensorAttr<dim, T>::set)
        .def("__getitem__", &tensor__getitem__<dim, T>)
        .def("__iter__", &Tensor_iterator<dim, T>::new_iterator, py::with_custodian_and_ward_postcall<0, 1>())
        .def("__len__", &tensor__len__<dim>)
        .def("__str__", &detail::TensorMethods<dim, T>::__str__)
        .def("__repr__", &detail::TensorMethods<dim, T>::__repr__)
        .def(py::self == py::other<VC>())
        .def(py::self == py::other<VR>())
        .def(py::self != py::other<VC>())
        .def(py::self != py::other<VR>())
        .def(py::self + py::other<VC>())
        .def(py::self + py::other<VR>())
        .def(py::self - py::other<VC>())
        .def(py::self - py::other<VR>())
        .def(-py::self)
        .def(py::self * dcomplex())
        .def(py::self * double())
        .def(dcomplex() * py::self)
        .def(double() * py::self)
        .def(py::self += py::other<V>())
        .def(py::self -= py::other<V>())
        .def(py::self *= T())
        .def("__div__", &tensor__div__float<dim, T>)
        .def("__truediv__", &tensor__div__float<dim, T>)
        .def("__div__", &tensor__div__complex<dim, T>)
        .def("__truediv__", &tensor__div__complex<dim, T>)
        .def("__pow__", &V::pow)
        // .def("conj", c,
        //      u8"Conjugate of the tensor. It can be called for real tensors, but then it\n"
        //      u8"simply returns `self`\n")
        .def("copy", &copy_tensor<dim, T>,
             u8"Copy of the tensor. Normally tensors behave like Python containers, and\n"
             u8"assignment operation makes shallow copy only. Use this method if you want\n"
             u8"to modify the copy without changing the source.\n")
        .add_static_property("dtype", &tensor_dtype<dim, T>,
                             u8"Type od the tensor components. This is always either ``float`` or ``complex``.\n")
        .def("__array__", &detail::TensorMethods<dim, T>::__array__, (py::arg("dtype")=py::object(), py::arg("copy")=py::object()));
    tensor_class.attr("__module__") = "plask";

    detail::TensorFromPython<dim, T>();

    register_vector_of<Tensor<dim, T>>(name).def("__array__", &detail::TensorMethods<dim, T>::list__array__,
                                                 (py::arg("dtype")=py::object(), py::arg("copy")=py::object()));

    py::scope tensor_scope = tensor_class;

    py::class_<Tensor_iterator<dim, T>>("_Iterator", py::no_init)
        .def("__iter__", &Tensor_iterator<dim, T>::__iter__, py::return_self<>())
        .def("__next__", &Tensor_iterator<dim, T>::next);

    return tensor_class;
}

// Python constructor
static py::object new_tensor(py::tuple args, py::dict kwargs) {
    auto n = py::len(args), nk = py::len(kwargs);

    py::list params;

    bool force_double = false;
    bool force_complex = false;

    if (kwargs.has_key("dtype")) {
        --nk;
        py::object dtype;
        dtype = kwargs["dtype"];
        if (!dtype.is_none()) {
            if (dtype.ptr() == reinterpret_cast<PyObject*>(&PyFloat_Type))
                force_double = true;
            else if (dtype.ptr() == reinterpret_cast<PyObject*>(&PyComplex_Type))
                force_complex = true;
            else {
                throw TypeError(u8"wrong dtype (can be only float or complex)");
            }
        }
    }

    if (n == 1 && nk == 0) {
        PyObject* obj = py::object(args[0]).ptr();
        if (PyArray_Check(obj) || PySequence_Check(obj)) {
            PyArrayObject* array;
            if (force_double)
                array = (PyArrayObject*)PyArray_ContiguousFromObject(obj, NPY_DOUBLE, 2, 2);
            else if (force_complex)
                array = (PyArrayObject*)PyArray_ContiguousFromObject(obj, NPY_CDOUBLE, 2, 2);
            else {
                force_double = true;
                array = (PyArrayObject*)PyArray_ContiguousFromObject(obj, NPY_DOUBLE, 2, 2);
                if (array == nullptr) {
                    PyErr_Clear();
                    force_double = false;
                    force_complex = true;
                    array = (PyArrayObject*)PyArray_ContiguousFromObject(obj, NPY_CDOUBLE, 2, 2);
                }
            }
            if (array == nullptr) throw plask::CriticalException(u8"Cannot create tensor from provided array");
            if (PyArray_NDIM(array) != 2 || PyArray_DIM(array, 0) != 3 || PyArray_DIM(array, 1) != 3)
                throw TypeError(u8"tensor array must be 2D and have exactly 3 rows and 3 columns");
            if (force_complex) {
                dcomplex* data = reinterpret_cast<dcomplex*>(PyArray_DATA(array));
                return py::object(Tensor<3, dcomplex>(data));
            }
            if (force_double) {
                double* data = reinterpret_cast<double*>(PyArray_DATA(array));
                return py::object(Tensor<3, double>(data));
            }
        }
    }

    if (n == 0) {  // Extract components from kwargs
        n = nk;
        py::object comp[9];

        py::stl_input_iterator<std::string> begin(kwargs.keys()), end;
        for (auto key = begin; key != end; ++key) {
            if (*key == "dtype") continue;
            py::object val = kwargs[*key];
            try {
                if (n == 2)
                    comp[tensor_attr_indx<2>(*key)] = val;
                else
                    comp[tensor_attr_indx<3>(*key)] = val;
            } catch (AttributeError&) {
                throw TypeError(u8"wrong component name for {:d}D tensor if config.axes = '{}'", n, current_axes.str());
            }
        }
        for (int i = 0; i < n; i++) params.append(comp[i]);

    } else if (nk > 0) {
        throw TypeError(u8"components must be provided entirely in a list or by names");
    } else {
        params = py::list(args);
    }

    // Now detect the dtype
    py::object result;
    try {
        if (force_complex) {
            PyErr_SetNone(PyExc_TypeError);
            throw py::error_already_set();
        }
        if (n == 2) return py::object(Tensor<2, double>(py::extract<double>(params[0]), py::extract<double>(params[1])));
        if (n == 3)
            return py::object(
                Tensor<3, double>(py::extract<double>(params[0]), py::extract<double>(params[1]), py::extract<double>(params[2])));
        if (n == 4)
            return py::object(Tensor<3, double>(py::extract<double>(params[0]), py::extract<double>(params[1]),
                                                py::extract<double>(params[2]), py::extract<double>(params[3])));
        if (n == 6)
            return py::object(Tensor<3, double>(py::extract<double>(params[0]), py::extract<double>(params[1]),
                                                py::extract<double>(params[2]), py::extract<double>(params[3]),
                                                py::extract<double>(params[4]), py::extract<double>(params[5])));
        if (n == 9)
            return py::object(
                Tensor<3, double>(py::extract<double>(params[0]), py::extract<double>(params[1]), py::extract<double>(params[2]),
                                  py::extract<double>(params[3]), py::extract<double>(params[4]), py::extract<double>(params[5]),
                                  py::extract<double>(params[6]), py::extract<double>(params[7]), py::extract<double>(params[8])));
        throw TypeError("wrong number of arguments");
    } catch (py::error_already_set&) {
        PyErr_Clear();
        try {
            if (force_double) {
                PyErr_SetNone(PyExc_TypeError);
                throw py::error_already_set();
            }
            if (n == 2) return py::object(Tensor<2, dcomplex>(py::extract<dcomplex>(params[0]), py::extract<dcomplex>(params[1])));
            if (n == 3)
                return py::object(Tensor<3, dcomplex>(py::extract<dcomplex>(params[0]), py::extract<dcomplex>(params[1]),
                                                      py::extract<dcomplex>(params[2])));
            if (n == 4)
                return py::object(Tensor<3, dcomplex>(py::extract<dcomplex>(params[0]), py::extract<dcomplex>(params[1]),
                                                      py::extract<dcomplex>(params[2]), py::extract<dcomplex>(params[3])));
            if (n == 6)
                return py::object(Tensor<3, dcomplex>(py::extract<dcomplex>(params[0]), py::extract<dcomplex>(params[1]),
                                                      py::extract<dcomplex>(params[2]), py::extract<dcomplex>(params[3]),
                                                      py::extract<dcomplex>(params[4]), py::extract<dcomplex>(params[5])));
            if (n == 9)
                return py::object(Tensor<3, dcomplex>(
                    py::extract<dcomplex>(params[0]), py::extract<dcomplex>(params[1]), py::extract<dcomplex>(params[2]),
                    py::extract<dcomplex>(params[3]), py::extract<dcomplex>(params[4]), py::extract<dcomplex>(params[5]),
                    py::extract<dcomplex>(params[6]), py::extract<dcomplex>(params[7]), py::extract<dcomplex>(params[8])));
            throw TypeError("wrong number of arguments");
        } catch (py::error_already_set&) {
            throw TypeError(u8"wrong tensor argument types");
        }
    }

    return py::object();
}

// Python doc
const static char* __doc__ =

    "tensor(xx, yy, zz, xy=0., xz=0., yz=0., dtype=None)\n"
    "tensor(zz, xx, yy, zx=0., zy=0., xy=0., dtype=None)\n"
    "tensor(rr, pp, zz, rp=0., rz=0., pz=0., dtype=None)\n"
    "tensor(xx, yy, zz, xy, yx, xz, zx, yz, zy, dtype=None)\n"
    "tensor(zz, xx, yy, zx, xz, zy, yz, xy, yx, dtype=None)\n"
    "tensor(rr, pp, zz, rp, pr, rz, zr, pz, zp, dtype=None)\n"
    "tensor(xx, yy, dtype=None)\n"
    "tensor(zz, xx, dtype=None)\n"
    "tensor(rr, zz, dtype=None)\n\n"

    "PLaSK tensor.\n\n"

    "The constructor arguments depend on the current value of\n"
    ":attr:`plask.config.axes`. However, you must either specify all the components\n"
    "either as the unnamed sequence or as the named keywords.\n\n"

    "Args:\n"
    "    _letters_ (dtype): Tensor components.\n"
    "        Their choice depends on the current value of :attr:`plask.config.axes`.\n"
    "    dtype (type): type of the tensor components.\n"
    "        If this argument is omitted or `None`, the type is determined\n"
    "        automatically.\n\n"

    "The order of 2D tensor components is always [`horizontal`, `vertical`],\n"
    "while for 3D tensors it is a two-dimensional matrix of Cartesian product\n"
    "of the [`longitudinal`, `transverse`, `vertical`] components.\n"
    "However, the component names depend on the :attr:`~plask.config.axes`\n"
    "configuration option. Changing this option will change the order of component\n"
    "names (even for existing tensors) accordingly:\n\n"

    "=================== =========== ======================================================\n"
    "plask.config.axes   2D tensor   3D tensor\n"
    "=================== =========== ======================================================\n"
    "`xyz`, `yz`, `z_up` [`yy`,`zz`] [[`xx`,`xy`,`xz`], [`yx`,`yy`,`yz`], [`zx`,`zy`,`zz`]]\n"
    "`zxy`, `xy`, `y_up` [`xx`,`yy`] [[`zz`,`zx`,`zy`], [`xz`,`xx`,`xy`], [`yz`,`yx`,`yy`]]\n"
    "`prz`, `rz`, `rad`  [`rr`,`zz`] [[`pp`,`pr`,`pz`], [`rp`,`rr`,`rz`], [`zp`,`zr`,`zz`]]\n"
    "`ltv`, `abs`        [`tt`,`vv`] [[`ll`,`lt`,`lv`], [`tl`,`tt`,`tv`], [`vl`,`vt`,`vv`]]\n"
    "=================== =========== ======================================================\n\n"

    "To access tensor components you may either use attribute names or numerical\n"
    "indexing. The ordering and naming rules are the same as for the construction.\n\n"

    "Examples:\n\n"

    "    >>> config.axes = 'xyz'\n"
    "    >>> v = tensor(1, 2, 3)\n"
    "    >>> v.zz\n"
    "    3\n"
    "    >>> v[0,0]\n"
    "    1\n\n";

void register_tensors() {
    register_tensor_class<2, double>("tensor");
    register_tensor_class<2, dcomplex>("tensor");
    register_tensor_class<3, double>("tensor");
    register_tensor_class<3, dcomplex>("tensor");

    py::def("tensor", py::raw_function(&new_tensor));
    py::scope().attr("tensor").attr("__doc__") = __doc__;
}

}  // namespace python
}  // namespace plask
