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
#ifndef PLASK__PYTHON_NUMPY_H
#define PLASK__PYTHON_NUMPY_H

#include "globals.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "ptr.hpp"

#if NPY_API_VERSION < 0x00000007
inline static void PyArray_SetBaseObject(PyArrayObject* arr, PyObject* obj) {
    PyArray_BASE(arr) = obj;
}
#endif

namespace plask { namespace python {

// ----------------------------------------------------------------------------------------------------------------------
// Get numpy typenums for some types
namespace detail {
    template <typename T> static inline constexpr int typenum() { return NPY_OBJECT; }
    template <> inline constexpr int typenum<int>() { return NPY_INT; }
    template <> inline constexpr int typenum<double>() { return NPY_DOUBLE; }
    template <> inline constexpr int typenum<dcomplex>() { return NPY_CDOUBLE; }
    template <> inline constexpr int typenum<Vec<2,double>>() { return NPY_DOUBLE; }
    template <> inline constexpr int typenum<Vec<2,dcomplex>>() { return NPY_CDOUBLE; }
    template <> inline constexpr int typenum<Vec<3,double>>() { return NPY_DOUBLE; }
    template <> inline constexpr int typenum<Vec<3,dcomplex>>() { return NPY_CDOUBLE; }
    template <> inline constexpr int typenum<const double>() { return NPY_DOUBLE; }
    template <> inline constexpr int typenum<const dcomplex>() { return NPY_CDOUBLE; }
    template <> inline constexpr int typenum<const Vec<2,double>>() { return NPY_DOUBLE; }
    template <> inline constexpr int typenum<const Vec<2,dcomplex>>() { return NPY_CDOUBLE; }
    template <> inline constexpr int typenum<const Vec<3,double>>() { return NPY_DOUBLE; }
    template <> inline constexpr int typenum<const Vec<3,dcomplex>>() { return NPY_CDOUBLE; }
    template <> inline constexpr int typenum<const Tensor2<double>>() { return NPY_DOUBLE; }
    template <> inline constexpr int typenum<const Tensor2<dcomplex>>() { return NPY_CDOUBLE; }
    template <> inline constexpr int typenum<const Tensor3<double>>() { return NPY_DOUBLE; }
    template <> inline constexpr int typenum<const Tensor3<dcomplex>>() { return NPY_CDOUBLE; }
}


// ----------------------------------------------------------------------------------------------------------------------
/**
 * Either make sure the object stays alive as long as array, or make a copy to the desired dtype
 */
template <typename T>
inline void confirm_array(PyObject*& arr, const py::object& self, const py::object& dtype, const py::object& copy) {
    PyHandle<PyArray_Descr> descr;
    if(!dtype.is_none() && PyArray_DescrConverter(dtype.ptr(), &descr.ref()) && descr->type_num != detail::typenum<T>()) {
        PyHandle<PyArrayObject> oarr = reinterpret_cast<PyArrayObject*>(arr);
        arr = PyArray_CastToType(oarr, descr, 1);
        if (arr == nullptr) throw TypeError(u8"cannot convert array to required dtype");
    } else if (!copy.is_none() && py::extract<bool>(copy)()) {
        arr = PyArray_FromArray(reinterpret_cast<PyArrayObject*>(arr), descr, NPY_ARRAY_ENSURECOPY);
        if (arr == nullptr) throw plask::CriticalException(u8"cannot copy array");
    } else {
        py::incref(self.ptr());
        PyArray_SetBaseObject((PyArrayObject*)arr, self.ptr()); // Make sure the data vector stays alive as long as the array
    }
}

// ----------------------------------------------------------------------------------------------------------------------
/**
 * Tool to make DataVector from numpy array
 */
namespace detail {
    struct NumpyDataDeleter {
        PyArrayObject* arr;
        NumpyDataDeleter(PyArrayObject* arr) : arr(arr) {
            OmpLockGuard lock(python_omp_lock);
            Py_XINCREF(arr);
        }
        void operator()(void*) {
            OmpLockGuard lock(python_omp_lock);
            Py_XDECREF(arr);
        }
    };
}

// ----------------------------------------------------------------------------------------------------------------------
/*
 * Import numpy (needs to be called in every cpp, which uses arrays)
 */
#ifndef NO_IMPORT_ARRAY
    static inline bool plask_import_array() {
        import_array1(false);
        return true;
    }
#endif

}} // namespace plask::python

#endif // PLASK__PYTHON_GLOBALS_H
