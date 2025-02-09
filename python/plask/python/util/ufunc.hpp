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
#ifndef PLASK__PYTHON__UFUNC_H
#define PLASK__PYTHON__UFUNC_H

#include "../numpy.hpp"

namespace plask { namespace python {

PLASK_PYTHON_API extern bool ufunc_ignore_errors;

template <typename OT, typename IT = OT, typename F> py::object UFUNC(F f, py::object input, const char* name, const char* arg) {
    try {
        return py::object(f(py::extract<IT>(input)));
    } catch (py::error_already_set&) {
        PyErr_Clear();

        PyArrayObject* inarr = (PyArrayObject*)PyArray_FROM_OT(input.ptr(), detail::typenum<IT>());

        if (inarr == NULL || PyArray_TYPE(inarr) != detail::typenum<IT>()) {
            Py_XDECREF(inarr);
            throw TypeError("cannot convert input array to required type");
        }

        PyArrayObject* op[2] = {inarr, NULL};
        PyArrayObject* ret;
        npy_uint32 flags = NPY_ITER_EXTERNAL_LOOP;
        npy_uint32 op_flags[2] = {NPY_ITER_READONLY, NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE};
        PyArray_Descr* op_dtypes[2] = {NULL, PyArray_DescrFromType(detail::typenum<OT>())};
        NpyIter* iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags, op_dtypes);
        if (iter == NULL) {
            throw CriticalException("error in array iteration");
        }
        NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, NULL);
        npy_intp innerstride = NpyIter_GetInnerStrideArray(iter)[0];
#if NPY_API_VERSION >= 0x00000012
        npy_intp itemsize = PyDataType_ELSIZE(op_dtypes[1]);
#else
        npy_intp itemsize = op_dtypes[1]->elsize;
#endif
        npy_intp* innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        char** dataptrarray = NpyIter_GetDataPtrArray(iter);
        do {
            npy_intp size = *innersizeptr;
            char *src = dataptrarray[0], *dst = dataptrarray[1];
            for (npy_intp i = 0; i < size; i++, src += innerstride, dst += itemsize) {
                try {
                    *((OT*)dst) = f(*((IT*)src));
                } catch (ComputationError err) {
                    writelog(LOG_ERROR, "{}({}={}): ComputationError: {}", name, arg, plask::str(*((IT*)src)), err.what());
                    if (ufunc_ignore_errors)
                        *((OT*)dst) = NAN;
                    else
                        throw;
                }
            }
        } while (iternext(iter));
        ret = NpyIter_GetOperandArray(iter)[1];
        Py_INCREF(ret);
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(ret);
            Py_DECREF(inarr);
            throw CriticalException("error in array iteration");
        }
        Py_DECREF(inarr);
        return py::object(py::handle<>((PyObject*)ret));
        ;
    }
    return py::object();
}

template <typename OT, typename IT = OT, typename F>
py::object PARALLEL_UFUNC(F f, py::object input, const char* name, const char* arg) {
    try {
        return py::object(f(py::extract<IT>(input)));
    } catch (py::error_already_set&) {
        PyErr_Clear();

        PyArrayObject* inarr = (PyArrayObject*)PyArray_FROM_OT(input.ptr(), detail::typenum<IT>());

        if (inarr == NULL || PyArray_TYPE(inarr) != detail::typenum<IT>()) {
            Py_XDECREF(inarr);
            throw TypeError("cannot convert input array to required type");
        }

        PyArrayObject* op[2] = {inarr, NULL};
        PyArrayObject* ret;
        npy_uint32 flags = NPY_ITER_EXTERNAL_LOOP;
        npy_uint32 op_flags[2] = {NPY_ITER_READONLY, NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE};
        PyArray_Descr* op_dtypes[2] = {NULL, PyArray_DescrFromType(detail::typenum<OT>())};
        NpyIter* iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags, op_dtypes);
        if (iter == NULL) {
            throw CriticalException("error in array iteration");
        }
        NpyIter_IterNextFunc* iternext = NpyIter_GetIterNext(iter, NULL);
        npy_intp innerstride = NpyIter_GetInnerStrideArray(iter)[0];
#if NPY_API_VERSION >= 0x00000012
        npy_intp itemsize = PyDataType_ELSIZE(op_dtypes[1]);
#else
        npy_intp itemsize = op_dtypes[1]->elsize;
#endif
        npy_intp* innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        char** dataptrarray = NpyIter_GetDataPtrArray(iter);
        std::exception_ptr error;
#ifndef _MSC_VER
#    pragma omp parallel
#endif
        {
#ifndef _MSC_VER
#    pragma omp single nowait
#endif
            do {
                npy_intp size = *innersizeptr;
                char *src = dataptrarray[0], *dst = dataptrarray[1];
                for (npy_intp i = 0; i < size; i++, src += innerstride, dst += itemsize) {
#ifndef _MSC_VER
#    pragma omp task firstprivate(src) firstprivate(dst)
#endif
                    {
                        if (!error) try {
                                *((OT*)dst) = f(*((IT*)src));
                            } catch (ComputationError err) {
                                if (ufunc_ignore_errors) {
                                    writelog(LOG_ERROR, "{}({}={}): ComputationError: {}", name, arg, plask::str(*((IT*)src)),
                                             err.what());
                                    *((OT*)dst) = NAN;
                                } else {
#ifndef _MSC_VER
#    pragma omp critical
#endif
                                    error = std::current_exception();
                                }
                            } catch (...) {
#ifndef _MSC_VER
#    pragma omp critical
#endif
                                error = std::current_exception();
                            }
                    }
                }
            } while (iternext(iter));
#ifndef _MSC_VER
#    pragma omp taskwait
#endif
        }
        if (error) {
            Py_XDECREF(inarr);
            std::rethrow_exception(error);
        }
        ret = NpyIter_GetOperandArray(iter)[1];
        Py_INCREF(ret);
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(ret);
            Py_DECREF(inarr);
            throw CriticalException("error in array iteration");
        }
        Py_DECREF(inarr);
        return py::object(py::handle<>((PyObject*)ret));
        ;
    }
    return py::object();
}

}}  // namespace plask::python

#endif  // PLASK__PYTHON__UFUNC_H
