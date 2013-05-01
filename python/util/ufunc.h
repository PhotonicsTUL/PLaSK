#ifndef PLASK__PYTHON__UFUNC_H
#define PLASK__PYTHON__UFUNC_H

#include "../plask/python_numpy.h"

namespace plask { namespace python {

template <typename T, typename F>
py::object UFUNC(F f, py::object input) {
    try {
        return py::object(f(py::extract<T>(input)));
    } catch (py::error_already_set) {
        PyErr_Clear();

        PyArrayObject* inarr = (PyArrayObject*)PyArray_FROM_OT(input.ptr(), detail::typenum<T>());
        if (inarr == NULL || PyArray_TYPE(inarr) != detail::typenum<T>()) {
            Py_XDECREF(inarr);
            throw TypeError("Cannot convert input array to required type");
        }

        NpyIter *iter;
        NpyIter_IterNextFunc *iternext;
        PyArrayObject *op[2], *ret;
        npy_uint32 flags;
        npy_uint32 op_flags[2];
        npy_intp itemsize, *innersizeptr, innerstride;
        char **dataptrarray;
        flags = NPY_ITER_EXTERNAL_LOOP;
        op[0] = inarr;
        op[1] = NULL;
        op_flags[0] = NPY_ITER_READONLY;
        op_flags[1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
        iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_NO_CASTING, op_flags, NULL);
        if (iter == NULL) { throw CriticalException("Error in array iteration"); }
        iternext = NpyIter_GetIterNext(iter, NULL);
        innerstride = NpyIter_GetInnerStrideArray(iter)[0];
        itemsize = NpyIter_GetDescrArray(iter)[0]->elsize;
        innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);
        dataptrarray = NpyIter_GetDataPtrArray(iter);
        npy_intp i;
        do {
            npy_intp size = *innersizeptr;
            char *src = dataptrarray[0], *dst = dataptrarray[1];
            for(i = 0; i < size; i++, src += innerstride, dst += itemsize) {
                *((T*)dst) = f(*((T*)src));
            }
        } while (iternext(iter));
        ret = NpyIter_GetOperandArray(iter)[1];
        Py_INCREF(ret);
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            Py_DECREF(ret);
            Py_DECREF(inarr);
            throw CriticalException("Error in array iteration");
        }
        Py_DECREF(inarr);
        return py::object(py::handle<>((PyObject*)ret));;
    }
    return py::object();
}

}} // namespace plask::python

#endif // PLASK__PYTHON__UFUNC_H