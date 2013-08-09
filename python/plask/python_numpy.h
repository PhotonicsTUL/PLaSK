#ifndef PLASK__PYTHON_NUMPY_H
#define PLASK__PYTHON_NUMPY_H

#include "python_globals.h"

#define NPY_NO_DEPRECATED_API
#include <numpy/arrayobject.h>

#ifndef NPY_1_7_API_VERSION
inline static void PyArray_SetBaseObject(PyArrayObject* arr, PyObject* obj) {
    PyArray_BASE(arr) = obj;
}
#endif

namespace plask { namespace python {

// ----------------------------------------------------------------------------------------------------------------------
// Get numpy typenums for some types
namespace detail {
    template <typename T> static inline constexpr int typenum();
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
    template <> inline constexpr int typenum<const Tensor3<dcomplex>>() { return NPY_CDOUBLE; }
}


// ----------------------------------------------------------------------------------------------------------------------
/**
 * Either make sure the object stays alive as long as array, or make a copy to the desired dtype
 */
template <typename T>
inline void confirm_array(PyObject*& arr, py::object& self, py::object& dtype) {
    PyArray_Descr* descr;
    if(PyArray_DescrConverter(dtype.ptr(), &descr) && descr->type_num != detail::typenum<T>()) {
        PyArrayObject* oarr = reinterpret_cast<PyArrayObject*>(arr);
        arr = PyArray_CastToType(oarr, descr, 1);
        Py_DECREF(oarr);
        if (arr == nullptr) throw TypeError("cannot convert array to required dtype");
    } else {
        py::incref(self.ptr());
        PyArray_SetBaseObject((PyArrayObject*)arr, self.ptr()); // Make sure the data vector stays alive as long as the array
    }
    Py_XDECREF(descr);
}

/*
 * Import numpy (needs to be called in every cpp, which uses arrays)
 */
static inline bool plask_import_array() {
    import_array1(false);
    return true;
}

}} // namespace plask::python

#endif // PLASK__PYTHON_GLOBALS_H
