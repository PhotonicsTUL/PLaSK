#ifndef PLASK__PYTHON_NUMPY_H
#define PLASK__PYTHON_NUMPY_H

#include "python_globals.h"
#include <numpy/arrayobject.h>


namespace plask { namespace python {

/**
 * Either make sure the object statys alive as long as array, or make a copy to the desired dtype
 */
template <typename T>
inline void confirm_array(PyObject*& arr, py::object& self, py::object& dtype) {
    PyArray_Descr* descr;
    if(PyArray_DescrConverter(dtype.ptr(), &descr) && descr->type_num != detail::typenum<T>()) {
        PyArrayObject* oarr = reinterpret_cast<PyArrayObject*>(arr);
        arr = PyArray_CastToType(oarr, descr, 1);
        Py_DECREF(oarr);
        if (arr == nullptr) throw TypeError("cannot convert array to required dtype");
    }
    else {
        py::incref(self.ptr());
        PyArray_BASE(arr) = self.ptr(); // Make sure the data vector stays alive as long as the array
    }
    Py_XDECREF(descr);
}

/*
 * Import numpy (needs to be called in every cpp, which users arrays)
 */
static inline bool plask_import_array() {
    import_array1(false);
    return true;
}

}} // namespace plask::python

#endif // PLASK__PYTHON_GLOBALS_H
