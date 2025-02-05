/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2025 Lodz University of Technology
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
#ifndef PYTHON_PLASK_PYTHON_OVERRIDE_HPP
#define PYTHON_PLASK_PYTHON_OVERRIDE_HPP

#include <frameobject.h>

#include "globals.hpp"

namespace plask { namespace python {

/**
 * Base class for methods that can be overridden form Python
 */
template <typename T> struct Overriden {
    PyObject* self;

    Overriden() {}

    Overriden(PyObject* self) : self(self) {}

    bool overriden(char const* name) const {
        OmpLockGuard lock(python_omp_lock);
        py::converter::registration const& r = py::converter::registered<T>::converters;
        PyTypeObject* class_object = r.get_class_object();
        if (self) {
            py::handle<> mh(PyObject_GetAttrString(self, const_cast<char*>(name)));
            if (mh && PyMethod_Check(mh.get())) {
                PyMethodObject* mo = (PyMethodObject*)mh.get();
                PyObject* borrowed_f = nullptr;
                if (mo->im_self == self && class_object->tp_dict != 0)
                    borrowed_f = PyDict_GetItemString(class_object->tp_dict, const_cast<char*>(name));
                if (borrowed_f != mo->im_func) return true;
            }
        }
        return false;
    }

    bool overriden_no_recursion(char const* name) const {
        py::converter::registration const& r = py::converter::registered<T>::converters;
        PyTypeObject* class_object = r.get_class_object();
        if (self) {
            py::handle<> mh(PyObject_GetAttrString(self, const_cast<char*>(name)));
            if (mh && PyMethod_Check(mh.get())) {
                PyMethodObject* mo = (PyMethodObject*)mh.get();
                PyObject* borrowed_f = nullptr;
                if (mo->im_self == self && class_object->tp_dict != 0)
                    borrowed_f = PyDict_GetItemString(class_object->tp_dict, const_cast<char*>(name));
                if (borrowed_f != mo->im_func) {
                    PyFrameObject* frame = PyEval_GetFrame();
                    if (frame == nullptr) return true;
                    bool result = true;
                    PyCodeObject* f_code =
#if PY_VERSION_HEX >= 0x030900B1
                        PyFrame_GetCode(frame);
#else
                        frame->f_code;
#endif
                    PyCodeObject* method_code = (PyCodeObject*)((PyFunctionObject*)mo->im_func)->func_code;
#if PY_VERSION_HEX >= 0x030b0000
                    if (f_code == method_code && f_code->co_argcount > 0) {
                        PyObject* f_locals = PyFrame_GetLocals(frame);
                        PyObject* co_varnames = PyCode_GetVarnames(f_code);
                        PyObject* self_name = PyTuple_GetItem(co_varnames, 0);
#    if PY_VERSION_HEX >= 0x030d0000
                        PyObject* getitem = PyObject_GetAttrString(f_locals, "__getitem__");
                        PyObject* arg0 = PyObject_CallFunctionObjArgs(getitem, self_name, NULL);
                        Py_XDECREF(getitem);
                        if (arg0 == self) result = false;
                        Py_XDECREF(arg0);
#    else
                        if (PyDict_GetItem(f_locals, self_name) == self) result = false;
#    endif
                        Py_XDECREF(co_varnames);
                        Py_XDECREF(f_locals);
                    }
#else
                    if (f_code == method_code && frame->f_localsplus[0] == self) result = false;
#    if PY_VERSION_HEX >= 0x030900B1
                    Py_XDECREF(f_code);
#    endif
#endif
                    return result;
                }
            }
        }
        return false;
    }

    template <typename R, typename... Args> inline R call_python(const char* name, Args... args) const {
        OmpLockGuard lock(python_omp_lock);
        if (overriden(name)) {
            return py::call_method<R>(self, name, args...);
        }
        py::handle<> __class__(PyObject_GetAttrString(self, "__class__"));
        py::handle<> __name__(PyObject_GetAttrString(__class__.get(), "__name__"));
        throw AttributeError("'{}' object has not attribute '{}'", std::string(py::extract<std::string>(py::object(__name__))),
                             name);
    }
};

}}  // namespace plask::python

#endif  // PYTHON_PLASK_PYTHON_OVERRIDE_HPP
