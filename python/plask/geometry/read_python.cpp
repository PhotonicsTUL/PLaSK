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
#include <plask/plask.hpp>
#include "../python_globals.hpp"

#define PLASK_GEOMETRY_PYTHON_TAG "python"
#define RETURN_VARIABLE "__object__"

namespace plask { namespace python {

extern AxisNames current_axes;
extern PLASK_PYTHON_API py::dict* pyXplGlobals;

namespace detail {
    struct SetPythonAxes {
        AxisNames saved;
        SetPythonAxes(GeometryReader& reader): saved(current_axes) {
            current_axes = reader.getAxisNames();
        }
        ~SetPythonAxes() {
            current_axes = saved;
        }
    };
}

shared_ptr<GeometryObject> read_python(GeometryReader& reader) {
    PyCodeObject* code = compilePythonFromXml(reader.source, reader.manager, true);
    if (!code) return shared_ptr<GeometryObject>();  // this can happen only in draft mode

    detail::SetPythonAxes setPythonAxes(reader);
    py::dict locals;

    PyObject* result = PyEval_EvalCode((PyObject*)code, pyXplGlobals->ptr(), locals.ptr());
    if (!result) {
        if (reader.manager.draft) {
            PyObject *value, *type;
            PyTracebackObject *traceback;
            PyErr_Fetch(&type, &value, (PyObject**)(&traceback));
            PyErr_NormalizeException(&type, &value, (PyObject**)(&traceback));
            py::handle<> value_h(value), type_h(type), traceback_h(py::allow_null((PyObject*)traceback));
            std::string message;
            PyObject* value_str = PyObject_Str(value);
            if (value_str) {
                message = py::extract<std::string>(value_str);
                Py_DECREF(value_str);
            }
            PyObject* type_name = PyObject_GetAttrString(type, "__name__");
            if (type_name) {
                PyObject* type_str = PyObject_Str(type_name);
                if (type_str) {
                    message = py::extract<std::string>(type_str)() + ": " + message;
                    Py_DECREF(type_str);
                }
                Py_DECREF(type_name);
            }
            int line = -1;
            if (traceback) {
                PyObject* original_filename = traceback->tb_frame->f_code->co_filename;
                while (traceback != NULL) {
                    if (traceback->tb_frame->f_code->co_filename == original_filename)
                        line = traceback->tb_lineno;
                    traceback = traceback->tb_next;
                };
            }
            reader.manager.pushError(message, line);
            PyErr_Clear();
            return shared_ptr<GeometryObject>();
        } else
            throw py::error_already_set();
    }

    if (result == Py_None) {
        Py_DECREF(result);
        if (locals.has_key(RETURN_VARIABLE)) {
            result = PyDict_GetItemString(locals.ptr(), RETURN_VARIABLE);
            Py_INCREF(result);
        } else {
            reader.manager.throwErrorIfNotDraft(XMLException(reader.source, "No geometry item defined"));
            return shared_ptr<GeometryObject>();
        }
    }
    py::handle<> hres(result);

    return py::extract<shared_ptr<GeometryObject>>(result);
}

static GeometryReader::RegisterObjectReader python_reader(PLASK_GEOMETRY_PYTHON_TAG, read_python);

}} // namespace plask::python
