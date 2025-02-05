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
#include "../python/globals.hpp"
#include "../python/manager.hpp"

#define PLASK_GEOMETRY_PYTHON_TAG "python"

namespace plask { namespace python {

extern AxisNames current_axes;

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
    assert(dynamic_cast<python::PythonManager*>(&reader.manager) != nullptr);
    python::PythonManager* python_manager = static_cast<python::PythonManager*>(&reader.manager);

    PyObject* code = compilePythonFromXml(reader.source, reader.manager, "", python_manager->globals);
    if (!code) return shared_ptr<GeometryObject>();  // this can happen only in draft mode

    detail::SetPythonAxes setPythonAxes(reader);

    PyObject* result = nullptr;
    if (PyCode_Check(code))
        result = PyEval_EvalCode(code, python_manager->globals.ptr(), nullptr);
    else
        result = PyObject_CallFunctionObjArgs(code, nullptr);
    Py_DECREF(code);
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
                PyCodeObject* f_code =
                    #if PY_VERSION_HEX >= 0x030900B1
                        PyFrame_GetCode(traceback->tb_frame);
                    #else
                        traceback->tb_frame->f_code;
                    #endif
                PyObject* original_filename = f_code->co_filename;
                #if PY_VERSION_HEX >= 0x030900B1
                    Py_XDECREF(f_code);
                #endif
                while (traceback != NULL) {
                    f_code =
                        #if PY_VERSION_HEX >= 0x030900B1
                            PyFrame_GetCode(traceback->tb_frame);
                        #else
                            traceback->tb_frame->f_code;
                        #endif
                    if (f_code->co_filename == original_filename) {
                        line = traceback->tb_lineno;
                        #if PY_VERSION_HEX >= 0x030a0000
                            if (line == -1) line = PyCode_Addr2Line(f_code, traceback->tb_lasti);
                        #endif
                    }
                    #if PY_VERSION_HEX >= 0x030900B1
                        Py_XDECREF(f_code);
                    #endif
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
        reader.manager.throwErrorIfNotDraft(XMLException(reader.source, "no geometry item returned by <python> tag"));
        return shared_ptr<GeometryObject>();
    }

    py::handle<> hres(result);
    return py::extract<shared_ptr<GeometryObject>>(result);
}

static GeometryReader::RegisterObjectReader python_reader(PLASK_GEOMETRY_PYTHON_TAG, read_python);

}} // namespace plask::python
