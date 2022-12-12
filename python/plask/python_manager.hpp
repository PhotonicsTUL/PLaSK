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
#ifndef PLASK__PYTHON_MANAGER_H
#define PLASK__PYTHON_MANAGER_H

#include "plask/manager.hpp"

namespace plask { namespace  python {


struct PLASK_PYTHON_API XMLExceptionWithCause: public XMLException {
    PyObject* cause;

    XMLExceptionWithCause(PyObject* cause, const XMLReader& reader, const std::string& msg):
        XMLException(reader, msg), cause(cause) {
        Py_XINCREF(cause);
    }

    XMLExceptionWithCause(XMLExceptionWithCause& src): XMLException(src), cause(src.cause) {
        Py_XINCREF(cause);
    }

    XMLExceptionWithCause(XMLExceptionWithCause&& src): XMLException(std::move(src)), cause(src.cause) {
        src.cause = nullptr;
    }

    virtual ~XMLExceptionWithCause() {
        Py_XDECREF(cause);
    }

    void print(const char* scriptname=nullptr, const char* top_frame=nullptr, int scriptline=0) {
        if (cause) {
            printPythonException(cause, scriptname, top_frame, scriptline);
            writelog(LOG_ERROR_DETAIL, "The above exception was the direct cause of the following exception:");
        }
        plask::writelog(plask::LOG_CRITICAL_ERROR, "{}, {}", scriptname, what());
    }

    void setPythonException();

    void throwPythonException() {
        setPythonException();
        throw py::error_already_set();
    }
};


struct PLASK_PYTHON_API PythonManager: public Manager {

//     /// List of constant profiles
//     py::dict profiles;

    /// List of overridden defines
    py::tuple overrites;

    /// Locals read from &lt;defines&gt; section and supplied by user
    py::dict defs;

    PythonManager(bool draft=false): Manager(draft) {}

    shared_ptr<Solver> loadSolver(const std::string& category, const std::string& lib, const std::string& solver_name, const std::string& name) override;

    void loadDefines(XMLReader& reader) override;

    void loadConnects(XMLReader& reader) override;

    void loadMaterial(XMLReader& reader) override;

    void loadMaterialModule(XMLReader& reader);

    void loadMaterials(XMLReader& reader) override;

    static void export_dict(py::object self, py::object dict);

    void loadScript(XMLReader& reader) override;
};

}} // namespace plask::python

#endif // PLASK__PYTHON_MANAGER_H
