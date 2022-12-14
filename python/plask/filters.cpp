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
#include "python_filters.hpp"

namespace plask { namespace python {

namespace detail {

    void PLASK_PYTHON_API filterin_parse_key(const py::object& key, shared_ptr<GeometryObject>& geom, PathHints*& path, size_t& points) {
        py::object object;
        path = nullptr;
        int pts = 10;
        if (PyTuple_Check(key.ptr())) {
            if (py::len(key) < 2 || py::len(key) > 3) throw KeyError(py::extract<std::string>(py::str(key)));
            object = key[0];
            if (py::len(key) == 3) {
                path = py::extract<PathHints*>(key[1]);
                pts = py::extract<int>(key[2]);
            } else {
                try {
                    path = py::extract<PathHints*>(key[1]);
                } catch (py::error_already_set&) {
                    PyErr_Clear();
                    try {
                        pts = py::extract<int>(key[1]);
                    } catch (py::error_already_set&) {
                        throw KeyError(py::extract<std::string>(py::str(key)));
                    }
                }
            }
            if (pts < 0) throw KeyError(py::extract<std::string>(py::str(key)));
            points = size_t(pts);
        } else {
            object = key;
        }
        geom = py::extract<shared_ptr<GeometryObject>>(object);
    }
}   // namespace detail

} } // namespace plask::python
