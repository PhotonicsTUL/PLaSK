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
/** \file
 * Python wrapper for optical/effective solvers.
 */
#include <cmath>
#include <plask/python.hpp>
#include <plask/python_util/ufunc.hpp>
#include <plask/mesh/generator_rectangular.hpp>

using namespace plask;
using namespace plask::python;

#define ROOTDIGGER_ATTRS_DOC                             \
    u8".. rubric:: Attributes:\n\n"                      \
    u8".. autosummary::\n\n"                             \
    u8"   ~optical.effective.RootParams.alpha\n"         \
    u8"   ~optical.effective.RootParams.lambd\n"         \
    u8"   ~optical.effective.RootParams.initial_range\n" \
    u8"   ~optical.effective.RootParams.maxiter\n"       \
    u8"   ~optical.effective.RootParams.maxstep\n"       \
    u8"   ~optical.effective.RootParams.method\n"        \
    u8"   ~optical.effective.RootParams.tolf_max\n"      \
    u8"   ~optical.effective.RootParams.tolf_min\n"      \
    u8"   ~optical.effective.RootParams.tolx\n\n"        \
    u8"   ~optical.effective.RootParams.stairs\n\n"      \
    u8":rtype: RootParams\n"

#define SEARCH_ARGS_DOC                                                                   \
    u8"    start (complex): Start of the search range (0 means automatic).\n"             \
    u8"    end (complex): End of the search range (0 means automatic).\n"                 \
    u8"    resteps (integer): Number of steps on the real axis during the search.\n"      \
    u8"    imsteps (integer): Number of steps on the imaginary axis during the search.\n" \
    u8"    eps (complex): required precision of the search.\n"


namespace plask { namespace optical { namespace effective { namespace python {

template <typename SolverT> static void Optical_setMesh(SolverT& self, py::object omesh) {
    try {
        shared_ptr<OrderedMesh1D> mesh = py::extract<shared_ptr<OrderedMesh1D>>(omesh);
        self.setHorizontalMesh(mesh);
    } catch (py::error_already_set&) {
        PyErr_Clear();
        try {
            shared_ptr<MeshGeneratorD<1>> meshg = py::extract<shared_ptr<MeshGeneratorD<1>>>(omesh);
            self.setMesh(plask::make_shared<RectangularMesh2DFrom1DGenerator>(meshg));
        } catch (py::error_already_set&) {
            PyErr_Clear();
            plask::python::detail::ExportedSolverDefaultDefs<SolverT>::Solver_setMesh(self, omesh);
        }
    }
}

template <typename Solver> static double Mode_total_absorption(typename Solver::Mode& self) {
    return self.solver->getTotalAbsorption(self);
}

template <typename Solver, typename Name> static py::object getDeltaNeff(Solver& self, py::object pos) {
    return UFUNC<dcomplex, double>([&](double p) { return self.getDeltaNeff(p); }, pos, Name::val, "pos");
}


}}}} // namespace plask::optical::effective::python
