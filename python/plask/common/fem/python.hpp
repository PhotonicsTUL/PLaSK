/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2023 Lodz University of Technology
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
#ifndef PLASK_COMMON_FEM_PYTHON_HPP
#define PLASK_COMMON_FEM_PYTHON_HPP

#include <plask/common/fem.hpp>

#include <plask/python_globals.hpp>
#include <plask/python_enum.hpp>

namespace plask {
namespace python {

template <typename SolverT> static IterativeMatrixParams* __get_iter_params(typename SolverT::Class& solver) {
    return &solver.iter_params;
}

void registerFemCommon();

template <typename SolverT> inline static void registerFemSolver(SolverT& solver) {
    solver.def_readwrite("algorithm", &SolverT::Class::algorithm, "Chosen matrix factorization algorithm");
    solver.add_property("iterative", py::make_function(&__get_iter_params<SolverT>, py::return_internal_reference<>()),
                        "Iterative matrix parameters (see :py:class:`~plask.IterativeParams`)");
}

template <typename SolverT> inline static void registerFemSolverWithMaskedMesh(SolverT& solver) {
    registerFemSolver(solver);
    solver.add_property("include_empty", &SolverT::Class::usingFullMesh, &SolverT::Class::useFullMesh,
                        "Should empty regions (e.g. air) be included into computation domain?");
}

}}  // namespace plask::python

#endif
