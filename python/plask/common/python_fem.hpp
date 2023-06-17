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

#include <plask/python_enum.hpp>

namespace plask { namespace python {

template <typename SolverT> inline static void registerFemSolver(SolverT& solver) {
    solver.def_readwrite("algorithm", &SolverT::Class::algorithm, "Chosen matrix factorization algorithm");
    solver.def_readwrite("itererr", &SolverT::Class::itererr, "Allowed residual iteration for iterative method");
    solver.def_readwrite("iterlim", &SolverT::Class::iterlim, "Maximum number of iterations for iterative method");

    // solver.def_readwrite("iter_accelerator" = &SolverT::Class::iter_accelerator, u8"Iterative solver accelerator");
    // solver.def_readwrite("iter_preconditioner" = &SolverT::Class::iter_preconditioner, u8"Iterative solver preconditioner");

    py::scope solver_scope = solver;

    py_enum<Algorithm>()
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
        .value("ITERATIVE", ALGORITHM_ITERATIVE);

    // py_enum<SparseBandMatrix::Accelelator>()
    //     .value("CG", SparseBandMatrix::ACCEL_CG)
    //     .value("SI", SparseBandMatrix::ACCEL_SI)
    //     .value("SOR", SparseBandMatrix::ACCEL_SOR)
    //     .value("SRCG", SparseBandMatrix::ACCEL_SRCG)
    //     .value("SRSI", SparseBandMatrix::ACCEL_SRSI)
    //     .value("BASIC", SparseBandMatrix::ACCEL_BASIC)
    //     .value("ME", SparseBandMatrix::ACCEL_ME)
    //     .value("CGNR", SparseBandMatrix::ACCEL_CGNR)
    //     .value("LSQR", SparseBandMatrix::ACCEL_LSQR)
    //     .value("ODIR", SparseBandMatrix::ACCEL_ODIR)
    //     .value("OMIN", SparseBandMatrix::ACCEL_OMIN)
    //     .value("ORES", SparseBandMatrix::ACCEL_ORES)
    //     .value("IOM", SparseBandMatrix::ACCEL_IOM)
    //     .value("GMRES", SparseBandMatrix::ACCEL_GMRES)
    //     .value("USYMLQ", SparseBandMatrix::ACCEL_USYMLQ)
    //     .value("USYMQR", SparseBandMatrix::ACCEL_USYMQR)
    //     .value("LANDIR", SparseBandMatrix::ACCEL_LANDIR)
    //     .value("LANMIN", SparseBandMatrix::ACCEL_LANMIN)
    //     .value("LANRES", SparseBandMatrix::ACCEL_LANRES)
    //     .value("CGCR", SparseBandMatrix::ACCEL_CGCR)
    //     .value("BCGS", SparseBandMatrix::ACCEL_BCGS)
    // ;

    // py_enum<SparseBandMatrix::Preconditioner>()
    //     .value("RICH", SparseBandMatrix::PRECOND_RICH)
    //     .value("JAC", SparseBandMatrix::PRECOND_JAC)
    //     .value("LJAC", SparseBandMatrix::PRECOND_LJAC)
    //     .value("LJACX", SparseBandMatrix::PRECOND_LJACX)
    //     .value("SOR", SparseBandMatrix::PRECOND_SOR)
    //     .value("SSOR", SparseBandMatrix::PRECOND_SSOR)
    //     .value("IC", SparseBandMatrix::PRECOND_IC)
    //     .value("MIC", SparseBandMatrix::PRECOND_MIC)
    //     .value("LSP", SparseBandMatrix::PRECOND_LSP)
    //     .value("NEU", SparseBandMatrix::PRECOND_NEU)
    //     .value("LSOR", SparseBandMatrix::PRECOND_LSOR)
    //     .value("LSSOR", SparseBandMatrix::PRECOND_LSSOR)
    //     .value("LLSP", SparseBandMatrix::PRECOND_LLSP)
    //     .value("LNEU", SparseBandMatrix::PRECOND_LNEU)
    //     .value("BIC", SparseBandMatrix::PRECOND_BIC)
    //     .value("BICX", SparseBandMatrix::PRECOND_BICX)
    //     .value("MBIC", SparseBandMatrix::PRECOND_MBIC)
    //     .value("MBICX", SparseBandMatrix::PRECOND_MBICX)
    // ;
}
template <typename SolverT>
inline static void registerFemSolverWithMaskedMesh(SolverT& solver) {
    registerFemSolver(solver);
    solver.add_property("include_empty", &SolverT::Class::usingFullMesh, &SolverT::Class::useFullMesh,
                        "Should empty regions (e.g. air) be included into computation domain?");
}

}}  // namespace plask::python

#endif
