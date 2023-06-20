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

namespace plask {
namespace python {

template <typename SolverT> static IterativeMatrixParams* __get_iter_params(typename SolverT::Class& solver) {
    return &solver.iter_params;
}

inline static void registerFemCommon() {
    py_enum<FemMatrixAlgorithm>()
        .value("CHOLESKY", ALGORITHM_CHOLESKY)
        .value("GAUSS", ALGORITHM_GAUSS)
        .value("ITERATIVE", ALGORITHM_ITERATIVE);

    py_enum<IterativeMatrixParams::Accelelator>()
        .value("CG", IterativeMatrixParams::ACCEL_CG)
        .value("SI", IterativeMatrixParams::ACCEL_SI)
        .value("SOR", IterativeMatrixParams::ACCEL_SOR)
        .value("SRCG", IterativeMatrixParams::ACCEL_SRCG)
        .value("SRSI", IterativeMatrixParams::ACCEL_SRSI)
        .value("BASIC", IterativeMatrixParams::ACCEL_BASIC)
        .value("ME", IterativeMatrixParams::ACCEL_ME)
        .value("CGNR", IterativeMatrixParams::ACCEL_CGNR)
        .value("LSQR", IterativeMatrixParams::ACCEL_LSQR)
        .value("ODIR", IterativeMatrixParams::ACCEL_ODIR)
        .value("OMIN", IterativeMatrixParams::ACCEL_OMIN)
        .value("ORES", IterativeMatrixParams::ACCEL_ORES)
        .value("IOM", IterativeMatrixParams::ACCEL_IOM)
        .value("GMRES", IterativeMatrixParams::ACCEL_GMRES)
        .value("USYMLQ", IterativeMatrixParams::ACCEL_USYMLQ)
        .value("USYMQR", IterativeMatrixParams::ACCEL_USYMQR)
        .value("LANDIR", IterativeMatrixParams::ACCEL_LANDIR)
        .value("LANMIN", IterativeMatrixParams::ACCEL_LANMIN)
        .value("LANRES", IterativeMatrixParams::ACCEL_LANRES)
        .value("CGCR", IterativeMatrixParams::ACCEL_CGCR)
        .value("BCGS", IterativeMatrixParams::ACCEL_BCGS);

    py_enum<IterativeMatrixParams::Preconditioner>()
        .value("RICH", IterativeMatrixParams::PRECOND_RICH)
        .value("JAC", IterativeMatrixParams::PRECOND_JAC)
        .value("LJAC", IterativeMatrixParams::PRECOND_LJAC)
        .value("LJACX", IterativeMatrixParams::PRECOND_LJACX)
        .value("SOR", IterativeMatrixParams::PRECOND_SOR)
        .value("SSOR", IterativeMatrixParams::PRECOND_SSOR)
        .value("IC", IterativeMatrixParams::PRECOND_IC)
        .value("MIC", IterativeMatrixParams::PRECOND_MIC)
        .value("LSP", IterativeMatrixParams::PRECOND_LSP)
        .value("NEU", IterativeMatrixParams::PRECOND_NEU)
        .value("LSOR", IterativeMatrixParams::PRECOND_LSOR)
        .value("LSSOR", IterativeMatrixParams::PRECOND_LSSOR)
        .value("LLSP", IterativeMatrixParams::PRECOND_LLSP)
        .value("LNEU", IterativeMatrixParams::PRECOND_LNEU)
        .value("BIC", IterativeMatrixParams::PRECOND_BIC)
        .value("BICX", IterativeMatrixParams::PRECOND_BICX)
        .value("MBIC", IterativeMatrixParams::PRECOND_MBIC)
        .value("MBICX", IterativeMatrixParams::PRECOND_MBICX);

    py_enum<IterativeMatrixParams::NoConvergenceBehavior>()
        .value("ERROR", IterativeMatrixParams::NO_CONVERGENCE_ERROR)
        .value("WARNING", IterativeMatrixParams::NO_CONVERGENCE_WARNING)
        .value("CONTINUE", IterativeMatrixParams::NO_CONVERGENCE_CONTINUE);

    py::class_<IterativeMatrixParams, boost::noncopyable>("FemIterativeMatrix", "Iterative matrix parameters", py::no_init)
        .def_readwrite("accelerator", &IterativeMatrixParams::accelerator,
                       "Solver accelerator\n\n"
                       "This is current iterative solver acceleration algorithm. Possible choices are:\n\n"
                       "+------------+-------------------------------------------------------------------------------------+\n"
                       "| ``cg``     | Conjugate Gradient acceleration - this is the default and should be preferred       |\n"
                       "| ``si``     | Chebyshev acceleration or Semi-Iteration                                            |\n"
                       "| ``sor``    | Successive Overrelaxation                                                           |\n"
                       "| ``srcg``   | Symmetric Successive Overrelaxation Conjugate Gradient Algorithm                    |\n"
                       "| ``srsi``   | Symmetric Successive Overrelaxation Semi-Iteration Algorithm                        |\n"
                       "| ``basic``  | Basic Iterative Method                                                              |\n"
                       "| ``me``     | Minimal Error Algorithm                                                             |\n"
                       "| ``cgnr``   | Conjugate Gradient applied to the Normal Equations                                  |\n"
                       "| ``lsqr``   | Least Squares Algorithm                                                             |\n"
                       "| ``odir``   | ORTHODIR, a truncated/restarted method useful for nonsymmetric systems of equations |\n"
                       "| ``omin``   | ORTHOMIN, a common truncated/restarted method used for nonsymmetric systems         |\n"
                       "| ``ores``   | ORTHORES, another truncated/restarted method for nonsymmetric systems               |\n"
                       "| ``iom``    | Incomplete Orthogonalization Method                                                 |\n"
                       "| ``gmres``  | Generalized Minimal Residual Method                                                 |\n"
                       "| ``usymlq`` | Unsymmetric LQ                                                                      |\n"
                       "| ``usymqr`` | Unsymmetric QR                                                                      |\n"
                       "| ``landir`` | Lanczos/ORTHODIR                                                                    |\n"
                       "| ``lanmin`` | Lanczos/ORTHOMIN or Biconjugate Gradient Method                                     |\n"
                       "| ``lanres`` | Lanczos/ORTHORES or “two-sided” Lanczos Method                                      |\n"
                       "| ``cgcr``   | Constrained Generalized Conjugate Residual Method                                   |\n"
                       "| ``bcgs``   | Biconjugate Gradient Squared Method                                                 |\n"
                       "+------------+-------------------------------------------------------------------------------------+\n")
        .def_readwrite("preconditioner", &IterativeMatrixParams::preconditioner,
                       "Solver preconditioner\n\n"
                       "Preconditioner used for iterative matrix solver. Possible choices are:\n\n"
                       "+-----------+---------------------------------------------+\n"
                       "| ``rich``  | Richardson's method                         |\n"
                       "| ``jac``   | Jacobi method                               |\n"
                       "| ``ljac``  | Line Jacobi method                          |\n"
                       "| ``ljacx`` | Line Jacobi method (approx. inverse)        |\n"
                       "| ``sor``   | Successive Overrelaxation                   |\n"
                       "| ``ssor``  | Symmetric SOR                               |\n"
                       "| ``ic``    | Incomplete Cholesky                         |\n"
                       "| ``mic``   | Modified Incomplete Cholesky                |\n"
                       "| ``lsp``   | Least Squares Polynomial                    |\n"
                       "| ``neu``   | Neumann Polynomial                          |\n"
                       "| ``lsor``  | Line SOR                                    |\n"
                       "| ``lssor`` | Line SSOR                                   |\n"
                       "| ``llsp``  | Line Least Squares Polynomial               |\n"
                       "| ``lneu``  | Line Neumann Polynomial                     |\n"
                       "| ``bic``   | Block Incomplete Cholesky (ver. 1)          |\n"
                       "| ``bicx``  | Block Incomplete Cholesky (ver. 2)          |\n"
                       "| ``mbic``  | Modified Block Incomplete Cholesky (ver. 1) |\n"
                       "| ``mbicx`` | Modified Block Incomplete Cholesky (ver. 2) |\n"
                       "+-----------+---------------------------------------------+\n")
        .def_readwrite("noconv", &IterativeMatrixParams::no_convergence_behavior,
                       "Desired behavior if the iterative solver does not converge.\n\n"
                       "Possible choices are: ``error``, ``warning``, ``continue``\n")
        .def_readwrite("maxit", &IterativeMatrixParams::maxit, "Maximum number of iterations")
        .def_readwrite("maxerr", &IterativeMatrixParams::err, "Maximum allowed residual iteration")
        .def_readwrite("nfact", &IterativeMatrixParams::nfact,
                       "Frequency of partial factorization\n\n"
                       "This number initializes the frequency of partial factorizations.\n"
                       "It specifies the number of linear system evaluatations between factorizations.\n"
                       "The default value is 1, which means that a factorization is performed at every\n"
                       "iteration.")
        // .def_readwrite("omega", &IterativeMatrixParams::omega, "Relaxation parameter")

        .def_readonly("converged", &IterativeMatrixParams::converged, "True if the solver converged")
        .def_readonly("iters", &IterativeMatrixParams::iters, "Number of iterations in the last run")
        .def_readonly("err", &IterativeMatrixParams::err, "Residual error in the last run");
}

template <typename SolverT> inline static void registerFemSolver(SolverT& solver) {
    solver.def_readwrite("algorithm", &SolverT::Class::algorithm, "Chosen matrix factorization algorithm");
    solver.add_property("iterative", py::make_function(&__get_iter_params<SolverT>, py::return_internal_reference<>()),
                        "Iterative matrix parameters (see :py:class:`~plask.FemIterativeMatrix`)");
}

template <typename SolverT> inline static void registerFemSolverWithMaskedMesh(SolverT& solver) {
    registerFemSolver(solver);
    solver.add_property("include_empty", &SolverT::Class::usingFullMesh, &SolverT::Class::useFullMesh,
                        "Should empty regions (e.g. air) be included into computation domain?");
}

}}  // namespace plask::python

#endif
