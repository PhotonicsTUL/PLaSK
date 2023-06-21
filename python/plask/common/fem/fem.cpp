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

#include <boost/algorithm/string.hpp>
#include <string>

#include <plask/common/fem.hpp>

#include <plask/python_enum.hpp>
#include <plask/python_globals.hpp>

namespace plask { namespace python {

static std::string IterativeMatrixParams__str__(const IterativeMatrixParams& self) {
    std::string accelerator = str(py::object(self.accelerator));
    std::string preconditioner = str(py::object(self.preconditioner));
    std::string noconv = str(py::object(self.no_convergence_behavior));

    return format(
        "Status:\n"
        "  converged: {}\n"
        "  iters: {}\n"
        "  err: {}\n"
        "Params:\n"
        "  maxit: {}\n"
        "  maxerr: {}\n"
        "  noconv: {}\n"
        "  accelerator: {}\n"
        "  preconditioner: {}\n"
        "  nfact: {}\n"
        "  ndeg: {}\n"
        "  lvfill: {}\n"
        "  ltrunc: {}\n"
        "  omega: {}\n"
        "  nsave: {}\n"
        "  nrestart: {}",
        self.converged? "True" : "False",
        self.iters,
        self.err,
        self.maxit,
        self.maxerr,
        noconv,
        accelerator,
        preconditioner,
        self.nfact,
        self.ndeg,
        self.lvfill,
        self.ltrunc,
        self.omega,
        self.ns1,
        self.ns2);
}

static std::string IterativeMatrixParams__repr__(const IterativeMatrixParams& self) {
    std::string accelerator = str(py::object(self.accelerator));
    std::string preconditioner = str(py::object(self.preconditioner));
    std::string noconv = str(py::object(self.no_convergence_behavior));
    boost::algorithm::to_lower(accelerator);
    boost::algorithm::to_lower(preconditioner);
    boost::algorithm::to_lower(noconv);

    return format(
        "IterativeParams("
        "maxit={}, "
        "maxerr={}, "
        "noconv='{}', "
        "accelerator='{}', "
        "preconditioner='{}', "
        "nfact={}, "
        "ndeg={}, "
        "lvfill={}, "
        "ltrunc={}, "
        "omega={}, "
        "nsave={}, "
        "nrestart={}, "
        "converged={}, "
        "iters={}, "
        "err={}"
        ")",
        self.maxit,
        self.maxerr,
        noconv,
        accelerator,
        preconditioner,
        self.nfact,
        self.ndeg,
        self.lvfill,
        self.ltrunc,
        self.omega,
        self.ns1,
        self.ns2,
        self.converged? "True" : "False",
        self.iters,
        self.err);
}

void registerFemCommon() {
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

    py::class_<IterativeMatrixParams, boost::noncopyable>(
        "IterativeParams",
        "Iterative matrix parameters\n\n"
        "This class holds parameters for iterative matrix used by solvers implementing\n"
        "Finite Element Method. PLaSK uses `NSPCG`_ package for performing iterations.\n"
        "Please refer to its documentation for explanation of most of the settings.\n\n"
        ".. _NSPCG: https://web.ma.utexas.edu/CNA/NSPCG/",
        py::no_init)
        .def("__str__", &IterativeMatrixParams__str__)
        .def("__repr__", &IterativeMatrixParams__repr__)
        .def_readwrite("preconditioner", &IterativeMatrixParams::preconditioner,
                       "Solver preconditioner\n\n"
                       "This is current preconditioner used for iterative matrix solver.\n\n"
                       ".. list-table:: Possible choices:\n\n"
                       "   * - ``rich``\n"
                       "     - Richardson's method\n"
                       "   * - ``jac``\n"
                       "     - Jacobi method\n"
                       "   * - ``ljac``\n"
                       "     - Line Jacobi method\n"
                       "   * - ``ljacx``\n"
                       "     - Line Jacobi method (approx. inverse)\n"
                       "   * - ``sor``\n"
                       "     - Successive Overrelaxation\n"
                       "       (can be used only with SOR accelerator)\n"
                       "   * - ``ssor``\n"
                       "     - Symmetric SOR\n"
                       "   * - ``ic``\n"
                       "     - Incomplete Cholesky (default)\n"
                       "   * - ``mic``\n"
                       "     - Modified Incomplete Cholesky\n"
                       "   * - ``lsp``\n"
                       "     - Least Squares Polynomial\n"
                       "   * - ``neu``\n"
                       "     - Neumann Polynomial\n"
                       "   * - ``lsor``\n"
                       "     - Line SOR\n"
                       "   * - ``lssor``\n"
                       "     - Line SSOR\n"
                       "   * - ``llsp``\n"
                       "     - Line Least Squares Polynomial\n"
                       "   * - ``lneu``\n"
                       "     - Line Neumann Polynomial\n"
                       "   * - ``bic``\n"
                       "     - Block Incomplete Cholesky (ver. 1)\n"
                       "   * - ``bicx``\n"
                       "     - Block Incomplete Cholesky (ver. 2)\n"
                       "   * - ``mbic``\n"
                       "     - Modified Block Incomplete Cholesky (ver. 1)\n"
                       "   * - ``mbicx``\n"
                       "     - Modified Block Incomplete Cholesky (ver. 2)\n")
        .def_readwrite("accelerator", &IterativeMatrixParams::accelerator,
                       "Solver accelerator\n\n"
                       "This is current iterative matrix solver acceleration algorithm.\n\n"
                       ".. list-table:: Possible choices:\n\n"
                       "   * - ``cg``\n"
                       "     - Conjugate Gradient acceleration (default)\n"
                       "   * - ``si``\n"
                       "     - Chebyshev acceleration or Semi-Iteration\n"
                       "   * - ``sor``\n"
                       "     - Successive Overrelaxation (can use only SOR preconditioner)\n"
                       "   * - ``srcg``\n"
                       "     - Symmetric Successive Overrelaxation Conjugate Gradient Algorithm\n"
                       "       (can use only SSOR preconditioner)\n"
                       "   * - ``srsi``\n"
                       "     - Symmetric Successive Overrelaxation Semi-Iteration Algorithm\n"
                       "       (can use only SSOR preconditioner)\n"
                       "   * - ``basic``\n"
                       "     - Basic Iterative Method\n"
                       "   * - ``me``\n"
                       "     - Minimal Error Algorithm\n"
                       "   * - ``cgnr``\n"
                       "     - Conjugate Gradient applied to the Normal Equations\n"
                       "   * - ``lsqr``\n"
                       "     - Least Squares Algorithm\n"
                       "   * - ``odir``\n"
                       "     - ORTHODIR, a truncated/restarted method useful for nonsymmetric systems of equations\n"
                       "   * - ``omin``\n"
                       "     - ORTHOMIN, a common truncated/restarted method used for nonsymmetric systems\n"
                       "   * - ``ores``\n"
                       "     - ORTHORES, another truncated/restarted method for nonsymmetric systems\n"
                       "   * - ``iom``\n"
                       "     - Incomplete Orthogonalization Method\n"
                       "   * - ``gmres``\n"
                       "     - Generalized Minimal Residual Method\n"
                       "   * - ``usymlq``\n"
                       "     - Unsymmetric LQ\n"
                       "   * - ``usymqr``\n"
                       "     - Unsymmetric QR\n"
                       "   * - ``landir``\n"
                       "     - Lanczos/ORTHODIR\n"
                       "   * - ``lanmin``\n"
                       "     - Lanczos/ORTHOMIN or Biconjugate Gradient Method\n"
                       "   * - ``lanres``\n"
                       "     - Lanczos/ORTHORES or “two-sided” Lanczos Method\n"
                       "   * - ``cgcr``\n"
                       "     - Constrained Generalized Conjugate Residual Method\n"
                       "   * - ``bcgs``\n"
                       "     - Biconjugate Gradient Squared Method\n")
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
        .def_readwrite("ndeg", &IterativeMatrixParams::ndeg,
                       "Polynomial degree\n\n"
                       "Degree of the polynomial to be used for the polynomial preconditioners.\n")
        .def_readwrite("lvfill", &IterativeMatrixParams::lvfill,
                       "Fill-in level\n\n"
                       "Level of fill-in for incomplete Cholesky preconditioners. Increasing this value\n"
                       "will result in more accurate factorizations at the expense of increased memory\n"
                       "usage and factorization time.\n")
        .def_readwrite("ltrunc", &IterativeMatrixParams::ltrunc,
                       "Truncation level\n\n"
                       "Truncation bandwidth to be used when approximating the inverses of matrices\n"
                       "with dense banded matrices. An increase in this value means a more accurate\n"
                       "factorization at the expense of increased storage.\n")
        .def_readwrite("omega", &IterativeMatrixParams::omega, "Relaxation parameter")
        .def_readwrite("nsave", &IterativeMatrixParams::ns1,
                       "Saved vectors number\n\n"
                       "The number of old vectors to be saved for the truncated acceleration methods.\n")
        .def_readwrite("nrestart", &IterativeMatrixParams::ns2,
                       "Restart frequency\n\n"
                       "The number of iterations between restarts for restarted acceleration methods.\n")

        .def_readonly("converged", &IterativeMatrixParams::converged, "True if the solver converged")
        .def_readonly("iters", &IterativeMatrixParams::iters, "Number of iterations in the last run")
        .def_readonly("err", &IterativeMatrixParams::err, "Residual error in the last run");
}

}}  // namespace plask::python
