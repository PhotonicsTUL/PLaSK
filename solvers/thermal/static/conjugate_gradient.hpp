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
#ifndef PLASK__MODULE_THERMAL_STATIC_CONJUGATE_GRADIENT_H
#define PLASK__MODULE_THERMAL_STATIC_CONJUGATE_GRADIENT_H

#include <nspcg/nspcg.hpp>
#include <plask/plask.hpp>

namespace plask { namespace thermal { namespace tstatic {

template <typename MatrixT> struct NspcgSolver {
    int nw, inw;
    aligned_unique_ptr<double> wksp;
    aligned_unique_ptr<int> iwksp;
    DataVector<double> U;

    NspcgSolver() : nw(0), inw(0) {}

    void reset() {
        nw = 0;
        inw = 0;
        wksp.reset();
        iwksp.reset();
        U.reset();
    }

    template <typename SolverT> void solve(const SolverT* solver, MatrixT& A, DataVector<double>& B) {
        iparm_t iparm;
        rparm_t rparm;
        dfault(iparm, rparm);

        iparm.itmax = solver->iterlim;
        rparm.zeta = solver->itererr;
        // iparm.ielim = 1;
        // iparm.iscale = 1;

#ifdef NDEBUG
        iparm.level = -1;
#else
        iparm.level = 3;
#endif

        int maxnz = A.nd;

        int NW = 3 * A.size + 2 * iparm.itmax + A.size * A.nd;
        int INW = A.nd + std::max(2 * A.size, int(A.nd + A.nd * A.nd));

        if (nw < NW) {
            nw = NW;
            wksp.reset(aligned_malloc<double>(nw));
        }

        if (inw < INW) {
            inw = INW;
            iwksp.reset(aligned_malloc<int>(inw));
        }

        // for (size_t r = 0; r < A.size; ++r) {
        //     for (size_t c = 0; c < A.size; ++c) {
        //         if (std::find(A.bno, A.bno+A.nd, std::abs(int(r)-int(c))) == A.bno+A.nd )
        //             std::cout << "    .    ";
        //         else
        //             std::cout << str(A(r, c), "{:8.3f}") << " ";
        //     }
        //     std::cout << "         " << str(B[r], "{:8.3f}") << std::endl;
        // }

        if (U.size() != B.size()) U.reset(B.size(), 300.);

        int ier;

        // TODO add choice of algorithms and predonditioners

        while (true) {
            nspcg(mic2, cg, A.size, A.nd, A.size, maxnz, A.data, A.bno, nullptr, nullptr, U.data(), nullptr, B.data(), wksp.get(),
                  iwksp.get(), nw, inw, iparm, rparm, ier);

            // Increase workspace if needed
            if (ier == -2 && nw)
                wksp.reset(aligned_malloc<double>(nw));
            else if (ier == -3 && inw)
                iwksp.reset(aligned_malloc<int>(inw));
            else
                break;
        }

        if (ier != 0) {
            switch (ier) {
                case -1: throw ComputationError(solver->getId(), "Nonpositive matrix size {}", A.size);
                case -2: throw ComputationError(solver->getId(), "Insufficient real workspace ({} required)", nw);
                case -3: throw ComputationError(solver->getId(), "Insufficient integer workspace ({} required)", inw);
                case -4: throw ComputationError(solver->getId(), "Nonpositive diagonal element in stiffness matrix");
                case -5: throw ComputationError(solver->getId(), "Nonexistent diagonal element in stiffness matrix");
                case -6: throw ComputationError(solver->getId(), "Stiffness matrix A is not positive definite");
                case -7: throw ComputationError(solver->getId(), "Preconditioned matrix Q is not positive definite");
                case -8: throw ComputationError(solver->getId(), "Cannot permute stiffness matrix as requested");
                case -9: throw ComputationError(solver->getId(), "Number of non-zero digonals is not large enough to allow expansion of matrix");
                case -10: throw ComputationError(solver->getId(), "Inadmissible parameter encountered");
                case -11: throw ComputationError(solver->getId(), "Incorrect storage mode for block method");
                case -12: throw ComputationError(solver->getId(), "Zero pivot encountered in factorization");
                case -13: throw ComputationError(solver->getId(), "Breakdown in direction vector calculation");
                case -14: throw ComputationError(solver->getId(), "Breakdown in attempt to perform rotation");
                case -15: throw ComputationError(solver->getId(), "Breakdown in iterate calculation");
                case -16: throw ComputationError(solver->getId(), "Unimplemented combination of parameters");
                case -18: throw ComputationError(solver->getId(), "Unable to perform eigenvalue estimation");
                case 1:
                    writelog(LOG_WARNING, solver->getId(), "Failure to converge in {} iterations", iparm.itmax);
                    break;
                case 2:
                    solver->writelog(LOG_WARNING, "`itererr` too small - reset to {}", 500 * std::numeric_limits<double>::epsilon());
                    break;
                case 3:
                    solver->writelog(LOG_DEBUG,
                        "ZBRENT failed to converge in MAXFN iterations (signifies difficulty in eigenvalue estimation)");
                    break;
                case 4:
                    solver->writelog(LOG_DEBUG,
                        "In ZBRENT, f (a) and f (b) have the same sign (signifies difficulty in eigenvalue estimation)");
                    break;
                case 5:
                    solver->writelog(LOG_DEBUG, "Negative pivot encountered in factorization");
                    break;
            }
        }

        if (ier != 1) solver->writelog(LOG_DETAIL, "Converged after {} iterations", iparm.itmax);

        std::swap(B, U);
    }
};

}}}  // namespace plask::thermal::tstatic

#endif  // PLASK__MODULE_THERMAL_STATIC_CONJUGATE_GRADIENT_H
