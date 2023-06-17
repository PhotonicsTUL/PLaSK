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
#ifndef PLASK_COMMON_FEM_ITERATIVE_MATRIX_H
#define PLASK_COMMON_FEM_ITERATIVE_MATRIX_H

#include <algorithm>

#include <nspcg/nspcg.hpp>

#include "matrix.hpp"

namespace plask {

struct SparseBandMatrix : FemMatrix {
    int* bno;  ///< Vector of non-zero band numbers (shift from diagonal)

    int iterlim;     ///< Maximum number of iterations
    double itererr;  ///< Maximum error of iteration

    /// NSPCG acceleration method
    enum Accelelator {
        ACCEL_CG,
        ACCEL_SI,
        ACCEL_SOR,
        ACCEL_SRCG,
        ACCEL_SRSI,
        ACCEL_BASIC,
        ACCEL_ME,
        ACCEL_CGNR,
        ACCEL_LSQR,
        ACCEL_ODIR,
        ACCEL_OMIN,
        ACCEL_ORES,
        ACCEL_IOM,
        ACCEL_GMRES,
        ACCEL_USYMLQ,
        ACCEL_USYMQR,
        ACCEL_LANDIR,
        ACCEL_LANMIN,
        ACCEL_LANRES,
        ACCEL_CGCR,
        ACCEL_BCGS
    };
    // Accelelator accelerator;  ///< NSPCG acceleration method

    /// NSPCG preconditioner
    enum Preconditioner {
        PRECOND_RICH,
        PRECOND_JAC,
        PRECOND_LJAC,
        PRECOND_LJACX,
        PRECOND_SOR,
        PRECOND_SSOR,
        PRECOND_IC,
        PRECOND_MIC,
        PRECOND_LSP,
        PRECOND_NEU,
        PRECOND_LSOR,
        PRECOND_LSSOR,
        PRECOND_LLSP,
        PRECOND_LNEU,
        PRECOND_BIC,
        PRECOND_BICX,
        PRECOND_MBIC,
        PRECOND_MBICX
    };
    // Preconditioner preconditioner;  ///< NSPCG preconditioner

  protected:
    int nw, inw;
    aligned_unique_ptr<double> wksp;
    aligned_unique_ptr<int> iwksp;

  public:
    /**
     * Create 2D matrix.
     * \param size size of the matrix
     * \param major shift of nodes to the next row (mesh[x,y+1])
     */
    template <typename SolverT>
    SparseBandMatrix(const SolverT* solver, size_t size, size_t major)
        : FemMatrix(solver, size, 4, 4), bno(aligned_malloc<int>(5)), iterlim(solver->iterlim), itererr(solver->itererr) {
        bno[0] = 0;
        bno[1] = 1;
        bno[2] = major - 1;
        bno[3] = major;
        bno[4] = major + 1;
    }

    /**
     * Create 3D matrix.
     * \param size size of the matrix
     * \param major shift of nodes to the next major row (mesh[x,y,z+1])
     * \param minor shift of nodes to the next minor row (mesh[x,y+1,z])
     */
    template <typename SolverT>
    SparseBandMatrix(const SolverT* solver, size_t size, size_t major, size_t minor)
        : FemMatrix(solver, size, 14, 14), bno(aligned_malloc<int>(14)), iterlim(solver->iterlim), itererr(solver->itererr) {
        bno[0] = 0;
        bno[1] = 1;
        bno[2] = minor - 1;
        bno[3] = minor;
        bno[4] = minor + 1;
        bno[5] = major - minor - 1;
        bno[6] = major - minor;
        bno[7] = major - minor + 1;
        bno[8] = major - 1;
        bno[9] = major;
        bno[10] = major + 1;
        bno[11] = major + minor - 1;
        bno[12] = major + minor;
        bno[13] = major + minor + 1;
    }

    ~SparseBandMatrix() { aligned_free<int>(bno); }

    /**
     * Return reference to array element.
     * \param r index of the element row
     * \param c index of the element column
     * \return reference to array element
     **/
    size_t index(size_t r, size_t c) override {
        if (r == c) return r;
        if (r < c) std::swap(r, c);
        size_t i = std::find(bno, bno + ld + 1, r - c) - bno;
        assert(i != ld + 1);
        return c + size * i;
    }

    int solverhs(DataVector<double>& B, DataVector<double>& X) override {
        iparm_t iparm;
        rparm_t rparm;
        nspcg_dfault(iparm, rparm);

        iparm.itmax = iterlim;
        rparm.zeta = itererr;

        solver->writelog(LOG_DETAIL, "Iterating linear system");


#ifdef NDEBUG
        iparm.level = -1;
#else
        iparm.level = 3;
#endif

        int n = size, maxnz = ld + 1;

        int NW = 3 * size + 2 * iparm.itmax + size * (ld + 1);
        int INW = ld + 1 + std::max(2 * size, (ld + 1) + (ld + 1) * (ld + 1));

        if (nw < NW) {
            nw = NW;
            wksp.reset(aligned_malloc<double>(nw));
        }

        if (inw < INW) {
            inw = INW;
            iwksp.reset(aligned_malloc<int>(inw));
        }

        // for (size_t r = 0; r < size; ++r) {
        //     for (size_t c = 0; c < size; ++c) {
        //         if (std::find(bno, A.bno+(ld+1), std::abs(int(r)-int(c))) == bno+(ld+1) )
        //             std::cout << "    .    ";
        //         else
        //             std::cout << str((*this))(r, c), "{:8.3f}") << " ";
        //     }
        //     std::cout << "         " << str(B[r], "{:8.3f}") << std::endl;
        // }

        assert(B.size() == size);

        DataVector<double> U;
        if (X.data() == nullptr || X.data() == B.data())
            U.reset(B.size(), 1.);
        else
            U = X;
        assert(U.size() == B.size());

        int ier;

        // TODO add choice of algorithms and predonditioners

        while (true) {
            nspcg(nspcg_mic2, nspcg_cg, n, ld + 1, n, maxnz, data, bno, nullptr, nullptr, U.data(), nullptr, B.data(),
                  wksp.get(), iwksp.get(), nw, inw, iparm, rparm, ier);

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
                case -1: throw ComputationError(solver->getId(), "Nonpositive matrix size {}", size);
                case -2: throw ComputationError(solver->getId(), "Insufficient real workspace ({} required)", nw);
                case -3: throw ComputationError(solver->getId(), "Insufficient integer workspace ({} required)", inw);
                case -4: throw ComputationError(solver->getId(), "Nonpositive diagonal element in stiffness matrix");
                case -5: throw ComputationError(solver->getId(), "Nonexistent diagonal element in stiffness matrix");
                case -6: throw ComputationError(solver->getId(), "Stiffness matrix A is not positive definite");
                case -7: throw ComputationError(solver->getId(), "Preconditioned matrix Q is not positive definite");
                case -8: throw ComputationError(solver->getId(), "Cannot permute stiffness matrix as requested");
                case -9:
                    throw ComputationError(solver->getId(),
                                           "Number of non-zero diagonals is not large enough to allow expansion of matrix");
                case -10: throw ComputationError(solver->getId(), "Inadmissible parameter encountered");
                case -11: throw ComputationError(solver->getId(), "Incorrect storage mode for block method");
                case -12: throw ComputationError(solver->getId(), "Zero pivot encountered in factorization");
                case -13: throw ComputationError(solver->getId(), "Breakdown in direction vector calculation");
                case -14: throw ComputationError(solver->getId(), "Breakdown in attempt to perform rotation");
                case -15: throw ComputationError(solver->getId(), "Breakdown in iterate calculation");
                case -16: throw ComputationError(solver->getId(), "Unimplemented combination of parameters");
                case -18: throw ComputationError(solver->getId(), "Unable to perform eigenvalue estimation");
                case 1: writelog(LOG_WARNING, solver->getId(), "Failed to converge in {} iterations", iparm.itmax); break;
                case 2:
                    solver->writelog(LOG_WARNING, "`itererr` too small - reset to {}",
                                     500 * std::numeric_limits<double>::epsilon());
                    break;
                case 3:
                    solver->writelog(LOG_DEBUG,
                                     "NSPGS: `zbrent` failed to converge in the maximum number of {} iterations (signifies "
                                     "difficulty in eigenvalue estimation)",
                                     std::max(iterlim, 50));
                    break;
                case 4:
                    solver->writelog(
                        LOG_DEBUG,
                        "NSPGS: In `zbrent`, f (a) and f (b) have the same sign (signifies difficulty in eigenvalue estimation)");
                    break;
                case 5: solver->writelog(LOG_DEBUG, "NSPGS: Negative pivot encountered in factorization"); break;
            }
        }

        if (ier != 1) solver->writelog(LOG_DETAIL, "Converged after {} iterations", iparm.itmax);

        if (X.data() != U.data()) X = U;

        return iparm.itmax;
    }

    void mult(const DataVector<const double>& vector, DataVector<double>& result) {
        std::fill(result.begin(), result.end(), 0.);
        SparseBandMatrix::addmult(vector, result);
    }

    void addmult(const DataVector<const double>& vector, DataVector<double>& result) {
        for (size_t d = 0; d <= ld; ++d) {
            for (size_t r = 0; r < size; ++r) {
                size_t c = r + bno[d];
                if (c >= size) break;
                result[r] += data[r + size * d] * vector[c];
            }
        }
    }

    void setBC(DataVector<double>& B, size_t r, double val) override {
        data[r] = 1.;
        B[r] = val;
        // above diagonal
        for (ptrdiff_t i = kd; i > 0; --i) {
            ptrdiff_t c = r - bno[i];
            if (c >= 0) {
                ptrdiff_t ii = c + size * i;
                B[c] -= data[ii] * val;
                data[ii] = 0.;
            }
        }
        // below diagonal
        for (ptrdiff_t i = 1; i <= ld; ++i) {
            ptrdiff_t c = r + bno[i];
            if (c < size) {
                size_t ii = r + size * i;
                B[c] -= data[ii] * val;
                data[ii] = 0.;
            }
        }
    }
};

}  // namespace plask

#endif  // PLASK_COMMON_FEM_ITERATIVE_MATRIX_H
