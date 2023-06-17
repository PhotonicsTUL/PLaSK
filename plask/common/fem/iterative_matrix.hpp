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
    Accelelator accelerator;  ///< NSPCG acceleration method

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
    Preconditioner preconditioner;  ///< NSPCG preconditioner

  protected:
    int nw = 0, inw = 0;
    double* wksp = nullptr;
    int* iwksp = nullptr;
    int* piv;
    int* ipiv;

  public:
    /**
     * Create 2D matrix.
     * \param size size of the matrix
     * \param major shift of nodes to the next row (mesh[x,y+1])
     */
    template <typename SolverT>
    SparseBandMatrix(const SolverT* solver, size_t size, size_t major)
        : FemMatrix(solver, size, 4, 4),
          bno(aligned_malloc<int>(5)),
          iterlim(solver->iterlim),
          itererr(solver->itererr),
          accelerator(solver->iter_accelerator),
          preconditioner(solver->iter_preconditioner) {
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
        : FemMatrix(solver, size, 13, 13),
          bno(aligned_malloc<int>(14)),
          iterlim(solver->iterlim),
          itererr(solver->itererr),
          accelerator(solver->iter_accelerator),
          preconditioner(solver->iter_preconditioner),
          piv(aligned_malloc<int>(size)),
          ipiv(aligned_malloc<int>(size)) {
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

    ~SparseBandMatrix() {
        aligned_free<int>(bno);
        aligned_free<double>(wksp);
        aligned_free<int>(iwksp);
        aligned_free<int>(piv);
        aligned_free<int>(ipiv);
    }

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

        int NW = 3 * size + 2 * iparm.itmax + size * maxnz;
        int INW = maxnz + std::max(2 * n, maxnz * maxnz + maxnz);

        if (nw < NW) {
            nw = NW;
            aligned_free<double>(wksp);
            wksp = aligned_malloc<double>(nw);
        }

        if (inw < INW) {
            inw = INW;
            aligned_free<int>(iwksp);
            iwksp = aligned_malloc<int>(inw);
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

        void (*precond_func)(...), (*accel_func)(...);

        // clang-format off
        switch (preconditioner) {
            case PRECOND_RICH: precond_func = nspcg_rich2; break;
            case PRECOND_JAC: precond_func = nspcg_jac2; break;
            case PRECOND_LJAC: precond_func = nspcg_ljac2; break;
            case PRECOND_LJACX: precond_func = nspcg_ljacx2; break;
            case PRECOND_SOR: precond_func = nspcg_sor2; break;
            case PRECOND_SSOR: precond_func = nspcg_ssor2; break;
            case PRECOND_IC: precond_func = nspcg_ic2; break;
            case PRECOND_MIC: precond_func = nspcg_mic2; break;
            case PRECOND_LSP: precond_func = nspcg_lsp2; break;
            case PRECOND_NEU: precond_func = nspcg_neu2; break;
            case PRECOND_LSOR: precond_func = nspcg_lsor2; break;
            case PRECOND_LSSOR: precond_func = nspcg_lssor2; break;
            case PRECOND_LLSP: precond_func = nspcg_llsp2; break;
            case PRECOND_LNEU: precond_func = nspcg_lneu2; break;
            case PRECOND_BIC: precond_func = nspcg_bic2; break;
            case PRECOND_BICX: precond_func = nspcg_bicx2; break;
            case PRECOND_MBIC: precond_func = nspcg_mbic2; break;
            case PRECOND_MBICX: precond_func = nspcg_mbicx2; break;
        };
        switch (accelerator) {
            case ACCEL_CG: accel_func = nspcg_cg; break;
            case ACCEL_SI: accel_func = nspcg_si; break;
            case ACCEL_SOR: accel_func = nspcg_sor; break;
            case ACCEL_SRCG: accel_func = nspcg_srcg; break;
            case ACCEL_SRSI: accel_func = nspcg_srsi; break;
            case ACCEL_BASIC: accel_func = nspcg_basic; break;
            case ACCEL_ME: accel_func = nspcg_me; break;
            case ACCEL_CGNR: accel_func = nspcg_cgnr; break;
            case ACCEL_LSQR: accel_func = nspcg_lsqr; break;
            case ACCEL_ODIR: accel_func = nspcg_odir; break;
            case ACCEL_OMIN: accel_func = nspcg_omin; break;
            case ACCEL_ORES: accel_func = nspcg_ores; break;
            case ACCEL_IOM: accel_func = nspcg_iom; break;
            case ACCEL_GMRES: accel_func = nspcg_gmres; break;
            case ACCEL_USYMLQ: accel_func = nspcg_usymlq; break;
            case ACCEL_USYMQR: accel_func = nspcg_usymqr; break;
            case ACCEL_LANMIN: accel_func = nspcg_lanmin; break;
            case ACCEL_LANRES: accel_func = nspcg_lanres; break;
            case ACCEL_CGCR: accel_func = nspcg_cgcr; break;
            case ACCEL_BCGS: accel_func = nspcg_bcgs; break;
        };
        // clang-format on

        while (true) {
            nspcg(precond_func, accel_func, n, maxnz, n, maxnz, data, bno, piv, ipiv, U.data(), nullptr, B.data(),
                  wksp, iwksp, nw, inw, iparm, rparm, ier);

            // Increase workspace if needed
            if (ier == -2 && nw) {
                aligned_free<double>(wksp);
                wksp = aligned_malloc<double>(nw);
            } else if (ier == -3 && inw) {
                aligned_free<int>(iwksp);
                iwksp = aligned_malloc<int>(inw);
            } else
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
                case 1:
                    solver->writelog(LOG_WARNING, "Failed to converge in {} iterations (err. is {})", iparm.itmax, rparm.zeta);
                    break;
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
        for (size_t r = 0; r < size; ++r) {
            result[r] += data[r] * vector[r];
        }
        for (size_t d = 1; d <= ld; ++d) {
            size_t sd = size * d;
            for (size_t r = 0; r < size; ++r) {
                size_t c = r + bno[d];
                if (c >= size) break;
                result[r] += data[r + sd] * vector[c];
                result[c] += data[r + sd] * vector[r];
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
                assert(ii < size * (ld + 1));
                B[c] -= data[ii] * val;
                data[ii] = 0.;
            }
        }
        // below diagonal
        for (ptrdiff_t i = 1; i <= ld; ++i) {
            ptrdiff_t c = r + bno[i];
            if (c < size) {
                size_t ii = r + size * i;
                assert(ii < size * (ld + 1));
                B[c] -= data[ii] * val;
                data[ii] = 0.;
            }
        }
    }
};

}  // namespace plask

#endif  // PLASK_COMMON_FEM_ITERATIVE_MATRIX_H
