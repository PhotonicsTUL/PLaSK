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

struct IterativeMatrixParams {
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
    Accelelator accelerator = ACCEL_CG;  ///< NSPCG acceleration method

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
    Preconditioner preconditioner = PRECOND_IC;  ///< NSPCG preconditioner

    int maxit = 1000;      ///< Maximum number of iterations
    double maxerr = 1e-6;  ///< Maximum allowed residual error of iteration
    int nfact = 10;        ///< Frequency of partial factorization

    int ns1 = 5;         ///< Number of old vectors to be saved for truncated acceleration methods
    int ns2 = 100000;    ///< Frequency of restarting restarted accelerations methods
    int lvfill = 0;      ///< Level of fill-in for incomplete Cholesky factorization
    int ltrunc = 0;      ///< Level of truncation for Cholesky factorization
    int ndeg = 1;        ///< Degree of the polynomial preconditioner
    double omega = 1.0;  ///< Relaxation parameter

    enum NoConvergenceBehavior { NO_CONVERGENCE_ERROR, NO_CONVERGENCE_WARNING, NO_CONVERGENCE_CONTINUE };
    NoConvergenceBehavior no_convergence_behavior = NO_CONVERGENCE_WARNING;  ///< What to do if the solution does not
                                                                             ///< converge

    // Output parameters
    bool converged = true;  ///< True if the solution converged
    int iters = 0;          ///< Number of iterations
    double err = 0;         ///< Residual error of the solution
};

struct SparseBandMatrix : FemMatrix {
    int* bno;  ///< Vector of non-zero band numbers (shift from diagonal)

  protected:
    IterativeMatrixParams* params;

    int ifact = 1;
    int nw = 0, inw = 0;
    double* wksp = nullptr;
    int* iwksp = nullptr;
    int kblsz = -1, nbl2d = -1;

  public:
    /**
     * Create 2D matrix.
     * \param size size of the matrix
     * \param major shift of nodes to the next row (mesh[x,y+1])
     */
    template <typename SolverT>
    SparseBandMatrix(SolverT* solver, size_t size, size_t major)
        : FemMatrix(solver, size, 4, 4), bno(aligned_malloc<int>(5)), params(&solver->iter_params) {
        bno[0] = 0;
        bno[1] = 1;
        bno[2] = major - 1;
        bno[3] = major;
        bno[4] = major + 1;
        kblsz = major - 1;
        nbl2d = major - 1;
    }

    /**
     * Create 3D matrix.
     * \param size size of the matrix
     * \param major shift of nodes to the next major row (mesh[x,y,z+1])
     * \param minor shift of nodes to the next minor row (mesh[x,y+1,z])
     */
    template <typename SolverT>
    SparseBandMatrix(SolverT* solver, size_t size, size_t major, size_t minor)
        : FemMatrix(solver, size, 13, 13), bno(aligned_malloc<int>(14)), params(&solver->iter_params) {
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
        kblsz = minor - 1;
        nbl2d = major - minor - 1;
    }

    ~SparseBandMatrix() {
        aligned_free<int>(bno);
        aligned_free<double>(wksp);
        aligned_free<int>(iwksp);
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

    void solverhs(DataVector<double>& B, DataVector<double>& X) override {
        iparm_t iparm;
        rparm_t rparm;
        nspcg_dfault(iparm, rparm);

        iparm.itmax = params->maxit;
        iparm.ipropa = 0;
        iparm.ifact = (--ifact) ? 0 : 1;
        if (ifact <= 0) ifact = params->nfact;
        rparm.zeta = params->maxerr;

        iparm.ns1 = params->ns1;
        iparm.ns2 = params->ns2;
        iparm.lvfill = params->lvfill;
        iparm.ltrunc = params->ltrunc;
        iparm.ndeg = params->ndeg;
        rparm.omega = params->omega;

        iparm.ns3 =
            (params->accelerator == IterativeMatrixParams::ACCEL_LANMIN || params->accelerator == IterativeMatrixParams::ACCEL_CGCR)
                ? 40
                : 0;
        iparm.kblsz = kblsz;
        iparm.nbl2d = nbl2d;

        solver->writelog(LOG_DETAIL, "Iterating linear system");

#ifdef NDEBUG
        iparm.level = -1;
#else
        iparm.level = 3;
#endif

        int n = size, maxnz = ld + 1;

        int NW = 3 * size + 2 * iparm.itmax + size * maxnz + iparm.kblsz;
        int INW = maxnz + std::max(2 * n, maxnz * maxnz + maxnz);

        if (nw < NW) {
            nw = NW;
            aligned_free<double>(wksp);
            wksp = aligned_malloc<double>(nw);
            iparm.ifact = 1;  // we need to do factorization with new workspace
        }

        if (inw < INW) {
            inw = INW;
            aligned_free<int>(iwksp);
            iwksp = aligned_malloc<int>(inw);
            iparm.ifact = 1;  // we need to do factorization with new workspace
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

        if ((params->accelerator == IterativeMatrixParams::ACCEL_SOR) !=
            (params->preconditioner == IterativeMatrixParams::PRECOND_SOR ||
             params->preconditioner == IterativeMatrixParams::PRECOND_LSOR)) {
            throw BadInput(solver->getId(), "SOR oraccelerator must be used with SOR or LSOR preconditioner");
        }
        if (params->accelerator == IterativeMatrixParams::ACCEL_SRCG &&
            params->preconditioner != IterativeMatrixParams::PRECOND_SSOR &&
            params->preconditioner != IterativeMatrixParams::PRECOND_LSSOR) {
            throw BadInput(solver->getId(), "SRCG accelerator must be used with SSOR or LSSOR preconditioner");
        }
        if (params->accelerator == IterativeMatrixParams::ACCEL_SRSI &&
            params->preconditioner != IterativeMatrixParams::PRECOND_SSOR &&
            params->preconditioner != IterativeMatrixParams::PRECOND_LSSOR) {
            throw BadInput(solver->getId(), "SRSI accelerator must be used with SSOR or LSSOR preconditioner");
        }
        // clang-format off
        switch (params->preconditioner) {
            case IterativeMatrixParams::PRECOND_RICH: precond_func = nspcg_rich2; break;
            case IterativeMatrixParams::PRECOND_JAC: precond_func = nspcg_jac2; break;
            case IterativeMatrixParams::PRECOND_LJAC: precond_func = nspcg_ljac2; break;
            case IterativeMatrixParams::PRECOND_LJACX: precond_func = nspcg_ljacx2; break;
            case IterativeMatrixParams::PRECOND_SOR: precond_func = nspcg_sor2; break;
            case IterativeMatrixParams::PRECOND_SSOR: precond_func = nspcg_ssor2; break;
            case IterativeMatrixParams::PRECOND_IC: precond_func = nspcg_ic2; break;
            case IterativeMatrixParams::PRECOND_MIC: precond_func = nspcg_mic2; break;
            case IterativeMatrixParams::PRECOND_LSP: precond_func = nspcg_lsp2; break;
            case IterativeMatrixParams::PRECOND_NEU: precond_func = nspcg_neu2; break;
            case IterativeMatrixParams::PRECOND_LSOR: precond_func = nspcg_lsor2; break;
            case IterativeMatrixParams::PRECOND_LSSOR: precond_func = nspcg_lssor2; break;
            case IterativeMatrixParams::PRECOND_LLSP: precond_func = nspcg_llsp2; break;
            case IterativeMatrixParams::PRECOND_LNEU: precond_func = nspcg_lneu2; break;
            case IterativeMatrixParams::PRECOND_BIC: precond_func = nspcg_bic2; break;
            case IterativeMatrixParams::PRECOND_BICX: precond_func = nspcg_bicx2; break;
            case IterativeMatrixParams::PRECOND_MBIC: precond_func = nspcg_mbic2; break;
            case IterativeMatrixParams::PRECOND_MBICX: precond_func = nspcg_mbicx2; break;
        };
        switch (params->accelerator) {
            case IterativeMatrixParams::ACCEL_CG: accel_func = nspcg_cg; break;
            case IterativeMatrixParams::ACCEL_SI: accel_func = nspcg_si; break;
            case IterativeMatrixParams::ACCEL_SOR: accel_func = nspcg_sor; break;
            case IterativeMatrixParams::ACCEL_SRCG: accel_func = nspcg_srcg; break;
            case IterativeMatrixParams::ACCEL_SRSI: accel_func = nspcg_srsi; break;
            case IterativeMatrixParams::ACCEL_BASIC: accel_func = nspcg_basic; break;
            case IterativeMatrixParams::ACCEL_ME: accel_func = nspcg_me; break;
            case IterativeMatrixParams::ACCEL_CGNR: accel_func = nspcg_cgnr; break;
            case IterativeMatrixParams::ACCEL_LSQR: accel_func = nspcg_lsqr; break;
            case IterativeMatrixParams::ACCEL_ODIR: accel_func = nspcg_odir; break;
            case IterativeMatrixParams::ACCEL_OMIN: accel_func = nspcg_omin; break;
            case IterativeMatrixParams::ACCEL_ORES: accel_func = nspcg_ores; break;
            case IterativeMatrixParams::ACCEL_IOM: accel_func = nspcg_iom; break;
            case IterativeMatrixParams::ACCEL_GMRES: accel_func = nspcg_gmres; break;
            case IterativeMatrixParams::ACCEL_USYMLQ: accel_func = nspcg_usymlq; break;
            case IterativeMatrixParams::ACCEL_USYMQR: accel_func = nspcg_usymqr; break;
            case IterativeMatrixParams::ACCEL_LANMIN: accel_func = nspcg_lanmin; break;
            case IterativeMatrixParams::ACCEL_LANRES: accel_func = nspcg_lanres; break;
            case IterativeMatrixParams::ACCEL_CGCR: accel_func = nspcg_cgcr; break;
            case IterativeMatrixParams::ACCEL_BCGS: accel_func = nspcg_bcgs; break;
        };
        // clang-format on

        while (true) {
            nspcg(precond_func, accel_func, n, ld + 1, n, maxnz, data, bno, nullptr, nullptr, U.data(), nullptr, B.data(), wksp,
                  iwksp, nw, inw, iparm, rparm, ier);

            // Increase workspace if needed
            if (ier == -2 && nw) {
                aligned_free<double>(wksp);
                wksp = aligned_malloc<double>(nw);
                iparm.ifact = 1;  // we need to do factorization with new workspace
            } else if (ier == -3 && inw) {
                aligned_free<int>(iwksp);
                iwksp = aligned_malloc<int>(inw);
                iparm.ifact = 1;  // we need to do factorization with new workspace
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
                    params->converged = false;
                    switch (params->no_convergence_behavior) {
                        case IterativeMatrixParams::NO_CONVERGENCE_ERROR:
                            throw ComputationError(solver->getId(), "Failed to converge in {} iterations (error {})", iparm.itmax,
                                                   rparm.zeta);
                        case IterativeMatrixParams::NO_CONVERGENCE_WARNING:
                            solver->writelog(LOG_WARNING, "Failed to converge in {} iterations (error {})", iparm.itmax,
                                             rparm.zeta);
                            break;
                        case IterativeMatrixParams::NO_CONVERGENCE_CONTINUE:
                            solver->writelog(LOG_DETAIL, "Did not converge yen in {} iterations (error {})", iparm.itmax,
                                             rparm.zeta);
                            break;
                    }
                    break;
                case 2:
                    solver->writelog(LOG_WARNING, "`maxerr` was too small, reset to {}", 3.55e-12);
                    break;
                case 3:
                    solver->writelog(LOG_DEBUG,
                                     "NSPGS: `zbrent` failed to converge in the maximum number of {} iterations (signifies "
                                     "difficulty in eigenvalue estimation)",
                                     std::max(params->maxit, 50));
                    break;
                case 4:
                    solver->writelog(LOG_DEBUG,
                                     "NSPGS: In `zbrent`, f (a) and f (b) have the same sign (signifies difficulty in "
                                     "eigenvalue estimation)");
                    break;
                case 5: solver->writelog(LOG_DEBUG, "NSPGS: Negative pivot encountered in factorization"); break;
            }
        }
        if (ier != 1) {
            solver->writelog(LOG_DETAIL, "Converged after {} iterations (error {})", iparm.itmax, rparm.zeta);
            params->converged = true;
        }

        params->iters = iparm.itmax;
        params->err = rparm.zeta;

        if (X.data() != U.data()) X = U;
    }

    void mult(const DataVector<const double>& vector, DataVector<double>& result) override {
        std::fill(result.begin(), result.end(), 0.);
        SparseBandMatrix::addmult(vector, result);
    }

    void addmult(const DataVector<const double>& vector, DataVector<double>& result) override {
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
