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

struct SparseMatrix : FemMatrix<> {
    typedef void (*NspcgFunc)(...);

  protected:
    int* icords;  ///< Vector of non-zero band numbers (shift from diagonal) or non-zero elements

    IterativeMatrixParams* params;

    const int nstore, ndim, mdim;

    int ifact = 1;
    int nw = 0, inw = 0;
    double* wksp = nullptr;
    int* iwksp = nullptr;
    int kblsz = -1, nbl2d = -1;

    virtual NspcgFunc get_preconditioner() = 0;

    virtual int get_maxnz() const { return mdim; }

  public:
    template <typename SolverT>
    SparseMatrix(SolverT* solver, size_t rank, size_t size, size_t isiz)
        : FemMatrix<>(solver, rank, size),
          icords(aligned_malloc<int>(isiz)),
          params(&solver->iter_params),
          nstore(2),
          ndim(rank),
          mdim(isiz) {}

    template <typename SolverT>
    SparseMatrix(SolverT* solver, size_t rank, size_t size)
        : FemMatrix<>(solver, rank, size),
          icords(aligned_malloc<int>(2 * size)),
          params(&solver->iter_params),
          nstore(4),
          ndim(size),
          mdim(size) {}

    ~SparseMatrix() {
        aligned_free<int>(icords);
        aligned_free<double>(wksp);
        aligned_free<int>(iwksp);
    }

    void solverhs(DataVector<double>& B, DataVector<double>& X) override {
        iparm_t iparm;
        rparm_t rparm;
        nspcg_dfault(iparm, rparm);

        iparm.nstore = nstore;
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

        int maxnz = get_maxnz();
        size_t default_nw = 3 * rank + 2 * params->maxit + rank * maxnz + std::max(kblsz, 1);
        size_t default_inw = maxnz + std::max(2 * int(rank), maxnz * maxnz + maxnz);

        if (nw < default_nw) {
            nw = default_nw;
            aligned_free<double>(wksp);
            wksp = aligned_malloc<double>(nw);
            iparm.ifact = 1;  // we need to do factorization with new workspace
        }

        if (inw < default_inw) {
            inw = default_inw;
            aligned_free<int>(iwksp);
            iwksp = aligned_malloc<int>(inw);
            iparm.ifact = 1;  // we need to do factorization with new workspace
        }

        // for (size_t r = 0; r < rank; ++r) {
        //     for (size_t c = 0; c < rank; ++c) {
        //         if (std::find(icords, A.icords+(ld+1), std::abs(int(r)-int(c))) == icords+(ld+1) )
        //             std::cout << "    .    ";
        //         else
        //             std::cout << str((*this))(r, c), "{:8.3f}") << " ";
        //     }
        //     std::cout << "         " << str(B[r], "{:8.3f}") << std::endl;
        // }

        assert(B.size() == rank);

        DataVector<double> U;
        if (X.data() == nullptr || X.data() == B.data())
            U.reset(B.size(), 1.);
        else
            U = X;
        assert(U.size() == B.size());

        int ier;

        // TODO add choice of algorithms and predonditioners

        NspcgFunc precond_func, accel_func;

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
        precond_func = this->get_preconditioner();

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
            case IterativeMatrixParams::ACCEL_LANDIR: accel_func = nspcg_landir; break;
            case IterativeMatrixParams::ACCEL_LANMIN: accel_func = nspcg_lanmin; break;
            case IterativeMatrixParams::ACCEL_LANRES: accel_func = nspcg_lanres; break;
            case IterativeMatrixParams::ACCEL_CGCR: accel_func = nspcg_cgcr; break;
            case IterativeMatrixParams::ACCEL_BCGS: accel_func = nspcg_bcgs; break;
        };
        // clang-format on

        while (true) {
            nspcg(precond_func, accel_func, ndim, mdim, rank, maxnz, data, icords, nullptr, nullptr, U.data(), nullptr, B.data(),
                  wksp, iwksp, nw, inw, iparm, rparm, ier);

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
                case -1: throw ComputationError(solver->getId(), "nonpositive matrix rank {}", rank);
                case -2: throw ComputationError(solver->getId(), "insufficient real workspace ({} required)", nw);
                case -3: throw ComputationError(solver->getId(), "insufficient integer workspace ({} required)", inw);
                case -4: throw ComputationError(solver->getId(), "nonpositive diagonal element in stiffness matrix");
                case -5: throw ComputationError(solver->getId(), "nonexistent diagonal element in stiffness matrix");
                case -6: throw ComputationError(solver->getId(), "stiffness matrix A is not positive definite");
                case -7: throw ComputationError(solver->getId(), "preconditioned matrix Q is not positive definite");
                case -8: throw ComputationError(solver->getId(), "cannot permute stiffness matrix as requested");
                case -9:
                    throw ComputationError(solver->getId(),
                                           "Number of non-zero diagonals is not large enough to allow expansion of matrix");
                case -10: throw ComputationError(solver->getId(), "inadmissible parameter encountered");
                case -11: throw ComputationError(solver->getId(), "incorrect storage mode for block method");
                case -12: throw ComputationError(solver->getId(), "zero pivot encountered in factorization");
                case -13: throw ComputationError(solver->getId(), "breakdown in direction vector calculation");
                case -14: throw ComputationError(solver->getId(), "breakdown in attempt to perform rotation");
                case -15: throw ComputationError(solver->getId(), "breakdown in iterate calculation");
                case -16: throw ComputationError(solver->getId(), "unimplemented combination of parameters");
                case -18: throw ComputationError(solver->getId(), "unable to perform eigenvalue estimation");
                case 1:
                    params->converged = false;
                    switch (params->no_convergence_behavior) {
                        case IterativeMatrixParams::NO_CONVERGENCE_ERROR:
                            throw ComputationError(solver->getId(), "failed to converge in {} iterations (error {})", iparm.itmax,
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
                case 2: solver->writelog(LOG_WARNING, "`maxerr` was too small, reset to {}", 3.55e-12); break;
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
        addmult(vector, result);
    }
};

struct SparseBandMatrix : SparseMatrix {
    /**
     * Create 2D matrix
     * \param solver solver
     * \param rank rank of the matrix
     * \param major shift of nodes to the next row (mesh[x,y+1])
     */
    template <typename SolverT>
    SparseBandMatrix(SolverT* solver, size_t rank, size_t major) : SparseMatrix(solver, rank, 5 * rank, 5) {
        icords[0] = 0;
        icords[1] = 1;
        icords[2] = major - 1;
        icords[3] = major;
        icords[4] = major + 1;
        kblsz = major - 1;
        nbl2d = major - 1;
    }

    /**
     * Create 3D matrix
     * \param solver solver
     * \param rank rank of the matrix
     * \param major shift of nodes to the next major row (mesh[x,y,z+1])
     * \param minor shift of nodes to the next minor row (mesh[x,y+1,z])
     */
    template <typename SolverT>
    SparseBandMatrix(SolverT* solver, size_t rank, size_t major, size_t minor) : SparseMatrix(solver, rank, 14 * rank, 14) {
        icords[0] = 0;
        icords[1] = 1;
        icords[2] = minor - 1;
        icords[3] = minor;
        icords[4] = minor + 1;
        icords[5] = major - minor - 1;
        icords[6] = major - minor;
        icords[7] = major - minor + 1;
        icords[8] = major - 1;
        icords[9] = major;
        icords[10] = major + 1;
        icords[11] = major + minor - 1;
        icords[12] = major + minor;
        icords[13] = major + minor + 1;
        kblsz = minor - 1;
        nbl2d = major - minor - 1;
    }

    /**
     * Create band matrix
     * \param solver solver
     * \param rank rank of the matrix
     * \param bands list of non-zero bands (shift from diagonal)
     */
    template <typename SolverT>
    SparseBandMatrix(SolverT* solver, size_t rank, std::initializer_list<int> bands)
        : SparseMatrix(solver, rank, bands.size() * rank, bands.size()) {
        size_t i = 0;
        for (int band : bands) icords[i++] = band;
        assert(icords[0] == 0);
    }

    /**
     * Return reference to array element.
     * \param r index of the element row
     * \param c index of the element column
     * \return reference to array element
     **/
    double& operator()(size_t r, size_t c) override {
        if (r == c) return data[r];
        if (r < c) std::swap(r, c);
        size_t i = std::find(icords, icords + mdim, r - c) - icords;
        assert(i != mdim);
        return data[c + rank * i];
    }

    void addmult(const DataVector<const double>& vector, DataVector<double>& result) override {
        for (size_t r = 0; r < rank; ++r) {
            result[r] += data[r] * vector[r];
        }
        for (size_t d = 1; d < mdim; ++d) {
            size_t sd = rank * d;
            for (size_t r = 0; r < rank; ++r) {
                size_t c = r + icords[d];
                if (c >= rank) break;
                result[r] += data[r + sd] * vector[c];
                result[c] += data[r + sd] * vector[r];
            }
        }
    }

  protected:
    NspcgFunc get_preconditioner() override {
        switch (params->preconditioner) {
            case IterativeMatrixParams::PRECOND_RICH: return nspcg_rich2;
            case IterativeMatrixParams::PRECOND_JAC: return nspcg_jac2;
            case IterativeMatrixParams::PRECOND_LJAC: return nspcg_ljac2;
            case IterativeMatrixParams::PRECOND_LJACX: return nspcg_ljacx2;
            case IterativeMatrixParams::PRECOND_SOR: return nspcg_sor2;
            case IterativeMatrixParams::PRECOND_SSOR: return nspcg_ssor2;
            case IterativeMatrixParams::PRECOND_IC: return nspcg_ic2;
            case IterativeMatrixParams::PRECOND_MIC: return nspcg_mic2;
            case IterativeMatrixParams::PRECOND_LSP: return nspcg_lsp2;
            case IterativeMatrixParams::PRECOND_NEU: return nspcg_neu2;
            case IterativeMatrixParams::PRECOND_LSOR: return nspcg_lsor2;
            case IterativeMatrixParams::PRECOND_LSSOR: return nspcg_lssor2;
            case IterativeMatrixParams::PRECOND_LLSP: return nspcg_llsp2;
            case IterativeMatrixParams::PRECOND_LNEU: return nspcg_lneu2;
            case IterativeMatrixParams::PRECOND_BIC: return nspcg_bic2;
            case IterativeMatrixParams::PRECOND_BICX: return nspcg_bicx2;
            case IterativeMatrixParams::PRECOND_MBIC: return nspcg_mbic2;
            case IterativeMatrixParams::PRECOND_MBICX: return nspcg_mbicx2;
        };
        assert(NULL);
        return nullptr;
    }

  public:
    void setBC(DataVector<double>& B, size_t r, double val) override {
        data[r] = 1.;
        B[r] = val;
        // above diagonal
        for (ptrdiff_t i = mdim - 1; i > 0; --i) {
            ptrdiff_t c = r - icords[i];
            if (c >= 0) {
                ptrdiff_t ii = c + rank * i;
                assert(ii < size);
                B[c] -= data[ii] * val;
                data[ii] = 0.;
            }
        }
        // below diagonal
        for (ptrdiff_t i = 1; i < mdim; ++i) {
            ptrdiff_t c = r + icords[i];
            if (c < rank) {
                size_t ii = r + rank * i;
                assert(ii < size);
                B[c] -= data[ii] * val;
                data[ii] = 0.;
            }
        }
    }
};

struct SparseFreeMatrix : SparseMatrix {
    int inz;  ///< Number of non-zero elements
    int* const ir;
    int* const ic;

    /**
     * Create sparse matrix
     * \param solver solver
     * \param rank rank of the matrix
     * \param maxnz maximum number of non-zero elements
     */
    template <typename SolverT>
    SparseFreeMatrix(SolverT* solver, size_t rank, size_t maxnz)
        : SparseMatrix(solver, rank, maxnz), inz(rank), ir(icords), ic(icords + maxnz) {
        assert(maxnz >= rank);
        for (size_t i = 0; i < rank; ++i) ir[i] = i + 1;
        for (size_t i = 0; i < rank; ++i) ic[i] = i + 1;
        if (params->preconditioner == IterativeMatrixParams::PRECOND_IC)
            params->preconditioner = IterativeMatrixParams::PRECOND_JAC;
    }

    /**
     * Return reference to array element
     * \param r index of the element row
     * \param c index of the element column
     * \return reference to array element
     **/
    double& operator()(size_t r, size_t c) override {
        if (r == c) return data[r];
        if (r > c) std::swap(r, c);
        assert(inz < size);
        ir[inz] = r + 1;
        ic[inz] = c + 1;
        return data[inz++];
    }

    void clear() override {
        std::fill_n(data, size, 0.);
        inz = rank;
    }

    void addmult(const DataVector<const double>& vector, DataVector<double>& result) override {
        for (size_t i = 0; i < rank; ++i) {
            result[i] += data[i] * vector[i];
        }
        for (size_t i = rank; i < inz; ++i) {
            size_t r = ir[i] - 1, c = ic[i] - 1;
            result[r] += data[i] * vector[c];
            result[c] += data[i] * vector[r];
        }
    }

  protected:
    NspcgFunc get_preconditioner() override {
        switch (params->preconditioner) {
            case IterativeMatrixParams::PRECOND_RICH: return nspcg_rich4;
            case IterativeMatrixParams::PRECOND_JAC: return nspcg_jac4;
            case IterativeMatrixParams::PRECOND_LSP: return nspcg_lsp4;
            case IterativeMatrixParams::PRECOND_NEU: return nspcg_neu4;
            default: throw NotImplemented("preconditioner not implemented for non-rectangular or masked mesh");
        };
        assert(NULL);
    }

    int get_maxnz() const override { return inz; }

  public:
    void setBC(DataVector<double>& B, size_t r, double val) override {
        data[r] = 1e32;
        B[r] = val * 1e32;
    }
};

}  // namespace plask

#endif  // PLASK_COMMON_FEM_ITERATIVE_MATRIX_H
