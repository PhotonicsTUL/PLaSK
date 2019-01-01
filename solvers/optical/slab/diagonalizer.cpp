#include "diagonalizer.h"
#include "expansion.h"

#include <plask/plask.hpp>

#include <algorithm>
#include <cstring>

#ifdef OPENMP_FOUND
#include <omp.h>
#endif

namespace plask { namespace optical { namespace slab {

Diagonalizer::Diagonalizer(Expansion* src) :
    src(src), diagonalized(src->solver->lcount, false), lcount(src->solver->lcount) {}

Diagonalizer::~Diagonalizer() {}


SimpleDiagonalizer::SimpleDiagonalizer(Expansion* g) :
    Diagonalizer(g),  gamma(lcount), Te(lcount), Th(lcount), Te1(lcount), Th1(lcount)
{
    const std::size_t N = src->matrixSize();         // Size of each matrix

    for (std::size_t i = 0; i < lcount; i++) {
        gamma[i] = cdiagonal(N);
        Th[i] = cmatrix(N, N);
        Th1[i] = cmatrix(N, N);
        Te[i] = cmatrix(N, N);
        Te1[i] = cmatrix(N, N);
    }
    #ifdef OPENMP_FOUND
        int nthr = std::min(omp_get_max_threads(), int(lcount));
        tmpmx = new cmatrix[nthr];
        tmplx = new omp_lock_t[nthr];
        writelog(LOG_DEBUG, "{}: Creating {:d} temporary matri{} for diagonalizer", src->solver->getId(), nthr, (nthr==1)?"x":"ces");
        for (int i = 0; i != nthr; ++i) {
            tmpmx[i] = cmatrix(N, N);
            omp_init_lock(tmplx+i);
        }
    #else
        writelog(LOG_DEBUG, "{}: Creating temporary matrix for diagonalizer", src->solver->getId());
        tmpmx = new cmatrix(N, N);
    #endif
}


SimpleDiagonalizer::~SimpleDiagonalizer()
{
    #ifdef OPENMP_FOUND
        int nthr = std::min(omp_get_max_threads(), int(lcount));
        for (int i = 0; i != nthr; ++i) {
            omp_destroy_lock(tmplx+i);
        }
        delete[] tmplx;
        delete[] tmpmx;
    #else
        delete tmpmx;
    #endif
}

std::size_t SimpleDiagonalizer::matrixSize() const
{
    return src->matrixSize();
}

void SimpleDiagonalizer::initDiagonalization()
{
    for (std::size_t layer = 0; layer < lcount; layer++)
        diagonalized[layer] = false;
}


bool SimpleDiagonalizer::diagonalizeLayer(size_t layer)
{
    if (diagonalized[layer]) return false;

    const size_t N = src->matrixSize();         // Size of each matrix
    const size_t NN = N*N;
    cdiagonal& gam = gamma[layer];

    #ifdef OPENMP_FOUND
        int nthr = std::min(omp_get_max_threads(), int(lcount));
        int mn;
        for (mn = 0; mn != nthr; ++mn)
            if (omp_test_lock(tmplx+mn)) break;
        assert(mn != nthr);
        cmatrix QE = tmpmx[mn];
        writelog(LOG_DEBUG, "{}: Diagonalizing matrix for layer {:d}/{:d} in thread {:d} [{:d}]",
                 src->solver->getId(), layer, lcount, omp_get_thread_num(), mn);
    #else
        cmatrix QE = *tmpmx;
        writelog(LOG_DEBUG, "{}: Diagonalizing matrix for layer {:d}/{:d}", src->solver->getId(), layer, lcount);
    #endif

    try {
        // First find necessary matrices
        cmatrix RE = Th1[layer], RH = Th[layer];

        src->getMatrices(layer, RE, RH);

        // Ugly hack to avoid singularities
        for (std::size_t i = 0; i != N; ++i) {
            if (RE(i,i) == 0.) RE(i,i) = SMALL;
            if (RH(i,i) == 0.) RH(i,i) = SMALL;
        }

        // std::cerr << "PLaSK\nRE:\n";
        // for (unsigned r = 0; r != N; ++r) {
        //     for (unsigned c = 0; c != N; ++c)
        //         std::cerr << format("{:7.1f} ", real(RE(r,c)));
        //     std::cerr << "\n";
        // }
        // std::cerr << "RH:\n";
        // for (unsigned r = 0; r != N; ++r) {
        //     for (unsigned c = 0; c != N; ++c)
        //         std::cerr << format("{:7.1f} ", real(RH(r,c)));
        //     std::cerr << "\n";
        // }
        assert(!RE.isnan());
        assert(!RH.isnan());

        if (src->diagonalQE(layer)) {

            // We are lucky - the QH matrix is diagonal so we can make it fast and easy

            // So we compute the diagonal elements of QH = RE*RH
            for (std::size_t ie = 0, ih = 0; ie < N; ie++, ih += N) {
                gam[ie] = 0;
                for (std::size_t jh = 0, je = 0; jh < N; jh++, je += N)
                    gam[ie] += RH[ie+je] * RE[ih+jh];
            }

            // Make Gamma of Gamma^2
            sqrtGamma(gam);

            src->getDiagonalEigenvectors(Te[layer], Te1[layer], RE, gam);

        } else {
            // We have to make the proper diagonalization
            // TODO: rewrite it to more low-level and more optimized computations
            mult_matrix_by_matrix(RH, RE, QE);  // QE = RH * RE

            // std::cerr << "PLaSK\nQE:\n";
            // for (unsigned r = 0; r != N; ++r) {
            //     for (unsigned c = 0; c != N; ++c)
            //         std::cerr << format("{:7.1f} ", real(QE(r,c)));
            //     std::cerr << "\n";
            // }

            // This is probably expensive but necessary check to avoid hangs
            if (QE.isnan()) throw ComputationError(src->solver->getId(), "SimpleDiagonalizer: NaN in Q matrix");

            // Here we make the actual diagonalization, i.e. compute the eigenvalues and eigenvectors of QE
            int info;
            if (N < 2) {
                dcomplex lwork[4];
                double rwork[2];
                zgeev('N', 'V', int(N), QE.data(), int(N), gam.data(), nullptr, int(N), Te[layer].data(), int(N),
                      lwork, 2, rwork, info);
            } else {
                // We use Th as work and Te1 as rwork (as N >= 2, their sizes are ok)
                zgeev('N', 'V', int(N), QE.data(), int(N), gam.data(), nullptr, int(N), Te[layer].data(), int(N),
                      Th[layer].data(), int(NN), reinterpret_cast<double*>(Te1[layer].data()), info);
            }
            if (info != 0) throw ComputationError(src->solver->getId(), "SimpleDiagonalizer: Could not compute {0}-th eignevalue of QE", info);

            // Find the inverse of Te in the classical way (maybe to be optimized in future)
            // TODO: eigenvectors should be built by hand based on Schur vectors
            std::copy_n(Te[layer].data(), NN, Th[layer].data());
            std::fill_n(Te1[layer].data(), NN, 0.);
            for (std::size_t i = 0; i < NN; i += (N+1))
                Te1[layer][i] = 1.;
            invmult(Th[layer], Te1[layer]);

            // Make Gamma of Gamma^2
            sqrtGamma(gam);
        }
        assert(!Te[layer].isnan());

        // So now there is the time to find TH = Re * Te * Gamma^(-1)
        mult_matrix_by_matrix(RE, Te[layer], Th[layer]);
        dcomplex* th = Th[layer].data();
        for (std::size_t j = 0; j < N; j++) {
            dcomplex g = 1. / gam[j];
            for (std::size_t i = 0; i < N; i++) *(th+i) *= g;
            th += N;
        }
        assert(!Th[layer].isnan());

        // Compute the Th1[layer] = Gamma[layer] * Te1[layer] * inv(RE)
        // we use the LU factorization of the RE matrix for this purpose and then solve Th1^T = inv(RE)^T * Te1^T * Gamma^T
        // the QE array is used as a temporary storage
        for (std::size_t i = 0; i < N; i++)
            for (std::size_t j = 0; j < N; j++)
                QE(i,j) = Te1[layer](j,i);
        // LU factorization of RE
        int ierr;
        std::unique_ptr<int[]> ipiv(new int[N]);
        zgetrf(int(N), int(N), RE.data(), int(N), ipiv.get(), ierr);
        if (ierr != 0) throw ComputationError(src->solver->getId(), "SimpleDiagonalizer: RE matrix singular");
        // the QE will contain inv(RE)^T * Te1^T
        zgetrs('t', int(N), int(N), RE.data(), int(N), ipiv.get(), QE.data(), int(N), ierr);
        if (ierr != 0) throw ComputationError(src->solver->getId(), "SimpleDiagonalizer: Could not compute inv(RE)");
        // compute QE^T and store it in Th1
        for (std::size_t j = 0; j < N; j++) {
            dcomplex g = gam[j];
            for (std::size_t i = 0; i < N; i++)
                Th1[layer](j,i) = QE(i,j) * g;
        }
        assert(!Th1[layer].isnan());
    } catch (...) {
        #ifdef OPENMP_FOUND
            omp_unset_lock(tmplx+mn);
        #endif
        throw;
    }

    #ifdef OPENMP_FOUND
        omp_unset_lock(tmplx+mn);
    #endif

    // Mark that layer has been diagonalized
    diagonalized[layer] = true;

    return true;
}

}}} // namespace plask::optical::slab
