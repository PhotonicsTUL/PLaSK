#include "diagonalizer.h"

#include <plask/plask.hpp>

#include <algorithm>
#include <cstring>

#ifdef OPENMP_FOUND
#include <omp.h>
#endif


namespace plask { namespace  solvers { namespace slab {


SimpleDiagonalizer::SimpleDiagonalizer(Expansion* g) :
    Diagonalizer(g),  gamma(lcount), Te(lcount), Th(lcount), Te1(lcount), Th1(lcount)
{
    int N = src->matrixSize();         // Size of each matrix

    //logger(LOG_BASIC) << "Creating simple diagonalizer...\n\n";

    for (int i = 0; i < lcount; i++) {
        gamma[i] = cdiagonal(N);
        Th[i] = cmatrix(N, N);
        Th1[i] = cmatrix(N, N);
        Te[i] = cmatrix(N, N);
        Te1[i] = cmatrix(N, N);
    }
#ifdef OPENMP_FOUND
    int nt = omp_get_max_threads();
    tmpmx = new cmatrix[nt];
    src->solver->writelog(LOG_DEBUG, "Creating %1% temporary matri%2% for diagonalizer", nt, (nt==1)?"x":"ces");
    for (size_t i = 0; i != nt; ++i) {
        tmpmx[i] = cmatrix(N, N);
    }
#else
    tmpmx = new cmatrix(N, N);
#endif
}


SimpleDiagonalizer::~SimpleDiagonalizer()
{
#ifdef OPENMP_FOUND
    delete[] tmpmx;
#else
    delete tmpmx;
#endif
}


void SimpleDiagonalizer::initDiagonalization(dcomplex ko, dcomplex kx, dcomplex ky)
{
    k0 = ko; Kx = kx, Ky = ky;

    for (int layer = 0; layer < lcount; layer++)
        diagonalized[layer] = false;
}


void SimpleDiagonalizer::diagonalizeLayer(size_t layer)
{
    if (diagonalized[layer]) return;
    
    int N = src->matrixSize();         // Size of each matrix

    #ifdef OPENMP_FOUND
        auto tn = omp_get_thread_num();
        cmatrix QE = tmpmx[tn];
        src->solver->writelog(LOG_DEBUG, "Diagonalizing matrix for layer %1% in thread %2%", layer, tn);
    #else
        cmatrix QE = *tmpmx;
        src->solver->writelog(LOG_DEBUG, "Diagonalizing matrix for layer %1%", layer);
    #endif
        
    // First find necessary matrices
    cmatrix RE = Th1[layer], RH = Th[layer];

    src->getMatrices(layer, k0, Kx, Ky, RE, RH);

    if (src->diagonalQE(layer)) {

        // We are lucky - the QH matrix is diagonal so we can make it fast and easy
        //logger(LOG_SHOWDIAGONALIZATION) << "    diagonalizer: using the uniform layer " << layer << "\n";

        // So we compute the diagonal elements of QH = RE*RH
        for (int ie = 0, ih = 0; ie < N; ie++, ih += N) {
            gamma[layer][ie] = 0;
            for (int jh = 0, je = 0; jh < N; jh++, je += N)
                gamma[layer][ie] += RH[ie+je] * RE[ih+jh];
        }

        // Eigenvector matrix is simply a unity matrix
        for (int i = 0; i < N; i++) {
                Te[layer](i, i) = Te1[layer](i, i) = 1.;
        }

    } else {
        // We have to make the proper diagonalization
        // TODO: rewrite it to more low-level and more optimized computations
        mult_matrix_by_matrix(RH, RE, QE);  // QE = RH * RE

        // This is probably expensive but necessary check to avoid hangs
        int NN = N*N;
        for (int i = 0; i < NN; i++) {
            if (isnan(real(QE[i])) || isnan(imag(QE[i])))
                throw ComputationError("SimpleDiagonalizer", "NaN in Q matrix");
        }

        // Here we make the actual diagonalization, i.e. compute the eigenvalues and eigenvectors of QE
        // we use Th as work and Te1 as rwork (as N >= 2, their sizes are ok)
        int info;
        zgeev('N', 'V', N, QE.data(), N, gamma[layer].data(), NULL, N,  Te[layer].data(), N,
            Th[layer].data(), NN, reinterpret_cast<double*>(Te1[layer].data()), info);

        // ...and rewrite the eigenvectors to their final locations
        memcpy(Th[layer].data(), Te[layer].data(), NN*sizeof(dcomplex));

        // Find the inverse of Te in the classical way (maybe to be optimized in future)
        // TODO: eigenvectors should be built by hand based on Schur vectors
        memset(Te1[layer].data(), 0., NN*sizeof(dcomplex));
        for (int i = 0; i < NN; i += (N+1))
            Te1[layer][i] = 1.;
        invmult(Th[layer], Te1[layer]);
    }

    // Make Gamma of Gamma^2
    cdiagonal& gam = gamma[layer];
    for (int j = 0; j < N; j++) {
        dcomplex g;
        g = sqrt(gam[j]);
        if (real(g) < -SMALL) g = -g;
        if (imag(g) > SMALL) g = -g;
        gam[j] = g;
    }

    // So now there is the time to find TH = Re * Te * Gamma^(-1)
    mult_matrix_by_matrix(RE, Te[layer], Th[layer]);
    dcomplex* th = Th[layer].data();
    for (int j = 0; j < N; j++) {
        dcomplex g = 1. / gam[j];
        for (int i = 0; i < N; i++) *(th+i) *= g;
        th += N;
    }

    // Compute the Th1[layer] = Gamma[layer] * Te1[layer] * inv(RE)
    // we use the LU factorization of the RE matrix for this purpose and then solve Th1^T = inv(RE)^T * Te1^T * Gamma^T
    // the QE array is used as a temporary storage
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            QE(i,j) = Te1[layer](j,i);
    // LU factorization of RE
    int ierr;
    int* ipiv = new int[N];
    zgetrf(N, N, RE.data(), N, ipiv, ierr);
    // the QE will contain inv(RE)^T * Te1^T
    zgetrs('t', N, N, RE.data(), N, ipiv, QE.data(), N, ierr);
    // compute QE^T and store it in Th1
    for (int j = 0; j < N; j++) {
        dcomplex g = gam[j];
        for (int i = 0; i < N; i++)
            Th1[layer](j,i) = QE(i,j) * g;
    }
    delete[] ipiv;

    // Mark that layer has been diagonalized
    diagonalized[layer] = true;
}

}}} // namespace plask::solvers::slab
