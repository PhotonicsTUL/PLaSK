#include "diagonalizer.h"

#include <plask/plask.hpp>

#include <algorithm>
#include <cstring>

namespace plask { namespace  solvers { namespace slab {


    SimpleDiagonalizer::SimpleDiagonalizer(GridBase& g) :
    DiagonalizerBase(g),  gamma(lcount), Te(lcount), Th(lcount), Te1(lcount), Th1(lcount)
{
    int N = grid.matrixSize();         // Size of each matrix

    //logger(LOG_BASIC) << "Creating simple diagonalizer...\n\n";

    for (int i = 0; i < lcount; i++) {
        gamma[i] = cdiagonal(N);
        Th[i] = cmatrix(N, N);
        Th1[i] = cmatrix(N, N);
        Te[i] = cmatrix(N, N);
        Te1[i] = cmatrix(N, N);
    }

    QE = cmatrix(N,N);
    tmp = cmatrix(N,N);
}


SimpleDiagonalizer::~SimpleDiagonalizer()
{
}


void SimpleDiagonalizer::diagonalizeLayer(int layer)
{
    // If diagonalization already done, do not repeat it
    if (diagonalized[layer]) return;

    int N = grid.matrixSize();         // Size of each matrix

    // First find necessary matrices
    cmatrix RH = grid.getRH(layer, k0, Kx, Ky, matgain);
    cmatrix RE = grid.getRE(layer, k0, Kx, Ky, matgain);

    if (grid.diagonalQE(layer)) {

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
                throw "SimpleDiagonalizer::diagonalizeLayer: NaN in Q matrix";
        }

        // Here we make the actual diagonalization, i.e. compute the eigenvalues and eigenvectors of QE
        // we use Te as work and Te1 as rwork (as N >= 2, their sizes are ok)
        int info;
        F(zgeev)('N', 'V', N, QE.data(), N, gamma[layer].data(), NULL, N,  tmp.data(), N,
                 Te[layer].data(), NN, reinterpret_cast<double*>(Te1[layer].data()), info);

        // ...and rewrite the eigenvectors to their final locations
        memcpy(Te[layer].data(), tmp.data(), NN*sizeof(dcomplex));

        // Find the inverse of Th in the classical way (maybe to be optimized in future)
        // TODO: eigenvectors should be built by hand based on Schur vectors
        memset(Te1[layer].data(), 0., NN*sizeof(dcomplex));
        for (int i = 0; i < NN; i += (N+1))
            Te1[layer][i] = 1.;
        invmult(tmp, Te1[layer]);
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
    // the RH array is used as a temporary storage
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            RH(i,j) = Te1[layer](j,i);
    // LU factorization of RE
    int ierr;
    int* ipiv = new int[N];
    F(zgetrf)(N, N, RE.data(), N, ipiv, ierr);
    // the RH will contain inv(RE)^T * Te1^T
    F(zgetrs)('t', N, N, RE.data(), N, ipiv, RH.data(), N, ierr);
    // compute RH^T and store it in Th1
    for (int j = 0; j < N; j++) {
        dcomplex g = gam[j];
        for (int i = 0; i < N; i++)
            Th1[layer](j,i) = RH(i,j) * g;
    }
    delete[] ipiv;

    // Mark that layer has been diagonalized
    diagonalized[layer] = true;
}

}}} // namespace plask::solvers::slab
