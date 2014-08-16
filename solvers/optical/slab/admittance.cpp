#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define BOOST_USE_WINDOWS_H
#endif
#include <boost/thread/mutex.hpp>

#include "admittance.h"
#include "solver.h"
#include "fortran.h"
#include "meshadapter.h"

namespace plask { namespace solvers { namespace slab {

AdmittanceTransfer::AdmittanceTransfer(SlabBase* solver, Expansion& expansion): Transfer(solver, expansion)
{
    writelog(LOG_DETAIL, "Initializing ADMITTANCE transfer");
    // Reserve space for matrix multiplications...
    int N = diagonalizer->matrixSize();
    Y = cmatrix(N,N);
    needAllY = false;
}


void AdmittanceTransfer::getFinalMatrix()
{
    int N = diagonalizer->matrixSize();
    int N0 = diagonalizer->source()->matrixSize();
    size_t count = solver->stack.size();

    // M = TH(interface) * Y(interface-1) * invTE(interface);
    findAdmittance(count-1, solver->interface-1);
    zgemm('n','n', N, N0, N, 1., Y.data(), N, diagonalizer->invTE(solver->stack[solver->interface]).data(), N, 0., work, N);
    zgemm('n','n', N0, N0, N, 1., diagonalizer->TH(solver->stack[solver->interface]).data(), N0, work, N, 0., M.data(), N0);

    // Find the(diagonalized field) admittance matrix and store it for the future reference
    findAdmittance(0, solver->interface);
    // M += TH(interface-1) * Y(interface) * invTE(interface-1);
    zgemm('n','n', N, N0, N, 1., Y.data(), N, diagonalizer->invTE(solver->stack[solver->interface-1]).data(), N, 0., work, N);
    zgemm('n','n', N0, N0, N, 1., diagonalizer->TH(solver->stack[solver->interface-1]).data(), N0, work, N, 1., M.data(), N0);
}


void AdmittanceTransfer::findAdmittance(int start, int end)
{
    const int inc = (start < end) ? 1 : -1;

    int N = diagonalizer->matrixSize();
    int NN = N*N;

    // Some temporary variables
    cdiagonal gamma, y1(N), y2(N);

    std::exception_ptr error;

    #ifdef OPENMP_FOUND
        std::vector<boost::mutex> layer_locks(diagonalizer->lcount);
        for (boost::mutex& mutex: layer_locks) mutex.lock();
    #endif

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic,1) nowait
        for (int l = 0; l < diagonalizer->lcount; ++l) {
            try {
                if (!error) diagonalizer->diagonalizeLayer(l);
                #ifdef OPENMP_FOUND
                layer_locks[l].unlock();
                #endif
            } catch(...) {
                error = std::current_exception();
            }
        }

        #pragma omp single
        if (!error) try {
            // Now iteratively we find matrices Y[i]

            // PML layer
            #ifdef OPENMP_FOUND
                layer_locks[solver->stack[start]].lock(); layer_locks[solver->stack[start]].unlock();
            #endif
            gamma = diagonalizer->Gamma(solver->stack[start]);
            std::fill_n(y2.data(), N, dcomplex(1.));                    // we use y2 for tracking sign changes
            for (int i = 0; i < N; i++) {
                y1[i] = gamma[i] * solver->vpml.factor;
                if (real(y1[i]) < -SMALL) { y1[i] = -y1[i]; y2[i] = -y2[i]; }
                if (imag(y1[i]) > SMALL) { y1[i] = -y1[i]; y2[i] = -y2[i]; }
            }
            get_y1(y1, solver->vpml.size, y1);
            std::fill_n(Y.data(), NN, dcomplex(0.));
            for (int i = 0; i < N; i++) Y(i,i) = - y1[i] * y2[i];

            // First layer
            double H = solver->vpml.shift;
            gamma = diagonalizer->Gamma(solver->stack[start]);
            get_y1(gamma, H, y1);
            get_y2(gamma, H, y2);
            for (int i = 0; i < N; i++) Y(i,i) = y1[i] - Y(i,i);        // off-diagonal elements of Y are 0
            for (int i = 0; i < N; i++) Y(i,i) = y2[i] * y2[i] / Y(i,i) - y1[i]; // Y = y2 * inv(Y) * y2 - y1

            // save the Y matrix for 1-st layer
            storeY(start);

            // Declare temporary matrixH) on 'work' array
            cmatrix wrk(N, N, work);

            for (int n = start+inc; n != end; n += inc)
            {
                #ifdef OPENMP_FOUND
                    layer_locks[solver->stack[n]].lock(); layer_locks[solver->stack[n]].unlock();
                #endif

                gamma = diagonalizer->Gamma(solver->stack[n]);

                H = solver->vbounds[n] - solver->vbounds[n-1];
                get_y1(gamma, H, y1);
                get_y2(gamma, H, y2);

                // The main equation
                // Y[n] = y2 * tE * inv(y1*tE - tH*Y[n-1]) * y2  -  y1

                mult_matrix_by_matrix(diagonalizer->TH(solver->stack[n-inc]), Y, temp);    // wrk = tH * Y[n-1]
                mult_matrix_by_matrix(diagonalizer->invTH(solver->stack[n]), temp, wrk);   // ...

                mult_matrix_by_matrix(diagonalizer->invTE(solver->stack[n]), diagonalizer->TE(solver->stack[n-inc]), temp); // compute tE

                for (int j = 0; j < N; j++)
                    for (int i = 0; i < N; i++) Y(i,j) = y1[i]*temp(i,j) - wrk(i,j);    // Y[n] = y1 * tE - wrk

                for (int i = 0; i < NN; i++) wrk[i] = 0.;
                for (int j = 0, i = 0; j < N; j++, i += N+1) wrk[i] = y2[j];            // wrk = y2

                invmult(Y, wrk);                                                        // wrk = inv(Y[n]) * (wrk = y2)
                mult_matrix_by_matrix(temp, wrk, Y);                                    // Y[n] = tE * wrk

                for (int j = 0; j < N; j++)
                    for (int i = 0; i < N; i++) Y(i,j) *= y2[i];                        // Y[n] = y2 * Y[n]

                for (int j = 0, i = 0; j < N; j++, i += N+1) Y[i] -= y1[j];             // Y[n] = Y[n] - y1

                // Save the Y matrix for n-th layer
                storeY(n);
            }
        } catch(...) {
                error = std::current_exception();
        }
    }
    if (error) std::rethrow_exception(error);
}


void AdmittanceTransfer::storeY(size_t n)
{
    if (needAllY) {
        int N = diagonalizer->matrixSize();
        if (memY.size() != solver->stack.size()) {
            // Allocate the storage for admittance matrices
            memY.resize(solver->stack.size());
            for (int i = 0; i < solver->stack.size(); i++) memY[i] = cmatrix(N,N);
        }
        memcpy(memY[n].data(), Y.data(), N*N*sizeof(dcomplex));
    }
}


void AdmittanceTransfer::determineFields()
{
    if (fields_determined == DETERMINED_RESONANT) return;

    writelog(LOG_DETAIL, solver->getId() + ": Determining optical fields");

    int N = diagonalizer->matrixSize();
    int N0 = diagonalizer->source()->matrixSize();
    size_t count = solver->stack.size();

    int NN = N*N;

    cdiagonal gamma;

    // [...]
    throw NotImplemented("AdmittanceTransfer::determineFields");

    needAllY = false;
    fields_determined = DETERMINED_RESONANT;
}



cvector AdmittanceTransfer::getFieldVectorE(double z, int n)
{
    throw NotImplemented("AdmittanceTransfer::getFieldVectorE");
}


cvector AdmittanceTransfer::getFieldVectorH(double z, int n)
{
    throw NotImplemented("AdmittanceTransfer::getFieldVectorH");
}

}}} // namespace plask::solvers::slab
