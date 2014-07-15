#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define BOOST_USE_WINDOWS_H
#endif
#include <boost/thread/mutex.hpp>

#include "reflection_base.h"
#include "fortran.h"
#include "mesh_adapter.h"

#define ee A
#define hh P

namespace plask { namespace solvers { namespace slab {

template <typename GeometryT>
void ReflectionSolver<GeometryT>::init()
{
    this->ensureInterface();

    // Reserve space for matrix multiplications...
    int N0 = this->diagonalizer->source()->matrixSize();
    int N = this->diagonalizer->matrixSize();
    M = cmatrix(N0, N0);

    // ...and eigenvalues determination
    assert(evals == nullptr);
    assert(rwork == nullptr);
    assert(work == nullptr);
    evals = aligned_new_array<dcomplex>(N0);
    rwork = aligned_new_array<double>(2*N0);
    lwork = N0*N0;
    work = aligned_new_array<dcomplex>(lwork);

    // Nothing found so far
    fields_determined = DETERMINED_NOTHING;
    interface_field = nullptr;

    this->writelog(LOG_DETAIL, "Initializing reflection-matrix solver (%1% layers in the stack, interface after %2% layer%3%)",
                               this->stack.size(), this->interface, (this->interface==1)? "" : "s");

    P = cmatrix(N,N);
    phas = cdiagonal(N);
    A = cmatrix(N,N);       // A is also temporary matrix

    assert(ipiv == nullptr);
    ipiv = aligned_new_array<int>(N);
    allP = false;

    this->recompute_coefficients = true;
}


template <typename GeometryT>
void ReflectionSolver<GeometryT>::cleanup() {
    fields.clear();
    memP.clear();
    if (this->diagonalizer) {
        int N0 = this->diagonalizer->source()->matrixSize();
        int N = this->diagonalizer->matrixSize();
        aligned_delete_array<dcomplex>(N0, evals); evals = nullptr;
        aligned_delete_array<double>(2*N0, rwork); rwork = nullptr;
        aligned_delete_array<dcomplex>(lwork, work); work = nullptr;
        aligned_delete_array<int>(N, ipiv); ipiv = nullptr;
    }
}


template <typename GeometryT>
ReflectionSolver<GeometryT>::~ReflectionSolver()
{
    cleanup();
    this->inTemperature.changedDisconnectMethod(this, &ReflectionSolver::onInputChanged);
    this->inGain.changedDisconnectMethod(this, &ReflectionSolver::onInputChanged);
}


template <typename GeometryT>
dcomplex ReflectionSolver<GeometryT>::determinant()
{
    // We change the matrices M and A so we will have to find the new fields
    fields_determined = DETERMINED_NOTHING;

    initDiagonalization();

    // Obtain admittance
    getFinalMatrix();

    int N = M.rows();

    // This is probably expensive but necessary check to avoid hangs
    int NN = N*N;
    for (int i = 0; i < NN; i++) {
        if (isnan(real(M[i])) || isnan(imag(M[i])))
            throw ComputationError(this->getId(), "NaN in discontinuity matrix");
    }

    // Find the eigenvalues of M using LAPACK
    dcomplex nth; int info;
    zgeev('N', 'N', N, M.data(), N, evals, &nth, 1, &nth, 1, work, lwork, rwork, info);
    if (info != 0) throw ComputationError(this->getId(), "eigenvalue determination failed");

    //TODO add some consideration for degenerate modes
    // Find the smallest eigenvalue
    dcomplex val, result;
    double min_mag = 1e32;
    for (int i = 0; i < N; i++) {
        val = evals[i];
        if (abs2(val) < min_mag) { min_mag = abs2(val); result = val; }
    }
    // // Find the determinant
    // dcomplex result = 1.;
    // for (int i = 0; i < N; i++) {
    //     result *= evals[i];
    // }
    // result = log(result);

    interface_field = nullptr;

    return result;
}


template <typename GeometryT>
void ReflectionSolver<GeometryT>::getAM(size_t start, size_t end, bool add, double mfac)
{
    // Get matrices sizes
    int N0 = this->diagonalizer->source()->matrixSize();
    int N = this->diagonalizer->matrixSize(); // <= N0
    int NN = N*N;
    cmatrix wrk(N, N0, work);    // matrix object for the workspace

    findReflection(start, end);

    cdiagonal gamma = this->diagonalizer->Gamma(this->stack[end]);

    double H = (end == 0 || end == this->vbounds.size())? 0. : abs(this->vbounds[end] - this->vbounds[end-1]);
    for (int i = 0; i < N; i++) phas[i] = exp(-I*gamma[i]*H);

    mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);     // P = phas * P * phas
    memcpy(A.data(), P.data(), NN*sizeof(dcomplex));                        // A = P

    // A = [ phas*P*phas - I ] [ phas*P*phas + I ]^{-1}
    for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] += 1;            // P = P + I
    for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) A[ii] -= 1;            // A = A - I
    int info;
    zgetrf(N, N, P.data(), N, ipiv, info);                                  // P = LU(P)
    ztrsm('R', 'U', 'N', 'N', N, N, 1., P.data(), N, A.data(), N);          // A = A * U^{-1}
    ztrsm('R', 'L', 'N', 'U', N, N, 1., P.data(), N, A.data(), N);          // A = A * L^{-1}
    // reorder columns (there is no such function in LAPACK)
    for (int j = N-1; j >=0 ; j--) {
        int jp = ipiv[j]-1;
        for (int i = 0; i < N; i++) std::swap(A(i,j), A(i,jp));
    }

    // M for the half of the structure
    mult_matrix_by_matrix(A, this->diagonalizer->invTE(this->stack[end]), wrk);    // wrk = A * invTE[end]

    zgemm('N','N', N0, N0, N, mfac, this->diagonalizer->TH(this->stack[end]).data(), N0,
          work, N, add?1.:0., M.data(), N0);                                 // M = mfac * TH[end] * wrk
}


template <typename GeometryT>
void ReflectionSolver<GeometryT>::findReflection(int start, int end)
{
    // Should be called from 0 to interface-1
    // and from count-1 to interface

    const int inc = (start < end) ? 1 : -1;

    int N0 = this->diagonalizer->source()->matrixSize();
    int N = this->diagonalizer->matrixSize();
    int NN = N*N;

    cmatrix wrk(N, N0, work);    // matrix object for the workspace

    cdiagonal gamma;

    // in the beginning the P matrix is zero
    memset(P.data(), 0, NN*sizeof(dcomplex));
    storeP(start);

    std::exception_ptr error;

    #ifdef OPENMP_FOUND
        std::vector<boost::mutex> layer_locks(this->diagonalizer->lcount);
        for (boost::mutex& mutex: layer_locks) mutex.lock();
    #endif

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic,1) nowait
        for (int l = 0; l < this->diagonalizer->lcount; ++l) {
            try {
                if (!error) this->diagonalizer->diagonalizeLayer(l);
                #ifdef OPENMP_FOUND
                layer_locks[l].unlock();
                #endif
            } catch(...) {
                error = std::current_exception();
            }
        }

        #pragma omp single
        if (!error) try {
            for (int n = start; n != end; n += inc) {
                #ifdef OPENMP_FOUND
                layer_locks[this->stack[n]].lock(); layer_locks[this->stack[n]].unlock();
                #endif
                gamma = this->diagonalizer->Gamma(this->stack[n]);
                assert(!gamma.isnan());

                double H = (n == start)? 0. : (this->vbounds[n] - this->vbounds[n-1]);

                if (!emitting || n != start) {
                    for (int i = 0; i < N; i++) phas[i] = exp(-I*gamma[i]*H);
                } else {
                    for (int i = 0; i < N; i++) {
                        dcomplex g = gamma[i];
                        if (real(g) < -SMALL) g = -g;
                        phas[i] = exp(-I*g*H);
                    }
                }
                assert(!P.isnan());
                assert(!phas.isnan());
                mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);         // P = phas * P * phas

                // ee = invTE(n+1)*TE(n) * [ phas*P*phas + I ]
                assert(!this->diagonalizer->TE(this->stack[n]).isnan());
                assert(!this->diagonalizer->invTE(this->stack[n]).isnan());
                for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] += 1.;               // P = P.orig + I
                mult_matrix_by_matrix(this->diagonalizer->TE(this->stack[n]), P, wrk);            // wrk = TE[n] * P
                #ifdef OPENMP_FOUND
                layer_locks[this->stack[n+inc]].lock(); layer_locks[this->stack[n+inc]].unlock();
                #endif
                mult_matrix_by_matrix(this->diagonalizer->invTE(this->stack[n+inc]), wrk, ee);    // ee = invTE[n+1] * wrk (= A)

                // hh = invTH(n+1)*TH(n) * [ phas*P*phas - I ]
                assert(!this->diagonalizer->TH(this->stack[n]).isnan());
                assert(!this->diagonalizer->invTH(this->stack[n]).isnan());
                for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] -= 2.;               // P = P - I

                // multiply rows of P by -1 where necessary for properly outgoing wave
                if (emitting && n == start) {
                    for (int i = 0; i < N; i++)
                        if (real(gamma[i]) < -SMALL)
                            for(int j = 0; j < N; j++) P(i,j) = -P(i,j);
                }

                mult_matrix_by_matrix(this->diagonalizer->TH(this->stack[n]), P, wrk);            // wrk = TH[n] * P
                mult_matrix_by_matrix(this->diagonalizer->invTH(this->stack[n+inc]), wrk, hh);    // hh = invTH[n+1] * wrk (= P)

                // ee := ee-hh, hh := ee+hh
                for (int i = 0; i < NN; i++) {
                    dcomplex e = ee[i], h = hh[i];
                    A[i] = e - h;
                    P[i] = e + h;
                }

                // There might appear a problem if the last layer and the one next to the last are the same.
                // It may happen that e ~ h and then A matrix is close to singular. This will yield unreliable results.
                // To avoid this never ever put two identical layers next to each other

                // P = P * inv(A)
                int info;
                zgetrf(N, N, A.data(), N, ipiv, info);                                      // A = LU(A)         (= A)
                if (info > 0) throw ComputationError(this->getId(), "findReflection: Matrix [e(n) - h(n)] is singular");
                assert(info == 0);
                ztrsm('R', 'U', 'N', 'N', N, N, 1., A.data(), N, P.data(), N);              // P = P * U^{-1}    (= P)
                ztrsm('R', 'L', 'N', 'U', N, N, 1., A.data(), N, P.data(), N);              // P = P * L^{-1}
                if (P.isnan()) throw ComputationError(this->getId(), "findReflection: NaN in reflection matrix");
                // reorder columns (there is no such function in LAPACK)
                for (int j = N-1; j >= 0; j--) {
                    int jp = ipiv[j]-1;
                    for (int i = 0; i < N; i++) std::swap(P(i,j), P(i,jp));
                }

                storeP(n+inc);
            }
        } catch(...) {
                error = std::current_exception();
        }
    }
    if (error) std::rethrow_exception(error);
}


template <typename GeometryT>
void ReflectionSolver<GeometryT>::storeP(size_t n) {
    if (allP) {
        int N = this->diagonalizer->matrixSize();
        if (memP.size() != this->stack.size()) {
            // Allocate the storage for admittance matrices
            memP.resize(this->stack.size());
            for (int i = 0; i < this->stack.size(); i++) memP[i] = cmatrix(N,N);
        }
        memcpy(memP[n].data(), P.data(), N*N*sizeof(dcomplex));
    }
}


template <typename GeometryT>
cvector ReflectionSolver<GeometryT>::getReflectionVector(const cvector& incident, IncidentDirection side)
{
    int last, first;

    initDiagonalization();
    switch (side) {
        case INCIDENCE_TOP:
            last = 0; first = this->stack.size()-1; break;
        case INCIDENCE_BOTTOM:
            last = this->stack.size()-1; first = 0; break;
    }
    findReflection(last, first);
    return P * incident;
}


template <typename GeometryT>
cvector ReflectionSolver<GeometryT>::getTransmissionVector(const cvector& incident, IncidentDirection side)
{
    determineReflectedFields(incident, side);
    size_t n = (side == INCIDENCE_BOTTOM)? this->stack.size()-1 : 0;
    return this->diagonalizer->TE(this->stack[n]) * fields[n].B;
}


template <typename GeometryT>
cvector ReflectionSolver<GeometryT>::getInterfaceVector()
{
    int N = M.rows();

    // Check if the necessary memory is already allocated
    if (interface_field_matrix.rows() != N) {
        interface_field_matrix = cmatrix(N,N);
        interface_field = nullptr;
    }

    // If the field already found, don't compute again
    if (!interface_field) {

        // Obtain admittance
        getFinalMatrix();

        // Find the eigenvalues of M using LAPACK
        dcomplex nth; int info;
        zgeev('N', 'V', N, M.data(), N, evals, &nth, 1, interface_field_matrix.data(), N, work, lwork, rwork, info);
        if (info != 0) throw ComputationError(this->getId(), "getInterfaceVector: zgeev failed");

        // Find the number of the smallest eigenvalue
        double mag, min_mag = 1e32;
        int n;
        for (int i = 0; i < N; i++) {
            mag = abs2(evals[i]);
            if (mag < min_mag) { min_mag = mag; n = i; }
        }

        // Error handling
        if (min_mag > this->root.tolf_max * this->root.tolf_max)
            throw BadInput(this->getId(), "getInterfaceVector: determinant not sufficiently close to 0");

        // Chose the eigenvector corresponding to the smallest eigenvalue
        interface_field = interface_field_matrix.data() + n*N;
    }

    // Make a copy of the interface vector
    cvector E(N);
    for (int i = 0; i < N; i++) E[i] = interface_field[i];

    return E;
}


template <typename GeometryT>
void ReflectionSolver<GeometryT>::determineFields()
{
    if (fields_determined == DETERMINED_RESONANT) return;

    this->writelog(LOG_DETAIL, "Determining optical fields");

    int N = this->diagonalizer->matrixSize();
    int N0 = this->diagonalizer->source()->matrixSize();
    int NN = N*N;

    cdiagonal gamma;

    size_t count = this->stack.size();

    // Assign the space for the field vectors
    fields.resize(count);
    // Obtain the physical fields at the last layer
    allP = true; interface_field = nullptr;

    cvector E = getInterfaceVector();
    cvector tmp(work, N);

    for (int pass = 0; pass < 1 || (pass < 2 && this->interface != count); pass++)
    {
        // each pass for below and above the interface

        int start, end, inc;
        switch (pass)
        {
            case 0: start = this->interface-1; end = 0;       inc = -1; break;
            case 1: start = this->interface;   end = count-1; inc = +1; break;
        }

        fields[start].F = cvector(N);
        fields[start].B = cvector(N);

        // compute B-field for the layer next to the interface
        int curr = this->stack[start];

        gamma = this->diagonalizer->Gamma(curr);
        double H = (start == 0 || start == count-1)? 0. : (this->vbounds[start] - this->vbounds[start-1]);
        for (int i = 0; i < N; i++)
            phas[i] = exp(-I*gamma[i]*H);

        // P = phas*P*phas + I
        memcpy(P.data(), memP[start].data(), NN*sizeof(dcomplex));
        mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);         // P := phas * P * phas
        for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] += 1.;               // P := P + I

        mult_matrix_by_vector(this->diagonalizer->invTE(curr), E, fields[start].B);       // B := invTE * E
        invmult(P, fields[start].B);                                                // B := inv(P) * B
        for (int i = 0; i < N; i++) fields[start].B[i] *= phas[i];                  // B := phas * B

        for (int n = start; n != end; n += inc)
        {
            // F-field for the current layer
            mult_matrix_by_vector(memP[n], fields[n].B, fields[n].F);

            // Compute B-field for the next (previous) layer

            fields[n+inc].F = cvector(N);
            fields[n+inc].B = cvector(N);

            // some aliases
            cvector& F1 = fields[n].F;
            cvector& B1 = fields[n].B;
            cvector& F2 = fields[n+inc].F;
            cvector& B2 = fields[n+inc].B;

            curr = this->stack[n];
            int next = this->stack[n+inc];

            gamma = this->diagonalizer->Gamma(next);

            for (int i = 0; i < N; i++) F2[i] = F1[i] - B1[i];              // F2 := F1 - B1
            mult_matrix_by_vector(this->diagonalizer->TH(curr), F2, tmp);         // tmp := TH * F2
            mult_matrix_by_vector(this->diagonalizer->invTH(next), tmp, B2);      // B2 := invTH * tmp
            // multiply rows of invTH by -1 where necessary for the outer layer
            if (n+inc == end && emitting) {
                for (int i = 0; i < N; i++)
                    if (real(gamma[i]) < -SMALL) B2[i] = -B2[i];
            }

            for (int i = 0; i < N; i++) F2[i] = F1[i] + B1[i];              // F2 := F1 + B1
            mult_matrix_by_vector(this->diagonalizer->TE(curr), F2, tmp);         // tmp := TE * F2
            zgemm('N','N', N, 1, N0, 1., this->diagonalizer->invTE(next).data(), N,
                  tmp.data(), N0, -1., B2.data(), N);                       // B2 := invTE * tmp - B2

            H = (n+inc == end)? 0. : (this->vbounds[n+inc] - this->vbounds[n+inc-1]);
            if (n+inc != end || !emitting) {
                for (int i = 0; i < N; i++)
                    B2[i] *= 0.5 * exp(-I*gamma[i]*H);                      // B2 := 1/2 * phas * B2
            } else {
                for (int i = 0; i < N; i++) {
                    dcomplex g = gamma[i];
                    if (real(g) < -SMALL) g = -g;
                    B2[i] *= 0.5 * exp(-I*g*H);                             // B2 := 1/2 * phas * B2
                }
            }
        }

        // mult_matrix_by_vector(memP[end], fields[end].B, fields[end].F);
        for (int i = 0; i < N; ++i) fields[end].F[i] = 0.; // in the outer layer there is no incoming field

        // In the outer layers replace F and B where necessary for consistent gamma handling
        if (emitting) {
            cvector& F2 = fields[end].F;
            cvector& B2 = fields[end].B;
            for (int i = 0; i < N; i++)
                if (real(gamma[i]) < -SMALL) std::swap(F2[i], B2[i]);
        }
    }

    allP = false;
    fields_determined = DETERMINED_RESONANT;
}

template <typename GeometryT>
void ReflectionSolver<GeometryT>::determineReflectedFields(const cvector& incident, IncidentDirection side)
{
    if (fields_determined == DETERMINED_REFLECTED) return;

    this->writelog(LOG_DETAIL, "Determining reflected optical fields");

    size_t count = this->stack.size();

    // Assign the space for the field vectors
    fields.resize(count);
    // Obtain the physical fields at the last layer
    allP = true;

    int start, end, inc;
    switch (side)
    {
        case INCIDENCE_TOP:    start = count-1; end = 0;       inc = -1; break;
        case INCIDENCE_BOTTOM: start = 0;       end = count-1; inc = +1; break;
    }

    // Compute reflection matrices
    initDiagonalization();
    findReflection(end, start);

    // Temporary and initial data
    int N = this->diagonalizer->matrixSize();
    int N0 = this->diagonalizer->source()->matrixSize();
    cvector tmp(work, N);
    cdiagonal gamma;

    int curr = this->stack[start];
    double H;

    fields[start].B = this->diagonalizer->invTE(curr) * incident; // diagonalized incident E-field
    fields[start].F = cvector(N);

    for (int n = start; n != end; n += inc)
    {
        // F-field for the current layer
        mult_matrix_by_vector(memP[n], fields[n].B, fields[n].F);

        // Compute B-field for the next (previous) layer

        fields[n+inc].F = cvector(N);
        fields[n+inc].B = cvector(N);

        // some aliases
        cvector& F1 = fields[n].F;
        cvector& B1 = fields[n].B;
        cvector& F2 = fields[n+inc].F;
        cvector& B2 = fields[n+inc].B;

        curr = this->stack[n];
        int next = this->stack[n+inc];

        gamma = this->diagonalizer->Gamma(next);

        for (int i = 0; i < N; i++) F2[i] = F1[i] - B1[i];              // F2 := F1 - B1
        mult_matrix_by_vector(this->diagonalizer->TH(curr), F2, tmp);         // tmp := TH * F2
        mult_matrix_by_vector(this->diagonalizer->invTH(next), tmp, B2);      // B2 := invTH * tmp
        // multiply rows of invTH by -1 where necessary for the outer layer
        if (n+inc == end) {
            for (int i = 0; i < N; i++)
                if (real(gamma[i]) < -SMALL) B2[i] = -B2[i];
        }

        for (int i = 0; i < N; i++) F2[i] = F1[i] + B1[i];              // F2 := F1 + B1
        mult_matrix_by_vector(this->diagonalizer->TE(curr), F2, tmp);         // tmp := TE * F2
        zgemm('N','N', N, 1, N0, 1., this->diagonalizer->invTE(next).data(), N,
              tmp.data(), N0, -1., B2.data(), N);                       // B2 := invTE * tmp - B2

        H = (n+inc != end)? this->vbounds[n+inc] - this->vbounds[n+inc-1] : 0.;
        if (n+inc != end) {
            for (int i = 0; i < N; i++)
                B2[i] *= 0.5 * exp(-I*gamma[i]*H);                      // B2 := 1/2 * phas * B2
        } else {
            for (int i = 0; i < N; i++) {
                dcomplex g = gamma[i];
                if (real(g) < -SMALL) g = -g;
                B2[i] *= 0.5 * exp(-I*g*H);                             // B2 := 1/2 * phas * B2
            }
        }
    }

    //mult_matrix_by_vector(getP(0), layerFields[0].B, layerFields[0].F);
    fields[end].F = cvector(N, 0.);

    // In the outer layers replace F and B where necessary for consistent gamma handling
    for (int n = 0; n < count; n += count-1) {
        gamma = this->diagonalizer->Gamma(this->stack[n]);
        for (int i = 0; i < N; i++) {
            if (real(gamma[i]) < -SMALL)
                std::swap(fields[n].F, fields[n].B);
        }
     }

    // Replace F and B before the interface for consistency in getFieldVectorE and getFieldVectorH
    switch (side)
    {
        case INCIDENCE_TOP:    start = this->interface; end = count; break;
        case INCIDENCE_BOTTOM: start = 0; end = this->interface; break;
    }
    for (int n = start; n < end; n++) {
        cvector& F2 = fields[n].F;
        cvector& B2 = fields[n].B;
        gamma = this->diagonalizer->Gamma(this->stack[n]);
        H = (n < count-1)? this->vbounds[n] - this->vbounds[n-1] : 0.;
        for (int i = 0; i < N; i++) {
                dcomplex phas = exp(-I*gamma[i]*H);
                dcomplex t = B2[i] / phas;
                B2[i] = F2[i] * phas;
                F2[i] = t;
        }
    }

    allP = false;
    fields_determined = DETERMINED_REFLECTED;
}


template <typename GeometryT>
cvector ReflectionSolver<GeometryT>::getFieldVectorE(double z, int n)
{
    assert(fields_determined != DETERMINED_NOTHING);

    cvector& FF = fields[n].F;
    cvector& BB = fields[n].B;

    if (n >= this->interface) {
        z = - z;
        if (n != 0 && n != this->vbounds.size())
            z += this->vbounds[n] - this->vbounds[n-1];
    }

    cdiagonal gamma = this->diagonalizer->Gamma(this->stack[n]);

    int N = gamma.size();
    cvector E(N);

    for (int i = 0; i < N; i++) {
        dcomplex phi = - I * gamma[i] * z;
        E[i] = FF[i] * exp(phi) + BB[i] * exp(-phi);
    }

    return this->diagonalizer->TE(this->stack[n]) * E;
}


template <typename GeometryT>
cvector ReflectionSolver<GeometryT>::getFieldVectorH(double z, int n)
{
    assert(fields_determined != DETERMINED_NOTHING);

    cvector& FF = fields[n].F;
    cvector& BB = fields[n].B;

    if (n >= this->interface) {
        z = - z;
        if (n != 0 && n != this->vbounds.size())
            z += this->vbounds[n] - this->vbounds[n-1];
    }

    cdiagonal gamma = this->diagonalizer->Gamma(this->stack[n]);

    int N = gamma.size();
    cvector H(N);

    for (int i = 0; i < N; i++) {
        dcomplex phi = - I * gamma[i] * z;
        H[i] = FF[i] * exp(phi) - BB[i] * exp(-phi);
    }

    if (emitting && (n == 0 || n == this->vbounds.size())) {
        // In the outer layers multiply H by -1 where necessary for propagating wave
        for (int i = 0; i < N; i++)
            if (real(gamma[i]) < -SMALL) H[i] = - H[i];
    }

    return this->diagonalizer->TH(this->stack[n]) * H;
}



template <typename GeometryT>
DataVector<Vec<3,dcomplex>> ReflectionSolver<GeometryT>::computeFieldE(const shared_ptr<const MeshD<GeometryT::DIM> > &dst_mesh, InterpolationMethod method)
{
    DataVector<Vec<3,dcomplex>> destination(dst_mesh->size());
    auto levels = makeLevelsAdapter<GeometryT::DIM>(dst_mesh);
    this->diagonalizer->source()->initField(Expansion::FieldParams::E, k0, klong, ktran, method);
    while (auto level = levels->yield()) {
        double z = level->vpos();
        size_t n = this->getLayerFor(z);
        cvector E = getFieldVectorE(z, n);
        cvector H = getFieldVectorH(z, n);
        if (n >= this->interface) for (auto& h: H) h = -h;
        size_t layer = this->stack[n];
        auto dest = this->diagonalizer->source()->getField(layer, level, E, H);
        for (size_t i = 0; i != level->size(); ++i) destination[level->index(i)] = dest[i];
    }
    this->diagonalizer->source()->cleanupField();
    return destination;
}


template <typename GeometryT>
DataVector<Vec<3,dcomplex>> ReflectionSolver<GeometryT>::computeFieldH(const shared_ptr<const MeshD<GeometryT::DIM>>& dst_mesh, InterpolationMethod method)
{
    DataVector<Vec<3,dcomplex>> destination(dst_mesh->size());
    auto levels = makeLevelsAdapter<GeometryT::DIM>(dst_mesh);
    this->diagonalizer->source()->initField(Expansion::FieldParams::H, k0, klong, ktran, method);
    while (auto level = levels->yield()) {
        double z = level->vpos();
        size_t n = this->getLayerFor(z);
        cvector E = getFieldVectorE(z, n);
        cvector H = getFieldVectorH(z, n);
        if (n >= this->interface) for (auto& h: H) h = -h;
        size_t layer = this->stack[n];
        auto dest = this->diagonalizer->source()->getField(layer, level, E, H);
        for (size_t i = 0; i != level->size(); ++i) destination[level->index(i)] = dest[i];
    }
    this->diagonalizer->source()->cleanupField();
    return destination;
}


template <typename GeometryT>
DataVector<double> ReflectionSolver<GeometryT>::computeFieldMagnitude(double power, const shared_ptr<const MeshD<GeometryT::DIM>>& dst_mesh, InterpolationMethod method)
{
    auto E = computeFieldE(dst_mesh, method);
    DataVector<double> result(E.size());
    for (size_t i = 0; i != E.size(); ++i) {
        result[i] = power * abs2(E[i]);
    }
    return result;
}


// cvector ReflectionSolver::getBackwardFieldVectorE(double z, int n)
// {
//     if (!fields_determined)
//         throw ComputationError(this->getId(), getBackwardFieldVectorE: Fields not determined);
//
//     cdiagonal gamma = this->diagonalizer->Gamma(this->stack[n]);
//     int N = gamma.size();
//
//     cvector FF = fields[n].F.copy();
//     for (int i = 0; i < N; i++)
//         if (real(gamma[i]) < -SMALL) FF[i] = fields[n].B[i];
//
//     cvector E(N);
//
//     z = this->stack[n].height - z;
//
//     for (int i = 0; i < N; i++) {
//         dcomplex phi = - I * gamma[i] * z;
//         E[i] = FF[i] * exp(phi);
//     }
//
//     return this->diagonalizer->TE(this->stack[n]) * E;
// }
//
// //**************************************************************************
// /// Return the electric field at the specific z point
// cvector ReflectionSolver::getForwardFieldVectorE(double z, int n)
// {
//     if (!fields_determined)
//         throw ComputationError(this->getId(), getForwardFieldVectorE: Fields not determined);
//
//     cdiagonal gamma = this->diagonalizer->Gamma(this->stack[n]);
//     int N = gamma.size();
//
//     cvector BB = fields[n].B.copy();
//     for (int i = 0; i < N; i++)
//         if (real(gamma[i]) < -SMALL) BB[i] = fields[n].F[i];
//
//     cvector E(N);
//
//     z = this->stack[n].height - z;
//
//     for (int i = 0; i < N; i++) {
//         dcomplex phi = - I * gamma[i] * z;
//         E[i] = BB[i] * exp(-phi);
//     }
//
//     return this->diagonalizer->TE(this->stack[n]) * E;
// }
//
// //--------------------------------------------------------------------------
// /// Return the electric field at the specific z point
// cvector ReflectionSolver::getBackwardFieldVectorH(double z, int n)
// {
//     if (!fields_determined)
//         throw ComputationError(this->getId(), getBackwardFieldVectorH: Fields not determined);
//
//     cdiagonal gamma = this->diagonalizer->Gamma(this->stack[n]);
//     int N = gamma.size();
//
//     cvector FF = fields[n].F.copy();
//     for (int i = 0; i < N; i++)
//         if (real(gamma[i]) < -SMALL) FF[i] = fields[n].B[i];
//
//     cvector H(N);
//
//     z = this->stack[n].height - z;
//
//     for (int i = 0; i < N; i++) {
//         dcomplex phi = - I * gamma[i] * z;
//         H[i] = FF[i] * exp(phi);
//     }
//
//     return this->diagonalizer->TH(this->stack[n]) * H;
// }
//
//
// //--------------------------------------------------------------------------
// /// Return the electric field at the specific z point
// cvector ReflectionSolver::getForwardFieldVectorH(double z, int n)
// {
//     if (!fields_determined)
//         throw ComputationError(this->getId(), getForwardFieldVectorH: Fields not determined);
//
//     cdiagonal gamma = this->diagonalizer->Gamma(this->stack[n]);
//     int N = gamma.size();
//
//     cvector BB = fields[n].B.copy();
//     for (int i = 0; i < N; i++)
//         if (real(gamma[i]) < -SMALL) BB[i] = fields[n].F[i];
//
//     cvector H(N);
//
//     z = this->stack[n].height - z;
//
//     for (int i = 0; i < N; i++) {
//         dcomplex phi = - I * gamma[i] * z;
//         H[i] = - BB[i] * exp(-phi);
//     }
//
//     return this->diagonalizer->TH(this->stack[n]) * H;
// }
//
//
//
// //**************************************************************************
// /// Return the intensity of the backward propagating field for a given layer
// double ReflectionSolver::backwardEIntegral(int n)
// {
//     if (!fields_determined)
//         throw ComputationError(this->getId(), backwardFieldIntegral: Fields not determined);
//
//     int layer = this->stack[n];
//
//     cdiagonal gamma = this->diagonalizer->Gamma(layer);
//     int N = this->diagonalizer->matrixSize();
//     int N0 = this->diagonalizer->source()->matrixSize();
//     double H = this->stack[n].height;
//
//     cvector E(N0), F;
//
//     if (n < interface) {
//         F = fields[n].F.copy();
//         for (int i = 0; i < N; i++)
//             if (real(gamma[i]) < -SMALL) F[i] = fields[n].B[i];
//     } else {
//         F = fields[n].B.copy();
//         for (int i = 0; i < N; i++)
//             if (real(gamma[i]) < -SMALL) F[i] = fields[n].F[i];
//     }
//
//     mult_matrix_by_vector(this->diagonalizer->TE(layer), F, E);
//
//     double integral = 0.;
//
//     for (int i = 0; i < N0; i++) {
//         integral += 0.5 * real(conj(E[i])* E[i]);
//     }
//
//     return integral;
// }
//
// //--------------------------------------------------------------------------
// /// Return the intensity of the forward propagating field for a given layer
// double ReflectionSolver::forwardEIntegral(int n)
// {
//     if (!fields_determined)
//         throw ComputationError(this->getId(), forwardFieldIntegral: Fields not determined);
//
//     int layer = this->stack[n];
//
//     cdiagonal gamma = this->diagonalizer->Gamma(layer);
//     int N = this->diagonalizer->matrixSize();
//     int N0 = this->diagonalizer->source()->matrixSize();
//     double H = this->stack[n].height;
//
//     cvector E(N0), B;
//
//     if (n < interface) {
//         B = fields[n].B.copy();
//         for (int i = 0; i < N; i++)
//             if (real(gamma[i]) < -SMALL) B[i] = fields[n].F[i];
//     } else {
//         B = fields[n].F.copy();
//         for (int i = 0; i < N; i++)
//             if (real(gamma[i]) < -SMALL) B[i] = fields[n].B[i];
//     }
//
//     mult_matrix_by_vector(this->diagonalizer->TE(layer), B, E);
//
//     double integral = 0.;
//
//     for (int i = 0; i < N0; i++) {
//         integral += 0.5 * real(conj(E[i])* E[i]);
//     }
//
//     return integral;
// }




//
// //******************************************************************************
// // Return the electric field components on a given grid
// void SlabSolverBase::backwardFieldE(const vector<double> X, const vector<double> Y, const vector<double> Z,
//                            dcomplex* outVectorField3D)
// {
//     int zsize = Z.size();
//     int xysize = X.size() * Y.size();
//     for (int i = 0; i < zsize; i++) {
//         double z = Z[i];
//         int n = layer_of_z(z);
//         int layer = this->stack[n];
//         dcomplex *field = outVectorField3D + 3 * i*xysize;
//         cvector E, H;
//         if (n >= interface || (interface==count && Z[i]>0)) {
//             E = getForwardFieldVectorE(z, n);
//             H = getForwardFieldVectorH(z, n);
//             int N = H.size(); for (int j = 0; j < N; j++) H[j] = -H[j];
//         } else {
//             E = getBackwardFieldVectorE(z, n);
//             H = getBackwardFieldVectorH(z, n);
//         }
//         this->diagonalizer->source()->fieldE(layer, X, Y, K0, Kx, Ky, E, H, field);
//     }
// }
//
// //------------------------------------------------------------------------------
// // Return the magnetic field components on a given grid
// void SlabSolverBase::backwardFieldH(const vector<double> X, const vector<double> Y, const vector<double> Z,
//                            dcomplex* outVectorField3D)
// {
//     int zsize = Z.size();
//     int xysize = X.size() * Y.size();
//     for (int i = 0; i < zsize; i++) {
//         double z = Z[i];
//         int n = layer_of_z(z);
//         int layer = this->stack[n];
//         dcomplex *field = outVectorField3D + 3 * i*xysize;
//         cvector E, H;
//         if (n >= interface || (interface==count && Z[i]>0)) {
//             E = getForwardFieldVectorE(z, n);
//             H = getForwardFieldVectorH(z, n);
//             int N = H.size(); for (int j = 0; j < N; j++) H[j] = -H[j];
//         } else {
//             E = getBackwardFieldVectorE(z, n);
//             H = getBackwardFieldVectorH(z, n);
//         }
//         this->diagonalizer->source()->fieldH(layer, X, Y, K0, Kx, Ky, E, H, field);
//     }
// }
//
// //******************************************************************************
//
// //******************************************************************************
// // Return the electric field components on a given grid
// void SlabSolverBase::forwardFieldE(const vector<double> X, const vector<double> Y, const vector<double> Z,
//                            dcomplex* outVectorField3D)
// {
//     int zsize = Z.size();
//     int xysize = X.size() * Y.size();
//     for (int i = 0; i < zsize; i++) {
//         double z = Z[i];
//         int n = layer_of_z(z);
//         int layer = this->stack[n];
//         dcomplex *field = outVectorField3D + 3 * i*xysize;
//         cvector E, H;
//         if (n >= interface || (interface==count && Z[i]>0)) {
//             E = getBackwardFieldVectorE(z, n);
//             H = getBackwardFieldVectorH(z, n);
//             int N = H.size(); for (int j = 0; j < N; j++) H[j] = -H[j];
//         } else {
//             E = getForwardFieldVectorE(z, n);
//             H = getForwardFieldVectorH(z, n);
//         }
//         this->diagonalizer->source()->fieldE(layer, X, Y, K0, Kx, Ky, E, H, field);
//     }
// }
//
// //------------------------------------------------------------------------------
// // Return the magnetic field components on a given grid
// void SlabSolverBase::forwardFieldH(const vector<double> X, const vector<double> Y, const vector<double> Z,
//                            dcomplex* outVectorField3D)
// {
//     int zsize = Z.size();
//     int xysize = X.size() * Y.size();
//     for (int i = 0; i < zsize; i++) {
//         double z = Z[i];
//         int n = layer_of_z(z);
//         int layer = this->stack[n];
//         dcomplex *field = outVectorField3D + 3 * i*xysize;
//         cvector E, H;
//         if (n >= interface || (interface==count && Z[i]>0)) {
//             E = getBackwardFieldVectorE(z, n);
//             H = getBackwardFieldVectorH(z, n);
//             int N = H.size(); for (int j = 0; j < N; j++) H[j] = -H[j];
//         } else {
//             E = getForwardFieldVectorE(z, n);
//             H = getForwardFieldVectorH(z, n);
//         }
//         this->diagonalizer->source()->fieldH(layer, X, Y, K0, Kx, Ky, E, H, field);
//     }
// }
//








// // Return the electic field components on a given grid
// void SlabSolverBase::interfaceFieldE(const std::vector<double> X, const std::vector<double> Y,
//                                     dcomplex* outVectorField2D)
// {
//     int layer = this->stack[interface-1];
//
//     int N = this->diagonalizer->matrixSize();
//     int N0 = this->diagonalizer->source()->matrixSize();
//
//     cvector tmp1(N), E, tmp2(N), H(N0);
//     E = getInterfaceVector(K0, Kx, Ky);
//
//     // H = TH * A * invTE * E
//     mult_matrix_by_vector(this->diagonalizer->invTE(this->stack[interface-1]), E, tmp1);
//     mult_matrix_by_vector(A, tmp1, tmp2);
//     mult_matrix_by_vector(this->diagonalizer->TH(this->stack[interface-1]), tmp2, H);
//
//     this->diagonalizer->source()->fieldE(layer, X, Y, K0, Kx, Ky, E, H, outVectorField2D);
// }
//
// // Return the magnetic field components on a given grid
// void SlabSolverBase::interfaceFieldH(const std::vector<double> X, const std::vector<double> Y,
//                                     dcomplex* outVectorField2D)
// {
//     int layer = this->stack[interface-1];
//
//     int N = this->diagonalizer->matrixSize();
//     int N0 = this->diagonalizer->source()->matrixSize();
//
//     cvector tmp1(N), E, tmp2(N), H(N0);
//     E = getInterfaceVector(K0, Kx, Ky);
//
//     // H = TH * A * invTE * E
//     mult_matrix_by_vector(this->diagonalizer->invTE(this->stack[interface-1]), E, tmp1);
//     mult_matrix_by_vector(A, tmp1, tmp2);
//     mult_matrix_by_vector(this->diagonalizer->TH(this->stack[interface-1]), tmp2, H);
//
//     this->diagonalizer->source()->fieldH(layer, X, Y, K0, Kx, Ky, E, H, outVectorField2D);
// }

template struct PLASK_SOLVER_API ReflectionSolver<Geometry2DCartesian>;
template struct PLASK_SOLVER_API ReflectionSolver<Geometry2DCylindrical>;
template struct PLASK_SOLVER_API ReflectionSolver<Geometry3D>;

}}} // namespace plask::solvers::slab
