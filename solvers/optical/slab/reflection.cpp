#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define BOOST_USE_WINDOWS_H
#endif
#include <boost/thread/mutex.hpp>

#include "reflection.h"
#include "solver.h"
#include "fortran.h"
#include "meshadapter.h"

namespace plask { namespace solvers { namespace slab {

ReflectionTransfer::ReflectionTransfer(SlabBase* solver, Expansion& expansion): Transfer(solver, expansion)
{
    writelog(LOG_DETAIL, "Initializing reflection transfer");
    int N = diagonalizer->matrixSize();
    P = cmatrix(N,N);
    phas = cdiagonal(N);
    ipiv = aligned_new_array<int>(N);
    allP = false;
}


ReflectionTransfer::~ReflectionTransfer() {
    int N = diagonalizer->matrixSize();
    aligned_delete_array<int>(N, ipiv); ipiv = nullptr;
}


void ReflectionTransfer::getAM(size_t start, size_t end, bool add, double mfac)
{
    // Get matrices sizes
    int N0 = diagonalizer->source()->matrixSize();
    int N = diagonalizer->matrixSize(); // <= N0
    int NN = N*N;
    cmatrix wrk(N, N0, work);    // matrix object for the workspace

    findReflection(start, end, false);

    cdiagonal gamma = diagonalizer->Gamma(solver->stack[end]);

    double H = (end == 0 || end == solver->vbounds.size())?
               0 : abs(solver->vbounds[end] - solver->vbounds[end-1]);
    for (int i = 0; i < N; i++) phas[i] = exp(-I*gamma[i]*H);

    mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);     // P = phas * P * phas
    memcpy(temp.data(), P.data(), NN*sizeof(dcomplex));                     // temp = P

    // temp = [ phas*P*phas - I ] [ phas*P*phas + I ]^{-1}
    for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] += 1;            // P = P + I
    for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) temp[ii] -= 1;         // temp = temp - I
    int info;
    zgetrf(N, N, P.data(), N, ipiv, info);                                  // P = LU(P)
    ztrsm('R', 'U', 'N', 'N', N, N, 1., P.data(), N, temp.data(), N);       // temp = temp * U^{-1}
    ztrsm('R', 'L', 'N', 'U', N, N, 1., P.data(), N, temp.data(), N);       // temp = temp * L^{-1}
    // reorder columns (there is no such function in LAPACK)
    for (int j = N-1; j >=0 ; j--) {
        int jp = ipiv[j]-1;
        for (int i = 0; i < N; i++) std::swap(temp(i,j), temp(i,jp));
    }

    // M for the half of the structure
    mult_matrix_by_matrix(temp, diagonalizer->invTE(solver->stack[end]), wrk);// wrk = temp * invTE[end]
    zgemm('N','N', N0, N0, N, mfac, diagonalizer->TH(solver->stack[end]).data(), N0,
          work, N, add?1.:0., M.data(), N0);                                 // M = mfac * TH[end] * wrk
}


void ReflectionTransfer::findReflection(int start, int end, bool emitting)
{
    // Should be called from 0 to interface-1
    // and from count-1 to interface

    write_debug("%s: searching for reflection for layers %d to %d", solver->getId(), start, end);

    const int inc = (start < end) ? 1 : -1;

    int N0 = diagonalizer->source()->matrixSize();
    int N = diagonalizer->matrixSize();
    int NN = N*N;

    cmatrix wrk(N, N0, work);    // matrix object for the workspace

    cdiagonal gamma;

    // in the beginning the P matrix is zero
    memset(P.data(), 0, NN*sizeof(dcomplex));

    std::exception_ptr error;

    #ifdef OPENMP_FOUND
        std::vector<boost::mutex> layer_locks(diagonalizer->lcount);
        for (boost::mutex& mutex: layer_locks) mutex.lock();
        #ifndef NDEBUG
            std::vector<bool> layer_accessed(diagonalizer->lcount, false);
        #endif
    #endif

    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic,1) nowait
        for (int l = 0; l < diagonalizer->lcount; ++l) {
            try {
                if (!error && diagonalizer->diagonalizeLayer(l)) {
                    #ifdef OPENMP_FOUND
                        layer_locks[l].unlock();
                        write_debug("%s: layer %d diagonalized", solver->getId(), l);
                    #endif
                }
            } catch(...) {
                error = std::current_exception();
            }
        }

        #pragma omp single
        if (!error) try {

            #ifdef OPENMP_FOUND
                write_debug("%s: entering into single region of reflection search in thread %d", solver->getId(), omp_get_thread_num());
                layer_locks[solver->stack[start]].lock(); layer_locks[solver->stack[start]].unlock();
                #ifndef NDEBUG
                    write_debug("%s: using diagonalized layer %d", solver->getId(), solver->stack[start]);
                    layer_accessed[solver->stack[start]] = true;
                #endif
            #endif

            // If we do not use emitting, we have to set field at the edge to 0 and the apply PML
            if (!emitting) {

                gamma = diagonalizer->Gamma(solver->stack[start]);

                // Aply PML
                // F(0) + B(0) = 0 ==> P(0) = -I
                for (int i = 0; i < N; i++) {
                    int s = 1;
                    dcomplex g = gamma[i] * solver->vpml.factor;
                    if (real(g) < -SMALL) { g = -g; s = -s; }
                    if (imag(g) > SMALL) { g = -g; s = -s; }
                    dcomplex p = - exp(-2.*I*g*solver->vpml.size);
                    if (s == 1) P(i,i) = p; // P = phas * (-I) * phas
                    else P(i,i) = 1./p;     // P = inv(phas * (-I) * phas)
                }
                assert(!P.isnan());

                // Shift matrix by `pmlshift`
                for (int i = 0; i < N; i++) phas[i] = exp(-I*gamma[i]*solver->vpml.shift);
                assert(!phas.isnan());
                mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas); // P = phas * P * phas
            }

            storeP(start);

            for (int n = start; n != end; n += inc) {
                gamma = diagonalizer->Gamma(solver->stack[n]);
                assert(!gamma.isnan());

                assert(!P.isnan());

                if (n != start) {
                    double H = solver->vbounds[n] - solver->vbounds[n-1];
                    for (int i = 0; i < N; i++) phas[i] = exp(-I*gamma[i]*H);
                    assert(!phas.isnan());
                    mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);         // P = phas * P * phas
                }

                // temp = invTE(n+1)*TE(n) * [ phas*P*phas + I ]
                assert(!diagonalizer->TE(solver->stack[n]).isnan());
                assert(!diagonalizer->invTE(solver->stack[n]).isnan());
                for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] += 1.;               // P = P.orig + I
                mult_matrix_by_matrix(diagonalizer->TE(solver->stack[n]), P, wrk);          // wrk = TE[n] * P
                #ifdef OPENMP_FOUND
                    layer_locks[solver->stack[n+inc]].lock(); layer_locks[solver->stack[n+inc]].unlock();
                    #ifndef NDEBUG
                        if (!layer_accessed[solver->stack[n+inc]]) {
                            write_debug("%s: using diagonalized layer %d", solver->getId(), solver->stack[n+inc]);
                            layer_accessed[solver->stack[n+inc]] = true;
                        }
                    #endif
                #endif
                mult_matrix_by_matrix(diagonalizer->invTE(solver->stack[n+inc]), wrk, temp);// temp = invTE[n+1] * wrk (= A)

                // P = invTH(n+1)*TH(n) * [ phas*P*phas - I ]
                assert(!diagonalizer->TH(solver->stack[n]).isnan());
                assert(!diagonalizer->invTH(solver->stack[n]).isnan());
                for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] -= 2.;               // P = P - I

                // multiply rows of P by -1 where necessary for properly outgoing wave
                if (emitting && n == start) {
                    for (int i = 0; i < N; i++)
                        if (real(gamma[i]) < -SMALL)
                            for(int j = 0; j < N; j++) P(i,j) = -P(i,j);
                }

                mult_matrix_by_matrix(diagonalizer->TH(solver->stack[n]), P, wrk);          // wrk = TH[n] * P
                mult_matrix_by_matrix(diagonalizer->invTH(solver->stack[n+inc]), wrk, P);   // P = invTH[n+1] * wrk (= P)

                // temp := temp-P, P := temp+P
                for (int i = 0; i < NN; i++) {
                    dcomplex e = temp[i], h = P[i];
                    temp[i] = e - h;
                    P[i] = e + h;
                }

                // P = P * inv(temp)
                int info;
                zgetrf(N, N, temp.data(), N, ipiv, info);                                   // temp = LU(temp)
                if (info > 0) throw ComputationError(solver->getId(), "findReflection: Matrix [e(n) - h(n)] is singular");
                assert(info == 0);
                ztrsm('R', 'U', 'N', 'N', N, N, 1., temp.data(), N, P.data(), N);           // P = P * U^{-1}
                ztrsm('R', 'L', 'N', 'U', N, N, 1., temp.data(), N, P.data(), N);           // P = P * L^{-1}
                if (P.isnan()) throw ComputationError(solver->getId(), "findReflection: NaN in reflection matrix");
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


void ReflectionTransfer::storeP(size_t n) {
    if (allP) {
        int N = diagonalizer->matrixSize();
        if (memP.size() != solver->stack.size()) {
            // Allocate the storage for admittance matrices
            memP.resize(solver->stack.size());
            for (int i = 0; i < solver->stack.size(); i++) memP[i] = cmatrix(N,N);
        }
        memcpy(memP[n].data(), P.data(), N*N*sizeof(dcomplex));
    }
}


cvector ReflectionTransfer::getReflectionVector(const cvector& incident, IncidentDirection side)
{
    int last, first;

    initDiagonalization();
    switch (side) {
        case INCIDENCE_TOP:
            last = 0; first = solver->stack.size()-1; break;
        case INCIDENCE_BOTTOM:
            last = solver->stack.size()-1; first = 0; break;
    }
    findReflection(last, first, true);
    return P * incident;
}


cvector ReflectionTransfer::getTransmissionVector(const cvector& incident, IncidentDirection side)
{
    determineReflectedFields(incident, side);
    size_t n = (side == INCIDENCE_BOTTOM)? solver->stack.size()-1 : 0;
    return diagonalizer->TE(solver->stack[n]) * fields[n].B;
}


void ReflectionTransfer::determineFields()
{
    if (fields_determined == DETERMINED_RESONANT) return;

    writelog(LOG_DETAIL, solver->getId() + ": Determining optical fields");

    int N = diagonalizer->matrixSize();
    int N0 = diagonalizer->source()->matrixSize();
    int NN = N*N;

    cdiagonal gamma;

    size_t count = solver->stack.size();

    // Assign the space for the field vectors
    fields.resize(count);
    // Obtain the physical fields at the last layer
    allP = true; interface_field = nullptr;

    auto E = getInterfaceVector();
    cvector temp(work, N);

    for (int pass = 0; pass < 1 || (pass < 2 && solver->interface != count); pass++)
    {
        // each pass for below and above the interface

        int start, end, inc;
        switch (pass)
        {
            case 0: start = solver->interface-1; end = 0;       inc = -1; break;
            case 1: start = solver->interface;   end = count-1; inc = +1; break;
        }

        fields[start].F = cvector(N);
        fields[start].B = cvector(N);

        // compute B-field for the layer next to the interface
        int curr = solver->stack[start];

        gamma = diagonalizer->Gamma(curr);

        double H = (start == 0 || start == count-1)? 0 : (solver->vbounds[start] - solver->vbounds[start-1]);
        for (int i = 0; i < N; i++)
            phas[i] = exp(-I*gamma[i]*H);

        // P = phas*P*phas + I
        memcpy(P.data(), memP[start].data(), NN*sizeof(dcomplex));
        mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);         // P := phas * P * phas
        for (int i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] += 1.;               // P := P + I

        mult_matrix_by_vector(diagonalizer->invTE(curr), E, fields[start].B);       // B := invTE * E
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

            curr = solver->stack[n];
            int next = solver->stack[n+inc];

            gamma = diagonalizer->Gamma(next);

            for (int i = 0; i < N; i++) F2[i] = F1[i] - B1[i];              // F2 := F1 - B1
            mult_matrix_by_vector(diagonalizer->TH(curr), F2, temp);         // temp := TH * F2
            mult_matrix_by_vector(diagonalizer->invTH(next), temp, B2);      // B2 := invTH * temp

            // // multiply rows of invTH by -1 where necessary for the outer layer
            // if (n+inc == end) {
            //     for (int i = 0; i < N; i++)
            //         if (real(gamma[i]) < -SMALL) B2[i] = -B2[i];
            // }

            for (int i = 0; i < N; i++) F2[i] = F1[i] + B1[i];              // F2 := F1 + B1
            mult_matrix_by_vector(diagonalizer->TE(curr), F2, temp);         // temp := TE * F2
            zgemm('N','N', N, 1, N0, 1., diagonalizer->invTE(next).data(), N,
                  temp.data(), N0, -1., B2.data(), N);                       // B2 := invTE * temp - B2

            H = (n+inc == end)? 0 : (solver->vbounds[n+inc] - solver->vbounds[n+inc-1]);
            // if (n+inc != end) {
                for (int i = 0; i < N; i++)
                    B2[i] *= 0.5 * exp(-I*gamma[i]*H);                      // B2 := 1/2 * phas * B2
            // } else {
            //     for (int i = 0; i < N; i++) {
            //         dcomplex g = gamma[i];
            //         if (real(g) < -SMALL) g = -g;
            //         B2[i] *= 0.5 * exp(-I*g*H);                             // B2 := 1/2 * phas * B2
            //     }
            // }
        }

        mult_matrix_by_vector(memP[end], fields[end].B, fields[end].F);
        // for (int i = 0; i < N; ++i) fields[end].F[i] = 0.; // in the outer layer there is no incoming field
        // // In the outer layers replace F and B where necessary for consistent gamma handling
        // cvector& F2 = fields[end].F;
        // cvector& B2 = fields[end].B;
        // for (int i = 0; i < N; i++)
        //     if (real(gamma[i]) < -SMALL) std::swap(F2[i], B2[i]);
    }

    allP = false;
    fields_determined = DETERMINED_RESONANT;
}

void ReflectionTransfer::determineReflectedFields(const cvector& incident, IncidentDirection side)
{
    if (fields_determined == DETERMINED_REFLECTED) return;

    writelog(LOG_DETAIL, solver->getId() + ": Determining reflected optical fields");

    size_t count = solver->stack.size();

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
    findReflection(end, start, true);

    // Temporary and initial data
    int N = diagonalizer->matrixSize();
    int N0 = diagonalizer->source()->matrixSize();
    cvector temp(work, N);
    cdiagonal gamma;

    int curr = solver->stack[start];
    double H;

    fields[start].B = diagonalizer->invTE(curr) * incident; // diagonalized incident E-field
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

        curr = solver->stack[n];
        int next = solver->stack[n+inc];

        gamma = diagonalizer->Gamma(next);

        for (int i = 0; i < N; i++) F2[i] = F1[i] - B1[i];              // F2 := F1 - B1
        mult_matrix_by_vector(diagonalizer->TH(curr), F2, temp);         // temp := TH * F2
        mult_matrix_by_vector(diagonalizer->invTH(next), temp, B2);      // B2 := invTH * temp
        // multiply rows of invTH by -1 where necessary for the outer layer
        if (n+inc == end) {
            for (int i = 0; i < N; i++)
                if (real(gamma[i]) < -SMALL) B2[i] = -B2[i];
        }

        for (int i = 0; i < N; i++) F2[i] = F1[i] + B1[i];              // F2 := F1 + B1
        mult_matrix_by_vector(diagonalizer->TE(curr), F2, temp);         // temp := TE * F2
        zgemm('N','N', N, 1, N0, 1., diagonalizer->invTE(next).data(), N,
              temp.data(), N0, -1., B2.data(), N);                       // B2 := invTE * temp - B2

        H = (n+inc != end)? solver->vbounds[n+inc] - solver->vbounds[n+inc-1] : 0.;
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
        gamma = diagonalizer->Gamma(solver->stack[n]);
        for (int i = 0; i < N; i++) {
            if (real(gamma[i]) < -SMALL)
                std::swap(fields[n].F, fields[n].B);
        }
     }

    // Replace F and B before the interface for consistency in getFieldVectorE and getFieldVectorH
    switch (side)
    {
        case INCIDENCE_TOP:    start = solver->interface; end = count; break;
        case INCIDENCE_BOTTOM: start = 0; end = solver->interface; break;
    }
    for (int n = start; n < end; n++) {
        cvector& F2 = fields[n].F;
        cvector& B2 = fields[n].B;
        gamma = diagonalizer->Gamma(solver->stack[n]);
        H = (n < count-1)? solver->vbounds[n] - solver->vbounds[n-1] : 0.;
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


cvector ReflectionTransfer::getFieldVectorE(double z, int n)
{
    assert(fields_determined != DETERMINED_NOTHING);

    cvector& FF = fields[n].F;
    cvector& BB = fields[n].B;

    if (n >= solver->interface) {
        z = - z;
        if (n != 0 && n != solver->vbounds.size())
            z += solver->vbounds[n] - solver->vbounds[n-1];
    }

    cdiagonal gamma = diagonalizer->Gamma(solver->stack[n]);

    int N = gamma.size();
    cvector E(N);

    for (int i = 0; i < N; i++) {
        dcomplex phi = - I * gamma[i] * z;
        E[i] = FF[i] * exp(phi) + BB[i] * exp(-phi);
    }

    return diagonalizer->TE(solver->stack[n]) * E;
}


cvector ReflectionTransfer::getFieldVectorH(double z, int n)
{
    assert(fields_determined != DETERMINED_NOTHING);

    cvector& FF = fields[n].F;
    cvector& BB = fields[n].B;

    if (n >= solver->interface) {
        z = - z;
        if (n != 0 && n != solver->vbounds.size())
            z += solver->vbounds[n] - solver->vbounds[n-1];
    }

    cdiagonal gamma = diagonalizer->Gamma(solver->stack[n]);

    int N = gamma.size();
    cvector H(N);

    for (int i = 0; i < N; i++) {
        dcomplex phi = - I * gamma[i] * z;
        H[i] = FF[i] * exp(phi) - BB[i] * exp(-phi);
    }

    if (n == 0 || n == solver->vbounds.size()) {
        // In the outer layers multiply H by -1 where necessary for propagating wave
        for (int i = 0; i < N; i++)
            if (real(gamma[i]) < -SMALL) H[i] = - H[i];
    }

    return diagonalizer->TH(solver->stack[n]) * H;
}



}}} // namespace plask::solvers::slab
