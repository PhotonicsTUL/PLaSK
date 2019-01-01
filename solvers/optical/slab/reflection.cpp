/*#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define BOOST_USE_WINDOWS_H
#endif*/

#include "reflection.h"
#include "solver.h"
#include "diagonalizer.h"
#include "expansion.h"
#include "fortran.h"
#include "meshadapter.h"

#include <plask/utils/openmp.h>

namespace plask { namespace optical { namespace slab {

ReflectionTransfer::ReflectionTransfer(SlabBase* solver, Expansion& expansion): Transfer(solver, expansion),
    storeP(STORE_NONE) {
    writelog(LOG_DETAIL, "{}: Initializing Reflection Transfer", solver->getId());
    size_t N = diagonalizer->matrixSize();
    P = cmatrix(N,N);
    phas = cdiagonal(N);
    ipiv = aligned_new_array<int>(N);
}


ReflectionTransfer::~ReflectionTransfer() {
    size_t N = diagonalizer->matrixSize();
    aligned_delete_array<int>(N, ipiv); ipiv = nullptr;
}


void ReflectionTransfer::getFinalMatrix() {
    getAM(0, solver->interface-1, false);
    getAM(solver->stack.size()-1, solver->interface, true);
}


void ReflectionTransfer::getAM(size_t start, size_t end, bool add, double mfac)
{
    // Get matrices sizes
    const std::size_t N0 = diagonalizer->source()->matrixSize();
    const std::size_t N = diagonalizer->matrixSize(); // <= N0
    const std::size_t NN = N*N;
    cmatrix work(N, N0, wrk);    // matrix object for the workspace

    findReflection(start, end, false, int(add)&1);

    cdiagonal gamma = diagonalizer->Gamma(solver->stack[end]);

    double H = (end == 0 || end == solver->vbounds->size())?
               0 : abs(solver->vbounds->at(end) - solver->vbounds->at(end-1));
    for (std::size_t i = 0; i < N; i++) phas[i] = exp(-I*gamma[i]*H);

    mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);         // P = phas * P * phas
    memcpy(temp.data(), P.data(), NN*sizeof(dcomplex));                         // temp = P

    // temp = [ phas*P*phas - I ] [ phas*P*phas + I ]^{-1}
    for (std::size_t i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] += 1;        // P = P + I
    for (std::size_t i = 0, ii = 0; i < N; i++, ii += (N+1)) temp[ii] -= 1;     // temp = temp - I
    int info;
    zgetrf(int(N), int(N), P.data(), int(N), ipiv, info);                       // P = LU(P)
    ztrsm('R', 'U', 'N', 'N', int(N), int(N), 1., P.data(), int(N), temp.data(), int(N));  // temp = temp * U^{-1}
    ztrsm('R', 'L', 'N', 'U', int(N), int(N), 1., P.data(), int(N), temp.data(), int(N));  // temp = temp * L^{-1}
    // reorder columns (there is no such function in LAPACK)
    for (std::ptrdiff_t j = N-1; j >=0; j--) {
        int jp = ipiv[j]-1;
        for (std::size_t i = 0; i < N; i++) std::swap(temp(i,j), temp(i,jp));
    }

    // M for the half of the structure
    mult_matrix_by_matrix(temp, diagonalizer->invTE(solver->stack[end]), work); // work = temp * invTE[end]
    zgemm('N','N', int(N0), int(N0), int(N), mfac, diagonalizer->TH(solver->stack[end]).data(), int(N0),
          wrk, int(N), add?1.:0., M.data(), int(N0));                           // M = mfac * TH[end] * work
}


void ReflectionTransfer::findReflection(std::size_t start, std::size_t end, bool emitting, int store)
{
    // Should be called from 0 to interface-1
    // and from count-1 to interface

    write_debug("{}: searching for reflection for layers {:d} to {:d}", solver->getId(), start, end);

    const std::ptrdiff_t inc = (start < end) ? 1 : -1;

    const std::size_t N0 = diagonalizer->source()->matrixSize();
    const std::size_t N = diagonalizer->matrixSize();
    const std::size_t NN = N*N;

    cmatrix work(N, N0, wrk);    // matrix object for the workspace

    cdiagonal gamma;

    // in the beginning the P matrix is zero
    std::fill_n(P.data(), NN, dcomplex(0.0));

    std::exception_ptr error;

    #pragma omp parallel for schedule(dynamic,1)
    for (int l = 0; l < int(diagonalizer->lcount); ++l) {
        try {
            if (!error) diagonalizer->diagonalizeLayer(l);
        } catch(...) {
            error = std::current_exception();
        }
    }
    if (error) std::rethrow_exception(error);

    #ifdef OPENMP_FOUND
        write_debug("{}: Entering into single region of reflection search", solver->getId());
    #endif

    // If we do not use emitting, we have to set field at the edge to 0 and the apply PML
    if (!emitting) {
        gamma = diagonalizer->Gamma(solver->stack[start]);
        // Aply PML
        // F(0) + B(0) = 0 ==> P(0) = -I
        for (std::size_t i = 0; i < N; i++) {
            dcomplex g = gamma[i] * solver->vpml.factor;
            P(i,i) = - exp(-2. * I * g * solver->vpml.size);                // P = phas * (-I) * phas
        }
        assert(!P.isnan());

        // Shift matrix by `pmldist`
        for (std::size_t i = 0; i < N; i++) phas[i] = exp(-I*gamma[i]*solver->vpml.dist);
        assert(!phas.isnan());
        mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas); // P = phas * P * phas
    }

    if (storeP == STORE_ALL) saveP(start);

    for (std::size_t n = start; n != end; n += inc) {
        gamma = diagonalizer->Gamma(solver->stack[n]);
        assert(!gamma.isnan());

        assert(!P.isnan());

        if (n != start) {
            double H = solver->vbounds->at(n) - solver->vbounds->at(n-1);
            for (std::size_t i = 0; i < N; i++) phas[i] = exp(-I*gamma[i]*H);
            assert(!phas.isnan());
            mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);         // P = phas * P * phas
        }

        // Further calculations must be done only if the adjacent layers are not the same
        if (solver->stack[n] != solver->stack[n+inc] || (emitting && n == start)) {
            // temp = invTE(n+1)*TE(n) * [ phas*P*phas + I ]
            assert(!diagonalizer->TE(solver->stack[n]).isnan());
            assert(!diagonalizer->invTE(solver->stack[n]).isnan());
            for (std::size_t i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] += 1.;               // P = P + I
            if (solver->stack[n] != solver->stack[n+inc]) {
                mult_matrix_by_matrix(diagonalizer->TE(solver->stack[n]), P, work);     // work = TE[n] * P
                mult_matrix_by_matrix(diagonalizer->invTE(solver->stack[n+inc]), work, temp);// temp = invTE[n+1] * work (= A)
            } else {
                std::copy_n(P.data(), NN, temp.data());
            }

            // P = invTH(n+1)*TH(n) * [ phas*P*phas - I ]
            assert(!diagonalizer->TH(solver->stack[n]).isnan());
            assert(!diagonalizer->invTH(solver->stack[n+inc]).isnan());
            for (std::size_t i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] -= 2.;               // P = P - I

            // multiply rows of P by -1 where necessary for properly outgoing wave
            if (emitting && n == start) {
                for (std::size_t i = 0; i < N; i++)
                    if (real(gamma[i]) < -SMALL)
                        for(std::size_t j = 0; j < N; j++) P(i,j) = -P(i,j);
            }

            if (solver->stack[n] != solver->stack[n+inc]) {
                mult_matrix_by_matrix(diagonalizer->TH(solver->stack[n]), P, work);     // work = TH[n] * P
                mult_matrix_by_matrix(diagonalizer->invTH(solver->stack[n+inc]), work, P);// P = invTH[n+1] * work (= P)
            }

            // temp := temp-P, P := temp+P
            for (std::size_t i = 0; i < NN; i++) {
                dcomplex e = temp[i], h = P[i];
                temp[i] = e - h;
                P[i] = e + h;
            }

            // P = P * inv(temp)
            int info;
            zgetrf(int(N), int(N), temp.data(), int(N), ipiv, info);                                   // temp = LU(temp)
            if (info > 0) throw ComputationError(solver->getId(), "findReflection: Matrix [e(n) - h(n)] is singular");
            assert(info == 0);
            ztrsm('R', 'U', 'N', 'N', int(N), int(N), 1., temp.data(), int(N), P.data(), int(N));    // P = P * U^{-1}
            ztrsm('R', 'L', 'N', 'U', int(N), int(N), 1., temp.data(), int(N), P.data(), int(N));           // P = P * L^{-1}
            if (P.isnan()) throw ComputationError(solver->getId(), "findReflection: NaN in reflection matrix");
            // reorder columns (there is no such function in LAPACK)
            for (std::ptrdiff_t j = N-1; j >= 0; j--) {
                int jp = ipiv[j]-1;
                for (std::size_t i = 0; i < N; i++) std::swap(P(i,j), P(i,jp));
            }
        }

        if (storeP == STORE_ALL) saveP(n+inc);
    }
    if (storeP == STORE_LAST) saveP(store);
}


cvector ReflectionTransfer::getReflectionVector(const cvector& incident, IncidentDirection side)
{
    std::size_t last, first;

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
    ptrdiff_t n = (side == INCIDENCE_BOTTOM)? solver->stack.size()-1 : 0;
    return
        (((side == INCIDENCE_BOTTOM && n < solver->interface) ||
         (side == INCIDENCE_TOP && n >= solver->interface))?
         fields[n].F : fields[n].B);
}


// Some aliases
#define F1 fields[n].F
#define B1 fields[n].B
#define F2 fields[n+inc].F
#define B2 fields[n+inc].B


void ReflectionTransfer::determineFields()
{
    if (fields_determined == DETERMINED_RESONANT) return;

    writelog(LOG_DETAIL, solver->getId() + ": Determining optical fields");

    const std::size_t N = diagonalizer->matrixSize();
    const std::size_t N0 = diagonalizer->source()->matrixSize();
    const std::size_t NN = N*N;
    cvector temp(wrk, N);

    cdiagonal gamma;

    size_t count = solver->stack.size();

    // Assign the space for the field vectors
    fields.resize(count);

    // Obtain the physical fields at the last layer
    storeP = STORE_LAST;
    memP.resize(2);
    interface_field = nullptr;
    auto E = getInterfaceVector();

    for (unsigned pass = 0; pass < 1 || (pass < 2 && solver->interface != std::ptrdiff_t(count)); pass++)
    {
        // each pass for below and above the interface

        std::size_t start, end;
        std::ptrdiff_t inc;
        switch (pass)
        {
            case 0: start = solver->interface-1; end = 0;       inc = -1; break;
            case 1: start = solver->interface;   end = count-1; inc = +1; break;
        }

        fields[start].F.reset(N);
        fields[start].B.reset(N);

        // compute B-field for the layer next to the interface
        std::size_t curr = solver->stack[start];

        gamma = diagonalizer->Gamma(curr);

        double H = (start == 0 || start == count-1)? 0. : (solver->vbounds->at(start) - solver->vbounds->at(start-1));
        for (std::size_t i = 0; i < N; i++)
            phas[i] = exp(-I*gamma[i]*H);

        // P = phas*P*phas + I
        assert(memP[pass].rows() == N && memP[pass].cols() == N);
        memcpy(P.data(), memP[pass].data(), NN*sizeof(dcomplex));
        mult_diagonal_by_matrix(phas, P); mult_matrix_by_diagonal(P, phas);         // P := phas * P * phas
        for (std::size_t i = 0, ii = 0; i < N; i++, ii += (N+1)) P[ii] += 1.;       // P := P + I

        mult_matrix_by_vector(diagonalizer->invTE(curr), E, fields[start].B);       // B := invTE * E
        invmult(P, fields[start].B);                                                // B := inv(P) * B
        for (std::size_t i = 0; i < N; i++) fields[start].B[i] *= phas[i];          // B := phas * B

        // F-field for the first layer
        mult_matrix_by_vector(memP[pass], fields[start].B, fields[start].F);

        for (std::size_t n = start; n != end; n += inc) {
            // Compute F and B field for the next (previous) layer

            F2.reset(N);
            B2.reset(N);

            curr = solver->stack[n];
            std::size_t next = solver->stack[n+inc];
            assert(diagonalizer->isDiagonalized(curr));
            assert(diagonalizer->isDiagonalized(next));

            gamma = diagonalizer->Gamma(next);

            if (next != curr) {
                for (std::size_t i = 0; i < N; i++) F2[i] = F1[i] - B1[i];          // F2 := F1 - B1
                mult_matrix_by_vector(diagonalizer->TH(curr), F2, temp);            // temp := TH * F2
                mult_matrix_by_vector(diagonalizer->invTH(next), temp, B2);         // B2 := invTH * temp

                for (std::size_t i = 0; i < N; i++) F2[i] = F1[i] + B1[i];          // F2 := F1 + B1
                mult_matrix_by_vector(diagonalizer->TE(curr), F2, temp);            // temp := TE * F2
                memcpy(F2.data(), B2.data(), N*sizeof(dcomplex));
                zgemm('N','N', int(N), 1, int(N0), 1., diagonalizer->invTE(next).data(), int(N),
                      temp.data(), int(N0), -1., B2.data(), int(N));                // B2 := invTE * temp - B2
                for (std::size_t i = 0; i < N; i++)
                    F2[i] += 0.5 * B2[i];                                           // F2 := B2 + tH (F1-B2)
            } else {
                for (std::size_t i = 0; i < N; i++) B2[i] = 2. * B1[i];
                memcpy(F2.data(), F1.data(), N*sizeof(dcomplex));
            }

            H = (n+inc == end)? 0. : (solver->vbounds->at(n+inc) - solver->vbounds->at(n+inc-1));
            for (std::size_t i = 0; i < N; i++) {
                dcomplex phas = exp(-I*gamma[i]*H);
                B2[i] *= 0.5 * phas;                                                // B2 := 1/2 * phas * B2
                F2[i] /= phas;                                                      // F2 := phas^(-1) * F2
            }
        }
    }

    storeP = STORE_NONE;
    fields_determined = DETERMINED_RESONANT;

    // Finally normalize fields
    if (solver->emission == SlabBase::EMISSION_BOTTOM || solver->emission == SlabBase::EMISSION_TOP) {
        size_t n = (solver->emission == SlabBase::EMISSION_BOTTOM)? 0 : count-1;

        double P = 1./Z0 * abs(diagonalizer->source()->integratePoyntingVert(getFieldVectorE(n, 0.),
                                                                             getFieldVectorH(n, 0.)));

        if (P < SMALL) {
            writelog(LOG_WARNING, "Device is not emitting to the {} side: skipping normalization",
                    (solver->emission == SlabBase::EMISSION_TOP)? "top" : "bottom");
        } else {
            P = 1. / sqrt(P);
            for (size_t n = 0; n < count; ++n) {
                F1 *= P;
                B1 *= P;
            }
        }
    }
}


void ReflectionTransfer::determineReflectedFields(const cvector& incident, IncidentDirection side)
{
    if (fields_determined == DETERMINED_REFLECTED && incident == incident_vector) return;
    incident_vector = incident.copy();

    writelog(LOG_DETAIL, solver->getId() + ": Determining reflected optical fields");

    size_t count = solver->stack.size();

    // Assign the space for the field vectors
    fields.resize(count);

    std::size_t start, end;
    std::ptrdiff_t inc;
    switch (side)
    {
        case INCIDENCE_TOP:    start = count-1; end = 0;       inc = -1; break;
        case INCIDENCE_BOTTOM: start = 0;       end = count-1; inc = +1; break;
    }

    // Store all reflectivities
    storeP = STORE_ALL;
    memP.resize(count);

    // Compute reflection matrices
    initDiagonalization();
    findReflection(end, start, true);

    // Temporary and initial data
    const std::size_t N = diagonalizer->matrixSize();
    const std::size_t N0 = diagonalizer->source()->matrixSize();
    cvector temp(wrk, N);
    cdiagonal gamma;

    std::size_t curr = solver->stack[start];
    double H;

    fields[start].B = incident.copy(); // diagonalized incident E-field
    fields[start].F.reset(N);

    for (std::size_t n = start; n != end; n += inc)
    {
        // F-field for the current layer
        mult_matrix_by_vector(memP[n], B1, F1);

        // Compute B field for the next (previous) layer

        F2.reset(N);
        B2.reset(N);

        curr = solver->stack[n];
        const std::size_t next = solver->stack[n+inc];

        gamma = diagonalizer->Gamma(next);

        if (next != curr || n+inc == end) {
            if (next != curr) {
                for (std::size_t i = 0; i < N; i++) F2[i] = F1[i] - B1[i];                  // F2 := F1 - B1
                mult_matrix_by_vector(diagonalizer->TH(curr), F2, temp);                    // temp := TH * F2
                mult_matrix_by_vector(diagonalizer->invTH(next), temp, B2);                 // B2 := invTH * temp
            } else {
                for (std::size_t i = 0; i < N; i++) B2[i] = F1[i] - B1[i];                  // B2 := F1 - B1
            }
            // multiply rows of invTH by -1 where necessary for the outer layer
            if (n+inc == end) {
                for (std::size_t i = 0; i < N; i++)
                    if (real(gamma[i]) < -SMALL) B2[i] = -B2[i];
            }

            for (std::size_t i = 0; i < N; i++) F2[i] = F1[i] + B1[i];                      // F2 := F1 + B1
            if (next != curr) {
                mult_matrix_by_vector(diagonalizer->TE(curr), F2, temp);                    // temp := TE * F2
                zgemm('N','N', int(N), 1, int(N0), 1., diagonalizer->invTE(next).data(), int(N),
                      temp.data(), int(N0), -1., B2.data(), int(N));                        // B2 := invTE * temp - B2
            } else {
                for (std::size_t i = 0; i < N; i++)
                    B2[i] = F2[i] - B2[i];                                                  // B2 := (F1+B1) + (F1-B1)
            }
        } else {
            for (std::size_t i = 0; i < N; i++) B2[i] = 2. * B1[i];
        }

        if (n+inc != end) {
            H = solver->vbounds->at(n+inc) - solver->vbounds->at(n+inc-1);
            for (std::size_t i = 0; i < N; i++)
                B2[i] *= 0.5 * exp(-I*gamma[i]*H);                                          // B2 := 1/2 * phas * B2
        } else {
            for (std::size_t i = 0; i < N; i++) B2[i] *= 0.5;                               // B2 := 1/2 * phas * B2
        }
    }

    fields[end].F = cvector(N, 0.);

    // In the outer layers replace F and B where necessary for consistent gamma handling
    for (std::size_t n = 0; n < count; n += count-1) {
        gamma = diagonalizer->Gamma(solver->stack[n]);
        for (std::size_t i = 0; i < N; i++) {
            if (real(gamma[i]) < -SMALL)
                std::swap(fields[n].F, fields[n].B);
        }
     }

    // Replace F and B at one side of the interface for consistency in getFieldVectorE and getFieldVectorH
    size_t interface = size_t(max(solver->interface, ptrdiff_t(0)));
    switch (side)
    {
        case INCIDENCE_TOP:    start = interface; end = count; break;
        case INCIDENCE_BOTTOM: start = 0; end = min(interface, count); break;
    }
    // start = size_t(max(solver->interface, ptrdiff_t(0))); end = count;
    for (std::size_t n = start; n < end; n++) {
        gamma = diagonalizer->Gamma(solver->stack[n]);
        H = (n < count-1 && n > 0)? solver->vbounds->at(n) - solver->vbounds->at(n-1) : 0.;
        for (std::size_t i = 0; i < N; i++) {
            dcomplex phas = exp(-I*gamma[i]*H);
            dcomplex t = B1[i] / phas;
            if (isnan(t) && B1[i] == 0.) t = 0.;
            B1[i] = F1[i] * phas;
            if (isnan(B1[i]) && F1[i] == 0.) B1[i] = 0.;
            F1[i] = t;
        }
    }

    storeP = STORE_NONE;
    fields_determined = DETERMINED_REFLECTED;
}


cvector ReflectionTransfer::getFieldVectorE(double z, std::size_t n)
{
    assert(fields_determined != DETERMINED_NOTHING);

    if (std::ptrdiff_t(n) >= solver->interface) {
        z = - z;
        if (n != 0 && n != solver->vbounds->size())
            z += solver->vbounds->at(n) - solver->vbounds->at(n-1);
    }

    cdiagonal gamma = diagonalizer->Gamma(solver->stack[n]);

    const std::size_t N = gamma.size();
    cvector E(N);

    for (std::size_t i = 0; i < N; i++) {
        dcomplex phi = - I * gamma[i] * z;
        dcomplex ef = F1[i] * exp(phi), eb = B1[i] * exp(-phi);
        if (isnan(ef) && F1[i] == 0.) ef = 0.;
        if (isnan(eb) && B1[i] == 0.) eb = 0.;
        // if (isnan(ef)) std::cerr << "ef" << F1[i] << exp(phi) << " ";
        // if (isnan(eb)) std::cerr << "eb" << B1[i] << exp(-phi) << " ";
        E[i] = ef + eb;
    }

    return diagonalizer->TE(solver->stack[n]) * E;
}


cvector ReflectionTransfer::getFieldVectorH(double z, std::size_t n)
{
    assert(fields_determined != DETERMINED_NOTHING);

    if (std::ptrdiff_t(n) >= solver->interface) {
        z = - z;
        if (n != 0 && n != solver->vbounds->size())
            z += solver->vbounds->at(n) - solver->vbounds->at(n-1);
    }

    cdiagonal gamma = diagonalizer->Gamma(solver->stack[n]);

    const std::size_t N = gamma.size();
    cvector H(N);

    for (std::size_t i = 0; i < N; i++) {
        dcomplex phi = - I * gamma[i] * z;
        dcomplex ef = F1[i] * exp(phi), eb = B1[i] * exp(-phi);
        if (isnan(ef) && F1[i] == 0.) ef = 0.;
        if (isnan(eb) && B1[i] == 0.) eb = 0.;
        // if (isnan(ef)) std::cerr << "ef" << F1[i] << exp(phi) << " ";
        // if (isnan(eb)) std::cerr << "eb" << B1[i] << exp(-phi) << " ";
        H[i] = ef - eb;
    }

    if (n == 0 || std::size_t(n) == solver->vbounds->size()) {
        // In the outer layers multiply H by -1 where necessary for propagating wave
        for (std::size_t i = 0; i < N; i++)
            if (real(gamma[i]) < -SMALL) H[i] = - H[i];
    }

    return diagonalizer->TH(solver->stack[n]) * H;
}



double ReflectionTransfer::integrateEE(double z1, double z2) {
    //TODO
    throw NotImplemented("integrateEE");
}

double ReflectionTransfer::integrateHH(double z1, double z2) {
    //TODO
    throw NotImplemented("integrateHH");
}

}}} // namespace plask::optical::slab
