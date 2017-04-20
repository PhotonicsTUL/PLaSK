#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define BOOST_USE_WINDOWS_H
#endif

#include "admittance.h"
#include "solver.h"
#include "expansion.h"
#include "fortran.h"
#include "meshadapter.h"

namespace plask { namespace solvers { namespace slab {

AdmittanceTransfer::AdmittanceTransfer(SlabBase* solver, Expansion& expansion): Transfer(solver, expansion)
{
    writelog(LOG_DETAIL, "Initializing Admittance Transfer");
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
    zgemm('n','n', N, N0, N, 1., Y.data(), N, diagonalizer->invTE(solver->stack[solver->interface]).data(), N, 0., wrk, N);
    zgemm('n','n', N0, N0, N, 1., diagonalizer->TH(solver->stack[solver->interface]).data(), N0, wrk, N, 0., M.data(), N0);

    // Find the(diagonalized field) admittance matrix and store it for the future reference
    findAdmittance(0, solver->interface);
    // M += TH(interface-1) * Y(interface) * invTE(interface-1);
    zgemm('n','n', N, N0, N, 1., Y.data(), N, diagonalizer->invTE(solver->stack[solver->interface-1]).data(), N, 0., wrk, N);
    zgemm('n','n', N0, N0, N, 1., diagonalizer->TH(solver->stack[solver->interface-1]).data(), N0, wrk, N, 1., M.data(), N0);
}


void AdmittanceTransfer::findAdmittance(int start, int end)
{
    const int inc = (start < end) ? 1 : -1;

    int N = diagonalizer->matrixSize();
    int NN = N*N;

    // Some temporary variables
    cdiagonal gamma, y1(N), y2(N);

    std::exception_ptr error;

    #pragma omp parallel for schedule(dynamic,1)
    for (int l = 0; l < diagonalizer->lcount; ++l) {
        try {
            if (!error) diagonalizer->diagonalizeLayer(l);
        } catch(...) {
            error = std::current_exception();
        }
    }
    if (error) std::rethrow_exception(error);

    // Now iteratively we find matrices Y[i]

    // PML layer
    #ifdef OPENMP_FOUND
        write_debug("{}: Entering into single region of admittance search", solver->getId());
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
    double H = solver->vpml.dist;
    gamma = diagonalizer->Gamma(solver->stack[start]);
    get_y1(gamma, H, y1);
    get_y2(gamma, H, y2);
    // off-diagonal elements of Y are 0
    for (int i = 0; i < N; i++) Y(i,i) = y2[i] * y2[i] / (y1[i] - Y(i,i)) - y1[i]; // Y = y2 * inv(y1-Y) * y2 - y1

    // save the Y matrix for 1-st layer
    storeY(start);

    if (start == end) return;

    // Declare temporary matrixH) on 'wrk' array
    cmatrix work(N, N, wrk);

    for (int n = start+inc; n != end; n += inc)
    {
        gamma = diagonalizer->Gamma(solver->stack[n]);

        H = solver->vbounds[n] - solver->vbounds[n-1];
        get_y1(gamma, H, y1);
        get_y2(gamma, H, y2);

        // The main equation
        // Y[n] = y2 * tE * inv(y1*tE - tH*Y[n-1]) * y2  -  y1

        mult_matrix_by_matrix(diagonalizer->TH(solver->stack[n-inc]), Y, temp);     // work = tH * Y[n-1]
        mult_matrix_by_matrix(diagonalizer->invTH(solver->stack[n]), temp, work);   // ...

        mult_matrix_by_matrix(diagonalizer->invTE(solver->stack[n]), diagonalizer->TE(solver->stack[n-inc]), temp); // compute tE

        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) Y(i,j) = y1[i]*temp(i,j) - work(i,j);       // Y[n] = y1 * tE - work

        for (int i = 0; i < NN; i++) work[i] = 0.;
        for (int j = 0, i = 0; j < N; j++, i += N+1) work[i] = y2[j];               // work = y2

        invmult(Y, work);                                                           // work = inv(Y[n]) * (work = y2)
        mult_matrix_by_matrix(temp, work, Y);                                       // Y[n] = tE * work

        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) Y(i,j) *= y2[i];                            // Y[n] = y2 * Y[n]

        for (int j = 0, i = 0; j < N; j++, i += N+1) Y[i] -= y1[j];                 // Y[n] = Y[n] - y1

        // Save the Y matrix for n-th layer
        storeY(n);
    }
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

    // Assign all the required space
    cdiagonal gamma, y1(N), y2(N);

    // Assign the space for the field vectors
    fields.resize(count);

    // Temporary vector for storing fields in the real domain
    cvector tv(N0);

    // Obtain the physical fields at the last layer
    needAllY = true;
    interface_field = nullptr;
    auto E = getInterfaceVector();

    // Declare temporary matrix on 'wrk' array
    cmatrix work(N, N, wrk);

    for (int pass = 0; pass < 1 || (pass < 2 && solver->interface != count); pass++)
    {
        // each pass for below and above the interface

        int start, end, inc;
        switch (pass) {
            case 0: start = solver->interface-1; end = -1;    inc =  1; break;
            case 1: start = solver->interface;   end = count; inc = -1; break;
        }

        // Ed[start] = invTE[start] E
        fields[start].Ed = cvector(N);
        mult_matrix_by_vector(diagonalizer->invTE(solver->stack[start]), E, fields[start].Ed);

        for (int n = start; n != end; n -= inc)
        {
            int curr = solver->stack[n];

            double H = (n == 0 || n == count-1)? solver->vpml.dist : solver->vbounds[n] - solver->vbounds[n-1];
            gamma = diagonalizer->Gamma(curr);
            get_y1(gamma, H, y1);
            get_y2(gamma, H, y2);

            // work = Y[n] + y1
            cmatrix Y = getY(n);
            for (int i = 0; i < NN; i++) work[i] = Y[i];
            for (int i = 0; i < N; i++) work (i,i) += y1[i];

            // E0[n] = work * Ed[n]
            fields[n].E0 = cvector(N);
            mult_matrix_by_vector(work, fields[n].Ed, fields[n].E0);

            // E0[n] = - inv(y2) * E0[0]
            for (int i = 0; i < N; i++) {
                if (abs(y2[i]) < SMALL)         // Actually we cannot really compute E0 in this case.
                    fields[n].E0[i] = 0.;       // So let's cheat a little, as the field cannot
                else                            // increase to the boundaries.
                    fields[n].E0[i] /= - y2[i];
            }

            if (n != end+inc) { // not the last layer
                int prev = solver->stack[n-inc];
                // Ed[n-inc] = invTE[n-inc] * TE[n] * E0[n]
                fields[n-inc].Ed = cvector(N);
                mult_matrix_by_vector(diagonalizer->TE(curr), fields[n].E0, tv);
                mult_matrix_by_vector(diagonalizer->invTE(prev), tv, fields[n-inc].Ed);
            } else {
                fields[n].H0 = cvector(N);
                for (int i = 0; i < N; i++)
                    //fields[end+inc].H0[i] = y2[i] * fields[end+inc].Ed[i];
                    fields[end+inc].H0[i] = double(inc) *
                                                 (y1[i] * fields[end+inc].E0[i] + y2[i] * fields[end+inc].Ed[i]);
            }

            // Now compute the magnetic fields

            // Hd[n] = Y[n] * Ed[n]
            fields[n].Hd = cvector(N);
            mult_matrix_by_vector(Y, fields[n].Ed, fields[n].Hd);

            if (n != start) {
                int next = solver->stack[n+inc];
                // H0[n+inc] = invTH[n+inc] * TH[n] * Hd[n]
                fields[n+inc].H0 = cvector(N);
                mult_matrix_by_vector(diagonalizer->TH(curr), fields[n].Hd, tv);
                mult_matrix_by_vector(diagonalizer->invTH(next), tv, fields[n+inc].H0);
            }

            // An alternative method is to find the H0 from the following equation:
            // H0 = y1 * E0 + y2 * Ed
            // for (int i = 0; i < N; i++)
            //     fields[n].H0[i] = y1[i] * fields[n].E0[i]  +  y2[i] * fields[n].Ed[i];
            // However in some cases this can make the magnetic field discontinous
        }
    }

    // Now fill the Y matrix with the one from the interface (necessary for interfaceField*)
    memcpy(Y.data(), getY(solver->interface-1).data(), NN*sizeof(dcomplex));

    needAllY = false;
    fields_determined = DETERMINED_RESONANT;

    // Finally normalize fields
    if (solver->emission == SlabBase::EMISSION_BOTTOM || solver->emission == SlabBase::EMISSION_TOP) {
        size_t n = (solver->emission == SlabBase::EMISSION_BOTTOM)? 0 : count-1;
        int l = solver->stack[n];

        cvector hv(N0);
        mult_matrix_by_vector(diagonalizer->TE(l), fields[n].Ed, tv);
        mult_matrix_by_vector(diagonalizer->TH(l), fields[n].Hd, hv);

        double P = 1./Z0 * abs(diagonalizer->source()->integratePoyntingVert(tv, hv));

        if (P < SMALL) {
            writelog(LOG_WARNING, "Device is not emitting to the {} side: skipping normalization",
                    (solver->emission == SlabBase::EMISSION_TOP)? "top" : "bottom");
        } else {
            P = 1. / sqrt(P);
            for (size_t i = 0; i < count; ++i) {
                fields[i].E0 *= P;
                fields[i].H0 *= P;
                fields[i].Ed *= P;
                fields[i].Hd *= P;
            }
        }
    }
}


cvector AdmittanceTransfer::getFieldVectorE(double z, int n)
{
    cvector E0 = fields[n].E0;
    cvector Ed = fields[n].Ed;

    cdiagonal gamma = diagonalizer->Gamma(solver->stack[n]);
    double d = (n == 0 || n == solver->vbounds.size())? solver->vpml.dist : solver->vbounds[n] - solver->vbounds[n-1];
    if (n >= solver->interface) z = d - z;
    else if (n == 0) z += d;

    if ((n == 0 || n == solver->vbounds.size()) && z < 0.)
        return cvector(diagonalizer->source()->matrixSize(), NAN);

    int N = gamma.size();
    cvector E(N);

    for (int i = 0; i < N; i++) {
        dcomplex g = gamma[i];
        //E[i] = (sin(g*(d-z)) * E0[i] + sin(g*z) * Ed[i]) / sin(g*d);

        double a = abs(exp(2.*I*g*d));
        if (isinf(a) || a < SMALL) {
            dcomplex d0p = exp(I*g*z) - exp(I*g*(z-2*d));
            dcomplex d0n = exp(I*g*(2*d-z)) - exp(-I*g*z);
            if (isinf(real(d0p)) || isinf(imag(d0p))) d0p = 0.; else d0p = 1./ d0p;
            if (isinf(real(d0n)) || isinf(imag(d0n))) d0n = 0.; else d0n = 1./ d0n;
            dcomplex ddp = exp(I*g*(d-z)) - exp(-I*g*(d+z));
            dcomplex ddn = exp(I*g*(d+z)) - exp(I*g*(z-d));
            if (isinf(real(ddp)) || isinf(imag(ddp))) ddp = 0.; else ddp = 1./ ddp;
            if (isinf(real(ddn)) || isinf(imag(ddn))) ddn = 0.; else ddn = 1./ ddn;
            E[i] = (d0p-d0n) * E0[i] + (ddp-ddn) * Ed[i];
        } else {
            E[i] = (sinh(I*g*(d-z)) * E0[i] + sinh(I*g*z) * Ed[i]) / sinh(I*g*d);
        }
    }

    cvector result(diagonalizer->source()->matrixSize());
    // result = diagonalizer->TE(n) * E;
    mult_matrix_by_vector(diagonalizer->TE(solver->stack[n]), E, result);
    return result;
}


cvector AdmittanceTransfer::getFieldVectorH(double z, int n)
{
    cvector H0 = fields[n].H0;
    cvector Hd = fields[n].Hd;

    cdiagonal gamma = diagonalizer->Gamma(solver->stack[n]);
    double d = (n == 0 || n == solver->vbounds.size())? solver->vpml.dist : solver->vbounds[n] - solver->vbounds[n-1];
    if (n >= solver->interface) z = d - z;
    else if (n == 0) z += d;

    if ((n == 0 || n == solver->vbounds.size()) && z < 0.)
        return cvector(diagonalizer->source()->matrixSize(), NAN);

    int N = gamma.size();
    cvector H(N);

    for (int i = 0; i < N; i++) {
        dcomplex g = gamma[i];
        //H[i] = (sin(g*(d-z)) * H0[i] + sin(g*z) * Hd[i]) / sin(g*d);

        double a = abs(exp(2.*I*g*d));
        if (isinf(a) || a < SMALL) {
            dcomplex d0p = exp(I*g*z) - exp(I*g*(z-2*d));
            dcomplex d0n = exp(I*g*(2*d-z)) - exp(-I*g*z);
            if (isinf(real(d0p)) || isinf(imag(d0p))) d0p = 0.; else d0p = 1./ d0p;
            if (isinf(real(d0n)) || isinf(imag(d0n))) d0n = 0.; else d0n = 1./ d0n;
            dcomplex ddp = exp(I*g*(d-z)) - exp(-I*g*(d+z));
            dcomplex ddn = exp(I*g*(d+z)) - exp(I*g*(z-d));
            if (isinf(real(ddp)) || isinf(imag(ddp))) ddp = 0.; else ddp = 1./ ddp;
            if (isinf(real(ddn)) || isinf(imag(ddn))) ddn = 0.; else ddn = 1./ ddn;
            H[i] = (d0p-d0n) * H0[i] + (ddp-ddn) * Hd[i];
        } else {
            H[i] = (sinh(I*g*(d-z)) * H0[i] + sinh(I*g*z) * Hd[i]) / sinh(I*g*d);
        }
    }

    cvector result(diagonalizer->source()->matrixSize());
    // result = diagonalizer->TH(n) * H;
    mult_matrix_by_vector(diagonalizer->TH(solver->stack[n]), H, result);
    return result;
}


cvector AdmittanceTransfer::getReflectionVector(const cvector& incident, IncidentDirection side)
{
    int curr, prev;

    initDiagonalization();

    switch (side) {
        case INCIDENCE_TOP:
            findAdmittance(0, solver->stack.size()-1);
            curr = solver->stack[solver->stack.size()-1];
            prev = solver->stack[solver->stack.size()-2];
            break;
        case INCIDENCE_BOTTOM:
            findAdmittance(solver->stack.size()-1, 0);
            curr = solver->stack[0];
            prev = solver->stack[1];
            break;
    }

    int N = diagonalizer->matrixSize();
    int NN = N * N;
    cmatrix work(N, N, wrk);  // we have Y, temp and work

    // Transfer to the outermost layer:
    if (prev != curr) {
        mult_matrix_by_matrix(diagonalizer->invTE(prev), diagonalizer->TE(curr), work); // work = tE¯¹
        mult_matrix_by_matrix(Y, work, temp);                                           // temp = Y tE¯¹
        mult_matrix_by_matrix(diagonalizer->invTH(curr), diagonalizer->TH(prev), work); // work = tH
        mult_matrix_by_matrix(work, temp, Y);                                           // Y = tH Y tE¯¹
    }

    std::copy_n(Y.data(), NN, temp.data());                                             // temp = Y
    // Use Jacobi preconditioner for temp
    for (size_t i = 0; i != N; ++i) {
        temp(i,i) -= 1.;                                                                // temp = Y - I
        dcomplex f = 1. / temp(i,i);
        wrk[i] = f;
        for (size_t j = 0; j != N; ++j)
            temp(i,j) *= f;
    }
    cvector reflected(N);
    for (size_t i = 0; i != N; ++i) reflected[i] = wrk[i] * incident[i];
    invmult(temp, reflected);
    for (size_t i = 0; i != N; ++i)
        reflected[i] = - 2. * reflected[i] - incident[i];                               // R = 2 [I-Y]¯¹ P - P
    return reflected;
}

void AdmittanceTransfer::determineReflectedFields(const cvector& incident, IncidentDirection side)
{
    if (fields_determined == DETERMINED_REFLECTED) return;

    writelog(LOG_DETAIL, solver->getId() + ": Determining reflected optical fields");

    int N = diagonalizer->matrixSize();
    int N0 = diagonalizer->source()->matrixSize();
    size_t count = solver->stack.size();

    int NN = N*N;

    // Assign all the required space
    cdiagonal gamma, y1(N), y2(N);

    // Assign the space for the field vectors
    fields.resize(count);

    // Temporary vector for storing fields in the real domain
    cvector tv(N0);

    int start, end, inc;
    switch (side) {
        case INCIDENCE_TOP:    start = count-1; end = -1; inc = 1; break;
        case INCIDENCE_BOTTOM: start = 0; end = count; inc = -1;   break;
    }

    // Obtain the physical fields at the incident-side layer
    needAllY = true;
    fields[start].E0 = getReflectionVector(incident, side);
    fields[start].E0 += incident;
    fields[start].H0 = Y * fields[start].E0;

    // E(z) =  cosh(iΓz) A + sinh(iΓz) B
    // H(z) = -sinh(iΓz) A - coshh(iΓz) B
    // A = E(0)   B = -H(0)

    gamma = diagonalizer->Gamma(solver->stack[start]);
    fields[start].Ed = cvector(N);
    fields[start].Hd = cvector(N);
    for (int i = 0; i != N; ++i) {
//TODO!!!!
        dcomplex gam = gamma[i];
        if (real(gam) < -SMALL) gam = -gam;
        if (imag(gam) > SMALL) gam = -gam;
        dcomplex p = I * gam * solver->vpml.dist;
        dcomplex ch = cosh(p), sh = sinh(p);
        fields[start].Ed[i] = fields[start].E0[i] * ch - fields[start].H0[i] * sh;
        fields[start].Hd[i] = fields[start].H0[i] * ch - fields[start].E0[i] * sh;
    }

    // Declare temporary matrix on 'wrk' array
    cmatrix work(N, N, wrk);

    mult_matrix_by_matrix(diagonalizer->invTE(solver->stack[start-inc]), diagonalizer->TE(solver->stack[start]), work);
    fields[start-inc].Ed = work * fields[start].E0;

    for (int n = start-inc; n != end; n -= inc)
    {
        int curr = solver->stack[n];

        double H = (n == 0 || n == count-1)? solver->vpml.dist : solver->vbounds[n] - solver->vbounds[n-1];
        gamma = diagonalizer->Gamma(curr);
        get_y1(gamma, H, y1);
        get_y2(gamma, H, y2);

        // work = Y[n] + y1
        cmatrix Y = getY(n);
        for (int i = 0; i < NN; i++) work[i] = Y[i];
        for (int i = 0; i < N; i++) work (i,i) += y1[i];

        // E0[n] = work * Ed[n]
        fields[n].E0 = cvector(N);
        mult_matrix_by_vector(work, fields[n].Ed, fields[n].E0);

        // E0[n] = - inv(y2) * E0[0]
        for (int i = 0; i < N; i++) {
            if (abs(y2[i]) < SMALL)         // Actually we cannot really compute E0 in this case.
                fields[n].E0[i] = 0.;       // So let's cheat a little, as the field cannot
            else                            // increase to the boundaries.
                fields[n].E0[i] /= - y2[i];
        }

        if (n != end+inc) {                 // not the last layer
            int prev = solver->stack[n-inc];
            // Ed[n-inc] = invTE[n-inc] * TE[n] * E0[n]
            fields[n-inc].Ed = cvector(N);
            mult_matrix_by_vector(diagonalizer->TE(curr), fields[n].E0, tv);
            mult_matrix_by_vector(diagonalizer->invTE(prev), tv, fields[n-inc].Ed);
        } else {
            fields[n].H0 = cvector(N);
            for (int i = 0; i < N; i++)
                //fields[end+inc].H0[i] = y2[i] * fields[end+inc].Ed[i];
                fields[end+inc].H0[i] = double(inc) *
                                                (y1[i] * fields[end+inc].E0[i] + y2[i] * fields[end+inc].Ed[i]);
        }

        // Now compute the magnetic fields

        // Hd[n] = Y[n] * Ed[n]
        fields[n].Hd = cvector(N);
        mult_matrix_by_vector(Y, fields[n].Ed, fields[n].Hd);

        if (n != start-inc) {
            int next = solver->stack[n+inc];
            // H0[n+inc] = invTH[n+inc] * TH[n] * Hd[n]
            fields[n+inc].H0 = cvector(N);
            mult_matrix_by_vector(diagonalizer->TH(curr), fields[n].Hd, tv);
            mult_matrix_by_vector(diagonalizer->invTH(next), tv, fields[n+inc].H0);
        }

        // An alternative method is to find the H0 from the following equation:
        // H0 = y1 * E0 + y2 * Ed
        // for (int i = 0; i < N; i++)
        //     fields[n].H0[i] = y1[i] * fields[n].E0[i]  +  y2[i] * fields[n].Ed[i];
        // However in some cases this can make the magnetic field discontinous
    }

    // Finally revert fields at one side of the interface
    switch (side) {
        case INCIDENCE_TOP:    start = solver->interface; end = count; break;
        case INCIDENCE_BOTTOM: start = 0; end = solver->interface; break;
    }
    for (size_t n = start; n < end; ++n) {
        std::swap(fields[n].E0, fields[n].Ed);
        std::swap(fields[n].H0, fields[n].Hd);
        // TODO should I revert H?
    }

    needAllY = false;
    fields_determined = DETERMINED_REFLECTED;
}

cvector AdmittanceTransfer::getTransmissionVector(const cvector& incident, IncidentDirection side)
{
    determineReflectedFields(incident, side);
    size_t n = (side == INCIDENCE_BOTTOM)? solver->stack.size()-1 : 0;
    return diagonalizer->TE(solver->stack[n]) * fields[n].E0;
}

}}} // namespace plask::solvers::slab
