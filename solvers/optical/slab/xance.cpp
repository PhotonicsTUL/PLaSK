/*#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define BOOST_USE_WINDOWS_H
#endif*/

#include "xance.hpp"
#include "solver.hpp"
#include "diagonalizer.hpp"
#include "expansion.hpp"
#include "fortran.hpp"
#include "meshadapter.hpp"

namespace plask { namespace optical { namespace slab {

XanceTransfer::XanceTransfer(SlabBase* solver, Expansion& expansion): Transfer(solver, expansion)
{
    // Reserve space for matrix multiplications...
    const std::size_t N = diagonalizer->matrixSize();
    Y = cmatrix(N,N);
    needAllY = false;
}


void XanceTransfer::storeY(size_t n)
{
    if (needAllY) {
        const std::size_t N = diagonalizer->matrixSize();
        if (memY.size() != solver->stack.size()) {
            // Allocate the storage for admittance matrices
            memY.resize(solver->stack.size());
            for (std::size_t i = 0; i < solver->stack.size(); i++) memY[i] = cmatrix(N,N);
        }
        memcpy(memY[n].data(), Y.data(), N*N*sizeof(dcomplex));
    }
}


cvector XanceTransfer::getFieldVectorE(double z, std::size_t n, PropagationDirection part)
{
    cvector E0 = fields[n].E0;
    cvector Ed = fields[n].Ed;

    cdiagonal gamma = diagonalizer->Gamma(solver->stack[n]);
    double d = get_d(n, z, part);

    if ((n == 0 || std::size_t(n) == solver->vbounds->size()) && z < 0.)
        return cvector(diagonalizer->source()->matrixSize(), NAN);

    const std::size_t N = gamma.size();
    cvector E(N);

    switch (part) {
        case PROPAGATION_TOTAL:
            for (std::size_t i = 0; i < N; i++) {
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
            break;
        case PROPAGATION_UPWARDS:
            for (std::size_t i = 0; i < N; i++) {
                dcomplex g = gamma[i]; if (g.real() < 0) g = -g;
                E[i] = 0.5 * (E0[i] * exp(I*g*d) - Ed[i]) * exp(-I*g*z) / sinh(I*g*d);
            }
            break;
        case PROPAGATION_DOWNWARDS:
            for (std::size_t i = 0; i < N; i++) {
                dcomplex g = gamma[i]; if (g.real() < 0) g = -g;
                E[i] = 0.5 * (Ed[i] - E0[i] * exp(-I*g*d)) * exp(I*g*z) / sinh(I*g*d);
            }
            break;
    }

    cvector result(diagonalizer->source()->matrixSize());
    // result = diagonalizer->TE(n) * E;
    mult_matrix_by_vector(diagonalizer->TE(solver->stack[n]), E, result);
    return result;
}


cvector XanceTransfer::getFieldVectorH(double z, std::size_t n, PropagationDirection part)
{
    cvector H0 = fields[n].H0;
    cvector Hd = fields[n].Hd;

    cdiagonal gamma = diagonalizer->Gamma(solver->stack[n]);
    double d = get_d(n, z, part);

    if ((n == 0 || std::size_t(n) == solver->vbounds->size()) && z < 0.)
        return cvector(diagonalizer->source()->matrixSize(), NAN);

    const std::size_t N = gamma.size();
    cvector H(N);

    switch (part) {
        case PROPAGATION_TOTAL:
            for (std::size_t i = 0; i < N; i++) {
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
            break;
        case PROPAGATION_UPWARDS:
            for (std::size_t i = 0; i < N; i++) {
                dcomplex g = gamma[i]; if (g.real() < 0) g = -g;
                H[i] = 0.5 * (H0[i] * exp(I*g*d) - Hd[i]) * exp(-I*g*z) / sinh(I*g*d);
            }
            break;
        case PROPAGATION_DOWNWARDS:
            for (std::size_t i = 0; i < N; i++) {
                dcomplex g = gamma[i]; if (g.real() < 0) g = -g;
                H[i] = 0.5 * (Hd[i] - H0[i] * exp(-I*g*d)) * exp(I*g*z) / sinh(I*g*d);
            }
            break;
    }

    cvector result(diagonalizer->source()->matrixSize());
    // result = diagonalizer->TH(n) * H;
    mult_matrix_by_vector(diagonalizer->TH(solver->stack[n]), H, result);
    return result;
}


cvector XanceTransfer::getTransmissionVector(const cvector& incident, IncidentDirection side)
{
    determineReflectedFields(incident, side);
    size_t n = (side == INCIDENCE_BOTTOM)? solver->stack.size()-1 : 0;
    return fields[n].E0;
}


double XanceTransfer::integrateField(WhichField field, size_t n, double z1, double z2) {
    // size_t layer = solver->stack[n];
    // size_t N = diagonalizer->matrixSize();

    // cvector F0, Fd;
    // if (field == FIELD_E) {
    //     F0 = fields[n].E0;
    //     Fd = fields[n].Ed;
    // } else {
    //     F0 = fields[n].H0;
    //     Fd = fields[n].Hd;
    // }

    // cmatrix TE = diagonalizer->TE(layer),
    //         TH = diagonalizer->TH(layer);
    // cdiagonal gamma = diagonalizer->Gamma(layer);

    // PropagationDirection part = PROPAGATION_TOTAL;
    // get_d(n, z1, part);
    // double d = get_d(n, z2, part);

    // if (std::ptrdiff_t(n) >= solver->interface) std::swap(z1, z2);

    // double result = 0.;
    // for (size_t i = 0; i != N; ++i) {
    //     cvector E(TE.data() + N*i, N),
    //             H(TH.data() + N*i, N);
    //     double TT = diagonalizer->source()->integrateField(field, layer, E, H);

    //     double gr = 2. * gamma[i].real(), gi = 2. * gamma[i].imag();
    //     double M = cosh(gi * d) - cos(gr * d);
    //     if (isinf(M)) {
    //         double VV = real(F0[i]*conj(F0[i])) + real(Fd[i]*conj(Fd[i])) - 2. * real(F0[i]*conj(Fd[i]));
    //         result += TT * VV;
    //     } else {
    //         double cos00, cosdd;
    //         dcomplex cos0d;
    //         if (is_zero(gr)) {
    //             cos00 = cosdd = z2-z1;
    //             cos0d = cos(gamma[i] * d) * (z2-z1);
    //         } else {
    //             cos00 = (sin(gr * (d-z1)) - sin(gr * (d-z2))) / gr;
    //             cosdd = (sin(gr * z2) - sin(gr * z1)) / gr;
    //             cos0d = (sin(gamma[i] * d - gr * z1) - sin(gamma[i] * d - gr * z2)) / gr;
    //         }
    //         double cosh00, coshdd;
    //         dcomplex cosh0d;
    //         if (is_zero(gi)) {
    //             cosh00 = coshdd = z2-z1;
    //             cosh0d = cos(gamma[i] * d) * (z2-z1);
    //         } else {
    //             cosh00 = (sinh(gi * (d-z1)) - sinh(gi * (d-z2))) / gi;
    //             coshdd = (sinh(gi * z2) - sinh(gi * z1)) / gi;
    //             cosh0d = (sin(gamma[i] * d - gi * z1) - sin(gamma[i] * d - gi * z2)) / gi;
    //         }
    //         double VV =      real(F0[i]*conj(F0[i])) * (cosh00 - cos00) +
    //                          real(Fd[i]*conj(Fd[i])) * (coshdd - cosdd) -
    //                     2. * real(F0[i]*conj(Fd[i])  * (cosh0d - cos0d));
    //         result += TT * VV / M;
    //     }
    // }

    // return result;
}

}}} // namespace plask::optical::slab
