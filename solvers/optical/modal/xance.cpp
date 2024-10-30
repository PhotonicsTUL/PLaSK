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
/*#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define BOOST_USE_WINDOWS_H
#endif*/

#include "xance.hpp"
#include "solver.hpp"
#include "diagonalizer.hpp"
#include "expansion.hpp"
#include "fortran.hpp"
#include "meshadapter.hpp"

namespace plask { namespace optical { namespace modal {

XanceTransfer::XanceTransfer(ModalBase* solver, Expansion& expansion): Transfer(solver, expansion)
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


inline static dcomplex _int_xance(double z1, double z2, dcomplex a, dcomplex b = 0.) {
    if (is_zero(a)) return 0.5 * (z2 - z1) * cosh(b);
    const dcomplex a2 = 0.5 * a;
    return cosh(a2 * (z1 + z2) + b) * sinh(a2 * (z2 - z1)) / a;
}

double XanceTransfer::integrateField(WhichField field, size_t n, double z1, double z2) {
    size_t layer = solver->stack[n];
    size_t N = diagonalizer->matrixSize();

    const cvector& E0 = fields[n].E0;
    const cvector& Ed = fields[n].Ed;
    const cvector& H0 = fields[n].H0;
    const cvector& Hd = fields[n].Hd;

    cmatrix TE = diagonalizer->TE(layer),
            TH = diagonalizer->TH(layer);
    cdiagonal gamma = diagonalizer->Gamma(layer);

    double d = get_d(n, z1, z2);

    return diagonalizer->source()->integrateField(field, layer, TE, TH,
        [z1, z2, d, gamma, E0, Ed, H0, Hd](size_t i, size_t j) -> std::pair<dcomplex,dcomplex> {
            const dcomplex igm = I * (gamma[i] - conj(gamma[j])), igp = I * (gamma[i] + conj(gamma[j]));
            const dcomplex igid = I * gamma[i] * d, igjd = I * conj(gamma[j]) * d;
            dcomplex E = 0.;
            dcomplex H = 0.;
            if (!((is_zero(E0[i]) || is_zero(E0[j])) && (is_zero(H0[i]) || is_zero(H0[j])))) {
                dcomplex val = _int_xance(z1, z2, -igm, igm*d) - _int_xance(z1, z2, -igp, igp*d);
                E += E0[i] * conj(E0[j]) * val; H += H0[i] * conj(H0[j]) * val;
            }
            if (!((is_zero(E0[i]) || is_zero(Ed[j])) && (is_zero(H0[i]) || is_zero(Hd[j])))) {
                dcomplex val = _int_xance(z1, z2, -igp, igid) - _int_xance(z1, z2, -igm, igid);
                E += E0[i] * conj(Ed[j]) * val; H += H0[i] * conj(Hd[j]) * val;
            }
            if (!((is_zero(Ed[i]) || is_zero(E0[j])) && (is_zero(Hd[i]) || is_zero(H0[j])))) {
                dcomplex val = _int_xance(z1, z2, igp, -igjd) - _int_xance(z1, z2, igm, igjd);
                E += Ed[i] * conj(E0[j]) * val; H += Hd[i] * conj(H0[j]) * val;
            }
            if (!((is_zero(Ed[i]) || is_zero(Ed[j])) && (is_zero(Hd[i]) || is_zero(Hd[j])))) {
                dcomplex val = _int_xance(z1, z2, igm) - _int_xance(z1, z2, igp);
                E += Ed[i] * conj(Ed[j]) * val; H += Hd[i] * conj(Hd[j]) * val;
            }
            dcomplex f = 1. / (sinh(igid) * sinh(-igjd));
            return std::make_pair(f * E, f * H);
        });
}

}}} // namespace plask::optical::modal
