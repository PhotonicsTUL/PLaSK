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

#include "impedance.hpp"
#include "solver.hpp"
#include "diagonalizer.hpp"
#include "expansion.hpp"
#include "fortran.hpp"
#include "meshadapter.hpp"

namespace plask { namespace optical { namespace slab {

ImpedanceTransfer::ImpedanceTransfer(SlabBase* solver, Expansion& expansion): XanceTransfer(solver, expansion)
{
    writelog(LOG_DETAIL, "{}: Initializing Impedance Transfer", solver->getId());
}


void ImpedanceTransfer::getFinalMatrix()
{
    int N = int(diagonalizer->matrixSize());    // for using with LAPACK it is better to have int instead of std::size_t
    int N0 = int(diagonalizer->source()->matrixSize());
    size_t count = solver->stack.size();

    // M = TE(interface) * Y(interface-1) * invTH(interface);
    findImpedance(count-1, solver->interface-1);
    zgemm('n','n', N, N0, N, 1., Y.data(), N, diagonalizer->invTH(solver->stack[solver->interface]).data(), N, 0., wrk, N);
    zgemm('n','n', N0, N0, N, 1., diagonalizer->TE(solver->stack[solver->interface]).data(), N0, wrk, N, 0., M.data(), N0);

    // Find the(diagonalized field) admittance matrix and store it for the future reference
    findImpedance(0, solver->interface);
    // M += TE(interface-1) * Y(interface) * invTH(interface-1);
    zgemm('n','n', N, N0, N, 1., Y.data(), N, diagonalizer->invTH(solver->stack[solver->interface-1]).data(), N, 0., wrk, N);
    zgemm('n','n', N0, N0, N, 1., diagonalizer->TE(solver->stack[solver->interface-1]).data(), N0, wrk, N, 1., M.data(), N0);
}


void ImpedanceTransfer::findImpedance(std::ptrdiff_t start, std::ptrdiff_t end)
{
    const std::ptrdiff_t inc = (start < end) ? 1 : -1;

    const std::size_t N = diagonalizer->matrixSize();
    const std::size_t NN = N*N;

    // Some temporary variables
    cdiagonal gamma, y1(N), y2(N);

    std::exception_ptr error;

    PLASK_OMP_PARALLEL_FOR_1
    for (int l = 0; l < int(diagonalizer->lcount); ++l) {
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
    for (std::size_t i = 0; i < N; i++) {
        y1[i] = gamma[i] * solver->vpml.factor;
        if (real(y1[i]) < -SMALL) { y1[i] = -y1[i]; y2[i] = -y2[i]; }
        if (imag(y1[i]) > SMALL) { y1[i] = -y1[i]; y2[i] = -y2[i]; }
    }
    get_y1(y1, solver->vpml.size, y1);
    std::fill_n(Y.data(), NN, dcomplex(0.));
    for (std::size_t i = 0; i < N; i++) Y(i,i) = - y2[i] / y1[i];

    // First layer
    double h = solver->vpml.dist;
    gamma = diagonalizer->Gamma(solver->stack[start]);
    get_y1(gamma, h, y1);
    get_y2(gamma, h, y2);
    // off-diagonal elements of Y are 0
    for (std::size_t i = 0; i < N; i++) Y(i,i) = y2[i] * y2[i] / (y1[i] - Y(i,i)) - y1[i]; // Y = y2 * inv(y1-Y) * y2 - y1

    // save the Y matrix for 1-st layer
    storeY(start);

    if (start == end) return;

    // Declare temporary matrixH) on 'wrk' array
    cmatrix work(N, N, wrk);

    for (std::ptrdiff_t n = start+inc; n != end; n += inc)
    {
        gamma = diagonalizer->Gamma(solver->stack[n]);

        h = solver->vbounds->at(n) - solver->vbounds->at(n-1);
        get_y1(gamma, h, y1);
        get_y2(gamma, h, y2);

        // The main equation
        // Y[n] = y2 * tH * inv(y1*tH - tE*Y[n-1]) * y2  -  y1

        mult_matrix_by_matrix(diagonalizer->TE(solver->stack[n-inc]), Y, temp);         // work = tE * Y[n-1]
        mult_matrix_by_matrix(diagonalizer->invTE(solver->stack[n]), temp, work);       // ...

        mult_matrix_by_matrix(diagonalizer->invTH(solver->stack[n]), diagonalizer->TH(solver->stack[n-inc]), temp); // compute tH

        for (std::size_t j = 0; j < N; j++)
            for (std::size_t i = 0; i < N; i++) Y(i,j) = y1[i]*temp(i,j) - work(i,j);   // Y[n] = y1 * tH - work

        for (std::size_t i = 0; i < NN; i++) work[i] = 0.;
        for (std::size_t j = 0, i = 0; j < N; j++, i += N+1) work[i] = y2[j];           // work = y2

        invmult(Y, work);                                                               // work = inv(Y[n]) * (work = y2)
        mult_matrix_by_matrix(temp, work, Y);                                           // Y[n] = tH * work

        for (std::size_t j = 0; j < N; j++)
            for (std::size_t i = 0; i < N; i++) Y(i,j) *= y2[i];                        // Y[n] = y2 * Y[n]

        for (std::size_t j = 0, i = 0; j < N; j++, i += N+1) Y[i] -= y1[j];             // Y[n] = Y[n] - y1

        // Save the Y matrix for n-th layer
        storeY(n);
    }
}



void ImpedanceTransfer::determineFields()
{
    if (fields_determined == DETERMINED_RESONANT) return;

    writelog(LOG_DETAIL, solver->getId() + ": Determining optical fields");

    const std::size_t N = diagonalizer->matrixSize();
    const std::size_t N0 = diagonalizer->source()->matrixSize();
    size_t count = solver->stack.size();

    const std::size_t NN = N*N;

    // Assign all the required space
    cdiagonal gamma, y1(N), y2(N);

    // Assign the space for the field vectors
    fields.resize(count);

    // Temporary vector for storing fields in the real domain
    cvector tv(N0);

    // Obtain the physical fields at the last layer
    needAllY = true;
    interface_field = nullptr;
    auto H = getInterfaceVector();

    // Declare temporary matrix on 'wrk' array
    cmatrix work(N, N, wrk);

    for (int pass = 0; pass < 1 || (pass < 2 && solver->interface != std::ptrdiff_t(count)); pass++)
    {
        // each pass for below and above the interface

        std::ptrdiff_t start, end;
        int inc;
        switch (pass) {
            case 0: start = solver->interface-1; end = -1;    inc =  1; break;
            case 1: start = solver->interface;   end = count; inc = -1; break;
        }

        // Hd[start] = invTH[start] H
        fields[start].Hd = cvector(N);
        mult_matrix_by_vector(diagonalizer->invTH(solver->stack[start]), H, fields[start].Hd);

        fields[start].Hd *= double(inc);

        for (std::ptrdiff_t n = start; n != end; n -= inc)
        {
            const std::size_t curr = solver->stack[n];

            double h = (n == 0 || n == std::ptrdiff_t(count)-1)? solver->vpml.dist : solver->vbounds->at(n) - solver->vbounds->at(n-1);
            gamma = diagonalizer->Gamma(curr);
            get_y1(gamma, h, y1);
            get_y2(gamma, h, y2);

            // work = Y[n] + y1
            cmatrix Y = getY(n);
            for (std::size_t i = 0; i < NN; i++) work[i] = Y[i];
            for (std::size_t i = 0; i < N; i++) work (i,i) += y1[i];

            // H0[n] = work * Hd[n]
            fields[n].H0 = cvector(N);
            mult_matrix_by_vector(work, fields[n].Hd, fields[n].H0);

            // H0[n] = - inv(y2) * H0[0]
            for (size_t i = 0; i < N; i++) {
                if (abs(y2[i]) < SMALL)         // Actually we cannot really compute H0 in this case.
                    fields[n].H0[i] = 0.;       // So let's cheat a little, as the field cannot
                else                            // increase to the boundaries.
                    fields[n].H0[i] /= - y2[i];
            }

            if (n != end+inc) { // not the last layer
                const std::size_t prev = solver->stack[n-inc];
                // Hd[n-inc] = invTH[n-inc] * TH[n] * H0[n]
                fields[n-inc].Hd = cvector(N);
                mult_matrix_by_vector(diagonalizer->TH(curr), fields[n].H0, tv);
                mult_matrix_by_vector(diagonalizer->invTH(prev), tv, fields[n-inc].Hd);
            }

            // Now compute the electric fields

            // Ed[n] = Y[n] * Hd[n]
            fields[n].Ed = cvector(N);
            mult_matrix_by_vector(Y, fields[n].Hd, fields[n].Ed);

            if (n != start) {
                std::size_t next = solver->stack[n+inc];
                // E0[n+inc] = invTE[n+inc] * TE[n] * Ed[n]
                fields[n+inc].E0 = cvector(N);
                mult_matrix_by_vector(diagonalizer->TE(curr), fields[n].Ed, tv);
                mult_matrix_by_vector(diagonalizer->invTE(next), tv, fields[n+inc].E0);
            }

            // An alternative method is to find the E0 from the following equation:
            // E0 = y1 * H0 + y2 * Hd
            // for (int i = 0; i < N; i++)
            //     fields[n].E0[i] = y1[i] * fields[n].H0[i]  +  y2[i] * fields[n].Hd[i];
            // However in some cases this can make the electric field discontinuous.
        }
        if (start != end) {
            // Zero electric field at the end
            std::ptrdiff_t n = end + inc;
            fields[n].E0 = cvector(N, 0.);
        }
    }

    // Now fill the Y matrix with the one from the interface (necessary for interfaceField*)
    memcpy(Y.data(), getY(solver->interface-1).data(), NN*sizeof(dcomplex));

    needAllY = false;
    fields_determined = DETERMINED_RESONANT;

    // Finally normalize fields
    if (solver->emission == SlabBase::EMISSION_BOTTOM || solver->emission == SlabBase::EMISSION_TOP) {
        const std::size_t n = (solver->emission == SlabBase::EMISSION_BOTTOM)? 0 : count-1;
        const std::size_t l = solver->stack[n];

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


cvector ImpedanceTransfer::getReflectionVector(const cvector& incident, IncidentDirection side)
{
    throw NotImplemented("reflection with impedance transfer");
}

void ImpedanceTransfer::determineReflectedFields(const cvector& incident, IncidentDirection side)
{
    throw NotImplemented("reflection with impedance transfer");
}


}}} // namespace plask::optical::slab
