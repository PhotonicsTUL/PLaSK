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
#include "metal.hpp"
#include "Faddeeva.hh"

namespace plask { namespace materials {

dcomplex LorentzDrudeMetal::opt_eps(double lam) const
{
    double wl = 1239.84193009 / lam; // angular frequency in eV

    dcomplex eps_j; // total permittivity

    size_t k = f.size();

    double Wp = sqrt(f[0]) * wp;

    dcomplex epsf = 1. - Wp*Wp / (wl*wl - I * G[0] * wl); // first component of relative permittivity
    dcomplex epsb = dcomplex(0., 0.); // second component of relative permittivity
    for (size_t j = 1; j < k; ++j)
        epsb += ((f[j] * wp*wp) / ((w[j] * w[j] - wl*wl) + I * G[j] * wl));

    return epsf + epsb; // total permittivity
}

dcomplex BrendelBormannMetal::opt_eps(double lam) const
{
    double wl = 1239.84193009 / lam; // angular frequency in eV

    size_t k = f.size();

    double Wp = sqrt(f[0]) * wp;

    dcomplex epsf = 1. - Wp*Wp / (wl*wl - I * G[0] * wl); // first component of relative permittivity
    dcomplex epsb = dcomplex(0., 0.); // second component of relative permittivity
    for (size_t j = 1; j < k; ++j)
    {
        dcomplex aj = sqrt(wl*wl - I * G[j] * wl);
        dcomplex zp = I * (aj + w[j]) / (sqrt(2.) * s[j]);
        dcomplex zm = I * (aj - w[j]) / (sqrt(2.) * s[j]);
        epsb += ((-I * sqrt(PI) * f[j] * wp*wp) / (2. * sqrt(2.) * aj * s[j])) * (exp(zm*zm) * Faddeeva::erfc(zm) + exp(zp*zp) * Faddeeva::erfc(zp));
    }
    return epsf + epsb; // total permittivity
}

}} // namespace plask::materials
