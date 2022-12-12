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
#include "GaSb_Te.hpp"

#include <cmath>
#include "plask/material/db.hpp"      // MaterialsDB::Register
#include "plask/material/info.hpp"    // MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaSb_Te::name() const { return NAME; }

std::string GaSb_Te::str() const { return StringBuilder("GaSb").dopant("Te", ND); }

GaSb_Te::GaSb_Te(double Val) {
    ND = Val;
    // based on: Chiu (1990) Te doping (1990) Appl. Phys. Lett. (Fig. 4), fit by L. Piskorski
    if (ND <= 1e18)
        Nf_RT = ND;
    else
    {
        double tNL = log10(ND);
        double tnL = 0.383027*tNL*tNL*tNL - 22.1278*tNL*tNL + 425.212*tNL - 2700.2222;
        Nf_RT = ( pow(10.,tnL) );
    }
    mob_RT = 550. + (6300. - 550.) / (1.+pow(Nf_RT/2e17,0.786));
}

MI_PROPERTY(GaSb_Te, mob,
            MISource("Chiu (1990) Te doping (1990) Appl. Phys. Lett. (Fig. 4)"),
            MINote("fit by Lukasz Piskorski")
            )
Tensor2<double> GaSb_Te::mob(double T) const {
    double tmob = mob_RT * pow(300./T,1.);
    return ( Tensor2<double>(tmob,tmob) ); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(GaSb_Te, Nf,
            MISource("TODO"),
            MINote("fit by Lukasz Piskorski")
            )
double GaSb_Te::Nf(double T) const {
    double tD = 0.4506*log10(Nf_RT)-7.95;
    return ( Nf_RT*pow(T/300.,tD) );
}

double GaSb_Te::doping() const {
    return ( ND );
}

MI_PROPERTY(GaSb_Te, cond,
            MINote("-") // TODO
            )
Tensor2<double> GaSb_Te::cond(double T) const {
    double tCond = phys::qe * Nf(T)*1e6 * (mob(T).c00)*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType GaSb_Te::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(GaSb_Te, nr,
            MISource("M. Munoz-Uribe et al., Electronics Letters 32 (1996) 262-264"),
            MISource("D.E. Aspnes et al., Phys. Rev. B 27 (1983) 985-1009"),
            MISource("S. Adachi, J. Appl. Phys. 66 (1989) 6030-6040"),
            MIArgumentRange(MaterialInfo::lam, 620, 2560),
            MINote("fit by Lukasz Piskorski"),
            MINote("no fitting data from 827-1798nm wavelength range"),
            MINote("basing on fig. 5a (Adachi,1989) nR(wv) relation can be used for 620-4700nm wavelength range")
            )
double GaSb_Te::nr(double lam, double T, double) const {
    double tE = phys::h_eVc1e9/lam; // lam -> E
    double nR300K = 0.502*tE*tE*tE - 1.216*tE*tE + 1.339*tE + 3.419;
    double nR = nR300K - 0.029*(Nf_RT*1e-18); // -2.9e-2 - fit by Lukasz Piskorski (based on: P.P. Paskov (1997) J. Appl. Phys. 81, 1890-1898)
    return ( nR + nR*8.2e-5*(T-300.) ); // 8.2e-5 - from Adachi (2005) ebook p.243 tab. 10.6
}

MI_PROPERTY(GaSb_Te, absp,
            MISource("A. Chandola et al., Semicond. Sci. Technol. 20 (2005) 886-893"),
            MIArgumentRange(MaterialInfo::lam, 1600, 4700),
            MINote("temperature dependence - assumed: (1/abs)(dabs/dT)=1e-3"),
            MINote("fit by Lukasz Piskorski")
            )
double GaSb_Te::absp(double lam, double T) const {
    double tAbs_RT = 1e24*exp(-lam/33.) + 1.7e-24*Nf_RT*pow(lam,1.95) + pow(20.*sqrt(Nf_RT*1e-18),1.05);
    return ( tAbs_RT + tAbs_RT*1e-3*(T-300.) );
}

bool GaSb_Te::isEqual(const Material &other) const {
    const GaSb_Te& o = static_cast<const GaSb_Te&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaSb::isEqual(other);
}

static MaterialsDB::Register<GaSb_Te> materialDB_register_GaSb_Te;

}} // namespace plask::materials
