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
#include "GaAs_Si.hpp"

#include <cmath>
#include "plask/material/db.hpp"      // MaterialsDB::Register
#include "plask/material/info.hpp"    // MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string GaAs_Si::name() const { return NAME; }

std::string GaAs_Si::str() const { return StringBuilder("GaAs").dopant("Si", ND); }

GaAs_Si::GaAs_Si(double Val) {
    Nf_RT = Val;
    ND = Val;
    mob_RT = 6600./(1+pow((Nf_RT/5e17),0.53));
}

MI_PROPERTY(GaAs_Si, EactA,
            MINote("this parameter will be removed")
            )
double GaAs_Si::EactA(double /*T*/) const {
    return 0.;
}

MI_PROPERTY(GaAs_Si, EactD,
            MISource("L. Piskorski, PhD thesis (2010)")
            )
double GaAs_Si::EactD(double /*T*/) const {
    return 1e-3;
}

MI_PROPERTY(GaAs_Si, mob,
            MISource("fit to n-GaAs:Si (based on 8 papers 1982 - 2003)"),
            MINote("no temperature dependence")
            )
Tensor2<double> GaAs_Si::mob(double T) const {
    double mob_T = mob_RT * pow(300./T,1.4);
    return ( Tensor2<double>(mob_T,mob_T) );
}

MI_PROPERTY(GaAs_Si, Nf,
            MISource("based on 3 papers 1982 - 1996"),
            MINote("no temperature dependence")
            )
double GaAs_Si::Nf(double /*T*/) const {
    return ( Nf_RT );
}

MI_PROPERTY(GaAs_Si, Na,
            MINote("-")
            )
double GaAs_Si::Na() const {
    return ( 0. );
}

MI_PROPERTY(GaAs_Si, Nd,
            MINote("-")
            )
double GaAs_Si::Nd() const {
    return ( ND );
}

double GaAs_Si::doping() const {
    return ( ND );
}

MI_PROPERTY(GaAs_Si, cond,
			MINote("no temperature dependence")
            )
Tensor2<double> GaAs_Si::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType GaAs_Si::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(GaAs_Si, absp,
            MISource("fit by Lukasz Piskorski") // TODO
            )
double GaAs_Si::absp(double lam, double T) const {
    double tDWl = phys::h_eVc1e9*(Eg(300.,0.,'G')-Eg(T,0.,'G'))/(Eg(300.,0.,'G')*Eg(T,0.,'G'));
    double tWl = (lam-tDWl)*1e-3;
    double tAbsp(0.);
    if (tWl <= 6.) // 0.85-6 um
        tAbsp = (Nf_RT/1e18)*(1e24*exp(-tWl/0.0169)+4.67+0.00211*pow(tWl,4.80));
    else if (tWl <= 27.) // 6-27 um
        tAbsp = (Nf_RT/1e18)*(-8.4+0.233*pow(tWl,2.6));
    return ( tAbsp );
}

bool GaAs_Si::isEqual(const Material &other) const {
    const GaAs_Si& o = static_cast<const GaAs_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && GaAs::isEqual(other);
}

static MaterialsDB::Register<GaAs_Si> materialDB_register_GaAs_Si;

}} // namespace plask::materials
