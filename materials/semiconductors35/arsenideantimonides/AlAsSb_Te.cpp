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
#include "AlAsSb_Te.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlAsSb_Te::name() const { return NAME; }

std::string AlAsSb_Te::str() const { return StringBuilder("Al")("As")("Sb", Sb).dopant("Te", ND); }

MI_PARENT(AlAsSb_Te, AlAsSb)

AlAsSb_Te::AlAsSb_Te(const Material::Composition& Comp, double Val): AlAsSb(Comp) //, mGaAs_Te(Val), mAlAs_Te(Val)
{
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
    double mob_RT_AlAs = 30. + (310. - 30.) / (1.+pow(ND/8e17,2.)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    double mob_RT_AlSb = 30. + (200. - 30.) / (1.+pow(ND/4e17,3.25)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    mob_RT = 1. / (As/mob_RT_AlAs + Sb/mob_RT_AlSb - 9.3e-7*As*Sb); // for small amount of arsenide
}

MI_PROPERTY(AlAsSb_Te, mob,
            MISource("Stirn 1966")
            )
Tensor2<double> AlAsSb_Te::mob(double T) const {
    double tmob = mob_RT * pow(300./T,1.8);
    return ( Tensor2<double>(tmob, tmob) );
}

MI_PROPERTY(AlAsSb_Te, Nf,
            MISource("-"),
            MINote("TODO")
            )
double AlAsSb_Te::Nf(double /*T*/) const {
    return ( Nf_RT );
}

double AlAsSb_Te::doping() const {
    return ( ND );
}

Tensor2<double> AlAsSb_Te::cond(double T) const {
    double tCond = phys::qe * Nf(T)*1e6 * (mob(T).c00)*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType AlAsSb_Te::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(AlAsSb_Te, nr,
            MISource("C. Alibert et al., Journal of Applied Physics 69 (1991) 3208-3211"),
            MIArgumentRange(MaterialInfo::lam, 500, 7000),
            MINote("TODO")
            )
double AlAsSb_Te::nr(double lam, double T, double /*n*/) const {
    double tE = phys::h_eVc1e9/lam; // lam -> E
    double tE0 = 3.2;
    double tEd = 28.;
    double tEG = 2.338;
    double nR300K2 = 1. + tEd/tE0 + tEd*tE*tE/pow(tE0,3.) + tEd*pow(tE,4.)/(2.*pow(tE0,3.)*(tE0*tE0-tEG*tEG)) * log((2.*tE0*tE0-tEG*tEG-tE*tE)/(tEG*tEG-tE*tE));

    double nR300K;
    if (nR300K2>0) nR300K = sqrt(nR300K2);
    else nR300K = 1.; // TODO

    double nR = nR300K; // TODO // for E << Eg: dnR/dn = 0
    double dnRdT = As*4.6e-5 + Sb*1.19e-5; // from Adachi (2005) ebook p.243 tab. 10.6
    return ( nR + nR*dnRdT*(T-300.) );
}

MI_PROPERTY(AlAsSb, absp,
            MISource("H. Hattasan (2013)"),
            //MIArgumentRange(MaterialInfo::lam, 2000, 20000),
            MINote("temperature dependence - assumed: (1/abs)(dabs/dT)=1e-3"),
            MINote("only free-carrier absorption assumed")
            )
double AlAsSb_Te::absp(double lam, double T) const {
    double tAbs_RT = 1.9e-24 * Nf_RT * pow(lam,2.);
    return ( tAbs_RT + tAbs_RT*1e-3*(T-300.) );
}

bool AlAsSb_Te::isEqual(const Material &other) const {
    const AlAsSb_Te& o = static_cast<const AlAsSb_Te&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlAsSb::isEqual(other);
}

static MaterialsDB::Register<AlAsSb_Te> materialDB_register_AlAsSb_Te;

}}       // namespace plask::materials
