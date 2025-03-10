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
#include "AlGaAsSb_Si.hpp"

#include <cmath>
#include <plask/material/db.hpp>      // MaterialsDB::Register
#include <plask/material/info.hpp>    // MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlGaAsSb_Si::name() const { return NAME; }

MI_PARENT(AlGaAsSb_Si, AlGaAsSb)

std::string AlGaAsSb_Si::str() const { return StringBuilder("Al", Al)("Ga")("As")("Sb", Sb).dopant("Si", NA); }

AlGaAsSb_Si::AlGaAsSb_Si(const Material::Composition& Comp, double Val): AlGaAsSb(Comp)
{
    NA = Val;
    if ( NA < pow(10.,((1.-2.27)/(-0.0731))) )
        Nf_RT = NA;
    else
        Nf_RT = ( (-0.0731*log10(NA)+2.27) * NA );
    double mob_RT_AlSb = 30. + (300. - 30.) / (1.+pow(NA/3e17,1.54)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    double mob_RT_GaSb = 95. + (565. - 95.) / (1.+pow(NA/4e18,0.85)); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
    mob_RT = 1. / (Al/mob_RT_AlSb + Ga/mob_RT_GaSb + 6e-8*Al*Ga); // for small amount of arsenide
}

MI_PROPERTY(AlGaAsSb_Si, mob,
            MISource("D. Martin et al., Semiconductors Science and Technology 19 (2004) 1040-1052"), // TODO
            MINote("fit by Lukasz Piskorski")
            )
Tensor2<double> AlGaAsSb_Si::mob(double T) const {
    double tmob = mob_RT * pow(300./T,1.2);
    return ( Tensor2<double>(tmob,tmob) ); // 1e-4: cm^2/(V*s) -> m^2/(V*s)
}

MI_PROPERTY(AlGaAsSb_Si, Nf,
            MISource("Mirowska et al., Domieszkowanie ..."), // TODO
            MINote("fit by Lukasz Piskorski")
            )
double AlGaAsSb_Si::Nf(double T) const {
    double tD;
    if (Nf_RT <= 6.4e17) tD = 1e17*pow(Nf_RT,-1.014);
    else tD = 0.088;
    return ( Nf_RT*pow(T/300.,tD) );
}

double AlGaAsSb_Si::doping() const {
    return ( NA );
}

MI_PROPERTY(AlGaAsSb_Si, cond,
            MINote("cond(T) = cond(300K)*(300/T)^d")
            )
Tensor2<double> AlGaAsSb_Si::cond(double T) const {
    double tCond = phys::qe * Nf(T)*1e6 * (mob(T).c00)*1e-4;
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType AlGaAsSb_Si::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(AlGaAsSb_Si, nr,
            MISource("Alibert, J. Appl. Phys (1991)"),
            //MIArgumentRange(MaterialInfo::lam, 620, 2560),
            MINote("for AlGaAsSb lattice matched to GaSb")
            )
double AlGaAsSb_Si::nr(double lam, double T, double) const {
    double tE = phys::h_eVc1e9/lam; // lam -> E
    double tE0 = 1.89*Ga+3.2*Al-0.36*Al*Ga;
    double tEd = 24.5*Ga+28.*Al-4.4*Al*Ga;
    double tEG = 0.725*Ga+2.338*Al-0.47*Al*Ga;
    double nR300K2 = 1. + tEd/tE0 + tEd*tE*tE/pow(tE0,3.) + tEd*pow(tE,4.)/(2.*pow(tE0,3.)*(tE0*tE0-tEG*tEG)) * log((2.*tE0*tE0-tEG*tEG-tE*tE)/(tEG*tEG-tE*tE));

    double nR300K;
    if (nR300K2>0) nR300K = sqrt(nR300K2);
    else nR300K = 1.; // TODO
    //taken from p-GaSb
    double nR = nR300K - 0.0074*(Nf_RT*1e-18); // -7.4e-3 - fit by Lukasz Piskorski (based on: P.P. Paskov (1997) J. Appl. Phys. 81, 1890-1898)
    double dnRdT = Al*As*4.6e-5 + Al*Sb*1.19e-5 + Ga*As*4.5e-5 + Ga*Sb*8.2e-5;
    return ( nR + nR*dnRdT*(T-300.) ); // 8.2e-5 - from Adachi (2005) ebook p.243 tab. 10.6
}

MI_PROPERTY(AlGaAsSb_Si, absp,
            MINote("fit by Lukasz Piskorski")
            )
double AlGaAsSb_Si::absp(double lam, double T) const {
    double tAbs_RT = 1e24*exp(-lam/33.) + 2.02e-24*Nf_RT*pow(lam,2.) + pow(20.*sqrt(Nf_RT*1e-18),1.05);
    return ( tAbs_RT + tAbs_RT*1e-3*(T-300.) );
}

bool AlGaAsSb_Si::isEqual(const Material &other) const {
    const AlGaAsSb_Si& o = static_cast<const AlGaAsSb_Si&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlGaAsSb::isEqual(other);
}

static MaterialsDB::Register<AlGaAsSb_Si> materialDB_register_AlGaAsSb_Si;

}} // namespace plask::materials
