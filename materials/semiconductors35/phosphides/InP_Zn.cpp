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
#include "InP_Zn.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InP_Zn::name() const { return NAME; }

std::string InP_Zn::str() const { return StringBuilder("InP").dopant("Zn", NA); }

InP_Zn::InP_Zn(double Val) {
    Nf_RT = 0.75*Val;
    NA = Val;
    mob_RT = 120./(1+pow((Nf_RT/2e18),1.00));
}

MI_PROPERTY(InP_Zn, mob,
            MISource("TODO"),
            MINote("no temperature dependence")
            )
Tensor2<double> InP_Zn::mob(double /*T*/) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
}

MI_PROPERTY(InP_Zn, Nf,
            MISource("TODO"),
            MINote("no temperature dependence")
            )
double InP_Zn::Nf(double /*T*/) const {
    return ( Nf_RT );
}

double InP_Zn::doping() const {
    return ( NA );
}

MI_PROPERTY(InP_Zn, cond,
			MINote("no temperature dependence")
            )
Tensor2<double> InP_Zn::cond(double /*T*/) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT*1e-4;
    return (Tensor2<double>(tCond, tCond));
}

Material::ConductivityType InP_Zn::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(InP_Zn, absp,
            MISource("TODO"),
            MINote("no temperature dependence")
            )
double InP_Zn::absp(double lam, double /*T*/) const {
    double tAbsp(0.);
    if ((lam > 1200.) && (lam < 1400.)) // only for 1300 nm TODO
        tAbsp = 23. * pow(Nf_RT/1e18, 0.7);
    else if ((lam > 1450.) && (lam < 1650.)) // only for 1550 nm TODO
        tAbsp = 38. * pow(Nf_RT/1e18, 0.7);
    else if ((lam > 2230.) && (lam < 2430.)) // only for 2330 nm TODO
        tAbsp = 52. * pow(Nf_RT/1e18, 1.2);
    else if ((lam > 8900.) && (lam < 9100.)) // only for 9000 nm TODO
        tAbsp = 200. * (Nf_RT/1e18);
    return ( tAbsp );
}

bool InP_Zn::isEqual(const Material &other) const {
    const InP_Zn& o = static_cast<const InP_Zn&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT;
}

static MaterialsDB::Register<InP_Zn> materialDB_register_InP_Zn;

}} // namespace plask::materials
