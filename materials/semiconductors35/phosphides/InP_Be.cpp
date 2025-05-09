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
#include "InP_Be.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InP_Be::name() const { return NAME; }

std::string InP_Be::str() const { return StringBuilder("InP").dopant("Be", NA); }

InP_Be::InP_Be(double Val) {
    Nf_RT = Val; // TODO (it is not from publication)
    NA = Val; // TODO (it is not from publication)
    mob_RT = 140./(1+pow((Nf_RT/1e18),0.50));
}

MI_PROPERTY(InP_Be, mob,
            MISource("TODO"),
            MINote("no temperature dependence")
            )
Tensor2<double> InP_Be::mob(double /*T*/) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
}

MI_PROPERTY(InP_Be, Nf,
            MISource("TODO"),
            MINote("no temperature dependence")
            )
double InP_Be::Nf(double /*T*/) const {
    return ( Nf_RT );
}

double InP_Be::doping() const {
    return ( NA );
}

MI_PROPERTY(InP_Be, cond,
			MINote("no temperature dependence")
            )
Tensor2<double> InP_Be::cond(double /*T*/) const {
    double tCond = phys::qe * Nf_RT*1e6 * mob_RT*1e-4;
    return (Tensor2<double>(tCond, tCond));
}

Material::ConductivityType InP_Be::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(InP_Be, absp,
            MISource("TODO"),
            MINote("no temperature dependence")
            )
double InP_Be::absp(double lam, double /*T*/) const {
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

bool InP_Be::isEqual(const Material &other) const {
    const InP_Be& o = static_cast<const InP_Be&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT;
}

static MaterialsDB::Register<InP_Be> materialDB_register_InP_Be;

}} // namespace plask::materials
