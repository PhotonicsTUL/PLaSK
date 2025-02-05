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
#include "GaN_Mg.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register


namespace plask { namespace materials {

MI_PARENT(GaN_Mg, GaN)

std::string GaN_Mg::name() const { return NAME; }

std::string GaN_Mg::str() const { return StringBuilder("GaN").dopant("Mg", NA); }

GaN_Mg::GaN_Mg(double Val) {
    Nf_RT = 0.65E4*std::pow(Val,0.71);
    NA = Val;
    mob_RT = 26.7*exp(-Nf_RT/1e18);
}

MI_PROPERTY(GaN_Mg, mob,
            MISource("P. Kozodoy et al., J. Appl. Phys. 87 (2000) 1832"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MINote("based on 9 papers (2000-2009): MBE-grown Mg-doped GaN"),
            MINote("Nf: 2e17 - 6e18 cm^-3")
            )
Tensor2<double> GaN_Mg::mob(double T) const {
    double tMob = mob_RT * (T*T*2.495E-5 -T*2.268E-2 +5.557);
    return Tensor2<double>(tMob,tMob);
}

MI_PROPERTY(GaN_Mg, Nf,
            MISource("P. Kozodoy et al., J. Appl. Phys. 87 (2000) 1832"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MINote("based on 4 papers (1998-2008): MBE-grown Mg-doped GaN"),
            MINote("Mg: 1e19 - 8e20 cm^-3")
            )
double GaN_Mg::Nf(double T) const {
	return  Nf_RT * (T*T*2.884E-4 -T*0.147 + 19.080);
}

MI_PROPERTY(GaN_Mg, Na,
            MINote("-")
            )
double GaN_Mg::Na() const {
    return ( NA );
}

MI_PROPERTY(GaN_Mg, Nd,
            MINote("-")
            )
double GaN_Mg::Nd() const {
    return ( 0. );
}

double GaN_Mg::doping() const {
    return NA;
}

MI_PROPERTY(GaN_Mg, cond,
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
Tensor2<double> GaN_Mg::cond(double T) const {
    return Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11);
}

Material::ConductivityType GaN_Mg::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(GaN_Mg, absp,
            MISeeClass<GaN>(MaterialInfo::absp)
            )
double GaN_Mg::absp(double lam, double T) const {
    double dE = phys::h_eVc1e9/lam - Eg(T, 0., 'G');
    double N = doping() * 1e-18;
    return  (19000.+200.*N) * exp(dE/(0.019+0.0001*N)) + (330.+30.*N) * exp(dE/(0.07+0.0008*N));
}

bool GaN_Mg::isEqual(const Material &other) const {
    const GaN_Mg& o = static_cast<const GaN_Mg&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && o.cond_RT == this->cond_RT;
}


Tensor2<double> GaN_Mg_bulk::thermk(double T, double /*t*/) const {
    return GaN_Mg::thermk(T, INFINITY);
}


static MaterialsDB::Register<GaN_Mg> materialDB_register_Mg;

static MaterialsDB::Register<GaN_Mg_bulk> materialDB_register_GaN_Mg_bulk;


}}       // namespace plask::materials
