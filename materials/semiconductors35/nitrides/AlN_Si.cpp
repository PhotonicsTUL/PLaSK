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
#include "AlN_Si.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlN_Si::name() const { return NAME; }

std::string AlN_Si::str() const { return StringBuilder("AlN").dopant("Si", ND); }

MI_PARENT(AlN_Si, AlN)

AlN_Si::AlN_Si(double Val) {
    Nf_RT = 6.197E-19*pow(Val,1.805);
    ND = Val;
    //mobRT(Nf_RT),
    mob_RT = 29.410*exp(-1.838E-17*Nf_RT);
}

MI_PROPERTY(AlN_Si, mob,
            MISource("K. Kusakabe et al., Physica B 376-377 (2006) 520"),
            MIArgumentRange(MaterialInfo::T, 270, 400),
            MINote("based on 4 papers (2004-2008): Si-doped AlN")
			)
Tensor2<double> AlN_Si::mob(double T) const {
    double tMob = mob_RT * (1.486 - T*0.00162);
    return (Tensor2<double>(tMob,tMob));
}

MI_PROPERTY(AlN_Si, Nf,
            MISource("Y. Taniyasu, Nature Letters 44 (2006) 325"),
            MIArgumentRange(MaterialInfo::T, 300, 400),
            MINote("based on 2 papers (2004-2008): Si-doped AlN")
            )
double AlN_Si::Nf(double T) const {
	return ( Nf_RT * 3.502E-27*pow(T,10.680) );
}

double AlN_Si::doping() const {
    return ND;
}

MI_PROPERTY(AlN_Si, cond,
            MIArgumentRange(MaterialInfo::T, 300, 400)
            )
Tensor2<double> AlN_Si::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

Material::ConductivityType AlN_Si::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(AlN_Si, absp,
            MISeeClass<AlN>(MaterialInfo::absp)
            )
double AlN_Si::absp(double lam, double /*T*/) const {
    const double a = phys::h_eVc1e9/lam - 6.28,
                 b = ND/1e18;
    return ( (19000+400*b)*exp(a/(0.019+0.001*b)) + (330+200*b)*exp(a/(0.07+0.016*b)) );
}

bool AlN_Si::isEqual(const Material &other) const {
    const AlN_Si& o = static_cast<const AlN_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT;
}

static MaterialsDB::Register<AlN_Si> materialDB_register_AlN_Si;

}}       // namespace plask::materials
