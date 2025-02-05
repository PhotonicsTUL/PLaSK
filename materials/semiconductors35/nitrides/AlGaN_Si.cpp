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
#include "AlGaN_Si.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlGaN_Si::name() const { return NAME; }

std::string AlGaN_Si::str() const { return StringBuilder("Al", Al)("Ga")("N").dopant("Si", ND); }

MI_PARENT(AlGaN_Si, AlGaN)

AlGaN_Si::AlGaN_Si(const Material::Composition& Comp, double Val): AlGaN(Comp), mGaN_Si(Val), mAlN_Si(Val)
{
    ND = Val;
}

MI_PROPERTY(AlGaN_Si, mob,
            MISource("based on 11 papers 1997-2008 about Si-doped AlGaN"),
            MISource("based on Si-doped GaN and AlN")
            )
Tensor2<double> AlGaN_Si::mob(double T) const {
    double lMob = Al*mAlN_Si.mob(T).c00 + pow(Ga,1.415+19.63*exp(-5.456*Al))*mGaN_Si.mob(T).c00,
           vMob = Al*mAlN_Si.mob(T).c11 + pow(Ga,1.415+19.63*exp(-5.456*Al))*mGaN_Si.mob(T).c11;
    return (Tensor2<double>(lMob, vMob));
}

MI_PROPERTY(AlGaN_Si, Nf,
            MINote("linear interpolation: Si-doped GaN, AlN")
            )
double AlGaN_Si::Nf(double T) const {
    return ( mAlN_Si.Nf(T)*Al + mGaN_Si.Nf(T)*Ga );
}

double AlGaN_Si::doping() const {
    return ND;
}

Tensor2<double> AlGaN_Si::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

Material::ConductivityType AlGaN_Si::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(AlGaN_Si, thermk,
            MISeeClass<AlGaN>(MaterialInfo::thermk),
            MINote("Si doping dependence for GaN")
            )
Tensor2<double> AlGaN_Si::thermk(double T, double t) const {
    double lCondT = 1/(Al/mAlN_Si.thermk(T,t).c00 + Ga/mGaN_Si.thermk(T,t).c00 + Al*Ga*0.4),
           vCondT = 1/(Al/mAlN_Si.thermk(T,t).c11 + Ga/mGaN_Si.thermk(T,t).c11 + Al*Ga*0.4);
    return(Tensor2<double>(lCondT, vCondT));
 }

MI_PROPERTY(AlGaN_Si, absp,
            MISeeClass<AlGaN>(MaterialInfo::absp)
            )
double AlGaN_Si::absp(double lam, double T) const {
    double E = phys::h_eVc1e9/lam;
    return ( (19000.+4000.*doping()/1e18)*exp((E-Eg(T,0.,'G'))/(0.019+0.001*doping()/1e18))+(330.+200.*doping()/1e18)*exp((E-Eg(T,0.,'G'))/(0.07+0.016*doping()/1e18)) );
}

bool AlGaN_Si::isEqual(const Material &other) const {
    const AlGaN_Si& o = static_cast<const AlGaN_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && AlGaN::isEqual(other);
}

static MaterialsDB::Register<AlGaN_Si> materialDB_register_AlGaN_Si;

}}       // namespace plask::materials
