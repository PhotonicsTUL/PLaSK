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
#include "AlGaN_Mg.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlGaN_Mg::name() const { return NAME; }

std::string AlGaN_Mg::str() const { return StringBuilder("Al", Al)("Ga")("N").dopant("Mg", NA); }

MI_PARENT(AlGaN_Mg, AlGaN)

AlGaN_Mg::AlGaN_Mg(const Material::Composition& Comp, double Val): AlGaN(Comp), mGaN_Mg(Val), mAlN_Mg(Val)
{
    NA = Val;
}

MI_PROPERTY(AlGaN_Mg, mob,
            MISource("based on 7 papers 1994-2010 about Mg-doped AlGaN"),
            MISource("based on Mg-doped GaN and AlN")
            )
Tensor2<double> AlGaN_Mg::mob(double T) const {
    double lMob = pow(Ga,28.856-16.793*(1-exp(-Al/0.056))-9.259*(1-exp(-Al/0.199)))*mGaN_Mg.mob(T).c00,
           vMob = pow(Ga,28.856-16.793*(1-exp(-Al/0.056))-9.259*(1-exp(-Al/0.199)))*mGaN_Mg.mob(T).c11;
    return (Tensor2<double>(lMob,vMob));
}

MI_PROPERTY(AlGaN_Mg, Nf,
            MINote("linear interpolation: Mg-doped GaN, AlN")
            )
double AlGaN_Mg::Nf(double T) const {
    return mGaN_Mg.Nf(T);
    //return mAlN_Mg.Nf(T)*Al + mGaN_Mg.Nf(T)*Ga;
}

double AlGaN_Mg::doping() const {
    return NA;
}

Tensor2<double> AlGaN_Mg::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

Material::ConductivityType AlGaN_Mg::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(AlGaN_Mg, absp,
            MISeeClass<AlGaN>(MaterialInfo::absp)
            )
double AlGaN_Mg::absp(double lam, double T) const {
    double E = phys::h_eVc1e9/lam;
    return ( (19000.+200.*doping()/1e18)*exp((E-Eg(T,0.,'G'))/(0.019+0.0001*doping()/1e18))+(330.+30.*doping()/1e18)*exp((E-Eg(T,0.,'G'))/(0.07+0.0008*doping()/1e18)) );
}

bool AlGaN_Mg::isEqual(const Material &other) const {
    const AlGaN_Mg& o = static_cast<const AlGaN_Mg&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && AlGaN::isEqual(other);
}

static MaterialsDB::Register<AlGaN_Mg> materialDB_register_AlGaN_Mg;

}}       // namespace plask::materials
