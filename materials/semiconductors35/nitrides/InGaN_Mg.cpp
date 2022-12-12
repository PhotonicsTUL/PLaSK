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
#include "InGaN_Mg.hpp"

#include <cmath>
#include "plask/material/db.hpp"  //MaterialsDB::Register
#include "plask/material/info.hpp"    //MaterialInfo::DB::Register

namespace plask { namespace materials {

MI_PARENT(InGaN_Mg, InGaN)

std::string InGaN_Mg::name() const { return NAME; }

std::string InGaN_Mg::str() const { return StringBuilder("In", In)("Ga")("N").dopant("Mg", NA); }

InGaN_Mg::InGaN_Mg(const Material::Composition& Comp, double Val): InGaN(Comp), mGaN_Mg(Val), mInN_Mg(Val)
{
    NA = Val;
}

MI_PROPERTY(InGaN_Mg, mob,
            MISource("B. N. Pantha et al., Applied Physics Letters 95 (2009) 261904"),
            MISource("K. Aryal et al., Applied Physics Letters 96 (2010) 052110")
            )
Tensor2<double> InGaN_Mg::mob(double T) const {
    double lMob = 1/(In/mInN_Mg.mob(T).c00 + Ga/mGaN_Mg.mob(T).c00 + In*Ga*(7.256E-19*Nf(T)+0.377)),
           vMob = 1/(In/mInN_Mg.mob(T).c11 + Ga/mGaN_Mg.mob(T).c11 + In*Ga*(7.256E-19*Nf(T)+0.377));
    return (Tensor2<double>(lMob, vMob));
}

MI_PROPERTY(InGaN_Mg, Nf,
            MINote("linear interpolation: Mg-doped GaN, InN")
            )
double InGaN_Mg::Nf(double T) const {
    return ( mInN_Mg.Nf(T)*In + mGaN_Mg.Nf(T)*Ga );
}

MI_PROPERTY(InGaN_Mg, Na,
            MINote("-")
            )
double InGaN_Mg::Na() const {
    return ( NA );
}

MI_PROPERTY(InGaN_Mg, Nd,
            MINote("-")
            )
double InGaN_Mg::Nd() const {
    return ( 0. );
}

double InGaN_Mg::doping() const {
    return NA;
}

Tensor2<double> InGaN_Mg::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

Material::ConductivityType InGaN_Mg::condtype() const { return Material::CONDUCTIVITY_P; }

MI_PROPERTY(InGaN_Mg, absp,
            MISeeClass<InGaN>(MaterialInfo::absp)
            )
double InGaN_Mg::absp(double lam, double T) const {
    double E = phys::h_eVc1e9/lam;
    return ( (19000.+200.*doping()/1e18)*exp((E-Eg(T,0.,'G'))/(0.019+0.0001*doping()/1e18))+(330.+30.*doping()/1e18)*exp((E-Eg(T,0.,'G'))/(0.07+0.0008*doping()/1e18)) );
}

bool InGaN_Mg::isEqual(const Material &other) const {
    const InGaN_Mg& o = static_cast<const InGaN_Mg&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && InGaN::isEqual(other);
}

static MaterialsDB::Register<InGaN_Mg> materialDB_register_InGaN_Mg;

}}       // namespace plask::materials
