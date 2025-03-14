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
#include "InGaN_Si.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

MI_PARENT(InGaN_Si, InGaN)

std::string InGaN_Si::name() const { return NAME; }

std::string InGaN_Si::str() const { return StringBuilder("In", In)("Ga")("N").dopant("Si", ND); }

InGaN_Si::InGaN_Si(const Material::Composition& Comp, double Val): InGaN(Comp), mGaN_Si(Val), mInN_Si(Val)
{
    ND = Val;
}

MI_PROPERTY(InGaN_Si, mob,
            MISource("based on 3 papers 2007-2009 about Si-doped InGaN/GaN/c-sapphire"),
            MISource("based on Si-doped GaN and InN")
            )
Tensor2<double> InGaN_Si::mob(double T) const {
    double lMob = 1/(In/mInN_Si.mob(T).c00 + Ga/mGaN_Si.mob(T).c00 + In*Ga*(-4.615E-21*Nf(T)+0.549)),
           vMob = 1/(In/mInN_Si.mob(T).c11 + Ga/mGaN_Si.mob(T).c11 + In*Ga*(-4.615E-21*Nf(T)+0.549));
    return (Tensor2<double>(lMob, vMob));
}

MI_PROPERTY(InGaN_Si, Nf,
            MINote("linear interpolation: Si-doped GaN, InN")
            )
double InGaN_Si::Nf(double T) const {
    return ( mInN_Si.Nf(T)*In + mGaN_Si.Nf(T)*Ga );
}

MI_PROPERTY(InGaN_Si, Na,
            MINote("-")
            )
double InGaN_Si::Na() const {
    return ( 0. );
}

MI_PROPERTY(InGaN_Si, Nd,
            MINote("-")
            )
double InGaN_Si::Nd() const {
    return ( ND );
}

double InGaN_Si::doping() const {
    return ND;
}

Tensor2<double> InGaN_Si::cond(double T) const {
    return (Tensor2<double>(phys::qe*100.*Nf(T)*mob(T).c00, phys::qe*100.*Nf(T)*mob(T).c11));
}

Material::ConductivityType InGaN_Si::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PROPERTY(InGaN_Si, thermk,
            MISeeClass<InGaN>(MaterialInfo::thermk),
            MINote("Si doping dependence for GaN")
            )
Tensor2<double> InGaN_Si::thermk(double T, double t) const {
    double lCondT = 1/(In/mInN_Si.thermk(T).c00 + Ga/mGaN_Si.thermk(T,t).c00 + In*Ga*0.215*exp(7.913*In)),
           vCondT = 1/(In/mInN_Si.thermk(T).c11 + Ga/mGaN_Si.thermk(T,t).c11 + In*Ga*0.215*exp(7.913*In));
    return(Tensor2<double>(lCondT, vCondT));
 }

MI_PROPERTY(InGaN_Si, absp,
            MISeeClass<InGaN>(MaterialInfo::absp)
            )
double InGaN_Si::absp(double lam, double T) const {
    double E = phys::h_eVc1e9/lam;
    return ( (19000.+4000.*doping()/1e18)*exp((E-Eg(T,0.,'G'))/(0.019+0.001*doping()/1e18))+(330.+200.*doping()/1e18)*exp((E-Eg(T,0.,'G'))/(0.07+0.016*doping()/1e18)) );
}

bool InGaN_Si::isEqual(const Material &other) const {
    const InGaN_Si& o = static_cast<const InGaN_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && InGaN::isEqual(other);
}


static MaterialsDB::Register<InGaN_Si> materialDB_register_InGaN_Si;

}}       // namespace plask::materials
