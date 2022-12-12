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
#include "InAs_C.hpp"

#include <cmath>
#include "plask/material/db.hpp"  //MaterialsDB::Register
#include "plask/material/info.hpp"    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InAs_C::name() const { return NAME; }

std::string InAs_C::str() const { return StringBuilder("InAs").dopant("C", NA); }

InAs_C::InAs_C(double Val) {
    Nf_RT = 0.; //TODO
    NA = 0.; //TODO
    mob_RT = 0.; //TODO
}

MI_PROPERTY(InAs_C, mob,
            MINote("TODO")
            )
Tensor2<double> InAs_C::mob(double /*T*/) const {
    return ( Tensor2<double>(mob_RT,mob_RT) );
}

MI_PROPERTY(InAs_C, Nf,
            MINote("TODO")
            )
double InAs_C::Nf(double /*T*/) const {
    return ( Nf_RT );
}

double InAs_C::doping() const {
    return ( NA );
}

MI_PROPERTY(InAs_C, cond, // TODO
            MINote("no temperature dependence")
            )
Tensor2<double> InAs_C::cond(double T) const {
    double tMob = mob(T).c00;
    double tCond = phys::qe * Nf_RT*1e6 * tMob;
    return (Tensor2<double>(tCond, tCond));
}

Material::ConductivityType InAs_C::condtype() const { return Material::CONDUCTIVITY_P; }

bool InAs_C::isEqual(const Material &other) const {
    const InAs_C& o = static_cast<const InAs_C&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && InAs::isEqual(other);
}

static MaterialsDB::Register<InAs_C> materialDB_register_InAs_C;

}} // namespace plask::materials
