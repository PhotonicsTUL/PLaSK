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
#include "InAs_Si.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string InAs_Si::name() const { return NAME; }

std::string InAs_Si::str() const { return StringBuilder("InAs").dopant("Si", ND); }

InAs_Si::InAs_Si(double Val) {
    Nf_RT = Val; //TODO
    ND = Val; //TODO
    mob_RT = 15000./(1.+pow((Nf_RT/1e18),0.81)); // 1e-4: cm^2/(V*s) -> m^2/(V*s);
}

MI_PROPERTY(InAs_Si, mob,
            MISource("L.-G. Li, Chin. Phys. Lett. 29 (2012) pp. 076801"),
            MINote("mob(T) assumed, TODO: find exp. data")
            )
Tensor2<double> InAs_Si::mob(double T) const {
    double tmob = mob_RT * pow(300./T,0.8);
    return ( Tensor2<double>(tmob,tmob) );
}

MI_PROPERTY(InAs_Si, Nf,
            MINote("Nf(ND) assumed, TODO: find exp. data"),
            MINote("no temperature dependence")
            )
double InAs_Si::Nf(double /*T*/) const {
    return ( Nf_RT );
}

double InAs_Si::doping() const {
    return ( ND );
}

MI_PROPERTY(InAs_Si, cond,
            MINote("cond(T) assumed, TODO: find exp. data")
            )
Tensor2<double> InAs_Si::cond(double T) const {
    double tCond_RT = phys::qe * Nf_RT*1e6 * mob_RT*1e-4;
    double tCond = tCond_RT * pow(300./T,1.5);
    return ( Tensor2<double>(tCond, tCond) );
}

Material::ConductivityType InAs_Si::condtype() const { return Material::CONDUCTIVITY_N; }

bool InAs_Si::isEqual(const Material &other) const {
    const InAs_Si& o = static_cast<const InAs_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && InAs::isEqual(other);
}

static MaterialsDB::Register<InAs_Si> materialDB_register_InAs_Si;

}} // namespace plask::materials
