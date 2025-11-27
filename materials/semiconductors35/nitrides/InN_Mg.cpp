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
#include "InN_Mg.hpp"

#include <cmath>
#include <plask/material/db.hpp>    //MaterialsDB::Register
#include <plask/material/info.hpp>  //MaterialInfo::DB::Register

namespace plask { namespace materials {

MI_PARENT(InN_Mg, InN)

std::string InN_Mg::name() const { return NAME; }

std::string InN_Mg::str() const { return StringBuilder("InN").dopant("Mg", NA); }

InN_Mg::InN_Mg(double Val) {
    Nf_RT = 3.311e-23 * pow(Val, 2.278);
    NA = Val;
    mob_RT = 5.739e13 * pow(Nf_RT, -0.663);
    cond_RT = phys::qe * 100. * Nf_RT * mob_RT;
}

MI_PROPERTY(InN_Mg,
            mob,
            MISource("based on 4 papers (2006-2010): MBE-grown Mg-doped InN"),
            MINote("No T Dependence based on K. Kumakura et al., J. Appl. Phys. 93 (2003) 3370"))
Tensor2<double> InN_Mg::mob(double /*T*/) const { return (Tensor2<double>(mob_RT, mob_RT)); }

MI_PROPERTY(InN_Mg,
            Nf,
            MISource("based on 2 papers (2008-2009): Mg-doped InN"),
            MINote("No T Dependence based on K. Kumakura et al., J. Appl. Phys. 93 (2003) 3370"))
double InN_Mg::Nf(double /*T*/) const { return (Nf_RT); }

MI_PROPERTY(InN_Mg, Na, MINote("-"))
double InN_Mg::Na() const { return (NA); }

MI_PROPERTY(InN_Mg, Nd, MINote("-"))
double InN_Mg::Nd() const { return (0.); }

double InN_Mg::doping() const { return NA; }

MI_PROPERTY(InN_Mg, cond, MINote("No T Dependence based on K. Kumakura et al., J. Appl. Phys. 93 (2003) 3370"))
Tensor2<double> InN_Mg::cond(double /*T*/) const { return (Tensor2<double>(cond_RT, cond_RT)); }

Material::ConductivityType InN_Mg::condtype() const { return Material::CONDUCTIVITY_P; }

bool InN_Mg::isEqual(const Material& other) const {
    const InN_Mg& o = static_cast<const InN_Mg&>(other);
    return o.NA == this->NA && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && o.cond_RT == this->cond_RT &&
           InN::isEqual(other);
}

MaterialsDB::Register<InN_Mg> materialDB_register_InN_Mg;

}}  // namespace plask::materials
