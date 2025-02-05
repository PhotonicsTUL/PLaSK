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
#include "AlInAs_Si.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string AlInAs_Si::name() const { return NAME; }

std::string AlInAs_Si::str() const { return StringBuilder("Al", Al)("In")("As").dopant("Si", ND); }

double AlInAs_Si::doping() const { return ND; }

Material::ConductivityType AlInAs_Si::condtype() const { return Material::CONDUCTIVITY_N; }

MI_PARENT(AlInAs_Si, AlInAs)

AlInAs_Si::AlInAs_Si(const Material::Composition& Comp, double Val): AlInAs(Comp), mAlAs_Si(Val), mInAs_Si(Val)
{
}

bool AlInAs_Si::isEqual(const Material &other) const {
    const AlInAs_Si& o = static_cast<const AlInAs_Si&>(other);
    return o.ND == this->ND && o.Nf_RT == this->Nf_RT && o.mob_RT == this->mob_RT && AlInAs::isEqual(other);
}

static MaterialsDB::Register<AlInAs_Si> materialDB_register_AlInAs_Si;

}}       // namespace plask::materials
