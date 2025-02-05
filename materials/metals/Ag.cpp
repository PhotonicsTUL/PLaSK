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
#include "Ag.hpp"

#include <cmath>
#include <plask/material/db.hpp>  //MaterialsDB::Register
#include <plask/material/info.hpp>    //MaterialInfo::DB::Register

namespace plask { namespace materials {

std::string Ag::name() const { return NAME; }


MI_PROPERTY(Ag, absp,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MINote("no temperature dependence")
)

MI_PROPERTY(Ag, nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MINote("no temperature dependence")
)

MI_PROPERTY(Ag, Nr,
    MISource("A. Rakic et al., Appl. Opt. 37(22) (1998) 5271-5283"),
    MINote("no temperature dependence")
)

Ag::Ag(): LorentzDrudeMetal(9.01,
                            {0.845, 0.065, 0.124, 0.011, 0.840, 5.646}, // f
                            {0.048, 3.886, 0.452, 0.065, 0.916, 2.419}, // G
                            {0.000, 0.816, 4.481, 8.185, 9.083, 20.29}  // w
) {}

// Ag::Ag(): BrendelBormannMetal(9.01,
//                               {0.821, 0.050, 0.133, 0.051, 0.467, 4.000}, // f
//                               {0.049, 0.189, 0.067, 0.019, 0.117, 0.052}, // G
//                               {0.000, 2.025, 5.185, 4.343, 9.809, 18.56}, // w
//                               {0.000, 1.894, 0.665, 0.189, 1.170, 0.516}  // s
// ) {}


bool Ag::isEqual(const Material &/*other*/) const {
    return true;
}


static MaterialsDB::Register<Ag> materialDB_register_Ag;

}} // namespace plask::materials
