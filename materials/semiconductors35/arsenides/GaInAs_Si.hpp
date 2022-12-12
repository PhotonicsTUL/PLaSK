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
#ifndef PLASK__GaInAs_Si_H
#define PLASK__GaInAs_Si_H

/** @file
This file contains Si-doped GaInAs
*/

#include "plask/material/material.hpp"
#include "GaInAs.hpp"
//#include "GaAs_Si.hpp"
//#include "InAs_Si.hpp"

namespace plask { namespace materials {

/**
 * Represent Si-doped GaInAs, its physical properties.
 */
struct GaInAs_Si: public GaInAs {

    static constexpr const char* NAME = "InGaAs:Si";

    GaInAs_Si(const Material::Composition& Comp, double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

    //GaAs_Si mGaAs_Si;
    //InAs_Si mInAs_Si;
};

}} // namespace plask::materials

#endif	//PLASK__GaInAs_Si_H
