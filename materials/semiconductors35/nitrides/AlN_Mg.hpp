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
#ifndef PLASK__AlN_Mg_H
#define PLASK__AlN_Mg_H

/** @file
This file contains Mg-doped AlN
*/

#include "plask/material/material.hpp"
#include "AlN.hpp"

namespace plask { namespace materials {

/**
 * Represent Mg-doped AlN, its physical properties.
 */
struct AlN_Mg: public AlN {

    static constexpr const char* NAME = "AlN:Mg";

    AlN_Mg(double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override;
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double NA,
           Nf_RT,
           mob_RT,
           cond_RT;

};


}} // namespace plask::materials

#endif	//PLASK__AlN_Mg_H
