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
#ifndef PLASK__GaN_Si_H
#define PLASK__GaN_Si_H

/** @file
This file contains Si-doped GaN
*/

#include <plask/material/material.hpp>
#include "GaN.hpp"

namespace plask { namespace materials {

/**
 * Represent Si-doped GaN, its physical properties.
 */
struct GaN_Si: public GaN {

    static constexpr const char* NAME = "GaN:Si";

    GaN_Si(double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO change to cm^(-3)
    double Na() const override;
    double Nd() const override;
    double doping() const override;
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    Tensor2<double> thermk(double T, double t) const override;
    double nr(double lam, double T, double n=0.) const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

};


/**
 * Represent Si-doped bulk (substrate) GaN, its physical properties.
 */
struct GaN_Si_bulk: public GaN_Si {

    static constexpr const char* NAME = "GaN_bulk:Si";

    GaN_Si_bulk(double val): GaN_Si(val) {}

    Tensor2<double> thermk(double T, double t) const override;

};


}} // namespace plask::materials

#endif	//PLASK__GaN_Si_H
