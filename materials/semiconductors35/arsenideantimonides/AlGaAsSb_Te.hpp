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
#ifndef PLASK__AlGaAsSb_Te_H
#define PLASK__AlGaAsSb_Te_H

/** @file
This file contains Te-doped AlGaAsSb
*/

#include <plask/material/material.hpp>
#include "AlGaAsSb.hpp"

namespace plask { namespace materials {

/**
 * Represent Te-doped AlGaAsSb, its physical properties.
 */
struct AlGaAsSb_Te: public AlGaAsSb {

    static constexpr const char* NAME = "AlGaAsSb:Te";

    AlGaAsSb_Te(const Composition &Comp, double Val);
    std::string name() const override;
    std::string str() const override;
    Tensor2<double> mob(double T) const override;
    double Nf(double T) const override; //TODO make sure the result is in cm^(-3)
    double doping() const override; //TODO Piotr: NEW method (there is no doping() in Material), maybe it should be non-virtual?
    Tensor2<double> cond(double T) const override;
    ConductivityType condtype() const override;
    double nr(double lam, double T, double n = .0) const override;
    double absp(double lam, double T) const override;

protected:
    bool isEqual(const Material& other) const override;

private:
    double ND,
           Nf_RT,
           mob_RT;

};

}} // namespace plask::materials

#endif	//PLASK__AlGaAsSb_Te_H
