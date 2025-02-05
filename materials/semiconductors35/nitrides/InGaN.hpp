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
#ifndef PLASK__InGaN_H
#define PLASK__InGaN_H

/** @file
This file contains undoped InGaN
*/

#include <plask/material/material.hpp>
#include "GaN.hpp"
#include "InN.hpp"

namespace plask { namespace materials {

/**
 * Represent undoped InGaN, its physical properties.
 */
struct InGaN: public Semiconductor {

    static constexpr const char* NAME = "InGaN";

    InGaN(const Material::Composition& Comp);
    std::string name() const override;
    std::string str() const override;
    Composition composition() const override;
    Tensor2<double> thermk(double T, double t) const override;
    double nr(double lam, double T, double n=0.) const override;
    double absp(double lam, double T) const override;
    double Eg(double T, double e, char point) const override;
    //double VB(double T, double e, char point, char hole) const override;
    double Dso(double T, double e) const override;
    Tensor2<double> Me(double T, double e, char point) const override;
    Tensor2<double> Mhh(double T, double e) const override;
    Tensor2<double> Mlh(double T, double e) const override;
    double CB(double T, double e, char point) const override;
    double VB(double T, double e, char point, char hole) const override;
    double lattC(double T, char x) const override;
    ConductivityType condtype() const override;

protected:
    bool isEqual(const Material& other) const override;

protected:
    double In,
           Ga;

    GaN mGaN;
    InN mInN;

};


}} // namespace plask::materials

#endif	//PLASK__InGaN_H
