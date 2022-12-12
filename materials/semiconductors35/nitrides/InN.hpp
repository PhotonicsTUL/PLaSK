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
#ifndef PLASK__InN_H
#define PLASK__InN_H

/** @file
This file contains undoped InN
*/

#include "plask/material/material.hpp"


namespace plask { namespace materials {

/**
 * Represent undoped InN, its physical properties.
 */
struct InN: public Semiconductor {

    static constexpr const char* NAME = "InN";

    std::string name() const override;
    Tensor2<double> thermk(double T, double h=INFINITY) const override;
    double lattC(double T, char x) const override;
    double Eg(double T, double e, char point) const override;
    //double VB(double T, double e, char point, char hole) const override;
    double Dso(double T, double e) const override;
    Tensor2<double> Me(double T, double e, char point) const override;
    Tensor2<double> Mhh(double T, double e) const override;
    Tensor2<double> Mlh(double T, double e) const override;
    //double CB(double T, double e, char point) const override; // commented out since no return leads to UB - Piotr Beling [25.02.2016]
    double VB(double T, double e, char point, char hole) const override;
    ConductivityType condtype() const override;

protected:
    bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__InN_H
