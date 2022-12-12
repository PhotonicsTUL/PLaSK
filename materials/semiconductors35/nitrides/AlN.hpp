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
#ifndef PLASK__AlN_H
#define PLASK__AlN_H

/** @file
This file contains undoped AlN
*/

#include "plask/material/material.hpp"

namespace plask { namespace materials {

/**
 * Represent undoped AlN, its physical properties.
 */
struct AlN: public Semiconductor {

    static constexpr const char* NAME = "AlN";

	std::string name() const override;
    Tensor2<double> thermk(double T, double t) const override;
    double nr(double lam, double T, double n=0.) const override;
    double absp(double lam, double T) const override;
    double lattC(double T, char x) const override;
    double Eg(double T, double e, char point) const override;
    double VB(double T, double e, char point, char hole) const override;
    double Dso(double T, double e) const override;
    Tensor2<double> Me(double T, double e, char point) const override;
    Tensor2<double> Mhh(double T, double e) const override;
    Tensor2<double> Mlh(double T, double e) const override;
    ConductivityType condtype() const override;
/*TODO
    double Mhh(double T, double e, char point) const override;
    double Mhh_l(double T, char point) const override;
    double Mhh_v(double T, char point) const override;
    double Mlh(double T, double e, char point) const override;
    double Mlh_l(double T, char point) const override;
    double Mlh_v(double T, char point) const override;
*/

protected:
    bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__AlN_H
