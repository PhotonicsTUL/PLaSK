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
#ifndef PLASK__aSi_H
#define PLASK__aSi_H

/** @file
This file contains a-Si
*/

#include <plask/material/material.hpp>

namespace plask { namespace materials {

/**
 * Represent a-Si, its physical properties.
 */
struct aSi: public Dielectric {

    static constexpr const char* NAME = "Si";

    virtual std::string name() const override;
    virtual Tensor2<double> cond(double T) const override;
    virtual Tensor2<double> thermk(double T, double h=INFINITY) const override;
    virtual ConductivityType condtype() const override;
    virtual double nr(double lam, double T, double n = .0) const override;
    virtual double absp(double lam, double T) const override;

protected:
    virtual bool isEqual(const Material& other) const override;

};


}} // namespace plask::materials

#endif	//PLASK__aSi_H
