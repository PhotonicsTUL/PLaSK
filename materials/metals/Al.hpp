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
#ifndef PLASK__Al_H
#define PLASK__Al_H

/** @file
This file contains Al
*/

#include "metal.hpp"

namespace plask { namespace materials {

/**
 * Represent Al, its physical properties.
 */
struct Al: public LorentzDrudeMetal {

    Al();

    static constexpr const char* NAME = "Al";

    virtual std::string name() const override;

  protected:
    virtual bool isEqual(const Material& other) const override;
};


}} // namespace plask::materials

#endif //PLASK__Al_H
