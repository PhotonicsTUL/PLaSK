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
#ifndef PLASK__GAIN_H
#define PLASK__GAIN_H

#include <plask/math.hpp>
#include <plask/provider/providerfor.hpp>

namespace plask {

/**
 * Material gain (1/cm).
 *
 * This is the gain property. It should have the same unit as absorption.
 * Providers must set it to NaN everywhere outside of the active region.
 * Optical solvers should thread NaNs as zeros.
 * It can also be set negative in case there is some absorption which is not
 * covered by the material database.
 *
 * It can also be a gain profile. Some optical solvers can determine
 * the threshold gain as a constant, which should be added to it in order to
 * obtain the zero modal gain (threshold). The regions where it is NaN should
 * not be affected.
 *
 * Providers of material gain should accept additional parameter,
 * which is the wavelength for which the gain should be computed.
 */
struct PLASK_API Gain : public MultiFieldProperty<Tensor2<double>, double> {
    enum EnumType {
        GAIN = 0,
        DGDN = 1
    };
    static constexpr size_t NUM_VALS = 2;
    static constexpr const char* NAME = "material gain";
    static constexpr const char* UNIT = "1/cm";
    // static inline double getDefaultValue() { return NAN; }
};

/**
 * Luminescence [?].
 */
struct PLASK_API Luminescence : public FieldProperty<Tensor2<double>, double> {
    static constexpr const char* NAME = "luminescence";
    static constexpr const char* UNIT = "a.u.";
    // static inline double getDefaultValue() { return NAN; }
};

} // namespace plask

#endif // PLASK__GAIN_H
