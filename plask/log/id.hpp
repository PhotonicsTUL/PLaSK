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
#ifndef PLASK__LOG_ID_H
#define PLASK__LOG_ID_H

#include <cstdint>
#include <string>

#include <plask/config.hpp>

namespace plask {

/**
 * Get unique number.
 *
 * This function is threads-safe.
 * @return unique number
 */
PLASK_API std::uint64_t getUniqueNumber();

/**
 * Get unique string.
 *
 * This function is threads-safe.
 * @return lexical_cast<std::string>(getUniqueNumber())
 */
PLASK_API std::string getUniqueString();

}   // namespace plask

#endif // PLASK__LOG_ID_H
