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
#include "manager.hpp"

#ifdef _MSC_VER
#include <algorithm>
#endif

namespace plask {

const DynamicLibrary& DynamicLibraries::load(const std::string &file_name, unsigned flags) {
    return *loaded.emplace(file_name, flags).first;
}

void DynamicLibraries::close(const DynamicLibrary &to_close) {
    loaded.erase(to_close);
}

void DynamicLibraries::closeAll() {
    loaded.clear();
}

DynamicLibraries &DynamicLibraries::defaultSet()
{
    static DynamicLibraries default_set;
    return default_set;
}



}   // namespace plask
