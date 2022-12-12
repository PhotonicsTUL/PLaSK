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
#include "factory.hpp"
#include "../utils/string.hpp"

namespace plask {

FiltersFactory &FiltersFactory::getDefault() {
    static FiltersFactory defaultDb;
    return defaultDb;
}

shared_ptr<Solver> FiltersFactory::get(XMLReader &reader, Manager& manager) {
    if (reader.getTagName() != "filter")
        return shared_ptr<Solver>();
    std::string typeName = reader.requireAttribute("for");
    auto it = filterCreators.find(typeName);
    if (it == filterCreators.end())
        throw Exception("No filter for {0}", typeName);
    return it->second(reader, manager);
}

void FiltersFactory::add(const std::string typeName, FiltersFactory::FilterCreator filterCreator) {
    filterCreators[typeName] = filterCreator;
}


}   // namespace plask
