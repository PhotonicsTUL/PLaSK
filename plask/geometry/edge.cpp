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
#include "edge.hpp"
#include <cmath>

#include <boost/algorithm/string.hpp>

namespace plask {

namespace edge {

bool Strategy::canMoveOutsideBoundingBox() const {
    return false;
}

Strategy* Strategy::fromStr(const std::string& str, const MaterialsDB& materialsDB) {
    std::string lower_name = boost::to_lower_copy(str);
    if (lower_name == "null") return new Null();
    if (lower_name == "periodic") return new Periodic();
    if (lower_name == "extend") return new Extend();
    if (lower_name == "mirror") return new Mirror();
    return new SimpleMaterial(materialsDB.get(str));
}

void SimpleMaterial::applyLo(double, double, double&, shared_ptr<plask::Material> &result_material, const Strategy*) const {
    result_material = material;
}
void SimpleMaterial::applyHi(double, double, double&, shared_ptr<plask::Material> &result_material, const Strategy*) const {
    result_material = material;
}

SimpleMaterial* SimpleMaterial::clone() const {
    return new SimpleMaterial(this->material);
}

std::string SimpleMaterial::str() const {
    return this->material->str();
}



void Null::applyLo(double, double, double&, shared_ptr<plask::Material> &, const Strategy*) const {}
void Null::applyHi(double, double, double&, shared_ptr<plask::Material> &, const Strategy*) const {}

Null* Null::clone() const {
    return new Null();
}

std::string Null::str() const {
    return "null";
}


void Extend::applyLo(double bbox_lo, double /*bbox_hi*/, double &p, shared_ptr<plask::Material>&, const Strategy*) const {
    p = bbox_lo + 1e-12;
}
void Extend::applyHi(double /*bbox_lo*/, double bbox_hi, double &p, shared_ptr<plask::Material>&, const Strategy*) const {
    p = bbox_hi - 1e-12;
}

Extend* Extend::clone() const {
    return new Extend();
}

std::string Extend::str() const {
    return "extend";
}


void Periodic::applyLo(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>&, const Strategy* opposite) const {
    if (opposite->type() == MIRROR) {
        double len = bbox_hi - bbox_lo;
        double len2 = 2 * len;
        p = std::fmod(p-bbox_lo, len2) + len2;
        if (p > len) p = len2 - p;
        p += bbox_lo;
    } else {
        p = std::fmod(p-bbox_lo, bbox_hi - bbox_lo) + bbox_hi;
    }
}
void Periodic::applyHi(double bbox_lo, double bbox_hi, double& p, shared_ptr<Material>&, const Strategy* opposite) const {
    if (opposite->type() == MIRROR) {
        double len = bbox_hi - bbox_lo;
        double len2 = 2 * len;
        p = std::fmod(p-bbox_lo, len2);
        if (p > len) p = len2 - p;
        p += bbox_lo;
    } else {
        p = std::fmod(p-bbox_lo, bbox_hi - bbox_lo) + bbox_lo;
    }
}

Periodic* Periodic::clone() const {
    return new Periodic();
}

std::string Periodic::str() const {
    return "periodic";
}

#define mirror_not_at_zero "Mirror is not located at the axis"

void Mirror::applyLo(double bbox_lo, double, double& p, shared_ptr<Material>&, const Strategy*) const {
    if (bbox_lo != 0.0)
        throw Exception(mirror_not_at_zero);
    p = -p;
    //p += 2.0 * (bbox_lo - p);
}
void Mirror::applyHi(double, double bbox_hi, double& p, shared_ptr<Material>&, const Strategy*) const {
    if (bbox_hi != 0.0)
        throw Exception(mirror_not_at_zero);
    p = -p;
    //p -= 2.0 * (p - bbox_hi);
}

bool Mirror::canMoveOutsideBoundingBox() const {
    return true;
}

Mirror* Mirror::clone() const {
    return new Mirror();
}

std::string Mirror::str() const {
    return "mirror";
}


}   // namespace edge

}   // namespace plask
