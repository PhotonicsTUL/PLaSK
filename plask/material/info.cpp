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
#include "info.hpp"
#include "db.hpp"

#include <limits>

namespace plask {

const char* MaterialInfo::PROPERTY_NAME_STRING[55] = {
    "kind",
    "lattC",
    "Eg",
    "CB",
    "VB",
    "Dso",
    "Mso",
    "Me",
    "Mhh",
    "Mlh",
    "Mh",
    "ac",
    "av",
    "b",
    "d",
    "c11",
    "c12",
    "c44",
    "eps",
    "chi",
    "Na",
    "Nd",
    "Ni",
    "Nf",
    "EactD",
    "EactA",
    "mob",
    "cond",
    "condtype",
    "A",
    "B",
    "C",
    "D",
    "thermk",
    "dens",
    "cp",
    "nr",
    "absp",
    "Nr",
    "NR",
    "mobe",
    "mobh",
    "taue",
    "tauh",
    "Ce",
    "Ch",
    "e13",
    "e33",
    "c13",
    "c33",
    "Psp",
    "y1",
    "y2",
    "y3"
};

/// Names of arguments for which we need to give the ranges
const char* MaterialInfo::ARGUMENT_NAME_STRING[7] = {
    "T",
    "e",
    "lam",
    "n",
    "h",
    "doping",
    "point"
};



MaterialInfo::PROPERTY_NAME MaterialInfo::parsePropertyName(const std::string &name)
{
    for (unsigned i = 0; i < sizeof (PROPERTY_NAME_STRING) / sizeof (PROPERTY_NAME_STRING[0]); ++i)
        if (name == PROPERTY_NAME_STRING[i]) return PROPERTY_NAME(i);
    throw plask::Exception("\"" + name + "\" is not a proper name of material's property.");
}

MaterialInfo::ARGUMENT_NAME MaterialInfo::parseArgumentName(const std::string &name)
{
    for (unsigned i = 0; i < sizeof (ARGUMENT_NAME_STRING) / sizeof (ARGUMENT_NAME_STRING[0]); ++i)
        if (name == ARGUMENT_NAME_STRING[i]) return ARGUMENT_NAME(i);
    throw plask::Exception("\"" + name + "\" is not a proper name of argument of material's method.");
}

MaterialInfo::Link::Link(const std::string &to_parse) {
    std::string s;
    std::tie(s, this->note) = plask::splitString2(to_parse, ' ');
    std::tie(this->className, s) = plask::splitString2(s, '.');
    this->property = MaterialInfo::parsePropertyName(s);
}

std::string MaterialInfo::Link::str() const {
    std::string result;
    ((result += this->className) += '.') += MaterialInfo::PROPERTY_NAME_STRING[this->property];
    (result += ' ') += this->note;
    return result;
}


void MaterialInfo::override(const MaterialInfo &to_override) {
    this->parent = to_override.parent;
    for (auto& prop: to_override.propertyInfo)
        this->propertyInfo[prop.first] = prop.second;
}

MaterialInfo::PropertyInfo& MaterialInfo::operator()(PROPERTY_NAME property) {
    return propertyInfo[property];
}

plask::optional<MaterialInfo::PropertyInfo> MaterialInfo::getPropertyInfo(MaterialInfo::PROPERTY_NAME property) const
{
    auto i = propertyInfo.find(property);
    return i == propertyInfo.end() ? plask::optional<MaterialInfo::PropertyInfo>() : plask::optional<MaterialInfo::PropertyInfo>(i->second);
}

/*const MaterialInfo::PropertyInfo& MaterialInfo::operator()(PROPERTY_NAME property) const {
    return propertyInfo[property];
}*/

const MaterialInfo::PropertyInfo::ArgumentRange MaterialInfo::PropertyInfo::NO_RANGE =
    ArgumentRange(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

boost::tokenizer<boost::char_separator<char> > MaterialInfo::PropertyInfo::eachLine() const {
    return boost::tokenizer<boost::char_separator<char>>(_info, boost::char_separator<char>("\n\r"));
}

std::vector<std::string> MaterialInfo::PropertyInfo::eachOfType(const std::string &type) const {
    std::vector<std::string> result;
    for (const std::string& line: eachLine()) {
        auto p = splitString2(line, ':');
        boost::trim(p.first); boost::trim(p.second);
        if (p.first == type)
            result.push_back(p.second);
    }
    return result;
}

std::string MaterialInfo::PropertyInfo::getSource() const {
    std::string result;
    for (const std::string& source: eachOfType("source")) {
        if (source.empty()) continue;
        if (!result.empty()) result += '\n';
        result += source;
    }
    return result;
}

std::string MaterialInfo::PropertyInfo::getNote() const {
    std::string result;
    for (const std::string& source: eachOfType("note")) {
        if (source.empty()) continue;
        if (!result.empty()) result += '\n';
        result += source;
    }
    return result;
}

MaterialInfo::PropertyInfo::ArgumentRange MaterialInfo::PropertyInfo::getArgumentRange(plask::MaterialInfo::ARGUMENT_NAME argument) const {
    for (const std::string& range_desc: eachOfType(ARGUMENT_NAME_STRING[argument] + std::string(" range"))) {
         std::string from, to;
         std::tie(from, to) = splitString2(range_desc, ':');
         try {
            return MaterialInfo::PropertyInfo::ArgumentRange(boost::lexical_cast<double>(from), boost::lexical_cast<double>(to));
         } catch (const std::exception&) {}
    }
    return NO_RANGE;
}

std::vector<MaterialInfo::Link> MaterialInfo::PropertyInfo::getLinks() const {
    std::vector<MaterialInfo::Link> result;
    for (const std::string& link_str: eachOfType("see"))
        try {
            result.push_back(MaterialInfo::Link(link_str));
        } catch (const std::exception&) {}
    return result;
}

MaterialInfo::PropertyInfo& MaterialInfo::PropertyInfo::setArgumentRange(MaterialInfo::ARGUMENT_NAME argument, MaterialInfo::PropertyInfo::ArgumentRange range) {
    if (range != NO_RANGE) {
        std::string s;
        (s += MaterialInfo::ARGUMENT_NAME_STRING[argument]) += " range: ";
        s += boost::lexical_cast<std::string>(range.first);
        s += ":";
        s += boost::lexical_cast<std::string>(range.second);
        add(std::move(s));
    }
    return *this;
}

MaterialInfo::DB& MaterialInfo::DB::getDefault() {
    return MaterialsDB::getDefault().info;
}

MaterialInfo & MaterialInfo::DB::add(const std::string &materialName, const std::string &parentMaterial) {
    MaterialInfo& result = materialInfo[materialName];
    result.parent = parentMaterial;
    return result;
}

MaterialInfo & MaterialInfo::DB::add(const std::string& materialName) {
    return materialInfo[materialName];
}

plask::optional<MaterialInfo> MaterialInfo::DB::get(const std::string &materialName, bool with_inherited_info) const {
    auto this_mat_info = materialInfo.find(materialName);
    if (this_mat_info == materialInfo.end())
        return plask::optional<MaterialInfo>();

    if (!with_inherited_info || this_mat_info->second.parent.empty())
        return plask::optional<MaterialInfo>(this_mat_info->second);

    plask::optional<MaterialInfo> parent_info = get(this_mat_info->second.parent, true);
    if (!parent_info)
        return plask::optional<MaterialInfo>(this_mat_info->second);
    parent_info->override(this_mat_info->second);
    return parent_info;
}

plask::optional<MaterialInfo::PropertyInfo> MaterialInfo::DB::get(const std::string &materialName, PROPERTY_NAME propertyName, bool with_inherited_info) const {
    auto this_mat_info = materialInfo.find(materialName);
    if (this_mat_info == materialInfo.end())
        return plask::optional<MaterialInfo::PropertyInfo>();

    auto res = this_mat_info->second.getPropertyInfo(propertyName);
    return res || !with_inherited_info || this_mat_info->second.parent.empty() ? res : get(this_mat_info->second.parent, propertyName, true);
}



}
