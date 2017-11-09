#include "info.h"

#include <limits>

namespace plask {

const char* MaterialInfo::PROPERTY_NAME_STRING[] = {
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
    "Psp"
};

/// Names of arguments for which we need to give the ranges
const char* MaterialInfo::ARGUMENT_NAME_STRING[] = {
    "T",
    "e",
    "lam",
    "n",
    "h",
    "doping"
};



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

const MaterialInfo::PropertyInfo::ArgumentRange& MaterialInfo::PropertyInfo::getArgumentRange(plask::MaterialInfo::ARGUMENT_NAME argument) const {
    auto r = _argumentRange.find(argument);
    return r != _argumentRange.end() ? r->second : NO_RANGE;
}

MaterialInfo::PropertyInfo& MaterialInfo::PropertyInfo::setArgumentRange(MaterialInfo::ARGUMENT_NAME argument, MaterialInfo::PropertyInfo::ArgumentRange range) {
    if (range == NO_RANGE)
        _argumentRange.erase(argument);
    else
        _argumentRange[argument] = range;
    return *this;
}

MaterialInfo::DB& MaterialInfo::DB::getDefault() {
    static MaterialInfo::DB defaultInfoDB;
    return defaultInfoDB;
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
        return parent ? parent->get(materialName, with_inherited_info) : plask::optional<MaterialInfo>();

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
        return parent ? parent->get(materialName, propertyName, with_inherited_info) : plask::optional<MaterialInfo::PropertyInfo>();

    auto res = this_mat_info->second.getPropertyInfo(propertyName);
    return res || !with_inherited_info || this_mat_info->second.parent.empty() ? res : get(this_mat_info->second.parent, propertyName, true);
}


}
