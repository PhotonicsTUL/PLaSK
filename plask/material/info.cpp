#include "info.h"

#include <limits>

namespace plask {

void MaterialInfo::override(const MaterialInfo &to_override) {
    this->parent = to_override.parent;
    for (auto& prop: to_override.propertyInfo)
        this->propertyInfo[prop.first] = prop.second;
}

MaterialInfo::PropertyInfo& MaterialInfo::operator()(PROPERTY_NAME property) {
    return propertyInfo[property];
}

/*const MaterialInfo::PropertyInfo& MaterialInfo::operator()(PROPERTY_NAME property) const {
    return propertyInfo[property];
}*/

const MaterialInfo::PropertyInfo::ArgumentRange MaterialInfo::PropertyInfo::NO_RANGE =
    ArgumentRange(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

const MaterialInfo::PropertyInfo::ArgumentRange& MaterialInfo::PropertyInfo::getArgumentRange(plask::MaterialInfo::ARGUMENT_NAME argument) {
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

boost::optional<MaterialInfo> MaterialInfo::DB::get(const std::string &materialName, bool with_inharited_info) {
    auto this_mat_info = materialInfo.find(materialName);
    if (this_mat_info == materialInfo.end())
        return boost::optional<MaterialInfo>();

    if (!with_inharited_info || this_mat_info->second.parent.empty())
        return boost::optional<MaterialInfo>(this_mat_info->second);

    boost::optional<MaterialInfo> parent_info = get(this_mat_info->second.parent, true);
    if (!parent_info)
        return boost::optional<MaterialInfo>(this_mat_info->second);
    parent_info->override(this_mat_info->second);
    return parent_info;
}


}
