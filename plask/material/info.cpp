#include "info.h"

#include <limits>

namespace plask {

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


}
