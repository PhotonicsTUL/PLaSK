#include "material.h"

std::string NameOnlyMaterial::name() const
{
    return _name;
}

plask::Material::Kind NameOnlyMaterial::kind() const {
    return plask::Material::NONE;
}


