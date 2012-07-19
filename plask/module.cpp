#include "module.h"
#include "utils/string.h"

namespace plask {

void Module::loadConfiguration(GeometryReader &conf) {
    conf.source.requireTagEnd();  //require empty configuration
}

std::string Module::getId() const {
    return replaceChars(getName(), [](char in) { return isspace(in) ? '_' : in; });
}

}   // namespace plask

