#include "module.h"
#include "utils/string.h"

namespace plask {

std::string Module::getId() const {
    return replaceChars(getName(), [](char in) { return isspace(in) ? '_' : in; });
}

}   // namespace plask

