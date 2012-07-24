#include "solver.h"
#include "utils/string.h"

namespace plask {

void Solver::loadConfiguration(GeometryReader &conf) {
    conf.source.requireTagEnd();  //require empty configuration
}

std::string Solver::getId() const {
    return replaceChars(getName(), [](char in) { return isspace(in) ? '_' : in; });
}

}   // namespace plask

