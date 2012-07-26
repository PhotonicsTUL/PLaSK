#include "solver.h"
#include "utils/string.h"

namespace plask {

void Solver::loadConfiguration(XMLReader& reader, Manager&) {
    reader.requireTagEnd();  //require empty configuration
}

std::string Solver::getId() const {
    return replaceChars(getName(), [](char in) { return isspace(in) ? '_' : in; });
}

}   // namespace plask

