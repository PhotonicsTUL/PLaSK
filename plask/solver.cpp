#include "solver.h"
#include "utils/string.h"

namespace plask {

void Solver::loadConfiguration(XMLReader& reader, Manager& manager) {
    while (reader.requireTagOrEnd()) {
        loadParam(reader.getNodeName(), reader, manager);
        reader.requireTagEnd();
    }
}

void Solver::loadParam(const std::string& param, XMLReader& reader, Manager& manager) {
    throw XMLUnexpectedElementException(reader, "no additional configuration for this solver");
}

}   // namespace plask

