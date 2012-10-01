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

bool Solver::initCalculation() {
    if (!initialized) {
        writelog(LOG_INFO, "Initializing solver");
        onInitialize();
        initialized = true;
        return false;
    } else {
        return true;
    }
}


}   // namespace plask

