#include "solver.h"
#include "utils/string.h"

namespace plask {


void Solver::loadConfiguration(XMLReader& reader, Manager& manager) {
    reader.requireTagEnd();
}

void Solver::parseStandardConfiguration(XMLReader& source, Manager& manager, const std::string& expected_msg) {
    throw XMLUnexpectedElementException(source, expected_msg);
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

