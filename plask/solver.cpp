#include "solver.hpp"
#include "utils/string.hpp"

#include "license/verify.hpp"

namespace plask {

void Solver::loadConfiguration(XMLReader& reader, Manager& /*manager*/) {
    reader.requireTagEnd();
}

void Solver::parseStandardConfiguration(XMLReader& source, Manager& /*manager*/, const std::string& expected_msg) {
    throw XMLUnexpectedElementException(source, expected_msg);
}

bool Solver::initCalculation() {
    #ifdef LICENSE_CHECK
        if (!verified) { license_verifier.verify(); verified = true; }
    #endif
    if (!initialized) {
        writelog(LOG_INFO, "Initializing solver");
        onInitialize();
        initialized = true;
        return true;
    } else {
        return false;
    }
}


}   // namespace plask

