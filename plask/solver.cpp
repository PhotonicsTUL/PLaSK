#include "solver.h"
#include "utils/string.h"

#include "license/verify.h"

namespace plask {

void Solver::loadConfiguration(XMLReader& reader, Manager& manager) {
    reader.requireTagEnd();
}

void Solver::parseStandardConfiguration(XMLReader& source, Manager& manager, const std::string& expected_msg) {
    throw XMLUnexpectedElementException(source, expected_msg);
}

bool Solver::initCalculation() {
    #ifdef LICENSE_CHECKING
        if (!verified) { license_verifier.verify(); verified = true; }
    #endif
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

