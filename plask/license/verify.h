#ifndef PLASK__LICENSE_VERIFY
#define PLASK__LICENSE_VERIFY

#include "../utils/xml.h"

#include <string>
#include "data.h"

namespace plask {

extern std::string info;

/**
 * Verify if the license is correct.
 * It checks for checksums and also validates expiration date
 */
PLASK_API void verifyLicense();


}
#endif

