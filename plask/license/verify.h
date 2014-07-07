#ifndef PLASK__LICENSE_VERIFY
#define PLASK__LICENSE_VERIFY

#include "../utils/xml.h"

#include <string>

namespace plask {

/**
 * Verify if the license is correct.
 * It checks for checksums and also validates expiration date.
 * Throw exception if verification fail.
 */
PLASK_API void verifyLicense();

}
#endif
