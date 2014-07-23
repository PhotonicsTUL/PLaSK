#ifndef PLASK__LICENSE_VERIFY
#define PLASK__LICENSE_VERIFY

#include <string>
#include <ctime>

#include "../utils/xml.h"

#define PLASK_LICENSE_EXPIRY_TAG_NAME "expiry"
#define PLASK_LICENSE_MAC_HWID_NAME "hwid"

namespace plask {

/**
 * License verifier
 * It checks for checksums and also validates expiration date.
 * Throw exception if verification fail.
 */
class PLASK_API LicenseVerifier {

    std::string filename, content;

    // function expects the string in format dd/mm/yyyy:
    // case from http://stackoverflow.com/questions/19482378/how-to-parse-and-validate-a-date-in-stdstring-in-c
    static std::time_t extractDate(const std::string& s);

    bool try_load_license(const std::string& fname);

  public:

    LicenseVerifier();

    void verify();
};

#ifdef LICENSE_CHECKING
    extern PLASK_API LicenseVerifier license_verifier;
#endif


}
#endif
