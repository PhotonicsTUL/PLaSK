#ifndef PLASK__LICENSE_VERIFY
#define PLASK__LICENSE_VERIFY

#include <string>
#include <ctime>

#include "../utils/xml.h"

namespace plask {

/**
 * License verifier.
 *
 * It checks for checksums (see processLicense function), validates expiration date and mac addresses.
 *
 * Expiration date is read from @c expiry tag. This tag should occur exactly once and its content should be in format DD/MM/YYYY (any non-digit character can be used instead of '/').
 *
 * Mac address is in @c mac tag. This tag can be placed in license file zero or more times and its content should be in format HH:HH:HH:HH:HH:HH.
 * Each time it is checked if computer has ethernet interface with given mac.
 */
class PLASK_API LicenseVerifier {

    std::string filename, content;

    /// Parse date from the string in format DD/MM/YYYY.
    static std::time_t extractDate(const std::string& s);

    /**
     * Try load license file (but not verify it). If success, fill @a filename and @a content members.
     * @param fname name of license file to load
     * @return @c true only after successful loading
     */
    bool try_load_license(const std::string& fname);

  public:

    LicenseVerifier();

    /// Verify license. Throw exception if verification fail.
    void verify();
    
    /// Get formatted name of the license user
    std::string getUser();
};

#ifdef LICENSE_CHECKING
    extern PLASK_API LicenseVerifier license_verifier;
#endif


}
#endif
