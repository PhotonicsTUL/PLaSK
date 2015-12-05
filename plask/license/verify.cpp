#include <fstream>
#include <sstream>

#include "../utils/system.h"

// This should not be included in any plask .h file because it can be not distributed to solvers' developers:
#include "../../license_sign/license.h"
#include "../../license_sign/getmac.h"

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   include <shlobj.h>
#endif

#include <plask/config.h>
#include <plask/log/log.h>
#include "verify.h"

#define PLASK_LICENSE_EXPIRY_TAG_NAME "expiry"
#define PLASK_LICENSE_MAC_TAG_NAME "mac"

#define PLASK_LICENSE_USER_TAG_NAME "name"
#define PLASK_LICENSE_EMAIL_TAG_NAME "email"
#define PLASK_LICENSE_ORGANISATION_TAG_NAME "organisation"

namespace plask {

#ifdef LICENSE_CHECKING
PLASK_API LicenseVerifier license_verifier;
#endif

// code mostly from http://stackoverflow.com/questions/19482378/how-to-parse-and-validate-a-date-in-stdstring-in-c
std::time_t LicenseVerifier::extractDate(const std::string &s) {
    std::istringstream is(s);
    char delimiter;
    int d, m, y;
    if (is >> d >> delimiter >> m >> delimiter >> y) {
        struct tm t = {-1};
        t.tm_mday = d;
        t.tm_mon = m - 1;
        t.tm_year = y - 1900;
        t.tm_isdst = -1;

        // normalize:
        time_t when = mktime(&t);
        const struct tm *norm = localtime(&when);
        // the actual date would be:
        // m = norm->tm_mon + 1;
        // d = norm->tm_mday;
        // y = norm->tm_year;
        // e.g. 29/02/2013 would become 01/03/2013

        // validate (is the normalized date still the same?):
        if (norm->tm_mday == d    &&
                norm->tm_mon  == m - 1 &&
                norm->tm_year == y - 1900)
            return when;
    }
    return (std::time_t) - 1;
}

bool LicenseVerifier::try_load_license(const std::string& fname) {
    std::ifstream in(fname);
    if (!in) return false;
    std::ostringstream out;
    out << in.rdbuf();
    in.close();
    content = out.str();
    filename = fname;
    return true;
}

LicenseVerifier::LicenseVerifier() {
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define V "\\"
    char home[MAX_PATH];
    SHGetFolderPath(NULL, CSIDL_PROFILE, NULL, 0, home);            // Depreciated
    //SHGetKnownFolderPath(FOLDERID_Profile, 0, NULL, home);        // Vista+
    try_load_license(std::string(home) + "\\plask_license.xml")  ||
    try_load_license(std::string(home) + "\\plask\\license.xml")  ||
#else
#   define V "/"
    char* home = getenv("HOME");
    try_load_license(std::string(home) + "/.plask_license.xml")  ||
    try_load_license(std::string(home) + "/.plask/license.xml")  ||
    try_load_license("/etc/plask_license.xml")  ||
    try_load_license("/etc/plask/license.xml")  ||
#endif
    try_load_license(prefixPath() + V "plask_license.xml") ||
    try_load_license(prefixPath() + V "etc" V "license.xml");
}

void LicenseVerifier::verify() {
    if (content == "")
        throw Exception("No valid license found");

    XMLReader r(std::unique_ptr<std::istringstream>(new std::istringstream(content, std::ios_base::binary)));

    boost::optional<std::string> expiry;
    if (!processLicense(r, nullptr,
                   [&] (XMLReader& src) {
                        boost::optional<std::vector<mac_address_t>> macs;
                        if (src.getNodeName() == PLASK_LICENSE_EXPIRY_TAG_NAME) {
                            if (expiry) src.throwException("duplicated <" PLASK_LICENSE_EXPIRY_TAG_NAME "> tag in license file");
                            expiry = src.getTextContent();
                        } else if (src.getNodeName() == PLASK_LICENSE_MAC_TAG_NAME) {
                            if (!macs) macs = getMacs();
                            if (std::find(macs->begin(), macs->end(), macFromString(src.getTextContent())) == macs->end())
                                src.throwException("License error: Hardware verification error.");
                        }
                    }
        ))
        throw Exception("License error: Invalid signature in file \"{0}\"", filename);

    if (!expiry)
        throw Exception("License error: No information about expiration date in file \"{0}\"", filename);

    std::time_t t = extractDate(*expiry);
    if (t == (std::time_t) (-1))
        throw Exception("License error: Ill-formatted expiration date \"{0}\"", *expiry);

    if (std::time(nullptr) + 24 * 3600 > t)
        throw Exception("License has expired");
}

std::string LicenseVerifier::getUser() {
    
    std::string user, organisation, email;
    if (content == "") return "";

    XMLReader r(std::unique_ptr<std::istringstream>(new std::istringstream(content, std::ios_base::binary)));

    readLicenseData(r, nullptr,
              [&] (XMLReader& src) {
                   if (src.getNodeName() == PLASK_LICENSE_USER_TAG_NAME)
                       user = src.getTextContent();
                   else if (src.getNodeName() == PLASK_LICENSE_EMAIL_TAG_NAME)
                       email = src.getTextContent();
                   else if (src.getNodeName() == PLASK_LICENSE_ORGANISATION_TAG_NAME)
                       organisation = src.getTextContent();
               }
    );
    
    if (email != "") {
        if (user != "") {
            user += " <"; user += email; user += ">";
        } else {
            user = email;
        }
    }
    if (organisation != "") {
        if (user != "") user += " ";
        user += organisation;
    }
    
    return user;
}


}   // namespace plask
