#include "../../license_sign/license.h"

#include "../utils/system.h"

#include <fstream>
#include <ctime>

namespace plask {

#define PLASK_LICENSE_EXPIRY_TAG_NAME "expiry"
//#define PLASK_LICENSE_MAC_HWID_NAME "hwid"

// function expects the string in format dd/mm/yyyy:
// cose from http://stackoverflow.com/questions/19482378/how-to-parse-and-validate-a-date-in-stdstring-in-c
std::time_t extractDate(const std::string& s){
    std::istringstream is(s);
    char delimiter;
    int d, m, y;
    if (is >> d >> delimiter >> m >> delimiter >> y) {
        struct tm t = {0};
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
    return (std::time_t) -1;
}

bool try_verify(const std::string& filename) {
    std::cout << filename << std::endl;

    std::unique_ptr<std::ifstream> f(new std::ifstream(filename));
    if (!*f) return false;
    XMLReader r(std::move(f));

    boost::optional<std::string> expiry;
    if (!processLicense(r, nullptr,
            [&] (XMLReader& src) {
                if (src.getNodeName() == PLASK_LICENSE_EXPIRY_TAG_NAME) {
                    if (expiry) src.throwException("duplicated <" PLASK_LICENSE_EXPIRY_TAG_NAME "> tag in license file");
                        expiry = src.getTextContent();
                }
            }
        ))
            throw Exception("License error: Invalid signature in file \"%1%\".", filename);

    if (!expiry)
        throw Exception("License error: No information about expiration date in file \"%1%\".", filename);

    std::time_t t = extractDate(*expiry);
    if (t == (std::time_t) (-1))
        throw Exception("License error: Ill-formated exparation date \"%1%\".", *expiry);

    if (std::time(nullptr) + 24 * 3600 > t)
        throw Exception("License has expired.");

    return true;
}

void verifyLicense() {
    std::string home = getenv("HOME");

    if (
            ! try_verify(home + "/.plask_license.xml")  &&
            ! try_verify(home + "/.plask/license.xml")  &&
            ! try_verify(home + "/etc/plask_license.xml")  &&
            ! try_verify(home + "/etc/plask/license.xml")  &&
            ! try_verify(prefixPath() + "/plask_license.xml") &&
            ! try_verify(prefixPath() + "/etc/license.xml")
       )
        throw Exception("Could not find license file.");
}



}   // namespace plask
