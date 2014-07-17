#include "../utils/system.h"

#include <fstream>
#include <ctime>

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   include <windows.h>
#endif

#include <plask/config.h>
#include <plask/log/log.h>
#include "verify.h"

namespace plask {

#ifdef LICENSE_CHECKING
    PLASK_API LicenseVerifier license_verifier;
#endif

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
    char* home[MAX_PATH];
    SHGetFolderPath(NULL, CSIDL_PROFILE, NULL, 0, path);
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



}   // namespace plask
