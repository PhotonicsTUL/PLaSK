#include <string>
#include <cstdio>
#include <cstdlib>
#include <boost/optional.hpp>

#include "../utils/system.h"
#include "data.h"
#include "base64.h"

namespace plask {

static boost::optional<std::string> readLicenseFile(const std::string& filename) {
    FILE* file = fopen(filename.c_str(), "r");
    if (file) {
        char header[30];
        fgets(header, 30, file);
        if (std::string(header) == "PLASK LICENSE DATA AH64C20D\n") {
            long pos = ftell(file);
            fseek(file, 0, SEEK_END);
            long size = ftell(file) - pos + 1;
            fseek(file, pos, SEEK_SET);
            char buffer[size];
            fgets(buffer, size, file);
            fclose(file);
            boost::optional<std::string>(base64::decode(buffer, size-1));
        }
        fclose(file);
    }
    return boost::optional<std::string>();
}

std::string info;

void loadInfo() {
    const char* home = getenv("HOME");
    if (auto data = readLicenseFile(std::string(home) + "/.plask.lic")) info = *data;
    else if (auto data = readLicenseFile("/etc/plask.lic")) info = *data;
    else if (auto data = readLicenseFile(prefixPath() + "/plask.lic")) info = *data;
}

}
