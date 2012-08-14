#include "system.h"

#if defined(_MSC_VER) || defined(__MINGW32__)

#define PLASK_SYSTEM_WINDOWS
#include <windows.h>

#else

#include <string>
#include <limits.h>
#include <unistd.h>

#endif

namespace plask {

//comes from: http://www.cplusplus.com/forum/general/11104/
std::string exePathAndName() {
#ifdef PLASK_SYSTEM_WINDOWS
    char result[ MAX_PATH ];
    return std::string( result, GetModuleFileName( NULL, result, MAX_PATH ) );
#else
    char result[ PATH_MAX ];
    ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
    return std::string( result, (count > 0) ? count : 0 );
#endif
}

std::string exePath() {
    std::string full = exePathAndName();
    std::string::size_type last_sep = full.find_last_of(FILE_PATH_SEPARATOR);
    return last_sep == std::string::npos ? full : full.substr(0, last_sep+1);
}

}   //namespace plask
