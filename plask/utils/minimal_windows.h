#ifndef PLASK__UTILS_MINIMAL_WINDOWS_H
#define PLASK__UTILS_MINIMAL_WINDOWS_H

// This file includes minimal possible version of windows.h.

// push_macro and pop_macro pragmas are not standard, but are supported by all compilers: MSVC, gcc, clang, intel

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#pragma push_macro("NOMINMAX")
#pragma push_macro("STRICT")
#pragma push_macro("WIN32_LEAN_AND_MEAN")
#pragma push_macro("NOCOMM")
#ifndef NOMINMAX
    #define NOMINMAX    //prevents windows.h from defining min, max macros, see http://stackoverflow.com/questions/1904635/warning-c4003-and-errors-c2589-and-c2059-on-x-stdnumeric-limitsintmax
#endif
#ifndef STRICT
    #define STRICT
#endif
#ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOCOMM
    #define NOCOMM
#endif

#include <windows.h>

//#ifdef copysign
//    #undef copysign
//#endif

#pragma pop_macro("NOMINMAX")
#pragma pop_macro("STRICT")
#pragma pop_macro("WIN32_LEAN_AND_MEAN")
#pragma pop_macro("NOCOMM")

#define BOOST_USE_WINDOWS_H
#endif

#endif // __PLASK__MINIMAL_WINDOWS_H

