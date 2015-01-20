#ifndef PLASK__UTILS_DEBUG_H
#define PLASK__UTILS_DEBUG_H

/** @file
This file contains debuging utils. For internal use only.
*/

#include <plask/config.h>

#if defined(PRINT_STACKTRACE_ON_EXCEPTION) && !(defined(_WIN32) || defined(__WIN32__) || defined(WIN32))

#include <backward.hpp>

#define PLASK_PRINT_STACK_HERE(d) { backward::StackTrace st; st.load_here(d); backward::Printer p; p.print(st); }

#else

#define PLASK_PRINT_STACK_HERE(d)

#endif

#endif // PLASK__UTILS_DEBUG_H
