#include "exceptions.h"

#include <plask/config.h>

#ifdef PRINT_STACKTRACE_ON_EXCEPTION
#include <backward.hpp>
namespace backward { backward::SignalHandling sh; } // print backtrace on segfault, etc.
#endif

namespace plask {

plask::Exception::Exception(const std::string &msg): std::runtime_error(msg) {
#ifdef PRINT_STACKTRACE_ON_EXCEPTION
    std::cerr << "Exception throwed: " << msg << std::endl;
    backward::StackTrace st; st.load_here(128);
    backward::Printer p; p.print(st);
#endif
}

}   // namespace plask
