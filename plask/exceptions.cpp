#include "exceptions.h"

#include <plask/config.h>
//#undef PRINT_STACKTRACE_ON_EXCEPTION

#ifdef PRINT_STACKTRACE_ON_EXCEPTION
#include <backward.hpp>
namespace backward {
    backward::SignalHandling sh;
} // print backtrace on segfault, etc.

/*void terminate_handler() {  //this can be eventually put: std::set_terminate(terminate_handler); but this is not required because signal is generated
    backward::StackTrace st; st.load_here(128);
    backward::Printer p; p.print(st);
    std::exception_ptr exptr = std::current_exception();
    if (exptr) {
        try {
            std::rethrow_exception(exptr);
        }
        catch (std::exception &ex) { std::cerr << "Terminated due to exception: " << ex.what();  }
        catch (...) { std::cerr << "Terminated due to non-std exception."; }
    } else { std::cerr << "Terminated, but not due to exception."; }
    std::abort();  // forces abnormal termination, see http://www.cplusplus.com/reference/exception/set_terminate/
}*/
#endif

namespace plask {

plask::Exception::Exception(const std::string &msg): std::runtime_error(msg) {
}

}   // namespace plask
