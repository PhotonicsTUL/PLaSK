#include "exceptions.h"

#include <plask/config.h>
//#undef PRINT_STACKTRACE_ON_EXCEPTION

#ifdef PRINT_STACKTRACE_ON_EXCEPTION

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) //win32 support

#include <win_printstack.hpp>
#include <signal.h>     /* signal, raise, sig_atomic_t */

void plask_win_signal_handler (int param) {
    //SIG_DFL(param); //call default signal handler
    printStack();   //print stack-trace
}

void plask_win_terminate_handler () {
  printStack();   //print stack-trace
  abort();  // forces abnormal termination
}

struct PlaskWinRegisterSignalHandler {
    PlaskWinRegisterSignalHandler() {
        signal(SIGABRT, plask_win_signal_handler);
        signal(SIGSEGV, plask_win_signal_handler);
        signal(SIGTERM, plask_win_signal_handler);
        std::set_terminate (plask_win_terminate_handler);
    }
} __plaskWinRegisterSignalHandler;

#else       //other systems:

#include <backward.hpp>
namespace backward {
    backward::SignalHandling sh;
} // print backtrace on segfault, etc.

#endif  //other systems support

#endif

namespace plask {

plask::Exception::Exception(const std::string &msg): std::runtime_error(msg) {
}

}   // namespace plask
