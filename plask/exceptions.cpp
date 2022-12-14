/* 
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) // win32 support
#include <win_printstack.hpp>
#include <signal.h>     /* signal, raise, sig_atomic_t */
#endif

#include "exceptions.hpp"

#include <plask/config.hpp>

#ifdef PRINT_STACKTRACE_ON_EXCEPTION

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32) // win32 support

inline void print_current_exception() {
    std::exception_ptr p = std::current_exception();
    if (p) {
        try {
             std::rethrow_exception (p);
        } catch (std::exception& e) {
             printf("Current exception: %s\n", e.what());
        } catch (...) {
             printf("%s\n", "Current exception is not std one.");
        }
    } else
        printf("%s\n", "There is no current exception.");
}

const char* sig_name(int sig_nr) {
    switch (sig_nr) {
        case SIGABRT: return "SIGABRT";
        case SIGSEGV: return "SIGSEGV";
        case SIGTERM: return "SIGTERM";
    }
    return "unknown";
}

void plask_win_signal_handler (int param) {
	#pragma omp critical (winbacktrace)
	{
        printf("Signal %s (%d) handler:\n", sig_name(param), param);
		print_current_exception();
		//SIG_DFL(param); //call default signal handler
		printStack();   //print stack-trace
	}
}

void plask_win_terminate_handler () {
	#pragma omp critical (winbacktrace)
	{
		printf("Terminate handler:\n");
		print_current_exception();
		printStack();   //print stack-trace
		abort();  // forces abnormal termination
	}
}

struct PlaskWinRegisterSignalHandler {
    PlaskWinRegisterSignalHandler() {
        signal(SIGABRT, plask_win_signal_handler);
        signal(SIGSEGV, plask_win_signal_handler);
        signal(SIGTERM, plask_win_signal_handler);
        std::set_terminate (plask_win_terminate_handler);
    }
} __plaskWinRegisterSignalHandler;

#else       //non-windows systems:

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
