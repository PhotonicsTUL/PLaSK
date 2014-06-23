#include "python_globals.h"
#include <boost/algorithm/string.hpp>
#include <boost/python/enum.hpp>

#include <plask/log/log.h>
#include <plask/log/data.h>

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   include <windows.h>
#   define BOOST_USE_WINDOWS_H
#else
#   include <unistd.h>
#endif

namespace plask { namespace python {

#define LOG_ENUM(v) loglevel.value(BOOST_PP_STRINGIZE(v), LOG_##v); scope.attr(BOOST_PP_STRINGIZE(LOG_##v)) = loglevel.attr(BOOST_PP_STRINGIZE(v));

typedef Data2DLog<std::string, std::string> LogOO;

void Data2DLog__call__(LogOO& self, py::object arg, py::object val) {
    self(py::extract<std::string>(py::str(arg)), py::extract<std::string>(py::str(val)));
}

int Data2DLog_count(LogOO& self, py::object arg, py::object val) {
    return self.count(py::extract<std::string>(py::str(arg)), py::extract<std::string>(py::str(val)));
}

// Logger
/// Class writing logs to Python sys.stderr
struct PythonSysLogger: public plask::Logger {

    enum ColorMode {
        COLOR_NONE,
        COLOR_ANSI
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
        , COLOR_WINDOWS
#       endif
    };

    enum Dest {
        DEST_STDERR,
        DEST_STDOUT
    };

    ColorMode color;
    Dest dest;

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
        void setcolor(unsigned short fg);
        unsigned short previous_color;
#   endif

    const char* head(plask::LogLevel level);

    PythonSysLogger();

    virtual void writelog(plask::LogLevel level, const std::string& msg);

};

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)

    PythonSysLogger::PythonSysLogger(): color(PythonSysLogger::COLOR_WINDOWS), dest(DEST_STDERR) {}

#   define COL_BLACK 0
#   define COL_BLUE 1
#   define COL_GREEN 2
#   define COL_CYAN 3
#   define COL_RED 4
#   define COL_MAGENTA 5
#   define COL_BROWN 6
#   define COL_WHITE 7
#   define COL_GRAY 8
#   define COL_BRIGHT_BLUE 9
#   define COL_BRIGHT_GREEN 10
#   define COL_BRIGHT_CYAN 11
#   define COL_BRIGHT_RED 12
#   define COL_BRIGHT_MAGENTA 13
#   define COL_YELLOW 14
#   define COL_BRIGHT_WHITE 15

    inline void PythonSysLogger::setcolor(unsigned short fg) {
        HANDLE handle = GetStdHandle((dest==DEST_STDERR)?STD_ERROR_HANDLE:STD_OUTPUT_HANDLE);
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        GetConsoleScreenBufferInfo(handle, &csbi);
        previous_color = csbi.wAttributes;
        SetConsoleTextAttribute(handle, (csbi.wAttributes & 0xF0) | fg);
    }

#else

    PythonSysLogger::PythonSysLogger(): color(isatty(fileno(stderr))? PythonSysLogger::COLOR_ANSI : PythonSysLogger::COLOR_NONE), dest(DEST_STDERR) {}

#endif

#define ANSI_DEFAULT "\033[00m"
#define ANSI_BLACK   "\033[30m"
#define ANSI_RED     "\033[31m"
#define ANSI_GREEN   "\033[32m"
#define ANSI_BROWN  "\033[33m"
#define ANSI_BLUE    "\033[34m"
#define ANSI_MAGENTA "\033[35m"
#define ANSI_CYAN    "\033[36m"
#define ANSI_WHITE   "\033[37m"
#define ANSI_GRAY   "\033[30;01m"
#define ANSI_BRIGHT_RED     "\033[31;01m"
#define ANSI_BRIGHT_GREEN   "\033[32;01m"
#define ANSI_YELLOW  "\033[33;01m"
#define ANSI_BRIGHT_BLUE    "\033[34;01m"
#define ANSI_BRIGHT_MAGENTA "\033[35;01m"
#define ANSI_BRIGHT_CYAN    "\033[36;01m"
#define ANSI_BRIGHT_WHITE   "\033[37;01m"
const char* PythonSysLogger::head(LogLevel level) {
    if (color == PythonSysLogger::COLOR_ANSI)
        switch (level) {
            case LOG_CRITICAL_ERROR:return ANSI_BRIGHT_RED  "CRITICAL ERROR";
            case LOG_ERROR:         return ANSI_BRIGHT_RED  "ERROR         ";
            case LOG_WARNING:       return ANSI_BROWN       "WARNING       ";
            case LOG_INFO:          return ANSI_BRIGHT_BLUE "INFO          ";
            case LOG_RESULT:        return ANSI_GREEN       "RESULT        ";
            case LOG_DATA:          return ANSI_CYAN        "DATA          ";
            case LOG_DETAIL:        return ANSI_DEFAULT     "DETAIL        ";
            case LOG_ERROR_DETAIL:  return ANSI_RED         "ERROR DETAIL  ";
            case LOG_DEBUG:         return ANSI_GRAY        "DEBUG         ";
        }
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    else if (color == PythonSysLogger::COLOR_WINDOWS)
        switch (level) {
            case LOG_ERROR:         setcolor(COL_BRIGHT_RED); return "ERROR         ";
            case LOG_CRITICAL_ERROR:setcolor(COL_BRIGHT_RED); return "CRITICAL ERROR";
            case LOG_WARNING:       setcolor(COL_BROWN); return "WARNING       ";
            case LOG_INFO:          setcolor(COL_BRIGHT_CYAN); return "INFO          ";
            case LOG_RESULT:        setcolor(COL_GREEN); return "RESULT        ";
            case LOG_DATA:          setcolor(COL_CYAN); return "DATA          ";
            case LOG_DETAIL:        return "DETAIL        ";
            case LOG_ERROR_DETAIL:  setcolor(COL_RED); return "ERROR DETAIL  ";
            case LOG_DEBUG:         setcolor(COL_GRAY); return "DEBUG         ";
        }
#endif
    else
        switch (level) {
            case LOG_CRITICAL_ERROR:return "CRITICAL ERROR";
            case LOG_ERROR:         return "ERROR         ";
            case LOG_WARNING:       return "WARNING       ";
            case LOG_INFO:          return "INFO          ";
            case LOG_RESULT:        return "RESULT        ";
            case LOG_DATA:          return "DATA          ";
            case LOG_DETAIL:        return "DETAIL        ";
            case LOG_ERROR_DETAIL:  return "ERROR DETAIL  ";
            case LOG_DEBUG:         return "DEBUG         ";
        }
    return "UNSPECIFIED   "; // mostly to silence compiler warning than to use in the real life
}

void PythonSysLogger::writelog(LogLevel level, const std::string& msg) {
    if (color == COLOR_ANSI) {
        if (dest == DEST_STDERR)
            PySys_WriteStderr("%s: %s" ANSI_DEFAULT "\n", head(level), msg.c_str());
        else
            PySys_WriteStdout("%s: %s" ANSI_DEFAULT "\n", head(level), msg.c_str());
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    } else if (color == COLOR_WINDOWS) {
        if (dest == DEST_STDERR) {
            PySys_WriteStderr("%s: %s\n", head(level), msg.c_str());
            SetConsoleTextAttribute(GetStdHandle(STD_ERROR_HANDLE), previous_color);
        } else {
            PySys_WriteStdout("%s: %s\n", head(level), msg.c_str());
            SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), previous_color);
        }
#endif
    } else {
        if (dest == DEST_STDERR)
            PySys_WriteStderr("%s: %s\n", head(level), msg.c_str());
        else
            PySys_WriteStdout("%s: %s\n", head(level), msg.c_str());
    }
}


#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
__declspec(dllexport)
#endif
shared_ptr<Logger> makePythonLogger() {
    return shared_ptr<Logger>(new PythonSysLogger);
}


py::object LoggingConfig::getLoggingColor() const {
    auto logger = dynamic_pointer_cast<PythonSysLogger>(default_logger);
    if (logger)
        switch (logger->color) {
            case PythonSysLogger::COLOR_ANSI: return py::str("ansi");
            case PythonSysLogger::COLOR_NONE: return py::str("none");
#           if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
            case PythonSysLogger::COLOR_WINDOWS: return py::str("windows");
#           endif
        }
    return py::object();
}

void LoggingConfig::setLoggingColor(std::string color) {
    boost::to_lower(color);
    if (auto logger = dynamic_pointer_cast<PythonSysLogger>(default_logger)) {
        if (color == "ansi")
            logger->color = PythonSysLogger::COLOR_ANSI;
#       if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
        else if (color == "windows")
            logger->color = PythonSysLogger::COLOR_WINDOWS;
#       endif
        else if (color == "none")
            logger->color = PythonSysLogger::COLOR_NONE;
        else
            throw ValueError("Wrong logging coloring specification.");
        return;
    }
    throw TypeError("Setting coloring for current logging system does not make sense.");
}

py::object LoggingConfig::getLoggingDest() const {
    auto logger = dynamic_pointer_cast<PythonSysLogger>(default_logger);
    if (logger)
        switch (logger->dest) {
            case PythonSysLogger::DEST_STDERR: return py::str("stderr");
            case PythonSysLogger::DEST_STDOUT: return py::str("stdout");
        }
    return py::object();
}

void LoggingConfig::setLoggingDest(py::object dest) {
    if (auto logger = dynamic_pointer_cast<PythonSysLogger>(default_logger)) {
        py::object sys = py::import("sys");
        std::string dst;
        try { dst = py::extract<std::string>(dest); }
        catch (py::error_already_set) { PyErr_Clear(); }
        if (dest == sys.attr("stderr") || dst == "stderr" || dst == "sys.stderr")
            logger->dest = PythonSysLogger::DEST_STDERR;
        else if (dest == sys.attr("stdout") || dst == "stdout" || dst == "sys.stdout")
            logger->dest = PythonSysLogger::DEST_STDOUT;
        else
            throw ValueError("Logging output can only be sys.stderr or sys.stdout.");
        return;
    }
    throw TypeError("Setting output for current logging system does not make sense.");
}

void print_log(LogLevel level, py::object msg) {
    writelog(level, py::extract<std::string>(py::str(msg)));
}

void register_python_log()
{
    py_enum<LogLevel> loglevel("loglevel", "Log levels used in PLaSK");
    py::scope scope;
    LOG_ENUM(CRITICAL_ERROR);
    LOG_ENUM(ERROR);
    LOG_ENUM(WARNING);
    LOG_ENUM(INFO);
    LOG_ENUM(RESULT);
    LOG_ENUM(DATA);
    LOG_ENUM(DETAIL);
    LOG_ENUM(ERROR_DETAIL);
    LOG_ENUM(DEBUG);

    py::def("print_log", print_log, "Print log message into specified log level", (py::arg("level"), "msg"));

    py::class_<LogOO>("DataLog2",
        "Class used to log relations between two variables (argument and value)\n\n"
        "DataLog2(prefix, arg_name, val_name)\n    Create log with specified prefix, name, and argument and value names\n",
        py::init<std::string,std::string,std::string,std::string>((py::arg("prefix"), "name", "arg_name", "val_name")))
        .def("__call__", &Data2DLog__call__, (py::arg("arg"), "val"), "Log value pair")
        .def("count", &Data2DLog_count, (py::arg("arg"), "val"), "Log value pair and count successive logs")
        .def("reset", &LogOO::resetCounter, "Reset logs counter")
    ;

    py::class_<LoggingConfig>("LoggingConfig", "Settings of the logging system", py::no_init)
        .add_property("color", &LoggingConfig::getLoggingColor, &LoggingConfig::setLoggingColor, "Output color type")
        .add_property("output", &LoggingConfig::getLoggingDest, &LoggingConfig::setLoggingDest, "Output destination")
        .add_property("level", &LoggingConfig::getLogLevel, &LoggingConfig::setLogLevel, "Maximum log level")
    ;
}



}} // namespace plask::python
