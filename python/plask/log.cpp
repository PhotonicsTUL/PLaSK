#include "python_globals.h"
#include <boost/python/enum.hpp>

#include <plask/log/log.h>
#include <plask/log/data.h>

namespace plask { namespace python {

#define LOG_ENUM(v) loglevel.value(BOOST_PP_STRINGIZE(v), LOG_##v); scope.attr(BOOST_PP_STRINGIZE(LOG_##v)) = loglevel.attr(BOOST_PP_STRINGIZE(v));

typedef Data2DLog<std::string, std::string> LogOO;

void Data2DLog__call__(LogOO& self, py::object arg, py::object val) {
    self(py::extract<std::string>(py::str(arg)), py::extract<std::string>(py::str(val)));
}

int Data2DLog_count(LogOO& self, py::object arg, py::object val) {
    return self.count(py::extract<std::string>(py::str(arg)), py::extract<std::string>(py::str(val)));
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

    py::def("print_log", (void(*)(LogLevel, const std::string&))&writelog, "Print log message into specified log level", (py::arg("level"), "msg"));

    py::class_<LogOO>("DataLog2",
        "Class used to log relations between two variables (argument and value)\n\n"
        "DataLog2(prefix, arg_name, val_name)\n    Create log with specified prefix, name, and argument and value names\n",
        py::init<std::string,std::string,std::string,std::string>((py::arg("prefix"), "name", "arg_name", "val_name")))
        .def("__call__", &Data2DLog__call__, (py::arg("arg"), "val"), "Log value pair")
        .def("count", &Data2DLog_count, (py::arg("arg"), "val"), "Log value pair and count successive logs")
        .def("reset", &LogOO::resetCounter, "Reset logs counter")
    ;
}

#define DEFAULT "\033[00m"
#define BLACK   "\033[30m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define BROWN  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define WHITE   "\033[37m"
#define GRAY   "\033[30;01m"
#define BRIGHT_RED     "\033[31;01m"
#define BRIGHT_GREEN   "\033[32;01m"
#define YELLOW  "\033[33;01m"
#define BRIGHT_BLUE    "\033[34;01m"
#define BRIGHT_MAGENTA "\033[35;01m"
#define BRIGHT_CYAN    "\033[36;01m"
#define BRIGHT_WHITE   "\033[37;01m"
const char* PythonSysLogger::head(LogLevel level) {
    switch (level) {
        case LOG_CRITICAL_ERROR:return BRIGHT_RED "CRITICAL ERROR";
        case LOG_ERROR:         return BRIGHT_RED "ERROR         ";
        case LOG_WARNING:       return BROWN "WARNING       ";
        case LOG_INFO:          return BRIGHT_BLUE "INFO          ";
        case LOG_RESULT:        return GREEN "RESULT        ";
        case LOG_DATA:          return CYAN "DATA          ";
        case LOG_DETAIL:        return DEFAULT "DETAIL        ";
        case LOG_ERROR_DETAIL:  return RED "ERROR DETAIL  ";
        case LOG_DEBUG:         return GRAY "DEBUG         ";
    }
    return "UNSPECIFIED   "; // mostly to silence compiler warning than to use in the real life
}

void PythonSysLogger::writelog(LogLevel level, const std::string& msg) {
    PySys_WriteStderr("%s: %s" DEFAULT "\n", head(level), msg.c_str());
}





}} // namespace plask::python