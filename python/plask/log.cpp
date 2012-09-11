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

}} // namespace plask::python