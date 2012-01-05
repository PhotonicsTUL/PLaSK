#ifndef PLASK__PYTHON_GLOBALS_H
#define PLASK__PYTHON_GLOBALS_H

#include <boost/python.hpp>
namespace py = boost::python;

#include <config.h>

namespace plask { namespace python {

// Some config variables
struct Config
{
    // Which axis is up (z or y)
    static bool z_up;

    std::string get_vaxis() {
        if (z_up) return "z"; else return "y";
    }
    void set_vaxis(std::string axis) {
        if (axis != "z" and axis != "y") {
            PyErr_SetString(PyExc_ValueError, "Only z or y allowed for axis_up");
            throw py::error_already_set();
        }
        z_up = axis == "z";
    }

    std::string __str__() {
        return std::string()
            + "axis_up:   " + (z_up?"z":"y")
        ;
    }

    std::string __repr__() {
        return std::string()
            + "config.axis_up = " + (z_up?"'z'":"'y'")
        ;
    }

};

extern Config config;

inline static void register_config()
{
    py::class_<Config>("config", "Global PLaSK configuration.", py::no_init)
        .def("__str__", &Config::__str__)
        .def("__repr__", &Config::__repr__)
        .add_property("axis_up", &Config::get_vaxis, &Config::set_vaxis,
                      "Denotes orientation of coordinate system. Holds the name a vertical axis which i.e. the one along layers growth direction.")
    ;
    py::scope().attr("config") = config;
}

}} // namespace plask::python

#endif // PLASK__PYTHON_GLOBALS_H