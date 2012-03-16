#ifndef PLASK__PYTHON_GLOBALS_H
#define PLASK__PYTHON_GLOBALS_H

#include <boost/python.hpp>
#include <plask/config.h>
#include <plask/math.h>
#include <plask/memory.h>

namespace plask { namespace python {

namespace py = boost::python;

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
            PyErr_SetString(PyExc_ValueError, "Only z or y allowed for vertical_axis");
            throw py::error_already_set();
        }
        z_up = axis == "z";
    }

    std::string __str__() {
        return std::string()
            + "vertical_axis:   " + (z_up?"z":"y")
        ;
    }

    std::string __repr__() {
        return std::string()
            + "config.vertical_axis = " + (z_up?"'z'":"'y'")
        ;
    }

};

extern Config config;

inline static void register_config()
{
    py::class_<Config>("config", "Global PLaSK configuration.", py::no_init)
        .def("__str__", &Config::__str__)
        .def("__repr__", &Config::__repr__)
        .add_property("vertical_axis", &Config::get_vaxis, &Config::set_vaxis,
                      "Denotes orientation of coordinate system. Holds the name a vertical axis which i.e. the one along layers growth direction.")
    ;
    py::scope().attr("config") = config;
}

// ----------------------------------------------------------------------------------------------------------------------

// Format complex numbers in Python way
namespace detail {

template <typename T>
struct Sc {
    T v;
    Sc(T c) : v(c) {}
    friend inline std::ostream& operator<<(std::ostream& out, const Sc& c) {
        out << c.v;
        return out;
    }
};
template <>
struct Sc<dcomplex> {
    dcomplex v;
    Sc(dcomplex c) : v(c) {}
    friend inline std::ostream& operator<<(std::ostream& out, const Sc& c) {
        double r = c.v.real(), i = c.v.imag();
        out << "(" << r << ((i>=0)?"+":"") << i << "j)";
        return out;
    }
};

} // namespace plask::python::detail

template <typename T>
inline detail::Sc<T> sc(const T& v) { return detail::Sc<T>(v); }


}} // namespace plask::python

#endif // PLASK__PYTHON_GLOBALS_H
