#ifndef PLASK__PYTHON_GLOBALS_H
#define PLASK__PYTHON_GLOBALS_H

#include <cmath>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <plask/config.h>
#include <plask/exceptions.h>
#include <plask/math.h>
#include <plask/memory.h>

namespace plask { namespace python {

namespace py = boost::python;

// ----------------------------------------------------------------------------------------------------------------------
// Exceptions

template <typename ExcType>
void register_exception(PyObject* py_exc) {
    py::register_exception_translator<ExcType>([=](const ExcType& err){ PyErr_SetString(py_exc, err.what()); });
}

struct ValueError: public Exception {
    template <typename... T>
    ValueError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct TypeError: public Exception {
    template <typename... T>
    TypeError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct IndexError: public Exception {
    template <typename... T>
    IndexError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct KeyError: public Exception {
    template <typename... T>
    KeyError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct AttributeError: public Exception {
    template <typename... T>
    AttributeError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct StopIteration: public Exception {
    template <typename... T>
    StopIteration(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

// ----------------------------------------------------------------------------------------------------------------------
// Config

struct Config
{
    // Which axis is up (z or y)
    static bool z_up;

    std::string get_vaxis() {
        if (z_up) return "z"; else return "y";
    }
    void set_vaxis(std::string axis) {
        if (axis != "z" and axis != "y") {
            throw ValueError("Only z or y allowed for vertical_axis");
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


// ----------------------------------------------------------------------------------------------------------------------
// Register vectors of something
template <typename T>
static std::string str__vector_of(const std::vector<T>& self) {
    std::string result = "[";
    int i = self.size()-1;
    for (auto v: self) {
        result += py::extract<std::string>(py::object(v).attr("__repr__")());
        result += (i)? ", " : "";
        --i;
    }
    return result + "]";
}
template <typename T>
static inline py::class_< std::vector<T>, shared_ptr<std::vector<T>> > register_vector_of(const std::string& name) {
    return py::class_< std::vector<T>, shared_ptr<std::vector<T>> >((name+"_list").c_str(), py::no_init)
        .def(py::vector_indexing_suite<std::vector<T>>())
        .def("__repr__", &str__vector_of<T>)
        .def("__str__", &str__vector_of<T>)
    ;
}

}} // namespace plask::python

#endif // PLASK__PYTHON_GLOBALS_H
