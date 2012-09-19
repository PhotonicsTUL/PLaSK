#ifndef PLASK__PYTHON_GLOBALS_H
#define PLASK__PYTHON_GLOBALS_H

#include <cmath>

// ----------------------------------------------------------------------------------------------------------------------
// Shared pointer
#include <plask/memory.h>

#ifdef PLASK_SHARED_PTR_STD
namespace boost { namespace python {
    template<class T> inline T* get_pointer(std::shared_ptr<T> const& p) { return p.get(); }
}}
#endif

// ----------------------------------------------------------------------------------------------------------------------
// Important includes
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/ndarraytypes.h>

#include <plask/exceptions.h>
#include <plask/math.h>
#include <plask/vec.h>
#include <plask/axes.h>
#include <plask/geometry/space.h>

#include "python_enum.h"

namespace plask { namespace python {

namespace py = boost::python;

// ----------------------------------------------------------------------------------------------------------------------
// Exceptions

template <typename ExcType>
void register_exception(PyObject* py_exc) {
    py::register_exception_translator<ExcType>( [=](const ExcType& err) { PyErr_SetString(py_exc, err.what()); } );
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

struct IOError: public Exception {
    template <typename... T>
    IOError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};


// ----------------------------------------------------------------------------------------------------------------------
// String functions for Python3
#if PY_VERSION_HEX >= 0x03000000
    inline auto PyString_Check(PyObject* o) -> decltype(PyUnicode_Check(o)) { return PyUnicode_Check(o); }
    inline const char* PyString_AsString(PyObject* o) { return py::extract<const char*>(o); }
    inline bool PyInt_Check(PyObject* o) { return PyLong_Check(o); }
    inline long PyInt_AsLong(PyObject* o) { return PyLong_AsLong(o); }
#   define PyCodeObject PyObject
#endif


// ----------------------------------------------------------------------------------------------------------------------
// Compare shared pointers
template <typename T>
bool __is__(const shared_ptr<T>& a, const shared_ptr<T>& b) {
    return a == b;
}


// ----------------------------------------------------------------------------------------------------------------------
// Format complex numbers in Python way
template <typename T>
inline std::string pyformat(const T& v) { std::stringstream s; s << v; return s.str(); }

template <>
inline std::string pyformat<dcomplex>(const dcomplex& v) { return format("(%g%+gj)", real(v), imag(v)); }


// ----------------------------------------------------------------------------------------------------------------------
// PLaSK str function for Python objects
inline std::string str(py::object obj) {
    return py::extract<std::string>(py::str(obj));
}


// ----------------------------------------------------------------------------------------------------------------------
// Typename
template <typename T>
constexpr std::string type_name() {
    const std::string s = py::type_id<T>().name();
    size_t n = s.find_last_of(':');
    return (n != s.npos)? s.substr(n+1) : s;
}


// ----------------------------------------------------------------------------------------------------------------------
// Get numpy typenums for some types
namespace detail {
    template <typename T> static inline constexpr int typenum();
    template <> inline constexpr int typenum<double>() { return NPY_DOUBLE; }
    template <> inline constexpr int typenum<dcomplex>() { return NPY_CDOUBLE; }
    template <> constexpr inline int typenum<Vec<2,double>>() { return NPY_DOUBLE; }
    template <> constexpr inline int typenum<Vec<2,dcomplex>>() { return NPY_CDOUBLE; }
    template <> constexpr inline int typenum<Vec<3,double>>() { return NPY_DOUBLE; }
    template <> constexpr inline int typenum<Vec<3,dcomplex>>() { return NPY_CDOUBLE; }
}


// ----------------------------------------------------------------------------------------------------------------------
// Get dtype for data
namespace detail {
    extern py::object vector2fClass;
    extern py::object vector2cClass;
    extern py::object vector3fClass;
    extern py::object vector3cClass;

    template <typename T> inline static py::handle<> dtype();
    template<> inline py::handle<> dtype<double>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyFloat_Type))); }
    template<> inline py::handle<> dtype<Vec<2,double>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(vector2fClass.ptr()))); }
    template<> inline py::handle<> dtype<Vec<3,double>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(vector2cClass.ptr()))); }
    template<> inline py::handle<> dtype<dcomplex>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyComplex_Type))); }
    template<> inline py::handle<> dtype<Vec<2,dcomplex>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(vector3fClass.ptr()))); }
    template<> inline py::handle<> dtype<Vec<3,dcomplex>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(vector3cClass.ptr()))); }
}


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


// ----------------------------------------------------------------------------------------------------------------------
// Get space names
template <typename SpaceT> static inline std::string spaceName();
template <> inline std::string spaceName<Geometry2DCartesian>() { return "Cartesian2D"; }
template <> inline std::string spaceName<Geometry2DCylindrical>() { return "Cylindrical2D"; }
template <> inline std::string spaceName<Geometry3D>() { return "Cartesian3D"; }

template <typename SpaceT> static inline std::string spaceSuffix();
template <> inline std::string spaceSuffix<Geometry2DCartesian>() { return "2D"; }
template <> inline std::string spaceSuffix<Geometry2DCylindrical>() { return "Cyl"; }
template <> inline std::string spaceSuffix<Geometry3D>() { return "3D"; }


// ----------------------------------------------------------------------------------------------------------------------
// Config
struct Config
{
    // Current axis names
    static AxisNames axes;

    std::string axes_name() {
        return axes.str();
    }
    void set_axes(std::string axis) {
        axes = AxisNames::axisNamesRegister.get(axis);
    }

    std::string __str__() {
        return std::string()
            + "axes:   " + axes_name();
        ;
    }

    std::string __repr__() {
        return
            format("config.axes = '%s'", axes_name())
        ;
    }

};
extern Config config;


}} // namespace plask::python

#endif // PLASK__PYTHON_GLOBALS_H
