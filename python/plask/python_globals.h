#ifndef PLASK__PYTHON_GLOBALS_H
#define PLASK__PYTHON_GLOBALS_H

#include <cmath>
#include <vector>

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
#   define _WIN32_WINNT 0x502
#   include <windows.h>
#   define BOOST_USE_WINDOWS_H
#endif

// ----------------------------------------------------------------------------------------------------------------------
// Shared pointer
#include <plask/memory.h>

#ifdef PLASK_SHARED_PTR_STD
namespace boost { namespace python {
    template<class T> inline T* get_pointer(std::shared_ptr<T> const& p) { return p.get(); }
}}
#endif

// ----------------------------------------------------------------------------------------------------------------------
// Important contains
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32)) && defined(hypot)
#   undef hypot
#endif


#include <plask/exceptions.h>
#include <plask/math.h>
#include <plask/vec.h>
#include <plask/axes.h>
#include <plask/geometry/space.h>
#include <plask/log/log.h>
#include <plask/parallel.h>

#include "python_enum.h"

namespace plask { namespace python {

namespace py = boost::python;

PLASK_PYTHON_API py::object py_eval(std::string string, py::object global=py::object(), py::object local=py::object());

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

PLASK_PYTHON_API std::string getPythonExceptionMessage();

// ----------------------------------------------------------------------------------------------------------------------
// String functions for Python3
#if PY_VERSION_HEX >= 0x03000000
    inline auto PyString_Check(PyObject* o) -> decltype(PyUnicode_Check(o)) { return PyUnicode_Check(o); }
    inline const char* PyString_AsString(PyObject* o) { return py::extract<const char*>(o); }
    inline bool PyInt_Check(PyObject* o) { return PyLong_Check(o); }
    inline long PyInt_AsLong(PyObject* o) { return PyLong_AsLong(o); }
#endif


// ----------------------------------------------------------------------------------------------------------------------
// Compare shared pointers
template <typename T>
bool __is__(const shared_ptr<T>& a, const shared_ptr<T>& b) {
    return a == b;
}

template <typename T>
long __hash__(const shared_ptr<T>& a) {
    auto p = a.get();
    return *reinterpret_cast<long*>(&p);
}


// ----------------------------------------------------------------------------------------------------------------------
inline py::object pass_through(const py::object& o) { return o; }


// ----------------------------------------------------------------------------------------------------------------------
struct PredicatePythonCallable {
    py::object callable;
    PredicatePythonCallable(const py::object& callable): callable(callable) {};
    bool operator()(const GeometryObject& obj) const {
        return py::extract<bool>(callable(const_pointer_cast<GeometryObject>(obj.shared_from_this())));
    }
};


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
// Get dtype for data
namespace detail {
    template <typename T> inline static py::handle<> dtype();
    template<> inline py::handle<> dtype<double>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyFloat_Type))); }
    template<> inline py::handle<> dtype<Vec<2,double>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(py::converter::registry::lookup(py::type_id<Vec<2,double>>()).get_class_object()))); }
    template<> inline py::handle<> dtype<Vec<3,double>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(py::converter::registry::lookup(py::type_id<Vec<3,double>>()).get_class_object()))); }
    template<> inline py::handle<> dtype<dcomplex>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyComplex_Type))); }
    template<> inline py::handle<> dtype<Vec<2,dcomplex>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(py::converter::registry::lookup(py::type_id<Vec<2,dcomplex>>()).get_class_object()))); }
    template<> inline py::handle<> dtype<Vec<3,dcomplex>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(py::converter::registry::lookup(py::type_id<Vec<3,dcomplex>>()).get_class_object()))); }
    template<> inline py::handle<> dtype<Tensor2<double>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyTuple_Type))); }
    template<> inline py::handle<> dtype<Tensor2<dcomplex>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyTuple_Type))); }
    template<> inline py::handle<> dtype<Tensor3<double>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyTuple_Type))); }
    template<> inline py::handle<> dtype<Tensor3<dcomplex>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyTuple_Type))); }
}


// ----------------------------------------------------------------------------------------------------------------------
// Register vectors of something

template <typename T>
struct VectorFromSequence {
    VectorFromSequence() {
        boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<std::vector<T>>());
    }
    static void* convertible(PyObject* obj) {
        if (!PySequence_Check(obj)) return NULL;
        return obj;
    }
    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
        void* storage = ((boost::python::converter::rvalue_from_python_storage<std::vector<T>>*)data)->storage.bytes;
        auto seq = py::object(py::handle<>(py::borrowed(obj)));
        py::stl_input_iterator<T> begin(seq), end;
        std::vector<T>* result = new(storage) std::vector<T>();
        result->reserve(py::len(seq)); for (auto iter = begin; iter != end; ++iter) result->push_back(*iter);
        data->convertible = storage;
    }
};

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
static inline py::class_<std::vector<T>, shared_ptr<std::vector<T>>> register_vector_of(const std::string& name) {
    VectorFromSequence<T>();
    py::class_<std::vector<T>, shared_ptr<std::vector<T>>> cls((name+"_list").c_str(), py::no_init); cls
        .def(py::vector_indexing_suite<std::vector<T>>())
        .def("__repr__", &str__vector_of<T>)
        .def("__str__", &str__vector_of<T>)
    ;
    py::scope scope;
    py::delattr(scope, py::str(name+"_list"));
    return cls;
}


// ----------------------------------------------------------------------------------------------------------------------
// Get space names
template <typename SpaceT> static inline std::string spaceName();
template <> inline std::string spaceName<Geometry2DCartesian>() { return "Cartesian2D"; }
template <> inline std::string spaceName<Geometry2DCylindrical>() { return "Cylindrical2D"; }
template <> inline std::string spaceName<Geometry3D>() { return "Cartesian3D"; }
template <> inline std::string spaceName<void>() { return ""; }

template <typename SpaceT> static inline std::string spaceSuffix();
template <> inline std::string spaceSuffix<Geometry2DCartesian>() { return "2D"; }
template <> inline std::string spaceSuffix<Geometry2DCylindrical>() { return "Cyl"; }
template <> inline std::string spaceSuffix<Geometry3D>() { return "3D"; }
template <> inline std::string spaceSuffix<void>() { return ""; }


// ----------------------------------------------------------------------------------------------------------------------
/// Class for setting logging configuration
struct LoggingConfig
{
    py::object getLoggingColor() const;
    void setLoggingColor(std::string color);

    py::object getLoggingDest() const;
    void setLoggingDest(py::object dest);

    LogLevel getLogLevel() const { return maxLoglevel; }
    void setLogLevel(LogLevel level) { if (!forcedLoglevel) maxLoglevel = level; }
};


/// Config class
struct Config
{
    std::string axes_name() const;

    void set_axes(std::string axis);

    std::string __str__() const;

    std::string __repr__() const;
};

extern PLASK_PYTHON_API AxisNames current_axes;

inline AxisNames* getCurrentAxes() {
    return &current_axes;
}

// ----------------------------------------------------------------------------------------------------------------------
// Helpers for parsing kwargs

namespace detail
{
    template <size_t i>
    static inline void _parse_kwargs(py::list& arglist, py::dict& kwargs) {}

    template <size_t i, typename... Names>
    static inline void _parse_kwargs(py::list& arglist, py::dict& kwargs, const std::string& name, const Names&... names) {
        py::object oname(name);
        if (kwargs.has_key(oname))
        {
            if (i < py::len(arglist)) {
                throw name;
            } else {
                arglist.append(kwargs[oname]);
                py::delitem(kwargs, oname);
            }
        }
        _parse_kwargs<i+1>(arglist, kwargs, names...);
    }
}

/// Helper for parsing arguments in raw_function
template <typename... Names>
static inline void parseKwargs(const std::string& fname, py::tuple& args, py::dict& kwargs, const Names&... names) {
    kwargs = kwargs.copy();
    py::list arglist(args);
    try {
        detail::_parse_kwargs<0>(arglist, kwargs, names...);
    } catch (const std::string& name) {
        throw TypeError("%1%() got multiple values for keyword argument '%2%'", fname, name);
    }
    if (py::len(arglist) != sizeof...(names))
        throw TypeError("%1%() takes exactly %2% non-keyword arguments (%3% given)", fname, sizeof...(names), py::len(arglist));
    args = py::tuple(arglist);
}

// ----------------------------------------------------------------------------------------------------------------------
// Parallel locking

extern PLASK_PYTHON_API OmpNestLock python_omp_lock;

}} // namespace python::plask


#endif // PLASK__PYTHON_GLOBALS_H
