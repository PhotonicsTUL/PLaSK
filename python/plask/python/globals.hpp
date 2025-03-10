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
#ifndef PLASK__PYTHON_GLOBALS_H
#define PLASK__PYTHON_GLOBALS_H

#include <cmath>
#include <plask/vec.hpp>
#include <vector>

// ----------------------------------------------------------------------------------------------------------------------
// Shared pointer
#include <plask/memory.hpp>

#ifdef PLASK_SHARED_PTR_STD
namespace boost { namespace python {
template <class T> inline T* get_pointer(std::shared_ptr<T> const& p) { return p.get(); }
}}  // namespace boost::python
#endif

// ----------------------------------------------------------------------------------------------------------------------
// Important contains
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <frameobject.h>

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32)) && defined(hypot)
#    undef hypot
#endif

#include <plask/axes.hpp>
#include <plask/exceptions.hpp>
#include <plask/geometry/space.hpp>
#include <plask/log/log.hpp>
#include <plask/manager.hpp>
#include <plask/math.hpp>
#include <plask/parallel.hpp>

#include "enum.hpp"

namespace plask { namespace python {

namespace py = boost::python;

PLASK_PYTHON_API py::object py_eval(std::string string, py::object global = py::object(), py::object local = py::object());

// ----------------------------------------------------------------------------------------------------------------------
// Exceptions

template <typename ExcType> void register_exception(PyObject* py_exc) {
    py::register_exception_translator<ExcType>([=](const ExcType& err) { PyErr_SetString(py_exc, err.what()); });
}

struct ValueError : public Exception {
    template <typename... T> ValueError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct TypeError : public Exception {
    template <typename... T> TypeError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct IndexError : public Exception {
    template <typename... T> IndexError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct KeyError : public Exception {
    template <typename... T> KeyError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct AttributeError : public Exception {
    template <typename... T> AttributeError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct StopIteration : public Exception {
    StopIteration() : Exception("") {}
    template <typename... T> StopIteration(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct IOError : public Exception {
    template <typename... T> IOError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

struct RecursionError : public Exception {
    template <typename... T> RecursionError(const std::string& msg, const T&... args) : Exception(msg, args...) {}
};

PLASK_PYTHON_API std::string getPythonExceptionMessage();

PLASK_PYTHON_API int printPythonException(PyObject* otype,
                                          PyObject* value,
                                          PyObject* otraceback,
                                          const char* scriptname = nullptr,
                                          const char* top_frame = nullptr,
                                          int scriptline = 0);

inline int printPythonException(PyObject* value,
                                const char* scriptname = nullptr,
                                const char* top_frame = nullptr,
                                int scriptline = 0) {
    PyObject* type = PyObject_Type(value);
    PyObject* traceback = PyException_GetTraceback(value);
    py::handle<> type_h(type), traceback_h(py::allow_null(traceback));
    return printPythonException(type, value, traceback, scriptname, top_frame, scriptline);
}

// ----------------------------------------------------------------------------------------------------------------------
// Compare shared pointers
template <typename T> bool __is__(const shared_ptr<T>& a, const shared_ptr<T>& b) { return a == b; }

template <typename T> long __hash__(const shared_ptr<T>& a) {
    auto p = a.get();
    return *reinterpret_cast<long*>(&p);
}

// ----------------------------------------------------------------------------------------------------------------------
inline py::object pass_through(const py::object& o) { return o; }

// ----------------------------------------------------------------------------------------------------------------------
struct PredicatePythonCallable {
    py::object callable;
    PredicatePythonCallable(const py::object& callable) : callable(callable){};
    bool operator()(const GeometryObject& obj) const {
        return py::extract<bool>(callable(const_pointer_cast<GeometryObject>(obj.shared_from_this())));
    }
};

// ----------------------------------------------------------------------------------------------------------------------
// Format complex numbers in Python way
template <typename T> inline std::string pyformat(const T& v) {
    std::stringstream s;
    s << v;
    return s.str();
}

template <> inline std::string pyformat<dcomplex>(const dcomplex& v) { return format("({:g}{:+g}j)", real(v), imag(v)); }

// ----------------------------------------------------------------------------------------------------------------------
// PLaSK str function for Python objects
inline std::string str(py::object obj) { return py::extract<std::string>(py::str(obj)); }

// ----------------------------------------------------------------------------------------------------------------------
// Get dtype for data
namespace detail {
template <typename T> inline py::handle<> dtype() {
    return py::handle<>(
        py::borrowed<>(reinterpret_cast<PyObject*>(py::converter::registry::lookup(py::type_id<T>()).get_class_object())));
}
template <> inline py::handle<> dtype<double>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyFloat_Type))); }
template <> inline py::handle<> dtype<dcomplex>() {
    return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyComplex_Type)));
}
// template<> inline py::handle<> dtype<Tensor2<double>>() { return
// py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyTuple_Type))); } template<> inline py::handle<>
// dtype<Tensor2<dcomplex>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyTuple_Type))); } template<> inline
// py::handle<> dtype<Tensor3<double>>() { return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyTuple_Type))); }
// template<> inline py::handle<> dtype<Tensor3<dcomplex>>() { return
// py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyTuple_Type))); }
}  // namespace detail

// ----------------------------------------------------------------------------------------------------------------------
// Geometry suffix

template <typename GeometryT> inline std::string format_geometry_suffix(const char*);
template <> inline std::string format_geometry_suffix<Geometry2DCartesian>(const char* fmt) { return format(fmt, "2D"); }
template <> inline std::string format_geometry_suffix<Geometry2DCylindrical>(const char* fmt) { return format(fmt, "Cyl"); }
template <> inline std::string format_geometry_suffix<Geometry3D>(const char* fmt) { return format(fmt, "3D"); }

// ----------------------------------------------------------------------------------------------------------------------
// Register vectors of something

template <typename T> struct VectorFromSequence {
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
        std::vector<T>* result = new (storage) std::vector<T>();
        result->reserve(py::len(seq));
        for (auto iter = begin; iter != end; ++iter) result->push_back(*iter);
        data->convertible = storage;
    }
};

template <typename T> static std::string str__vector_of(const std::vector<T>& self) {
    std::string result = "[";
    int i = int(self.size()) - 1;
    for (auto v : self) {
        result += py::extract<std::string>(py::object(v).attr("__repr__")());
        result += (i) ? ", " : "";
        --i;
    }
    return result + "]";
}

template <typename T>
static inline py::class_<std::vector<T>, shared_ptr<std::vector<T>>> register_vector_of(const std::string& name) {
    VectorFromSequence<T>();
    py::class_<std::vector<T>, shared_ptr<std::vector<T>>> cls((name + "_list").c_str(), py::no_init);
    cls.def(py::vector_indexing_suite<std::vector<T>>()).def("__repr__", &str__vector_of<T>).def("__str__", &str__vector_of<T>);
    py::scope scope;
    py::delattr(scope, py::str(name + "_list"));
    return cls;
}

// ----------------------------------------------------------------------------------------------------------------------
// Get space names
template <typename SpaceT> static inline std::string spaceName();
template <> inline std::string spaceName<Geometry2DCartesian>() { return "Cartesian2D"; }
template <> inline std::string spaceName<Geometry2DCylindrical>() { return "Cylindrical"; }
template <> inline std::string spaceName<Geometry3D>() { return "Cartesian3D"; }
template <> inline std::string spaceName<void>() { return ""; }

template <typename SpaceT> static inline std::string spaceSuffix();
template <> inline std::string spaceSuffix<Geometry2DCartesian>() { return "2D"; }
template <> inline std::string spaceSuffix<Geometry2DCylindrical>() { return "Cyl"; }
template <> inline std::string spaceSuffix<Geometry3D>() { return "3D"; }
template <> inline std::string spaceSuffix<void>() { return ""; }

// ----------------------------------------------------------------------------------------------------------------------
/// Class for setting logging configuration
struct LoggingConfig {
    py::object getLoggingColor() const;
    void setLoggingColor(std::string color);

    py::object getLoggingDest() const;
    void setLoggingDest(py::object dest);

    LogLevel getLogLevel() const { return maxLoglevel; }
    void setLogLevel(LogLevel level) {
        if (!forcedLoglevel) maxLoglevel = level;
    }

    void forceLogLevel(LogLevel level) { maxLoglevel = level; }

    std::string __str__() const;

    std::string __repr__() const;
};

/// Config class
struct Config {
    std::string axes_name() const;

    void set_axes(std::string axis);

    bool getUfuncIgnoreError() const;
    void setUfuncIgnoreError(bool value);

    std::string __str__() const;

    std::string __repr__() const;
};

extern PLASK_PYTHON_API AxisNames current_axes;

inline AxisNames* getCurrentAxes() { return &current_axes; }

// ----------------------------------------------------------------------------------------------------------------------
// Helpers for parsing kwargs

namespace detail {
template <size_t i> static inline void _parse_kwargs(py::list& PLASK_UNUSED(arglist), py::dict& PLASK_UNUSED(kwargs)) {}

template <size_t i, typename... Names>
static inline void _parse_kwargs(py::list& arglist, py::dict& kwargs, const std::string& name, const Names&... names) {
    py::object oname(name);
    if (kwargs.has_key(oname)) {
        if (i < std::size_t(py::len(arglist))) {
            throw name;
        } else {
            arglist.append(kwargs[oname]);
            py::delitem(kwargs, oname);
        }
    }
    _parse_kwargs<i + 1>(arglist, kwargs, names...);
}
}  // namespace detail

/// Helper for parsing arguments in raw_function
template <typename... Names>
static inline void parseKwargs(const std::string& fname, py::tuple& args, py::dict& kwargs, const Names&... names) {
    kwargs = kwargs.copy();
    py::list arglist(args);
    try {
        detail::_parse_kwargs<0>(arglist, kwargs, names...);
    } catch (const std::string& name) {
        throw TypeError(u8"{0}() got multiple values for keyword argument '{1}'", fname, name);
    }
    if (py::len(arglist) != sizeof...(names))
        throw TypeError(u8"{0}() takes exactly {1} non-keyword arguments ({2} given)", fname, sizeof...(names), py::len(arglist));
    args = py::tuple(arglist);
}

/// Convert Python dict to std::map
template <typename TK, typename TV> std::map<TK, TV> dict_to_map(PyObject* obj) {
    std::map<TK, TV> map;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj, &pos, &key, &value)) {
        map[py::extract<TK>(key)] = py::extract<TV>(value);
    }
    return map;
}

template <typename TK, typename TV> std::map<TK, TV> dict_to_map(const py::object& obj) { return dict_to_map<TK, TV>(obj.ptr()); }

// ----------------------------------------------------------------------------------------------------------------------
// Parallel locking

#ifdef OPENMP_FOUND

struct OmpPythonLockState : OmpLockState {
    PyGILState_STATE gil_state;
    OmpPythonLockState(const PyGILState_STATE& state) : gil_state(state) {}
};

class OmpPythonLock : public OmpLock {
    OmpLockState* lock() override { return new OmpPythonLockState(PyGILState_Ensure()); }

    void unlock(OmpLockState* state) override {
        OmpPythonLockState* python_state = static_cast<OmpPythonLockState*>(state);
        PyGILState_Release(python_state->gil_state);
        delete python_state;
    }
};

#else

#    define OmpPythonLock OmpLock

#endif

extern PLASK_PYTHON_API OmpPythonLock python_omp_lock;

// ----------------------------------------------------------------------------------------------------------------------
// Helper for XML reads

/// Evaluate common Python types
inline static py::object eval_common_type(const std::string& value) {
    if (value == "" || value == "None") return py::object();
    if (value == "yes" || value == "true" || value == "True") return py::object(true);
    if (value == "no" || value == "false" || value == "False") return py::object(false);
    try {
        py::object val = py::eval(value.c_str());
        if (PyLong_Check(val.ptr()) || PyFloat_Check(val.ptr()) || PyComplex_Check(val.ptr()) || PyTuple_Check(val.ptr()) ||
            PyList_Check(val.ptr()))
            return val;
        else
            return py::str(value);
    } catch (py::error_already_set&) {
        PyErr_Clear();
        return py::str(value);
    }
}

/**
 * Remove indentation of the Python part (based on the indentation of the first line)
 * \param text text to fix
 * \param xmlline line number for error messages
 * \param tag tag name for error messages
 * \params indent if not 0, the indentation is set to this value
 */
std::string dedent(const std::string& text, unsigned xmlline, const char* tag = nullptr, unsigned indent = 0);

struct PythonManager;

/**
 * Read Python code from reader (either for eval or exec).
 * \param reader XML reader
 * \param manager XPL manager
 * \param args for the returned function
 * \param globals globals for eval
 * \return compiled PyCodeObject
 */
PyObject* compilePythonFromXml(XMLReader& reader, Manager& manager, const char* args, const py::dict& globals);

}}  // namespace plask::python

#endif  // PLASK__PYTHON_GLOBALS_H
