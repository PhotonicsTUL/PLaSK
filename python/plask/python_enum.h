#ifndef PLASK__PYTHON_ENUM_H
#define PLASK__PYTHON_ENUM_H

#include <boost/python.hpp>
#include <boost/algorithm/string.hpp>

namespace plask { namespace python {

template <class T>
struct py_enum
{
    typedef boost::python::objects::enum_base base;

    // Declare a new enumeration type in the current scope()
    py_enum();

    // Add a new enumeration value with the given name and value.
    inline py_enum<T>& value(char const* name, T);

    // Add all of the defined enumeration values to the current scope with the
    // same names used here.
    inline py_enum<T>& export_values();

  private:
    static std::map<std::string,T>& names();
    static void* convertible(PyObject* obj);
    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data);

  public:
    static PyObject* convert(T const& x);
};

template <class T>
inline py_enum<T>::py_enum() {
    boost::python::to_python_converter<T, py_enum<T>>();
    boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<T>());
}

template <typename T>
std::map<std::string,T>& py_enum<T>::names() {
    static std::map<std::string,T> value;
    return value;
}

// This is the conversion function that gets registered for converting
// these enums to Python.
template <class T>
PyObject* py_enum<T>::convert(T const& x)
{
    for(auto item: names()) {
        if (item.second == x) {
            boost::python::object value(item.first);
            return boost::python::incref(value.ptr());
        }
    }
    PyErr_SetString(PyExc_ValueError, "wrong enumeration value");
    throw boost::python::error_already_set();
    return nullptr;
}

//
// The following two static functions serve as the objects of an
// rvalue from_python converter for the enumeration type.
//

// This checks that a given Python object can be converted to the
// enumeration type.
template <class T>
void* py_enum<T>::convertible(PyObject* obj)
{
    return
#       if PY_VERSION_HEX >= 0x03000000
        (PyUnicode_Check(obj))
#       else
        (PyString_Check(obj))
#       endif
        ? obj : 0;
}

// Constructs an instance of the enumeration type in the from_python
// data.
template <class T>
void py_enum<T>::construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
{
    std::string key = boost::python::extract<std::string>(obj);
    boost::algorithm::to_upper(key);
    boost::algorithm::replace_all(key, " ", "_");
    boost::algorithm::replace_all(key, "-", "_");

    auto item = names().find(key);
    if (item == names().end()) {
        std::string msg = "Bad parameter value '" + key + "'";
        PyErr_SetString(PyExc_ValueError, msg.c_str());
        throw boost::python::error_already_set();
    }

    void* const storage = ((boost::python::converter::rvalue_from_python_storage<T>*)data)->storage.bytes;
    new (storage) T(item->second);
    data->convertible = storage;
}

template <class T>
inline py_enum<T>& py_enum<T>::value(char const* name, T x)
{
    names()[name] = x;
    return *this;
}


}} // namespace plask::python

#endif // PLASK__PYTHON_ENUM_H
