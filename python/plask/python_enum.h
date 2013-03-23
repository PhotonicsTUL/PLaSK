#ifndef PLASK__PYTHON_ENUM_H
#define PLASK__PYTHON_ENUM_H

# include <boost/python/detail/prefix.hpp>

# include <boost/python/object/enum_base.hpp>
# include <boost/python/converter/rvalue_from_python_data.hpp>
# include <boost/python/converter/registered.hpp>
#include <boost/algorithm/string.hpp>

namespace plask { namespace python {

template <class T>
struct py_enum : public boost::python::objects::enum_base
{
    typedef boost::python::objects::enum_base base;

    // Declare a new enumeration type in the current scope()
    py_enum(char const* name, char const* doc = 0);

    // Add a new enumeration value with the given name and value.
    inline py_enum<T>& value(char const* name, T);

    // Add all of the defined enumeration values to the current scope with the
    // same names used here.
    inline py_enum<T>& export_values();

  private:
    static boost::python::object& names_dict();
    static PyObject* to_python(void const* x);
    static void* convertible_from_python(PyObject* obj);
    static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data);
};

template <class T>
inline py_enum<T>::py_enum(char const* name, char const* doc)
    : base(
        name
        , &py_enum<T>::to_python
        , &py_enum<T>::convertible_from_python
        , &py_enum<T>::construct
        , boost::python::type_id<T>()
        , doc
        )
{
    names_dict() = this->attr("names");
}

template <typename T>
boost::python::object& py_enum<T>::names_dict() {
    static boost::python::object value;
    return value;
}

// This is the conversion function that gets registered for converting
// these enums to Python.
template <class T>
PyObject* py_enum<T>::to_python(void const* x)
{
    return base::to_python(
        boost::python::converter::registered<T>::converters.m_class_object, static_cast<long>(*(T const*)x));
}

//
// The following two static functions serve as the objects of an
// rvalue from_python converter for the enumeration type.
//

// This checks that a given Python object can be converted to the
// enumeration type.
template <class T>
void* py_enum<T>::convertible_from_python(PyObject* obj)
{
    return PyObject_IsInstance(obj, boost::python::upcast<PyObject>(boost::python::converter::registered<T>::converters.m_class_object)) ||
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
#if PY_VERSION_HEX >= 0x03000000
    if (PyUnicode_Check(obj)) {
#else
    if (PyString_Check(obj)) {
#endif
        std::string key = boost::python::extract<std::string>(obj);
        boost::algorithm::to_upper(key);
        obj = boost::python::object(names_dict()[key]).ptr();
    }
#if PY_VERSION_HEX >= 0x03000000
    T x = static_cast<T>(PyLong_AS_LONG(obj));
#else
    T x = static_cast<T>(PyInt_AS_LONG(obj));
#endif
    void* const storage = ((boost::python::converter::rvalue_from_python_storage<T>*)data)->storage.bytes;
    new (storage) T(x);
    data->convertible = storage;
}

template <class T>
inline py_enum<T>& py_enum<T>::value(char const* name, T x)
{
    this->add_value(name, static_cast<long>(x));
    return *this;
}

template <class T>
inline py_enum<T>& py_enum<T>::export_values()
{
    this->base::export_values();
    return *this;
}


}} // namespace plask::python

#endif // PLASK__PYTHON_ENUM_H
