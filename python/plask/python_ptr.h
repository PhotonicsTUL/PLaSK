#ifndef PLASK__PYTHON_PTR_H
#define PLASK__PYTHON_PTR_H

#include <Python.h>

namespace plask { namespace python {

/**
 * Lite holder of pointer to python object. It calls Py_XDECREF in destructor and Py_XINCREF in copy constructor.
 *
 * Designed to work with raw python C API.
 *
 * It is similar to boost::python::handle but has less limitation, e.g. can work with numpy handlers,
 * and behave more similar to raw PyObject* (it is auto-castable to this type).
 *
 * @tparam T type to which pointer should be hold, PyObject by default
 */
template <typename T = PyObject>
class PyHandle {

    /// Held pointer.
    T* ptr;

public:

    /**
     * Construct wrapper over @p ptr. Do not call Py_XINCREF.
     * @param ptr pointer to wrap
     */
    PyHandle(T* ptr = nullptr): ptr(ptr) {}

    /**
     * Copy constructor, call Py_XINCREF.
     * @param o object to copy
     */
    PyHandle(const PyHandle<T>& o) { this->ptr = o.ptr; Py_XINCREF(this->ptr); }

    /// Call Py_XDECREF.
    ~PyHandle() { Py_XDECREF(ptr); }

    /**
     * Copy operator, call Py_XDECREF(this->ptr) and Py_XINCREF(o->ptr) if @p o holds different pointer than this holds.
     * @param o object to assign
     */
    PyHandle<T>& operator=(const PyHandle<T>& o) {
        if (ptr == o.ptr) return *this;
        Py_XDECREF(ptr);
        ptr = o.ptr;
        Py_XINCREF(ptr);
        return *this;
    }

    /**
     * Get reference to the held pointer.
     *
     * Generally it can can be unsafe to change it if it is not nullptr (see reset).
     * @return reference to holded pointer
     */
    T*& ref() { return ptr; }

    /**
     * Get const reference to the held pointer.
     * @return const reference to the held pointer
     */
    const T*& ref() const { return ptr; }

    /**
     * Cast this to the held pointer.
     */
    operator T*() const { return ptr; }

    /**
     * Get field or method of the held object.
     * @return field or method of the held object
     */
    T* operator->() const { return ptr; }

    /**
     * Get reference to the held object.
     * @return reference to the held object
     */
    T& operator*() { return *ptr; }

    /**
     * Get const reference to the held object.
     * @return const reference to the held object
     */
    const T& operator*() const { return *ptr; }

    /**
     * Cast the pointer to some other type.
     * \return casted pointer
     */
    template <typename C>
    C* ptr_cast() const { return reinterpret_cast<C*>(ptr); }

    /**
     * Stop managing the current pointer. Set self to nullptr.
     * @return current pointer
     */
    T* release() { T* r = ptr; ptr = nullptr; return r; }

    /**
     * Release the current pointer and set it to new one.
     * @param new_ptr new pointer to hold
     */
    void reset(T* new_ptr = nullptr) { Py_XDECREF(ptr); ptr = new_ptr; }
};

} }

#endif // PLASK__PYTHON_PTR_H
