#ifndef PLASK__PYTHON_VECTOR_H
#define PLASK__PYTHON_VECTOR_H

#include <sstream>

#include <boost/python.hpp>
namespace py = boost::python;

#include <config.h>
#include <plask/vector/2d.h>
#include <plask/vector/3d.h>

namespace plask { namespace python {

template <int dim, typename T> struct Vec {};
template <typename T> struct Vec<2,T> { typedef Vec2<T> type; typedef py::init<T,T> init; };
template <typename T> struct Vec<3,T> { typedef Vec3<T> type; typedef py::init<T,T,T> init; };

// v = vector[i]
template <int dim, typename T>
static T __getitem__(typename Vec<dim,T>::type& self, int i) {
    if (i < 0) i = dim - i;
    if (i >= dim) {
        const char message[] = "vector index out of range";
        PyErr_SetString(PyExc_IndexError, message);
        throw py::error_already_set();
    }
    return self.coordinate[i];
}

// vector[i] = v
template <int dim, typename T>
static T __setitem__(typename Vec<dim,T>::type& self, int i, T v) {
    if (i < 0) i = dim - i;
    if (i >= dim) {
        const char message[] = "vector index out of range";
        PyErr_SetString(PyExc_IndexError, message);
        throw py::error_already_set();
    }
    self.coordinate[i] = v;
}

// for __str__(v)
template <int dim, typename T>
static std::string __str__(const typename Vec<dim,T>::type & to_print) {
    std::stringstream out;
    out << to_print;
    return out.str();
}

// for __repr__(v)
template <int dim, typename T>
static std::string __repr__(const typename Vec<dim,T>::type & to_print) {
    std::stringstream out;
    out << "vector(" << to_print << ")";
    return out.str();
}

// Register vector class to python
template <int dim, typename T>
inline static void py_vector_class_common(const char* name, const char* doc)
{
    typedef typename Vec<dim,T>::type V;
    typedef typename Vec<dim,double>::type VR;
    typedef typename Vec<dim,complex<double>>::type VC;

    T (*d)(const V&, const V&) = &dot<T>;

    py::class_<V>(name, doc, typename Vec<dim,T>::init())
        .def_readwrite("x", &V::x)
        .def_readwrite("y", &V::y)
        .def_readwrite("r", &V::r)
        .def_readwrite("z", &V::z)
        .def("__getitem__", __getitem__<dim, T>)
        .def("__setitem__", __getitem__<dim, T>)
        .def(py::self == py::other<VR>())
        .def(py::self == py::other<VC>())
        .def(py::self != py::other<VR>())
        .def(py::self != py::other<VC>())
        .def("magnitude", &V::magnitude, "Vector magnitue")
        .def("magnitude2", &V::magnitude2, "Squared vector magnitude (little faster than self.magnitude)")
        .def(py::self + py::other<VR>())
        .def(py::self + py::other<VC>())
        .def(py::self - py::other<VR>())
        .def(py::self - py::other<VC>())
        .def( - py::self)
        .def(py::self * T())
        .def(T() * py::self)
        .def(py::self += py::other<V>())
        .def(py::self -= py::other<V>())
        .def(py::self *= T())
        .def("__str__", __str__<dim, T>)
        .def("__repr__", __repr__<dim, T>)
        .def("dot", d, "Dot product with another vector");
    ;
}


inline static void register_vector_h()
{
    py_vector_class_common<2,double>("vector2d_float",
        "Float vector in 2D space. Create new instance as v = vector(x,y, float).");

    py_vector_class_common<2,complex<double>>("vector2d_complex",
        "Complex vector in 2D space. Create new instance as v = vector(x,y, float).");

    py_vector_class_common<3,double>("vector3d_float",
        "Float vector in 3D space. Create new instance as v = vector(x,y, float).");

    py_vector_class_common<3,complex<double>>("vector3d_complex",
        "Complex vector in 3D space. Create new instance as v = vector(x,y, float).");

}

}} // namespace plask::python
#endif // PLASK__PYTHON_VECTOR_H
