#include <sstream>
#include <iostream>

#include <plask/vec.h>
#include <config.h>

#include "globals.h"
#include "../util/raw_constructor.h"


namespace plask { namespace python {

class PyVec
{
  private:

    // Names of the hold types
    static const std::string names[];

    // Type of hold value
    enum VectorType { DOUBLE_2D = 0, DOUBLE_3D = 1, COMPLEX_2D = 2, COMPLEX_3D = 3 };
    const VectorType type;

    // Pointer to the hold value
    union {
        Vec<2,double>* ptr2d;
        Vec<3,double>* ptr3d;
        Vec<2,dcomplex>* ptr2c;
        Vec<3,dcomplex>* ptr3c;
    };

  public:

    // Constructors
    PyVec(const Vec<2,double>& vec) : type(DOUBLE_2D), ptr2d(new Vec<2,double>(vec)) {}
    PyVec(const Vec<3,double>& vec) : type(DOUBLE_3D), ptr3d(new Vec<3,double>(vec)) {}
    PyVec(const Vec<2,dcomplex>& vec) : type(COMPLEX_2D), ptr2c(new Vec<2,dcomplex>(vec)) {}
    PyVec(const Vec<3,dcomplex>& vec) : type(COMPLEX_3D), ptr3c(new Vec<3,dcomplex>(vec)) {}

    PyVec(const PyVec& src) : type(src.type) {
        switch (type) {
            case DOUBLE_2D: ptr2d = new Vec<2,double>(*src.ptr2d); break;
            case DOUBLE_3D: ptr3d = new Vec<3,double>(*src.ptr3d); break;
            case COMPLEX_2D: ptr2c = new Vec<2,dcomplex>(*src.ptr2c); break;
            case COMPLEX_3D: ptr3c = new Vec<3,dcomplex>(*src.ptr3c); break;
        }
    }

    // Destructor
    ~PyVec() {
        if (ptr2d != nullptr) { switch (type) {
                case DOUBLE_2D: delete ptr2d; break;
                case DOUBLE_3D: delete ptr3d; break;
                case COMPLEX_2D: delete ptr2c; break;
                case COMPLEX_3D: delete ptr3c; break;
        } }
    }

    // Converters

    operator Vec<2,double> () {
        if (type != DOUBLE_2D) {
            PyErr_SetString(PyExc_TypeError, ("can't convert " + names[(size_t)type] + " vector to 2D float").c_str());
            throw py::error_already_set();
        }
        return *ptr2d;
    }

    operator Vec<3,double> () {
        if (type != DOUBLE_3D) {
            PyErr_SetString(PyExc_TypeError, ("can't convert " + names[(size_t)type] + " vector to 3D float").c_str());
            throw py::error_already_set();
        }
        return *ptr3d;
    }

    operator Vec<2,dcomplex> () {
        if (type != COMPLEX_2D && type != DOUBLE_2D) {
            PyErr_SetString(PyExc_TypeError, ("can't convert "+ names[(size_t)type] + " vector to 2D complex").c_str());
            throw py::error_already_set();
        }
        return type == DOUBLE_2D? Vec<2,dcomplex>(*ptr2d) : (*ptr2c);
    }

    operator Vec<3,dcomplex> () {
        if (type != COMPLEX_3D && type != DOUBLE_3D) {
            PyErr_SetString(PyExc_TypeError, ("can't convert "+ names[(size_t)type] + " vector to 3D complex").c_str());
            throw py::error_already_set();
        }
        return type == DOUBLE_3D? Vec<3,dcomplex>(*ptr3d) : (*ptr3c);
    }


    // Python constructor
    static shared_ptr<PyVec> __init__(py::tuple args, py::dict kwargs)
    {
        int n = py::len(args);
        if (n != 2 && n != 3 & n != 0) {
            PyErr_SetString(PyExc_TypeError, "wrong number of arguments");
            throw py::error_already_set();
        }

        bool is_complex = false;
        shared_ptr<PyVec> pyvec;

        if (n == 0) { // Extract components from kwargs
        } else if (kwargs) {
            PyErr_SetString(PyExc_TypeError, "components must be provided entirely in a list or by names");
            throw py::error_already_set();
        } else {
        }

        PyErr_SetString(PyExc_RuntimeError, "not implemented");
        throw py::error_already_set();

        return pyvec;
    }

    // Operations

//     __getitem__
//     __setitem__
//     __str__
//     __repr__
//     __abs__
//     __neg__
//     __nonzero__ // compare abs with small
//     __getslice__
//     __setslice__
//     __iter__
//
//     __eq__
//     __ne__
//     __add__
//     __sub__
//     __iadd__
//     __isub__
//
//     __mul__
//     __rmul__
//
//     abs2
//     copy
//     dot
//     conj




};
const std::string PyVec::names[] = { "2D float", "3D float", "2D complex", "3D complex" };

// // v = vector[i]
// template <int dim, typename T>
// static T __getitem__(Vec<dim,T>& self, int i) {
//     if (i < 0) i = dim + i;
//     if (i >= dim || i < 0) {
//         const char message[] = "vector index out of range";
//         PyErr_SetString(PyExc_IndexError, message);
//         throw py::error_already_set();
//     }
//     return self[i];
// }
//
// // vector[i] = v
// template <int dim, typename T>
// static T __setitem__(Vec<dim,T>& self, int i, T v) {
//     if (i < 0) i = dim + i;
//     if (i >= dim || i < 0) {
//         const char message[] = "vector index out of range";
//         PyErr_SetString(PyExc_IndexError, message);
//         throw py::error_already_set();
//     }
//     self.component[i] = v;
// }
//
// // __str__(v)
// template <int dim, typename T>
// static std::string __str__(const Vec<dim,T>& to_print) {
//     std::stringstream out;
//     out << to_print;
//     return out.str();
// }
//
// // __repr__(v)
// template <int dim, typename T>
// static std::string __repr__(const Vec<dim,T>& to_print) {
//     std::stringstream out;
//     out << "vector(" << to_print << ")";
//     return out.str();
// }
//
// // v.__iter__
// template <int dim, typename T>
// struct Vec_iterator
// {
//     Vec<dim,T>* vec;
//     int i;
//
//     static Vec_iterator<dim, T> new_iterator(Vec<dim,T>& v) {
//         return Vec_iterator<dim, T>(v);
//     }
//
//     Vec_iterator(Vec<dim,T>& v) : vec(&v), i(0) {}
//
//     Vec_iterator<dim, T>* __iter__() { return this; }
//
//     T next() {
//         if (i >= dim) {
//             PyErr_SetString(PyExc_StopIteration, "No more components.");
//             boost::python::throw_error_already_set();
//         }
//         return vec->components[i++];
//     }
// };
//
// // copy v
// template <int dim, typename T>
// static Vec<dim,T> copy_vec(const Vec<dim,T>& v) {
//     return v;
// }
//
// // Register vector class to python
// template <int dim, typename T, typename Init>
// inline static py::class_<Vec<dim,T>>  py_vector_class_common(const char* name, const char* doc)
// {
//     typedef Vec<dim,T> V;
//     typedef Vec<dim,double> VR;
//     typedef Vec<dim,dcomplex> VC;
//
//     T (*dr)(const V&, const VR&) = &dot<T,double>;
//     dcomplex (*dc)(const V&, const VC&) = &dot<T,dcomplex>;
//
//     V (*c)(const V&) = &plask::conj<T>;
//
//     py::class_<V> vec_class = py::class_<V>(name, doc, Init())
//         .def_readwrite("x", &V::x)
//         .def_readwrite("y", &V::y)
//         .def_readwrite("r", &V::r)
//         .def_readwrite("z", &V::z)
//         .def("__getitem__", __getitem__<dim,T>)
//         .def("__setitem__", __getitem__<dim,T>)
//         .def("__iter__", &Vec_iterator<dim,T>::new_iterator, py::with_custodian_and_ward_postcall<0,1>())
//         .def(py::self == py::other<VR>())
//         .def(py::self == py::other<VC>())
//         .def(py::self != py::other<VR>())
//         .def(py::self != py::other<VC>())
//         .def("abs2", &abs2<dim,V>, "Squared vector magnitude")
//         .def("copy", &copy_vec<dim,T>)
//         .def("__abs__", &V::magnitude, "Vector magnitue")
//         .def(py::self + py::other<VR>())
//         .def(py::self + py::other<VC>())
//         .def(py::self - py::other<VR>())
//         .def(py::self - py::other<VC>())
//         .def( - py::self)
//         .def(py::self * T())
//         .def(T() * py::self)
//         .def(py::self += py::other<V>())
//         .def(py::self -= py::other<V>())
//         .def(py::self *= T())
//         .def("__str__", __str__<dim,T>)
//         .def("__repr__", __repr__<dim,T>)
//         .def("dot", dr, "Dot product with another vector a.conj(b)")
//         .def("dot", dc, "Dot product with another vector a.conj(b)")
//         .def("__mul__", dr)
//         .def("__mul__", dc)
//         .def("conjugate", c)
//         .def("conj", c)
//     ;
//
//     py::scope vec_scope = vec_class;
//
//     py::class_<Vec_iterator<dim,T>>("_Iterator", py::no_init)
//         .def("__iter__", &Vec_iterator<dim,T>::__iter__, py::return_self<>())
//         .def("next", &Vec_iterator<dim,T>::next)
//     ;
//
//     return vec_class;
// }

void register_vector()
{

    py::class_<PyVec, shared_ptr<PyVec>> pyvec("vector",
        "General vector class in PLaSK. It can inteligently hold both 2D and 3D vectors. "
        "The order of its componets always corresponds to the structure orientation (with the last component parallel to the epitaxial growth direction. "
        "However, the component names depend on the config.axis_up configuration option. Changing this option will change the component names accordingly. "
        "For config.axis_up = 'z', the component_name -> component_number mapping is following: (y,z)->(0,1) or (r,z)->(0,1) for 2D vectors and "
        "(x,y,z)->(0,1,2), (r,phi,z)->(0,1,2) for 3D ones. For config.axis_up = 'z' it becomes: (x,y)->(0,1) and (x,y,z)->(1,2,0), respectively. "
        "In the latter case, cylindrical component names (r,z) and (r,phi,z) are not allowed.", py::no_init);
    pyvec
        .def("__init__", raw_constructor(&PyVec::__init__, 0))
    ;

    py::scope().attr("vec") = pyvec;

    py::implicitly_convertible<PyVec,Vec<2,double>>();
    py::implicitly_convertible<Vec<2,double>,PyVec>();

    py::implicitly_convertible<PyVec,Vec<3,double>>();
    py::implicitly_convertible<Vec<3,double>,PyVec>();

    py::implicitly_convertible<PyVec,Vec<2,dcomplex>>();
    py::implicitly_convertible<Vec<2,dcomplex>,PyVec>();

    py::implicitly_convertible<PyVec,Vec<3,dcomplex>>();
    py::implicitly_convertible<Vec<3,dcomplex>,PyVec>();
}

}} // namespace plask::python
