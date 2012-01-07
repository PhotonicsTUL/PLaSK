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

    // Dimenstions of the hold types
    static const int dimensions[];

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

    // Format complex numbers in Python way
    struct sc {
        dcomplex v;
        sc(dcomplex c) : v(c) {}
        friend inline std::ostream& operator<<(std::ostream& out, const sc& c) {
            double r = c.v.real(), i = c.v.imag();
            out << "(" << r << ((i>=0)?"+":"") << i << "j)";
            return out;
        }
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
        bool force_complex = false, force_double = false;

        if (py::object dtype = kwargs.get("dtype")) {
            py::delitem(kwargs, py::object("dtype"));
             if (dtype.ptr() == (PyObject*)&PyFloat_Type) force_double = true;
             else if (dtype.ptr() == (PyObject*)&PyComplex_Type) force_complex = true;
        }

        py::tuple params;

        if (n == 0) { // Extract components from kwargs

            //TODO
            params = py::make_tuple(1,2,3);

        } else if (kwargs) {
            PyErr_SetString(PyExc_TypeError, "components must be provided entirely in a list or by names");
            throw py::error_already_set();
        } else {
            if (n != 2 && n != 3) {
                PyErr_SetString(PyExc_TypeError, "wrong number of arguments");
                throw py::error_already_set();
            }
            params = args;
        }

        if (!force_complex) {
            try {
                double cmp[3];
                for (int i = 0; i < n; ++i) cmp[i] = py::extract<double>(params[i]);
                if (n == 2) return shared_ptr<PyVec>( new PyVec(Vec<2,double>(cmp[0], cmp[1])) );
                else return shared_ptr<PyVec>( new PyVec(Vec<3,double>(cmp[0], cmp[1], cmp[2])) );
            } catch (py::error_already_set) {
                PyErr_Clear();
            }
        }

        if (!force_double) {
            try {
                dcomplex cmp[3];
                for (int i = 0; i < n; ++i) cmp[i] = py::extract<dcomplex>(params[i]);
                if (n == 2) return shared_ptr<PyVec>( new PyVec(Vec<2,dcomplex>(cmp[0], cmp[1])) );
                else return shared_ptr<PyVec>( new PyVec(Vec<3,dcomplex>(cmp[0], cmp[1], cmp[2])) );
            } catch (py::error_already_set) {
                PyErr_Clear();
                PyErr_SetString(PyExc_TypeError, "at least one of the vector arguments of invalid type");
                throw py::error_already_set();
            }
        }

        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "at least one of the vector arguments of invalid type");
        throw py::error_already_set();
        return shared_ptr<PyVec>();
    }

    // Operations

//     __getitem__
//     __setitem__

    std::string __str__() {
        std::stringstream out;
        switch (type) {
            case DOUBLE_2D: out << "[" << ptr2d->c0 << ", " << ptr2d->c1 << "]" ; break;
            case DOUBLE_3D: out << "[" << ptr3d->c0 << ", " << ptr3d->c1 << ", " << ptr3d->c2 << "]";; break;
            case COMPLEX_2D: out << "[" << sc(ptr2c->c0) << ", " << sc(ptr2c->c1) << "]"; break;
            case COMPLEX_3D: out << "[" << sc(ptr3c->c0) << ", " << sc(ptr3c->c1) << ", " << sc(ptr3c->c2) << "]"; break;
        }
        return out.str();
    }

    std::string __repr__() {
        std::stringstream out;
        switch (type) {
            case DOUBLE_2D: out << "vec(" << ptr2d->c0 << ", " << ptr2d->c1 << ")" ; break;
            case DOUBLE_3D: out << "vec(" << ptr3d->c0 << ", " << ptr3d->c1 << ", " << ptr3d->c2 << ")";; break;
            case COMPLEX_2D: out << "vec(" << sc(ptr2c->c0) << ", " << sc(ptr2c->c1) << ")"; break;
            case COMPLEX_3D: out << "vec(" << sc(ptr3c->c0) << ", " << sc(ptr3c->c1) << ", " << sc(ptr3c->c2) << ")"; break;
        }
        return out.str();
    }

    double abs() {
        switch (type) {
            case DOUBLE_2D: return plask::abs(*ptr2d);
            case DOUBLE_3D: return plask::abs(*ptr3d);
            case COMPLEX_2D: return plask::abs(*ptr2c);
            case COMPLEX_3D: return plask::abs(*ptr3c);
        }
        return 0;
     }

    double abs2() {
        switch (type) {
            case DOUBLE_2D: return plask::abs2(*ptr2d);
            case DOUBLE_3D: return plask::abs2(*ptr3d);
            case COMPLEX_2D: return plask::abs2(*ptr2c);
            case COMPLEX_3D: return plask::abs2(*ptr3c);
        }
        return 0;
     }

     PyVec __neg__() {
     }

    int __len__() { return dimensions[(int)type]; }

    bool __nonzero__() {
        switch (type) {
            case DOUBLE_2D: return !( is_zero(ptr2d->c0) && is_zero(ptr2d->c1) );
            case DOUBLE_3D: return !( is_zero(ptr3d->c0) && is_zero(ptr3d->c1) && is_zero(ptr3d->c2) );
            case COMPLEX_2D: return !( is_zero(ptr2c->c0) && is_zero(ptr2c->c1) );
            case COMPLEX_3D: return !( is_zero(ptr3c->c0) && is_zero(ptr3c->c1) && is_zero(ptr3c->c2) );
        }
        return false;
    }

    bool __eq__(const PyVec& other) {
        if (dimensions[(int)type] != dimensions[(int)other.type]) return false;
        switch (other.type) {
            case DOUBLE_2D: return type == DOUBLE_2D ? *ptr2d == *other.ptr2d : *ptr2c == *other.ptr2d;
            case DOUBLE_3D: return type == DOUBLE_3D ? *ptr3d == *other.ptr3d : *ptr3c == *other.ptr3d;
            case COMPLEX_2D: return type == COMPLEX_2D ? *ptr2c == *other.ptr2c : *ptr2d == *other.ptr2c;
            case COMPLEX_3D: return type == COMPLEX_3D ? *ptr3c == *other.ptr3c : *ptr3d == *other.ptr3c;
        }
        return false;
    }

    bool __ne__(const PyVec& other) {
        return ! __eq__(other);
    }

//     __add__
//     __sub__
//
//     __mul__
//     __rmul__
//
//     abs2
//     copy
//     dot
//     conj

    py::handle<> dtype() {
        PyObject* obj;
        if (type == COMPLEX_2D || type == COMPLEX_3D)
            obj = reinterpret_cast<PyObject*>(&PyComplex_Type);
        else
            obj = reinterpret_cast<PyObject*>(&PyFloat_Type);
        return py::handle<>(py::borrowed<>(obj));
    }


//    __iter__
};
const std::string PyVec::names[] = { "2D float", "3D float", "2D complex", "3D complex" };
const int PyVec::dimensions[] = { 2, 3, 2, 3 };

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
//     py::scope vec_scope = vec_class;
//
//     py::class_<Vec_iterator<dim,T>>("_Iterator", py::no_init)
//         .def("__iter__", &Vec_iterator<dim,T>::__iter__, py::return_self<>())
//         .def("next", &Vec_iterator<dim,T>::next)
//     ;
//
//     return vec_class;
// }

template<int dim, typename T>
struct VecConverter {
    static PyObject* convert(const Vec<dim,T>& vec) {
        shared_ptr<PyVec> pyvec { new PyVec(vec) };
        return py::incref<>(py::object(pyvec).ptr());
    }
};


void register_vector()
{
    py::class_<PyVec, shared_ptr<PyVec>>("vec",
        "General vector class in PLaSK. It can inteligently hold both 2D and 3D vectors.\n\n"
        "vec(#, #[, #]) -> initialize with ordered components\n"
        "vec(x=#, y=#, z=#) -> initialize with Cartesian components (z or x skipped for 2D)\n"
        "vec(r=#, phi=#, z=#) -> initialize with cylindrical components (phi skipped for 2D)\n"
        "The order of its components always corresponds to the structure orientation\n"
        "(with the last component parallel to the epitaxial growth direction.\n\n"
        "However, the component names depend on the config.axis_up configuration option.\n"
        "Changing this option will change the order of component names accordingly:\n\n"
        "For config.vertical_axis = 'z', the component name mapping is following:\n"
        "in 2D vectors it is [y,z] (x-component skipped) or [r,z] (you can refer to\n"
        "Cartesian or cylindrical cooridinates at your preference). In 3D vectors it is\n"
        "[x,y,z] or [r,phi,z].\n\n"
        "For config.vertical_axis = 'z' the order becomes: [x,y] and [z,x,y] for 2D and 3D vectors, "
        "respectively. In this case, cylindrical component names are not allowed.", py::no_init)
        .def("__init__", raw_constructor(&PyVec::__init__, 0))
        .def("__str__", &PyVec::__str__)
        .def("__repr__", &PyVec::__repr__)
        .def("__abs__", &PyVec::abs)
        .def("abs", &PyVec::abs)
        .def("abs2", &PyVec::abs2)
        .def("__len__", &PyVec::__len__)
        .def("__nonzero__", &PyVec::__nonzero__)
        .def("__eq__", &PyVec::__eq__)
        .def("__ne__", &PyVec::__ne__)
        .add_property("dtype", &PyVec::dtype)
    ;

    py::implicitly_convertible<PyVec,Vec<2,double>>();
    py::implicitly_convertible<PyVec,Vec<3,double>>();
    py::implicitly_convertible<PyVec,Vec<2,dcomplex>>();
    py::implicitly_convertible<PyVec,Vec<3,dcomplex>>();

    py::to_python_converter<Vec<2,double>, VecConverter<2,double>>();
    py::to_python_converter<Vec<3,double>, VecConverter<3,double>>();
    py::to_python_converter<Vec<2,dcomplex>, VecConverter<2,dcomplex>>();
    py::to_python_converter<Vec<3,dcomplex>, VecConverter<3,dcomplex>>();
}

}} // namespace plask::python
