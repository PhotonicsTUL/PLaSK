#include <sstream>

#include <plask/vec.h>
#include <config.h>

#include "globals.h"
#include "../util/raw_constructor.h"

#include <boost/python/stl_iterator.hpp>


namespace plask { namespace python {

// v = vector[i]
template <int dim, typename T>
static T vec__getitem__(Vec<dim,T>& self, int i) {
    if (i < 0) i = dim + i;
    if (i >= dim || i < 0) {
        const char message[] = "vector index out of range";
        PyErr_SetString(PyExc_IndexError, message);
        throw py::error_already_set();
    }
    return self[i];
}

// vector[i] = v
template <int dim, typename T>
static T vec__setitem__(Vec<dim,T>& self, int i, T v) {
    if (i < 0) i = dim + i;
    if (i >= dim || i < 0) {
        const char message[] = "vector index out of range";
        PyErr_SetString(PyExc_IndexError, message);
        throw py::error_already_set();
    }
    self.component[i] = v;
}

// __str__(v)
template <int dim, typename T>
static std::string vec__str__(const Vec<dim,T>& to_print) {
    std::stringstream out;
    out << "[";
    for (int i = 0; i < dim; ++i) out << sc(to_print.components[i]) << (i!=dim-1 ? ", " : "]");
    return out.str();
}

// __repr__(v)
template <int dim, typename T>
static std::string vec__repr__(const Vec<dim,T>& to_print) {
    std::stringstream out;
    out << "vector(";
    for (int i = 0; i < dim; ++i) out << sc(to_print.components[i]) << (i!=dim-1 ? ", " : ")");
    return out.str();
}

// v.__iter__
template <int dim, typename T>
struct Vec_iterator
{
    Vec<dim,T>* vec;
    int i;

    static Vec_iterator<dim, T> new_iterator(Vec<dim,T>& v) {
        return Vec_iterator<dim, T>(v);
    }

    Vec_iterator(Vec<dim,T>& v) : vec(&v), i(0) {}

    Vec_iterator<dim, T>* __iter__() { return this; }

    T next() {
        if (i >= dim) {
            PyErr_SetString(PyExc_StopIteration, "No more components.");
            boost::python::throw_error_already_set();
        }
        return vec->components[i++];
    }
};

// copy v
template <int dim, typename T>
static Vec<dim,T> copy_vec(const Vec<dim,T>& v) {
    return v;
}

// Multiplication
template <int dim, typename T, typename OtherT, typename ResultT>
static Vec<dim,ResultT> vec__mul__(const Vec<dim,T>& v, OtherT c) { return v * c; }

template <int dim, typename T, typename OtherT, typename ResultT>
static Vec<dim,ResultT> vec__rmul__(OtherT c, const Vec<dim,T>& v) { return c * v; }


// dtype
template <typename T> inline static py::handle<> plain_vec_dtype() { return py::handle<>(); }
template <> inline py::handle<> plain_vec_dtype<double>() {
    return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyFloat_Type)));
}
template <> inline py::handle<> plain_vec_dtype<dcomplex>() {
    return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyComplex_Type)));
}
template <int dim, typename T> py::handle<> vec_dtype(const Vec<dim,T>&) { return plain_vec_dtype<T>(); }

// Python doc
const static char* __doc__ =
        "General vector class in PLaSK. It can inteligently hold both 2D and 3D vectors.\n\n"
        "vector(#, #[, #]) -> initialize with ordered components\n"
        "vector(x=#, y=#, z=#) -> initialize with Cartesian components (z or x skipped for 2D)\n"
        "vector(r=#, phi=#, z=#) -> initialize with cylindrical components (phi skipped for 2D)\n"
        "The order of its components always corresponds to the structure orientation\n"
        "(with the last component parallel to the epitaxial growth direction.\n\n"
        "However, the component names depend on the config.axis_up configuration option.\n"
        "Changing this option will change the order of component names accordingly:\n\n"
        "For config.vertical_axis = 'z', the component name mapping is following:\n"
        "in 2D vectors it is [y,z] (x-component skipped) or [r,z] (you can refer to\n"
        "Cartesian or cylindrical cooridinates at your preference). In 3D vectors it is\n"
        "[x,y,z] or [r,phi,z].\n\n"
        "For config.vertical_axis = 'z' the order becomes: [x,y] and [z,x,y] for 2D and 3D vectors, "
        "respectively. In this case, cylindrical component names are not allowed.";

// Register vector class to python
template <int dim, typename T>
inline static py::class_<Vec<dim,T>> register_vector_class()
{
    typedef Vec<dim,T> V;
    typedef Vec<dim,double> VR;
    typedef Vec<dim,dcomplex> VC;

    T (*dr)(const V&, const VR&) = &dot<T,double>;
    dcomplex (*dc)(const V&, const VC&) = &dot<T,dcomplex>;

    V (*c)(const V&) = &plask::conj<T>;

    py::class_<V> vec_class = py::class_<V>("vector", __doc__, py::no_init)
//         .def_readwrite("x", &V::x)
//         .def_readwrite("y", &V::y)
//         .def_readwrite("r", &V::r)
//         .def_readwrite("z", &V::z)
        .def("__getitem__", vec__getitem__<dim,T>)
        .def("__setitem__", vec__getitem__<dim,T>)
        .def("__iter__", &Vec_iterator<dim,T>::new_iterator, py::with_custodian_and_ward_postcall<0,1>())
        .def(py::self == py::other<VR>())
        .def(py::self == py::other<VC>())
        .def(py::self != py::other<VR>())
        .def(py::self != py::other<VC>())
//         .def("abs2", &V::abs2, "Squared vector abs")
//         .def("__abs__", &V::abs, "Vector magnitue")
//         .def("abs", &V::abs, "Vector magnitue")
        .def("copy", &copy_vec<dim,T>)
        .def(py::self + py::other<VR>())
        .def(py::self + py::other<VC>())
        .def(py::self - py::other<VR>())
        .def(py::self - py::other<VC>())
        .def( - py::self)
        .def("__mul__", &vec__mul__<dim,T,double,T>)
        .def("__mul__", &vec__mul__<dim,T,dcomplex,dcomplex>)
        .def("__rmul__", &vec__rmul__<dim,T,double,T>)
        .def("__rmul__", &vec__rmul__<dim,T,dcomplex,dcomplex>)
        .def(py::self += py::other<V>())
        .def(py::self -= py::other<V>())
        .def(py::self *= T())
        .def("__str__", &vec__str__<dim,T>)
        .def("__repr__", &vec__repr__<dim,T>)
        .def("dot", dr, "Dot product with another vector")
        .def("dot", dc, "Dot product with another vector")
        .def("__mul__", dr)
        .def("__mul__", dc)
        .def("conjugate", c)
        .def("conj", c)
        .add_property("dtype", &vec_dtype<dim,T>)
    ;

    py::scope vec_scope = vec_class;

    py::class_<Vec_iterator<dim,T>>("_Iterator", py::no_init)
        .def("__iter__", &Vec_iterator<dim,T>::__iter__, py::return_self<>())
        .def("next", &Vec_iterator<dim,T>::next)
    ;


    return vec_class;
}


struct PyVec
{
    py::list components;
    const int dim;

    PyVec(const py::list& comps) : components(comps) , dim(py::len(comps)) {}

    template <int vdim, typename T>
    operator Vec<vdim,T> () {
        if (dim != vdim) {
            std::stringstream out;
            out << "cannot convert " << py::len(components) << "D vector to " << dim << "D";
            PyErr_SetString(PyExc_TypeError, out.str().c_str());
        }
        T c[vdim];
        for (int i = 0; i < vdim; ++i)
            c[i] = py::extract<T>(components[i]);
        return VecReturner<vdim,T>::result(c);
    }

    int len() { return dim; }

    py::object getitem(int i) {
        if (i < 0) i = dim + i;
        if (i >= dim || i < 0) {
            const char message[] = "vector index out of range";
            PyErr_SetString(PyExc_IndexError, message);
            throw py::error_already_set();
        }
        return components[i];
    }

    // vector[i] = v
    void setitem(int i, py::object v) {
        if (i < 0) i = dim + i;
        if (i >= dim || i < 0) {
            const char message[] = "vector index out of range";
            PyErr_SetString(PyExc_IndexError, message);
            throw py::error_already_set();
        }
        components[i] = v;
    }

    // __str__(v)
    std::string str() {
        std::stringstream out;
        out << "[";
        for (int i = 0; i < dim; ++i) out << std::string(py::extract<std::string>(py::str(components[i]))) << (i!=dim-1 ? ", " : "]");
        return out.str();
    }

    // __repr__(v)
    std::string repr() {
        std::stringstream out;
        out << "vector(";
        for (int i = 0; i < dim; ++i) out << std::string(py::extract<std::string>(py::str(components[i]))) << (i!=dim-1 ? ", " : ")");
        return out.str();
    }

    // __eq__(v)
    bool eq(const PyVec& v) {
        if (dim != v.dim) return false;
        for (int i = 0; i < v.dim; ++i)
            if (components[i] != v.components[i]) return false;
        return true;
    }

    // __ne__(v)
    bool ne(const PyVec& v) { return ! eq(v); }


    py::object dtype() { return py::object(); }


  private:
      template <int dim, typename T> struct VecReturner { inline static Vec<dim,T> result(T* c) {} };
      template <typename T> struct VecReturner<2,T> { inline static Vec<2,T> result(T c[]) { return Vec<2,T>(c[0],c[1]); } };
      template <typename T> struct VecReturner<3,T> { inline static Vec<3,T> result(T c[]) { return Vec<3,T>(c[0],c[1],c[2]); } };

};


// Python constructor
static shared_ptr<PyVec> pyvec__init__(py::tuple args, py::dict kwargs)
{
    int n = py::len(args);

    py::list params;

    if (n == 0) { // Extract components from kwargs

        bool cart = false, cylind = false;
        n = py::len(kwargs);
        py::object comp[3];

        py::stl_input_iterator<std::string> begin(kwargs.keys()), end;
        for (auto key = begin; key != end; ++key) {
            py::object val = kwargs[*key];

            if (Config::z_up) {
                if (*key == "x") {
                    if (n == 3) {
                        comp[0] = val;
                    } else {
                        PyErr_SetString(PyExc_TypeError,
                            "x component not allowed for 2D vectors if config.vertical_axis is 'z'");
                        throw py::error_already_set();
                    }
                    cart = true;
                } else if (*key == "y") {
                    comp[n==2?0:1] = val;
                    cart = true;
                } else if (*key == "z") {
                    comp[n==2?1:2] = val;
                } else if (*key == "phi") {
                    if (n == 3) {
                        comp[1] = val;
                    } else {
                        PyErr_SetString(PyExc_TypeError,
                            "phi component not allowed for 2D vectors");
                        throw py::error_already_set();
                    }
                    cylind = true;
                } else if (*key == "r") {
                    comp[0] = val;
                    cylind = true;
                } else {
                    PyErr_SetString(PyExc_TypeError, ("unrecognized component name '" + *key + "'").c_str());
                    throw py::error_already_set();
                }
            } else {
                if (*key == "z") {
                    if (n == 3) {
                        comp[0] = val;
                    } else {
                        PyErr_SetString(PyExc_TypeError,
                            "z component not allowed for 2D vectors if config.vertical_axis is 'y'");
                        throw py::error_already_set();
                    }
                } else if (*key == "x") {
                    comp[n==2?0:1] = val;
                } else if (*key == "y") {
                    comp[n==2?1:2] = val;
                } else if (*key == "phi" || *key == "r") {
                    PyErr_SetString(PyExc_TypeError, "radial components not allowed if config.vertical_axis is 'z'");
                    throw py::error_already_set();
                } else {
                    PyErr_SetString(PyExc_TypeError, ("unrecognized component name '" + *key + "'").c_str());
                    throw py::error_already_set();
                }
            }
        }

        if (cart && cylind) {
            PyErr_SetString(PyExc_TypeError, "mixed cylindrical and Cartesian component names");
            throw py::error_already_set();
        }

        for (int i = 0; i < n; i++)
            params.append(comp[i]);

    } else if (kwargs) {
        PyErr_SetString(PyExc_TypeError, "components must be provided entirely in a list or by names");
        throw py::error_already_set();
    } else {
        params = py::list(args);
    }

    if (n != 2 && n != 3) {
        PyErr_SetString(PyExc_TypeError, "wrong number of arguments");
        throw py::error_already_set();
    }

    return shared_ptr<PyVec>(new PyVec(params));
}



void register_vector()
{
    register_vector_class<2,double>();
    register_vector_class<2,dcomplex>();
    register_vector_class<3,double>();
    register_vector_class<3,dcomplex>();

//     py::implicitly_convertible<Vec<2,double>,Vec<2,dcomplex>>();
//     py::implicitly_convertible<Vec<3,double>,Vec<3,dcomplex>>();

    py::class_<PyVec, shared_ptr<PyVec>> pyvec("vector", __doc__, py::no_init);
    pyvec
        .def("__init__", raw_constructor(&pyvec__init__, 0))
        .def("__getitem__", &PyVec::getitem)
        .def("__setitem__", &PyVec::setitem)
        .def("__str__", &PyVec::str)
        .def("__repr__", &PyVec::repr)
        .def("__eq__", &PyVec::eq)
        .def("__ne__", &PyVec::ne)
        .add_property("dtype", &PyVec::dtype)
    ;

    py::scope().attr("vec") = pyvec;

    py::implicitly_convertible<PyVec,Vec<2,double>>();
    py::implicitly_convertible<PyVec,Vec<3,double>>();
    py::implicitly_convertible<PyVec,Vec<2,dcomplex>>();
    py::implicitly_convertible<PyVec,Vec<3,dcomplex>>();
}

}} // namespace plask::python
