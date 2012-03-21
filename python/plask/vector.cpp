#include <sstream>

#include <plask/vec.h>
#include <plask/exceptions.h>
#include <plask/config.h>

#include "globals.h"
#include "../util/raw_constructor.h"

#include <numpy/arrayobject.h>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

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
    self[i] = v;
}

// len(v)
template <int dim>
static int vec__len__() { return dim; }

// __str__(v)
template <int dim, typename T>
std::string vec__str__(const Vec<dim,T>& to_print) {
    std::stringstream out;
    out << "[";
    for (int i = 0; i < dim; ++i) out << sc(to_print[i]) << (i!=dim-1 ? ", " : "]");
    return out.str();
}

// __repr__(v)
template <int dim, typename T>
std::string vec__repr__(const Vec<dim,T>& to_print) {
    std::stringstream out;
    out << "vector(";
    for (int i = 0; i < dim; ++i) out << sc(to_print[i]) << (i!=dim-1 ? ", " : ")");
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

// dtype
template <typename T> inline static py::handle<> plain_vec_dtype() { return py::handle<>(); }
template <> inline py::handle<> plain_vec_dtype<double>() {
    return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyFloat_Type)));
}
template <> inline py::handle<> plain_vec_dtype<dcomplex>() {
    return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyComplex_Type)));
}
template <int dim, typename T> py::handle<> vec_dtype(const Vec<dim,T>&) { return plain_vec_dtype<T>(); }

// helpers for __array__
template <typename T> inline static int get_typenum();
template <> int get_typenum<double>() { return NPY_DOUBLE; }
template <> int get_typenum<dcomplex>() { return NPY_CDOUBLE; }

// vector.__array__
template <int dim, typename T>  py::object vec__array__(py::object self) {
    Vec<dim,T>* vec = py::extract<Vec<dim,T>*>(self);
    npy_intp dims[] = { dim };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, get_typenum<T>(), (void*)vec->components);
    if (arr == nullptr) throw plask::CriticalException("Cannot create array from vector");
    py::incref(self.ptr()); PyArray_BASE(arr) = self.ptr(); // Make sure the vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}

// vector_list.__array__
template <int dim, typename T>  py::object vec_list__array__(py::object self) {
    std::vector<Vec<dim,T>>* list = py::extract<std::vector<Vec<dim,T>>*>(self);
    npy_intp dims[] = { list->size(), dim };
    PyObject* arr = PyArray_SimpleNewFromData(2, dims, get_typenum<T>(), (void*)(&(*list)[0].components));
    if (arr == nullptr) throw plask::CriticalException("Cannot create array from vector list");
    py::incref(self.ptr()); PyArray_BASE(arr) = self.ptr(); // Make sure the vector list stays alive as long as the array
    return py::object(py::handle<>(arr));
}


// Access components by name
template <typename T, typename V, int dim, char c, bool z_up>
struct VecAccessor {
    inline static T getComponent(const V& self) {
        std::stringstream out;
        out << "component " << c << " does not make sense for this vector if config.vertical_axis = '" << (z_up?'z':'y') << "'";
        PyErr_SetString(PyExc_AttributeError, out.str().c_str());
        throw py::error_already_set();
        return T(); // Make compiler happy, never reached anyway
    }
    inline static void setComponent(V& self, T val) {
        std::stringstream out;
        out << "component " << c << " does not make sense for this vector if config.vertical_axis = '" << (z_up?'z':'y') << "'";
        PyErr_SetString(PyExc_AttributeError, out.str().c_str());
        throw py::error_already_set();
    }
};

// Implementations for particular names
#define COMP(dim, name, z_up, i) \
    template <typename T, typename V> \
    struct VecAccessor<T,V, dim,name,z_up> { \
        inline static T getComponent(const V& self) { return self.components[i]; } \
        inline static void setComponent(V& self, T val) { self.components[i] = val; } \
    }

COMP(2, 'x', false, 0);
COMP(2, 'y', false, 1);

COMP(2, 'y', true, 0);
COMP(2, 'r', true, 0);
COMP(2, 'z', true, 1);

COMP(3, 'z', false, 0);
COMP(3, 'x', false, 1);
COMP(3, 'y', false, 2);

COMP(3, 'x', true, 0);
COMP(3, 'r', true, 0);
COMP(3, 'y', true, 1);
COMP(3, 'p', true, 1);
COMP(3, 'z', true, 2);


// Getter and setter functions
template <int dim, typename T, char c>
T get_vec_component(Vec<dim,T>& self) {
    if (Config::z_up) return VecAccessor<T, Vec<dim,T>, dim, c, true>::getComponent(self);
    return VecAccessor<T, Vec<dim,T>, dim, c, false>::getComponent(self);
}
template <int dim, typename T, char c>
void set_vec_component(Vec<dim,T>& self, const T& val) {
    if (Config::z_up) VecAccessor<T, Vec<dim,T>, dim, c, true>::setComponent(self, val);
    else VecAccessor<T, Vec<dim,T>, dim, c, false>::setComponent(self, val);
}

#define vec_component_property(name) &get_vec_component<dim,T,name>, &set_vec_component<dim,T,name>


// Register vector class to python
template <int dim, typename T>
inline static py::class_<Vec<dim,T>> register_vector_class(std::string name="vector")
{
    typedef Vec<dim,T> V;
    typedef Vec<dim,double> VR;
    typedef Vec<dim,dcomplex> VC;

    T (*dr)(const V&, const VR&) = &dot<T,double>;
    dcomplex (*dc)(const V&, const VC&) = &dot<T,dcomplex>;

    V (*c)(const V&) = &plask::conj<T>;

    py::class_<V> vec_class = py::class_<V>(name.c_str(),
        "PLaSK vector.\n\n"
        "See Also\n"
        "--------\n"
        "vector\t:\tcreate a new vector.\n"
        , py::no_init)
        .add_property("x", vec_component_property('x'))
        .add_property("y", vec_component_property('y'))
        .add_property("z", vec_component_property('z'))
        .add_property("r", vec_component_property('r'))
        .add_property("phi", vec_component_property('p'))
        .def("__getitem__", vec__getitem__<dim,T>)
        .def("__setitem__", vec__setitem__<dim,T>)
        .def("__iter__", &Vec_iterator<dim,T>::new_iterator, py::with_custodian_and_ward_postcall<0,1>())
        .def("__len__", &vec__len__<dim>)
        .def("__str__", &vec__str__<dim,T>)
        .def("__repr__", &vec__repr__<dim,T>)
        .def(py::self == py::other<VC>())
        .def(py::self == py::other<VR>())
        .def(py::self != py::other<VC>())
        .def(py::self != py::other<VR>())
        .def(py::self + py::other<VC>())
        .def(py::self + py::other<VR>())
        .def(py::self - py::other<VC>())
        .def(py::self - py::other<VR>())
        .def( - py::self)
        .def(py::self * dcomplex())
        .def(py::self * double())
        .def(dcomplex() * py::self)
        .def(double() * py::self)
        .def(py::self += py::other<V>())
        .def(py::self -= py::other<V>())
        .def(py::self *= T())
        .def("__mul__", dc)
        .def("__mul__", dr)
        .def("dot", dc, "Dot product with another vector")
        .def("dot", dr, "Dot product with another vector")
        .def("conjugate", c)
        .def("conj", c)
        .def("abs2", (double (*)(const Vec<dim,T>&))&abs2<dim,T>, "Squared vector abs")
        .def("abs", (double (*)(const Vec<dim,T>&))&abs<dim,T>, "Vector magnitue")
        .def("__abs__", (double (*)(const Vec<dim,T>&))&abs<dim,T>, "Vector magnitue")
        .def("copy", &copy_vec<dim,T>)
        .add_property("dtype", &vec_dtype<dim,T>)
        .def("__array__", &vec__array__<dim,T>)
    ;

    py::class_< std::vector<Vec<dim,T>>, shared_ptr<std::vector<Vec<dim,T>>> >((name+"_list").c_str())
        .def(py::vector_indexing_suite< std::vector<Vec<dim,T>> >())
        .def("__array__", &vec_list__array__<dim,T>)
    ;

    py::scope vec_scope = vec_class;

    py::class_<Vec_iterator<dim,T>>("_Iterator", py::no_init)
        .def("__iter__", &Vec_iterator<dim,T>::__iter__, py::return_self<>())
        .def("next", &Vec_iterator<dim,T>::next)
    ;

    return vec_class;
}


// Python constructor
static py::object new_vector(py::tuple args, py::dict kwargs)
{
    int n = py::len(args), nk = py::len(kwargs);

    py::list params;

    bool force_double = false;
    bool force_complex = false;

    if (kwargs.has_key("dtype")) {
        --nk;
        py::object dtype;
        dtype = kwargs["dtype"];
        if (dtype.ptr() == reinterpret_cast<PyObject*>(&PyFloat_Type)) force_double = true;
        else if (dtype.ptr() == reinterpret_cast<PyObject*>(&PyComplex_Type)) force_complex = true;
        else {
            PyErr_SetString(PyExc_TypeError, "wrong dtype (can be only double or complex)");
            throw py::error_already_set();
        }
    }

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
                } else if (*key != "dtype") {
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

    } else if (nk > 0) {
        PyErr_SetString(PyExc_TypeError, "components must be provided entirely in a list or by names");
        throw py::error_already_set();
    } else {
        params = py::list(args);
    }

    if (n != 2 && n != 3) {
        PyErr_SetString(PyExc_TypeError, "wrong number of arguments");
        throw py::error_already_set();
    }

    // Now detect the dtype
    py::object result;
    try {
        if (force_complex) { PyErr_SetNone(PyExc_TypeError); throw py::error_already_set(); }
        double cmps[n];
        for (int i = 0; i < n; i++) cmps[i] = py::extract<double>(params[i]);
        if (n == 2) return py::object(Vec<2,double>::fromIterator(cmps));
        return py::object(Vec<3,double>::fromIterator(cmps));
    } catch (py::error_already_set) { PyErr_Clear(); try {
        if (force_double) { PyErr_SetNone(PyExc_TypeError); throw py::error_already_set(); }
        dcomplex cmps[n];
        for (int i = 0; i < n; i++) cmps[i] = py::extract<dcomplex>(params[i]);
        if (n == 2) return py::object(Vec<2,dcomplex>::fromIterator(cmps));
        return py::object(Vec<3,dcomplex>::fromIterator(cmps));
    } catch (py::error_already_set) {
        PyErr_SetString(PyExc_TypeError, "wrong vector argument types");
        throw py::error_already_set();
    }}

    return py::object();
}


// Python doc
const static std::string __doc__ =

    "Create PLaSK vector.\n\n"

    "vector(#, #[, #])\n"
    "    initialize with ordered components\n"
    "vector(x=#, y=#, z=#)\n"
    "    initialize with Cartesian components (z or x skipped in 2D)\n"
    "vector(r=#, phi=#, z=#)\n"
    "    initialize with cylindrical components (phi skipped in 2D)\n\n"

    "The order of its components always corresponds to the structure orientation\n"
    "(with the last component parallel to the epitaxial growth direction.\n\n"

    "However, the component names depend on the config.axis_up configuration option.\n"
    "Changing this option will change the order of component names accordingly:\n\n"

    "For config.vertical_axis = 'z', the component name mapping is following:\n"
    "in 2D vectors it is [y,z] (x-component skipped) or [r,z] (you can refer to\n"
    "Cartesian or cylindrical cooridinates at your preference). In 3D vectors it is\n"
    "[x,y,z] or [r,phi,z].\n\n"

    "For config.vertical_axis = 'z' the order becomes: [x,y] and [z,x,y] for 2D and 3D vectors,\n"
    "respectively. In this case, cylindrical component names are not allowed.\n\n"

    "Examples\n"
    "--------\n\n"
    ">>> vector(1,2)\n"
    "vector(1,2)\n    \n"
    "Create two-dimensional vector.\n    \n"

    ">>> config.vertical_axis = 'y'\n"
    ">>> vector(x=1, y=2, z=3)\n"
    "vector(3,2,1)\n    \n"
    "Create 3D vector specifying components in rotated coordinate system.\n    \n"

    ">>> config.vertical_axis = 'z'\n"
    ">>> vector(x=1, z=2, y=3)\n"
    "vector(1,3,2)\n    \n"
    "Create 3D vector specifying components.\n    \n"

    ">>> vector(r=2, z=0, dtype=complex)\n"
    "vector(2,0)\n    \n"
    "Create 2D vector in cylindrical coordinates, specifying dtype.\n"



    ;

void register_vector()
{
    // Initialize numpy
    import_array();

    register_vector_class<2,double>("vector2f");
    register_vector_class<2,dcomplex>("vector2fc");
    register_vector_class<3,double>("vector3f");
    register_vector_class<3,dcomplex>("vector2c");

    py::implicitly_convertible<Vec<2,double>,Vec<2,dcomplex>>();
    py::implicitly_convertible<Vec<3,double>,Vec<3,dcomplex>>();


    py::def("vector", py::raw_function(&new_vector));
    py::scope().attr("vector").attr("__doc__") = __doc__.c_str();
    py::scope().attr("vec") = py::scope().attr("vector");
}

}} // namespace plask::python
