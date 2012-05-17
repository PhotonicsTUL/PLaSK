#include <sstream>

#include <plask/vec.h>
#include <plask/exceptions.h>
#include <plask/config.h>

#include "python_globals.h"
#include "../util/raw_constructor.h"

#include <numpy/arrayobject.h>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/concept_check.hpp>

namespace plask { namespace python {

// v = vector[i]
template <int dim, typename T>
static T vec__getitem__(Vec<dim,T>& self, int i) {
    if (i < 0) i = dim + i;
    if (i >= dim || i < 0) throw IndexError("vector index out of range");
    return self[i];
}

// vector[i] = v
template <int dim, typename T>
static T vec__setitem__(Vec<dim,T>& self, int i, T v) {
    if (i < 0) i = dim + i;
    if (i >= dim || i < 0) throw IndexError("vector index out of range");
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
    for (int i = 0; i < dim; ++i) out << pyformat(to_print[i]) << (i!=dim-1 ? ", " : "]");
    return out.str();
}

// __repr__(v)
template <int dim, typename T>
std::string vec__repr__(const Vec<dim,T>& to_print) {
    std::stringstream out;
    out << "plask.vec(";
    for (int i = 0; i < dim; ++i) out << pyformat(to_print[i]) << (i!=dim-1 ? ", " : ")");
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
    if (arr == nullptr) throw plask::CriticalException("cannot create array from vector");
    py::incref(self.ptr()); PyArray_BASE(arr) = self.ptr(); // Make sure the vector stays alive as long as the array
    return py::object(py::handle<>(arr));
}

// vector_list.__array__
template <int dim, typename T>  py::object vec_list__array__(py::object self) {
    std::vector<Vec<dim,T>>* list = py::extract<std::vector<Vec<dim,T>>*>(self);
    npy_intp dims[] = { (npy_int)list->size(), dim };
    PyObject* arr = PyArray_SimpleNewFromData(2, dims, get_typenum<T>(), (void*)(&(*list)[0].components));
    if (arr == nullptr) throw plask::CriticalException("cannot create array from vector list");
    py::incref(self.ptr()); PyArray_BASE(arr) = self.ptr(); // Make sure the vector list stays alive as long as the array
    return py::object(py::handle<>(arr));
}


// Access components by name
template <int dim>
inline static int vec_attr_indx(const std::string& attr) {
    int i = config.axes[attr] - 3 + dim;
    if (i < 0 || i >= dim) {
        if (attr == "x" || attr == "y" || attr == "z" || attr == "r" || attr == "phi" ||
            attr == "lon" || attr == "tran" || attr == "up")
            throw AttributeError("attribute '%s' has no sense for %dD vector if config.axes = '%s'", attr, dim, config.axes_name());
        else
            throw AttributeError("'vector' object has no attribute '%s'", attr);
    }
    return i;
}

template <int dim, typename T>
struct VecAttr {
    typedef Vec<dim, T> V;
    static T get(const V& self, const std::string& attr) { return self[vec_attr_indx<dim>(attr)]; }
    static void set(V& self, const std::string& attr, T val) { self[vec_attr_indx<dim>(attr)] = val; }
};



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
        .def("__getattr__", &VecAttr<dim,T>::get)
        .def("__setattr__", &VecAttr<dim,T>::set)
        .def("__getitem__", &vec__getitem__<dim,T>)
        .def("__setitem__", &vec__setitem__<dim,T>)
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

    register_vector_of<Vec<dim,T>>(name)
        .def("__array__", &vec_list__array__<dim,T>)
    ;

    py::scope vec_scope = vec_class;

    py::class_<Vec_iterator<dim,T>>("_Iterator", py::no_init)
        .def("__iter__", &Vec_iterator<dim,T>::__iter__, py::return_self<>())
        .def("__next__", &Vec_iterator<dim,T>::next)
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
            throw TypeError("wrong dtype (can be only double or complex)");
        }
    }

    if (n == 0) { // Extract components from kwargs

        n = py::len(kwargs);
        py::object comp[3];

        py::stl_input_iterator<std::string> begin(kwargs.keys()), end;
        for (auto key = begin; key != end; ++key) {
            py::object val = kwargs[*key];
            try {
                if (n == 2) comp[vec_attr_indx<2>(*key)] = val;
                else if (n == 3) comp[vec_attr_indx<3>(*key)] = val;
            } catch (AttributeError) {
                throw TypeError("wrong component name for %dD vector if config.axes = '%s'", n, config.axes_name());
            }

        }
        for (int i = 0; i < n; i++)
            params.append(comp[i]);

    } else if (nk > 0) {
        throw TypeError("components must be provided entirely in a list or by names");
    } else {
        params = py::list(args);
    }

    if (n != 2 && n != 3) {
        throw TypeError("wrong number of arguments");
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
        throw TypeError("wrong vector argument types");
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
    "vector(r=#, p=#, z=#)\n"
    "    initialize with cylindrical components (p skipped in 2D)\n\n"

    "The order of its components always corresponds to the structure orientation\n"
    "(with the last component parallel to the epitaxial growth direction.\n\n"

    "However, the component names depend on the config.axes configuration option.\n"
    "Changing this option will change the order of component names accordingly:\n\n"

    "config.axes = 'xyz' (equivalents are 'yz' or 'z_up'):\n"
    "   2D vectors: [y,z], 3D vectors: [x,y,z]\n"
    "config.axes = 'zxy' (equivalents are 'xy' or 'y_up'):\n"
    "   2D vectors: [x,y], 3D vectors: [z,x,y]\n"
    "config.axes = 'prz' (equivalents are 'rz' or 'rad'):\n"
    "   2D vectors: [r,z], 3D vectors: [p,r,z]\n"
    "config.axes = 'lon,tran,up' (equivalent is 'absolute'):\n"
    "   2D vectors: [tran,up], 3D vectors: [lon,tran,up]\n"

    "Examples\n"
    "--------\n\n"
    ">>> vector(1, 2)\n"
    "vector(1, 2)\n    \n"
    "Create two-dimensional vector.\n    \n"

    ">>> config.axes = 'xy'\n"
    ">>> vector(x=1, y=2, z=3)\n"
    "vector(3, 2, 1)\n    \n"
    "Create 3D vector specifying components in rotated coordinate system.\n    \n"

    ">>> config.axes = 'xyz'\n"
    ">>> vector(x=1, z=2, y=3)\n"
    "vector(1, 3, 2)\n    \n"
    "Create 3D vector specifying components.\n    \n"

    ">>> config.axes = 'rz'\n"
    ">>> vector(r=2, z=0, dtype=complex)\n"
    "vector(2, 0)\n    \n"
    "Create 2D vector in cylindrical coordinates, specifying dtype.\n"

    ;

static inline bool plask_import_array() {
    import_array1(false);
    return true;
}

void register_vector()
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

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
