#include <sstream>

#include <plask/vec.h>
#include <plask/exceptions.h>
#include <plask/config.h>

#include "python_globals.h"
#include "python_numpy.h"
#include "python_util/raw_constructor.h"

#include <boost/python/stl_iterator.hpp>
#include <boost/concept_check.hpp>

#if PY_VERSION_HEX >= 0x03000000
#   define NEXT "__next__"
#else
#   define NEXT "next"
#endif

namespace plask { namespace python {


namespace detail {
    template <int dim, typename T> struct MakeVecFromNumpyImpl;

    template <int dim> struct MakeVecFromNumpyImpl<dim,double> {
        static inline void call(void* storage, PyObject* obj) {
            if (PyArray_DESCR((PyArrayObject*)obj)->type_num == NPY_LONG) {
                new(storage) Vec<dim,double>(Vec<dim,double>::fromIterator(static_cast<long*>(PyArray_DATA((PyArrayObject*)obj))));
            } else if (PyArray_DESCR((PyArrayObject*)obj)->type_num == NPY_DOUBLE)
                new(storage) Vec<dim,double>(Vec<dim,double>::fromIterator(static_cast<double*>(PyArray_DATA((PyArrayObject*)obj))));
            else
                throw py::error_already_set();
        }
    };

    template <int dim> struct MakeVecFromNumpyImpl<dim,dcomplex> {
        static inline void call(void* storage, PyObject* obj) {
            if (PyArray_DESCR((PyArrayObject*)obj)->type_num == NPY_LONG) {
                Vec<dim,dcomplex> *vec = new(storage) Vec<dim,dcomplex>; for (int i = 0; i < dim; ++i) (*vec)[i] = double( *(static_cast<long*>(PyArray_DATA((PyArrayObject*)obj)) + i) );
            } else if (PyArray_DESCR((PyArrayObject*)obj)->type_num == NPY_DOUBLE)
                new(storage) Vec<dim,dcomplex>(Vec<dim,dcomplex>::fromIterator(static_cast<double*>(PyArray_DATA((PyArrayObject*)obj))));
            else if (PyArray_DESCR((PyArrayObject*)obj)->type_num == NPY_CDOUBLE)
                new(storage) Vec<dim,dcomplex>(Vec<dim,dcomplex>::fromIterator(static_cast<dcomplex*>(PyArray_DATA((PyArrayObject*)obj))));
            else
                throw py::error_already_set();
        }
    };

    template <int dim, typename T>
    struct Vec_from_Sequence {
        Vec_from_Sequence() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<Vec<dim,T>>());
        }

        // Determine if obj can be converted into an Aligner
        static void* convertible(PyObject* obj) {
            if (!PyList_Check(obj) && !PyTuple_Check(obj) && !PyArray_Check(obj)) return NULL;
            return obj;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data) {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<Vec<dim,T>>*)data)->storage.bytes;
            try {
                if (PyArray_Check(obj)) {
                    if (PyArray_NDIM((PyArrayObject*)obj) != 1 || PyArray_DIMS((PyArrayObject*)obj)[0] != dim) throw py::error_already_set();
                    MakeVecFromNumpyImpl<dim,T>::call(storage, obj);
                } else {
                    auto seq = py::object(py::handle<>(py::borrowed(obj)));
                    if (py::len(seq) != dim || (PyArray_Check(obj) && PyArray_NDIM((PyArrayObject*)obj) != 1)) throw py::error_already_set();
                    py::stl_input_iterator<T> begin(seq);
                    new(storage) Vec<dim,T>(Vec<dim,T>::fromIterator(begin));
                }
                data->convertible = storage;
            } catch (py::error_already_set) {
                throw TypeError("Must provide either plask.vector or a sequence of length {0} of proper dtype", dim);
            }
        }
    };

    /*template <typename T>
    struct Vec1_to_Python {
        static PyObject* convert(Vec<1,T> const& v) {
            return boost::python::incref(boost::python::object(T(v)).ptr());
        }
    };*/

}


// v = vector[i]
template <int dim, typename T>
static T vec__getitem__(Vec<dim,T>& self, int i) {
    if (i < 0) i = dim + i;
    if (i >= dim || i < 0) throw IndexError("vector index out of range");
    return self[i];
}

// vector[i] = v
template <int dim, typename T>
static void vec__setitem__(Vec<dim,T>& self, int i, T v) {
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
            PyErr_SetString(PyExc_StopIteration, u8"No more components.");
            boost::python::throw_error_already_set();
        }
        return (*vec)[i++];
    }
};

// copy v
template <int dim, typename T>
static Vec<dim,T> copy_vec(const Vec<dim,T>& v) {
    return v;
}

// dtype
template <int dim, typename T> inline static py::handle<> vec_dtype() { return detail::dtype<T>(); }


// vector.__array__
template <int dim, typename T>  py::object vec__array__(py::object self, py::object dtype) {
    Vec<dim,T>* vec = py::extract<Vec<dim,T>*>(self);
    npy_intp dims[] = { dim };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, detail::typenum<T>(), (void*)vec->begin());
    if (arr == nullptr) throw plask::CriticalException(u8"cannot create array from vector");
    confirm_array<T>(arr, self, dtype);
    return py::object(py::handle<>(arr));
}

// vector_list.__array__
template <int dim, typename T>  py::object vec_list__array__(py::object self, py::object dtype) {
    std::vector<Vec<dim,T>>* list = py::extract<std::vector<Vec<dim,T>>*>(self);
    npy_intp dims[] = { (npy_int)list->size(), dim };
    PyObject* arr = PyArray_SimpleNewFromData(2, dims, detail::typenum<T>(), (void*)((*list)[0].begin()));
    if (arr == nullptr) throw plask::CriticalException(u8"cannot create array from vector list");
    confirm_array<T>(arr, self, dtype);
    return py::object(py::handle<>(arr));
}


// Access components by name
template <int dim>
inline static int vec_attr_indx(const std::string& attr) {
    int i = int(current_axes[attr]) - 3 + dim;
    if (i < 0 || i >= dim) {
        if (attr == "x" || attr == "y" || attr == "z" || attr == "r" || attr == "phi" ||
            attr == "lon" || attr == "tran" || attr == "up")
            throw AttributeError(u8"vector attribute '{}' has no sense for {:d}D vector if config.axes = '{}'", attr, dim, current_axes.str());
        else
            throw AttributeError(u8"'vec' object has no attribute '{}'", attr);
    }
    return i;
}

template <int dim, typename T>
struct VecAttr {
    typedef Vec<dim, T> V;
    static T get(const V& self, const std::string& attr) { return self[vec_attr_indx<dim>(attr)]; }
//     static void set(V& self, const std::string& attr, T val) { self[vec_attr_indx<dim>(attr)] = val; }
    static void set(V& self, const std::string& attr, T val) { throw TypeError("vector attribute '{}' cannot be changed", attr); }
};

template <int dim, typename T>
static Vec<dim,T> vector__div__float(const Vec<dim,T>& self, double f) { return self/f; }

template <int dim, typename T>
static Vec<dim,dcomplex> vector__div__complex(const Vec<dim,T>& self, dcomplex f) { return self * (1./f); }


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
        "See Also:\n"
        "    vec: create a new vector.\n"
        , py::no_init);
    vec_class
        .def("__getattr__", &VecAttr<dim,T>::get)
        .def("__setattr__", &VecAttr<dim,T>::set)
        .def("__getitem__", &vec__getitem__<dim,T>)
        // .def("__setitem__", &vec__setitem__<dim,T>)
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
        .def("__div__", &vector__div__float<dim,T>)
        .def("__truediv__", &vector__div__float<dim,T>)
        .def("__div__", &vector__div__complex<dim,T>)
        .def("__truediv__", &vector__div__complex<dim,T>)
        .def("__mul__", dc)
        .def("__mul__", dr)
        .def("dot", dc, py::arg("other"))
        .def("dot", dr, py::arg("other"),
             u8"Dot product with another vector. It is equal to `self` * `other`, so the\n"
             u8"`self` vector is conjugated.\n")
        .def("conjugate", c,
             u8"Conjugate of the vector. Alias for :meth:`conj`.\n")
        .def("conj", c,
             u8"Conjugate of the vector. It can be called for real vectors, but then it\n"
             u8"simply returns `self`\n")
        .def("abs2", (double (*)(const Vec<dim,T>&))&abs2<dim,T>,
             u8"Squared magnitude of the vector. It is always a real number equal to\n"
             u8"``v * v``.\n")
        .def("abs", (double (*)(const Vec<dim,T>&))&abs<dim,T>,
             u8"Magnitude of the vector. It is always a real number.\n")
        .def("__abs__", (double (*)(const Vec<dim,T>&))&abs<dim,T>)
        .def("copy", &copy_vec<dim,T>,
             u8"Copy of the vector. Normally vectors behave like Python containers, and\n"
             u8"assignement operation makes shallow copy only. Use this method if you want\n"
             u8"to modify the copy without changing the source.\n")
        .add_static_property("dtype", &vec_dtype<dim,T>,
             u8"Type od the vector components. This is always either ``float`` or ``complex``.\n")
        .def("__array__", &vec__array__<dim,T>, py::arg("dtype")=py::object())
    ;
    vec_class.attr("__module__") = "plask";

    detail::Vec_from_Sequence<dim,T>();

    register_vector_of<Vec<dim,T>>(name)
        .def("__array__", &vec_list__array__<dim,T>, py::arg("dtype")=py::object())
    ;

    py::scope vec_scope = vec_class;

    py::class_<Vec_iterator<dim,T>>("_Iterator", py::no_init)
        .def("__iter__", &Vec_iterator<dim,T>::__iter__, py::return_self<>())
        .def(NEXT, &Vec_iterator<dim,T>::next)
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
        if (dtype != py::object()) {
            if (dtype.ptr() == reinterpret_cast<PyObject*>(&PyFloat_Type)) force_double = true;
            else if (dtype.ptr() == reinterpret_cast<PyObject*>(&PyComplex_Type)) force_complex = true;
            else {
                throw TypeError(u8"wrong dtype (can be only float or complex)");
            }
        }
    }

    if (n == 0) { // Extract components from kwargs

        n = nk;
        py::object comp[3];

        py::stl_input_iterator<std::string> begin(kwargs.keys()), end;
        for (auto key = begin; key != end; ++key) {
            if (*key == "dtype") continue;
            py::object val = kwargs[*key];
            try {
                if (n == 2) comp[vec_attr_indx<2>(*key)] = val;
                else if (n == 3) comp[vec_attr_indx<3>(*key)] = val;
            } catch (AttributeError) {
                throw TypeError(u8"wrong component name for {:d}D vector if config.axes = '{}'", n, current_axes.str());
            }

        }
        for (int i = 0; i < n; i++)
            params.append(comp[i]);

    } else if (nk > 0) {
        throw TypeError(u8"components must be provided entirely in a list or by names");
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
        if (n == 2) return py::object(Vec<2,double>(py::extract<double>(params[0]), py::extract<double>(params[1])));
        return py::object(Vec<3,double>(py::extract<double>(params[0]), py::extract<double>(params[1]), py::extract<double>(params[2])));
    } catch (py::error_already_set) { PyErr_Clear(); try {
        if (force_double) { PyErr_SetNone(PyExc_TypeError); throw py::error_already_set(); }
        if (n == 2) return py::object(Vec<2,dcomplex>(py::extract<dcomplex>(params[0]), py::extract<dcomplex>(params[1])));
        return py::object(Vec<3,dcomplex>(py::extract<dcomplex>(params[0]), py::extract<dcomplex>(params[1]), py::extract<dcomplex>(params[2])));
    } catch (py::error_already_set) {
        throw TypeError(u8"wrong vector argument types");
    }}

    return py::object();
}


// Python doc
const static char* __doc__ =

    "vec(x,y,z, dtype=None)\n"
    "vec(z,x,y, dtype=None)\n"
    "vec(r,p,z, dtype=None)\n"
    "vec(x,y, dtype=None)\n"
    "vec(z,x, dtype=None)\n"
    "vec(r,z, dtype=None)\n\n"

    "PLaSK vector.\n\n"

    "The constructor arguments depend on the current value of\n"
    ":attr:`plask.config.axes`. However, you must either specify all the components\n"
    "either as the unnamed sequence or as the named keywords.\n\n"

    "Args:\n"
    "    _letter_ (dtype): Vector components.\n"
    "        Their choice depends on the current value of :attr:`plask.config.axes`.\n"
    "    dtype (type): type of the vector components.\n"
    "        If this argument is omitted or `None`, the type is determined\n"
    "        automatically.\n\n"

    "The order of vector components is always [`longitudinal`, `transverse`,\n"
    "`vertical`] for 3D vectors or [`transverse`, `vertical`] for 2D vectors.\n"
    "However, the component names depend on the :attr:`~plask.config.axes`\n"
    "configuration option. Changing this option will change the order of component\n"
    "names (even for existing vectors) accordingly:\n\n"

    "============================== ======================== ========================\n"
    "plask.config.axes value        2D vector components     3D vector components\n"
    "============================== ======================== ========================\n"
    "`xyz`, `yz`, `z_up`            [`y`, `z`]               [`x`, `y`, `z`]\n"
    "`zxy`, `xy`, `y_up`            [`x`, `y`]               [`z`, `x`, `y`]\n"
    "`prz`, `rz`, `rad`             [`r`, `z`]               [`p`, `r`, `z`]\n"
    "`ltv`, `abs`                   [`t`, `v`]               [`l`, `t`, `v`]\n"
    "`long,tran,vert`, `absolute`   [`tran`, `vert`]         [`long`, `tran`, `vert`]\n"
    "============================== ======================== ========================\n\n"

    "Examples:\n"
    "    Create two-dimensional vector:\n\n"

    "    >>> vector(1, 2)\n"
    "    vector(1, 2)\n\n"

    "    Create 3D vector specifying components in rotated coordinate system:\n\n"

    "    >>> config.axes = 'xy'\n"
    "    >>> vec(x=1, y=2, z=3)\n"
    "    plask.vec(3, 1, 2)\n\n"

    "    Create 3D vector specifying components:\n\n"

    "    >>> config.axes = 'xyz'\n"
    "    >>> vec(x=1, z=2, y=3)\n"
    "    plask.vec(1, 3, 2)\n\n"

    "    Create 2D vector in cylindrical coordinates, specifying dtype:\n\n"

    "    >>> config.axes = 'rz'\n"
    "    >>> vec(r=2, z=0, dtype=complex)\n"
    "    plask.vec((2+0j), (0+0j))\n\n"

    "To access vector components you may either use attribute names or numerical\n"
    "indexing. The ordering and naming rules are the same as for the construction.\n\n"

    "Examples:\n\n"

    "    >>> config.axes = 'xyz'\n"
    "    >>> v = vec(1, 2, 3)\n"
    "    >>> v.z\n"
    "    3\n"
    "    >>> v[0]\n"
    "    1\n\n"

    "You may perform all the proper algebraic operations on PLaSK vectors like\n"
    "addition, subtraction, multiplication by scalar, multiplication by another\n"
    "vector (which results in a dot product).\n\n"

    "Example:\n\n"

    "    >>> v1 = vec(1, 2, 3)\n"
    "    >>> v2 = vec(10, 20, 30)\n"
    "    >>> v1 + v2\n"
    "    plask.vec(11, 22, 33)\n"
    "    >>> 2 * v1\n"
    "    plask.vec(2, 4, 6)\n"
    "    >>> v1 * v2\n"
    "    140.0\n"
    "    >>> abs(v1)\n"
    "    >>> v3 = vec(0, 1+2j)\n"
    "    >>> v3.conj()\n"
    "    plask.vec(0, 1-2j)\n"
    "    >>> v3.abs2()\n"
    "    5.0\n\n"
    ;

void register_vectors()
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    //py::to_python_converter<Vec<1,double>, detail::Vec1_to_Python<double>>();

    register_vector_class<2,double>("vec");
    register_vector_class<2,dcomplex>("vec");
    register_vector_class<3,double>("vec");
    register_vector_class<3,dcomplex>("vec");

    py::def("vec", py::raw_function(&new_vector));
    py::scope().attr("vec").attr("__doc__") = __doc__;
}

}} // namespace plask::python
