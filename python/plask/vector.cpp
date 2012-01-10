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

// len(v)
template <int dim>
static int vec__len__() { return dim; }

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

// dtype
template <typename T> inline static py::handle<> plain_vec_dtype() { return py::handle<>(); }
template <> inline py::handle<> plain_vec_dtype<double>() {
    return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyFloat_Type)));
}
template <> inline py::handle<> plain_vec_dtype<dcomplex>() {
    return py::handle<>(py::borrowed<>(reinterpret_cast<PyObject*>(&PyComplex_Type)));
}
template <int dim, typename T> py::handle<> vec_dtype(const Vec<dim,T>&) { return plain_vec_dtype<T>(); }



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
        .add_property("x", vec_component_property('x'))
        .add_property("y", vec_component_property('y'))
        .add_property("z", vec_component_property('z'))
        .add_property("r", vec_component_property('r'))
        .add_property("phi", vec_component_property('p'))
        .def("__getitem__", vec__getitem__<dim,T>)
        .def("__setitem__", vec__getitem__<dim,T>)
        .def("__iter__", &Vec_iterator<dim,T>::new_iterator, py::with_custodian_and_ward_postcall<0,1>())
        .def("__len__", &vec__len__<dim>)
        .def("__str__", &vec__str__<dim,T>)
        .def("__repr__", &vec__repr__<dim,T>)
        .def(py::self == py::other<VR>())
        .def(py::self == py::other<VC>())
        .def(py::self != py::other<VR>())
        .def(py::self != py::other<VC>())
        .def(py::self + py::other<VR>())
        .def(py::self + py::other<VC>())
        .def(py::self - py::other<VR>())
        .def(py::self - py::other<VC>())
        .def( - py::self)
        .def(py::self * dcomplex())
        .def(py::self * double())
        .def(dcomplex() * py::self)
        .def(double() * py::self)
        .def(py::self += py::other<V>())
        .def(py::self -= py::other<V>())
        .def(py::self *= T())
        .def("__mul__", dr)
        .def("__mul__", dc)
        .def("dot", dr, "Dot product with another vector")
        .def("dot", dc, "Dot product with another vector")
        .def("conjugate", c)
        .def("conj", c)
        .def("abs2", (double (*)(const Vec<dim,T>&))&abs2<dim,T>, "Squared vector abs")
        .def("abs", (double (*)(const Vec<dim,T>&))&abs<dim,T>, "Vector magnitue")
        .def("__abs__", (double (*)(const Vec<dim,T>&))&abs<dim,T>, "Vector magnitue")
        .def("copy", &copy_vec<dim,T>)
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

    py::object __getitem__(int i) {
        if (i < 0) i = dim + i;
        if (i >= dim || i < 0) {
            const char message[] = "vector index out of range";
            PyErr_SetString(PyExc_IndexError, message);
            throw py::error_already_set();
        }
        return components[i];
    }

    // vector[i] = v
    void __setitem__(int i, py::object v) {
        if (i < 0) i = dim + i;
        if (i >= dim || i < 0) {
            const char message[] = "vector index out of range";
            PyErr_SetString(PyExc_IndexError, message);
            throw py::error_already_set();
        }
        components[i] = v;
    }

    // Getter and setter functions
    template <char c>
    py::object getComponent() {
        if (Config::z_up) {
            if (dim == 2)
                return VecAccessor<py::object, PyVec, 2, c, true>::getComponent(*this);
            else
                return VecAccessor<py::object, PyVec, 3, c, true>::getComponent(*this);
        } else {
            if (dim == 2)
                return VecAccessor<py::object, PyVec, 2, c, false>::getComponent(*this);
            else
                return VecAccessor<py::object, PyVec, 3, c, false>::getComponent(*this);
        }
        return py::object(); // make compiler happy
    }

    template <char c>
    void setComponent(py::object val) {
        if (Config::z_up) {
            if (dim == 2)
                VecAccessor<py::object, PyVec, 2, c, true>::setComponent(*this, val);
            else
                VecAccessor<py::object, PyVec, 3, c, true>::setComponent(*this, val);
        } else {
            if (dim == 2)
                VecAccessor<py::object, PyVec, 2, c, false>::setComponent(*this, val);
            else
                VecAccessor<py::object, PyVec, 3, c, false>::setComponent(*this, val);
        }
    }

    py::object __iter__() {
        return components.attr("__iter__")();
    }

    int __len__() { return dim; }

    std::string __str__() {
        std::stringstream out;
        out << "[";
        for (int i = 0; i < dim; ++i) out << std::string(py::extract<std::string>(py::str(components[i]))) << (i!=dim-1 ? ", " : "]");
        return out.str();
    }

    std::string __repr__() {
        std::stringstream out;
        out << "vector(";
        for (int i = 0; i < dim; ++i) out << std::string(py::extract<std::string>(py::str(components[i]))) << (i!=dim-1 ? ", " : ")");
        return out.str();
    }

    bool __eq__(const PyVec& v) {
        if (dim != v.dim) return false;
        for (int i = 0; i < v.dim; ++i)
            if (components[i] != v.components[i]) return false;
        return true;
    }

    bool __ne__(const PyVec& v) { return ! __eq__(v); }

    PyVec __add__(const PyVec& v) {
        if (v.dim != dim) {
            PyErr_SetString(PyExc_TypeError, "incompatibile vector dimensions");
            throw py::error_already_set();
        }
        py::list result;
        for (int i = 0; i < dim; i++) result.append(components[i] + v.components[i]);
        return PyVec(result);
    }

    PyVec __sub__(const PyVec& v) {
        if (v.dim != dim) {
            PyErr_SetString(PyExc_TypeError, "incompatibile vector dimensions");
            throw py::error_already_set();
        }
        py::list result;
        for (int i = 0; i < dim; i++) result.append(components[i] - v.components[i]);
        return PyVec(result);
    }

    PyVec __neg__() {
        py::list result;
        py::object zero = py::object(0.);
        for (int i = 0; i < dim; i++) result.append(zero - components[i]);
        return PyVec(result);
    }

    PyVec __mul__(py::object c) {
        py::list result;
        for (int i = 0; i < dim; i++) result.append(components[i] * c);
        return PyVec(result);
    }

    PyVec __rmul__(py::object c) {
        py::list result;
        for (int i = 0; i < dim; i++) result.append(c * components[i]);
        return PyVec(result);
    }

    py::object dot(const PyVec& v) {
        if (v.dim != dim) {
            PyErr_SetString(PyExc_TypeError, "incompatibile vector dimensions");
            throw py::error_already_set();
        }
        py::object result = py::object(0.);
        for (int i = 0; i < dim; i++) result += components[i] * v.components[i];
        return result;
    }

    PyVec conj() {
        py::list result;
        for (int i = 0; i < dim; i++) result.append(components[i].attr("conjugate")());
        return PyVec(result);
    }

    py::object abs2() {
        return this->dot(*this).attr("real");
    }

    double abs() {
        return sqrt(py::extract<double>(abs2()));
    }

    PyVec copy() {
        py::list result;
        for (int i = 0; i < dim; i++) result.append(components[i].attr("conjugate")());
        return PyVec(result);
    }

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


#define pyvec_component_property(name) &PyVec::getComponent<name>, &PyVec::setComponent<name>

void register_vector()
{
    register_vector_class<2,double>();
    register_vector_class<2,dcomplex>();
    register_vector_class<3,double>();
    register_vector_class<3,dcomplex>();

    py::implicitly_convertible<Vec<2,double>,Vec<2,dcomplex>>();
    py::implicitly_convertible<Vec<3,double>,Vec<3,dcomplex>>();

    py::class_<PyVec, shared_ptr<PyVec>> pyvec("vector", __doc__, py::no_init);
    pyvec
        .def("__init__", raw_constructor(&pyvec__init__, 0))
        .def("__getitem__", &PyVec::__getitem__)
        .def("__setitem__", &PyVec::__setitem__)
        .add_property("x", pyvec_component_property('x'))
        .add_property("y", pyvec_component_property('y'))
        .add_property("z", pyvec_component_property('z'))
        .add_property("r", pyvec_component_property('r'))
        .add_property("phi", pyvec_component_property('p'))
        .def("__iter__", &PyVec::__iter__)
        .def("__len__", &PyVec::__len__)
        .def("__str__", &PyVec::__str__)
        .def("__repr__", &PyVec::__repr__)
        .def("__eq__", &PyVec::__eq__)
        .def("__ne__", &PyVec::__ne__)
        .def("__add__", &PyVec::__add__)
        .def("__sub__", &PyVec::__sub__)
        .def("__neg__", &PyVec::__neg__)
        .def("__mul__", &PyVec::__mul__)
        .def("__rmul__", &PyVec::__rmul__)
        .def("__mul__", &PyVec::dot)
        .def("dot", &PyVec::dot)
        .def("conjugate", &PyVec::conj)
        .def("conj", &PyVec::conj)
        .def("abs2", &PyVec::abs2)
        .def("abs", &PyVec::abs)
        .def("__abs__", &PyVec::abs)
        .def("copy", &PyVec::copy)
        .add_property("dtype", &PyVec::dtype)
    ;

    py::scope().attr("vec") = pyvec;

    py::implicitly_convertible<PyVec,Vec<2,double>>();
    py::implicitly_convertible<PyVec,Vec<3,double>>();
    py::implicitly_convertible<PyVec,Vec<2,dcomplex>>();
    py::implicitly_convertible<PyVec,Vec<3,dcomplex>>();
}

}} // namespace plask::python
