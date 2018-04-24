#include "../python_globals.h"
#include "../python_numpy.h"
#include "../python_mesh.h"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/generator_rectangular.h>

#if PY_VERSION_HEX >= 0x03000000
#   define NEXT "__next__"
#else
#   define NEXT "next"
#endif

#define DIM RectangularMeshRefinedGenerator<dim>::DIM

namespace plask { namespace python {

extern AxisNames current_axes;


template <typename T>
shared_ptr<T> __init__empty() {
    return plask::make_shared<T>();
}


template <typename T>
static std::string __str__(const T& self) {
    std::stringstream out;
    out << self;
    return out.str();
}


template <typename To, typename From=To>
static shared_ptr<To> Mesh__init__(const From& from) {
    return plask::make_shared<To>(from);
}


namespace detail {
    struct OrderedAxis_from_Sequence
    {
        OrderedAxis_from_Sequence() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<OrderedAxis>());
        }

        static void* convertible(PyObject* obj) {
            if (!PySequence_Check(obj)) return NULL;
            return obj;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<OrderedAxis>*)data)->storage.bytes;
            py::stl_input_iterator<double> begin(py::object(py::handle<>(py::borrowed(obj)))), end;
            new(storage) OrderedAxis(std::vector<double>(begin, end));
            data->convertible = storage;
        }
    };
}

static py::object OrderedAxis__array__(py::object self, py::object dtype) {
    OrderedAxis* axis = py::extract<OrderedAxis*>(self);
    npy_intp dims[] = { npy_intp(axis->size()) };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)&(*axis->begin()));
    if (arr == nullptr) throw TypeError("cannot create array");
    confirm_array<double>(arr, self, dtype);
    return py::object(py::handle<>(arr));
}

template <typename RectilinearT>
shared_ptr<RectilinearT> Rectilinear__init__seq(py::object seq) {
    py::stl_input_iterator<double> begin(seq), end;
    return plask::make_shared<RectilinearT>(std::vector<double>(begin, end));
}

static std::string OrderedAxis__repr__(const OrderedAxis& self) {
    return "Rectilinear(" + __str__(self) + ")";
}

static double OrderedAxis__getitem__(const OrderedAxis& self, int i) {
    if (i < 0) i += int(self.size());
    if (i < 0 || std::size_t(i) >= self.size()) throw IndexError("axis/mesh index out of range");
    return self[i];
}

static void OrderedAxis__delitem__(OrderedAxis& self, int i) {
    if (i < 0) i += int(self.size());
    if (i < 0 || std::size_t(i) >= self.size()) throw IndexError("axis/mesh index out of range");
    self.removePoint(i);
}

static void OrderedAxis_extend(OrderedAxis& self, py::object sequence) {
    py::stl_input_iterator<double> begin(sequence), end;
    std::vector<double> points(begin, end);
    std::sort(points.begin(), points.end());
    self.addOrderedPoints(points.begin(), points.end());
}


/*namespace detail {
    struct RegularAxisFromTupleOrFloat
    {
        RegularAxisFromTupleOrFloat() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<RegularAxis>());
        }

        static void* convertible(PyObject* obj) {
            if (PyTuple_Check(obj) || PyFloat_Check(obj) || PyInt_Check(obj)) return obj;
            if (PySequence_Check(obj) && PySequence_Length(obj) == 1) return obj;
            return NULL;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<RegularAxis>*)data)->storage.bytes;
            auto tuple = py::object(py::handle<>(py::borrowed(obj)));
            try {
                if (PyFloat_Check(obj) || PyInt_Check(obj)) {
                    double val = py::extract<double>(tuple);
                    new(storage) RegularAxis(val, val, 1);
                } else if (py::len(tuple) == 1) {
                    double val = py::extract<double>(tuple[0]);
                    new(storage) RegularAxis(val, val, 1);
                } else if (py::len(tuple) == 3) {
                    new(storage) RegularAxis(py::extract<double>(tuple[0]), py::extract<double>(tuple[1]), py::extract<unsigned>(tuple[2]));
                } else
                    throw py::error_already_set();
                data->convertible = storage;
            } catch (py::error_already_set) {
                throw TypeError("Must provide either mesh.Regular or a tuple (first[, last=first, count=1])");
            }
        }
    };
}*/

template <typename RegularT>
shared_ptr<RegularT> Regular__init__one_param(double val) {
    return plask::make_shared<RegularT>(val, val, 1);
}

template <typename RegularT>
shared_ptr<RegularT> Regular__init__params(double first, double last, int count) {
    return plask::make_shared<RegularT>(first, last, count);
}

static std::string RegularAxis__repr__(const RegularAxis& self) {
    return format("Regular({0}, {1}, {2})", self.first(), self.last(), self.size());
}

static double RegularAxis__getitem__(const RegularAxis& self, int i) {
    if (i < 0) i += int(self.size());
    if (i < 0 || std::size_t(i) >= self.size()) throw IndexError("axis/mesh index out of range");
    return self[i];
}

static void RegularAxis_resize(RegularAxis& self, int count) {
    self.reset(self.first(), self.last(), count);
}

static void RegularAxis_setFirst(RegularAxis& self, double first) {
    self.reset(first, self.last(), self.size());
}

static void RegularAxis_setLast(RegularAxis& self, double last) {
    self.reset(self.first(), last, self.size());
}


template <typename MeshT, typename AxesT>
static shared_ptr<MeshT> RectangularMesh1D__init__axis(const AxesT& axis) {
    return plask::make_shared<MeshT>(axis);
}


static void RectangularMesh2D__setOrdering(RectangularMesh<2>& self, std::string order) {
    if (order == "best" || order == "optimal") self.setOptimalIterationOrder();
    else if (order == "10") self.setIterationOrder(RectangularMesh<2>::ORDER_10);
    else if (order == "01") self.setIterationOrder(RectangularMesh<2>::ORDER_01);
    else {
        throw ValueError("order must be '01', '10' or 'best'");
    }
}

template <typename MeshT>
static shared_ptr<MeshT> RectangularMesh2D__init__empty(std::string order) {
    auto mesh = plask::make_shared<MeshT>();
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<MeshAxis> extract_axis(const py::object& axis) {
    py::extract<shared_ptr<MeshAxis>> convert(axis);
    if (convert.check())
        return convert;
    else if (PySequence_Check(axis.ptr())) {
        py::stl_input_iterator<double> begin(axis), end;
        return plask::make_shared<OrderedAxis>(std::vector<double>(begin, end));
    } else {
        throw TypeError("Wrong type of axis, it must derive from Rectangular1D or be a sequence.");
    }
}

static shared_ptr<RectangularMesh<2>> RectangularMesh2D__init__axes(py::object axis0, py::object axis1, std::string order) {
    auto mesh = plask::make_shared<RectangularMesh<2>>(extract_axis(axis0), extract_axis(axis1));
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

static Vec<2,double> RectangularMesh2D__getitem__(const RectangularMesh<2>& self, py::object index) {
    try {
        int indx = py::extract<int>(index);
        if (indx < 0) indx += int(self.size());
        if (indx < 0 || std::size_t(indx) >= self.size()) throw IndexError("mesh index out of range");
        return self[indx];
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 += int(self.axis[0]->size());
    if (index0 < 0 || index0 >= int(self.axis[0]->size())) {
        throw IndexError("first mesh index ({0}) out of range (0<=index<{1})", index0, self.axis[0]->size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 += int(self.axis[1]->size());
    if (index1 < 0 || index1 >= int(self.axis[1]->size())) {
        throw IndexError("second mesh index ({0}) out of range (0<=index<{1})", index1, self.axis[1]->size());
    }
    return self(index0, index1);
}

static std::string RectangularMesh2D__getOrdering(RectangularMesh<2>& self) {
    return (self.getIterationOrder() == RectangularMesh<2>::ORDER_10) ? "10" : "01";
}

void RectangularMesh3D__setOrdering(RectangularMesh<3>& self, std::string order) {
    if (order == "best" || order == "optimal") self.setOptimalIterationOrder();
    else if (order == "012") self.setIterationOrder(RectangularMesh<3>::ORDER_012);
    else if (order == "021") self.setIterationOrder(RectangularMesh<3>::ORDER_021);
    else if (order == "102") self.setIterationOrder(RectangularMesh<3>::ORDER_102);
    else if (order == "120") self.setIterationOrder(RectangularMesh<3>::ORDER_120);
    else if (order == "201") self.setIterationOrder(RectangularMesh<3>::ORDER_201);
    else if (order == "210") self.setIterationOrder(RectangularMesh<3>::ORDER_210);
    else {
        throw ValueError("order must be any permutation of '012' or 'best'");
    }
}

std::string RectangularMesh3D__getOrdering(RectangularMesh<3>& self) {
    switch (self.getIterationOrder()) {
        case RectangularMesh<3>::ORDER_012: return "012";
        case RectangularMesh<3>::ORDER_021: return "021";
        case RectangularMesh<3>::ORDER_102: return "102";
        case RectangularMesh<3>::ORDER_120: return "120";
        case RectangularMesh<3>::ORDER_201: return "201";
        case RectangularMesh<3>::ORDER_210: return "210";
    }
    return "unknown";
}

template <typename MeshT>
shared_ptr<MeshT> RectangularMesh3D__init__empty(std::string order) {
    auto mesh = plask::make_shared<MeshT>();
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectangularMesh<3>> RectangularMesh3D__init__axes(py::object axis0, py::object axis1, py::object axis2, std::string order) {
    auto mesh = plask::make_shared<RectangularMesh<3>>(extract_axis(axis0), extract_axis(axis1), extract_axis(axis2));
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}

template <typename MeshT>
Vec<3,double> RectangularMesh3D__getitem__(const MeshT& self, py::object index) {
    try {
        int indx = py::extract<int>(index);
        if (indx < 0) indx += int(self.size());
        if (indx < 0 || std::size_t(indx) >= self.size()) throw IndexError("mesh index out of range");
        return self[indx];
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 += int(self.axis[0]->size());
    if (index0 < 0 || index0 >= int(self.axis[0]->size())) {
        throw IndexError("first mesh index ({0}) out of range (0<=index<{1})", index0, self.axis[0]->size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 += int(self.axis[1]->size());
    if (index1 < 0 || index1 >= int(self.axis[1]->size())) {
        throw IndexError("second mesh index ({0}) out of range (0<=index<{1})", index1, self.axis[1]->size());
    }
    int index2 = py::extract<int>(index[2]);
    if (index2 < 0) index2 = int(self.axis[2]->size());
    if (index2 < 0 || index2 >= int(self.axis[2]->size())) {
        throw IndexError("third mesh index ({0}) out of range (0<=index<{1})", index2, self.axis[2]->size());
    }
    return self(index0, index1, index2);
}





shared_ptr<RectangularMesh<2>> RectilinearMesh2D__init__geometry(const shared_ptr<GeometryObjectD<2>>& geometry, std::string order) {
    auto mesh = RectangularMesh2DSimpleGenerator().generate_t< RectangularMesh<2> >(geometry);
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectangularMesh<3>> RectilinearMesh3D__init__geometry(const shared_ptr<GeometryObjectD<3>>& geometry, std::string order) {
    auto mesh = RectangularMesh3DSimpleGenerator().generate_t< RectangularMesh<3> >(geometry);
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}


namespace detail {

    template <typename T, int dim, typename GT>
    struct AxisParamProxy {

        typedef AxisParamProxy<T,dim,GT> ThisT;

        typedef T(GT::*GetF)(typename Primitive<DIM>::Direction)const;
        typedef void(GT::*SetF)(typename Primitive<DIM>::Direction,T);

        GT& obj;
        GetF getter;
        SetF setter;

        AxisParamProxy(GT& obj, GetF get, SetF set):
            obj(obj), getter(get), setter(set) {}

        T get(int i) const { return (obj.*getter)(typename Primitive<DIM>::Direction(i)); }

        T __getitem__(int i) const {
            if (i < 0) i += dim; if (i > dim || i < 0) throw IndexError("tuple index out of range");
            return get(i);
        }

        void set(int i, T v) { (obj.*setter)(typename Primitive<DIM>::Direction(i), v); }

        void __setitem__(int i, T v) {
            if (i < 0) i += dim; if (i > dim || i < 0) throw IndexError("tuple index out of range");
            set(i, v);
        }

        py::tuple __mul__(T f) const;

        py::tuple __div__(T f) const;

        std::string __str__() const;

        struct Iter {
            const ThisT& obj;
            int i;
            Iter(const ThisT& obj): obj(obj), i(-1) {}
            T next() {
                ++i; if (i == dim) throw StopIteration(""); return obj.get(i);
            }
        };

        shared_ptr<Iter> __iter__() {
            return plask::make_shared<Iter>(*this);
        }

        // even if unused, scope argument is important as it sets python scope
        static void register_proxy(py::scope /*scope*/) {
            py::class_<ThisT, shared_ptr<ThisT>, boost::noncopyable> cls("_Proxy", py::no_init); cls
                .def("__getitem__", &ThisT::__getitem__)
                .def("__setitem__", &ThisT::__setitem__)
                .def("__mul__", &ThisT::__mul__)
                .def("__div__", &ThisT::__div__)
                .def("__truediv__", &ThisT::__div__)
                .def("__floordiv__", &ThisT::__div__)
                .def("__iter__", &ThisT::__iter__, py::with_custodian_and_ward_postcall<0,1>())
                .def("__str__", &ThisT::__str__)
            ;
            py::delattr(py::scope(), "_Proxy");

            py::scope scope2 = cls;
            (void) scope2;   // don't warn about unused variable scope2
            py::class_<Iter, shared_ptr<Iter>, boost::noncopyable>("Iterator", py::no_init)
                .def(NEXT, &Iter::next)
                .def("__iter__", pass_through)
            ;
        }
    };

    template <> py::tuple AxisParamProxy<size_t,2,RectangularMeshDivideGenerator<2>>::__mul__(size_t f) const {
        return py::make_tuple(get(0) * f, get(1) * f);
    }
    template <> py::tuple AxisParamProxy<size_t,3,RectangularMeshDivideGenerator<3>>::__mul__(size_t f) const {
        return py::make_tuple(get(0) * f, get(1) * f, get(2) * f);
    }

    template <> py::tuple AxisParamProxy<size_t,2,RectangularMeshDivideGenerator<2>>::__div__(size_t f) const {
        if (get(0) < f || get(1) < f) throw ValueError("Refinement already too small.");
        return py::make_tuple(get(0) / f, get(1) / f);
    }
    template <> py::tuple AxisParamProxy<size_t,3,RectangularMeshDivideGenerator<3>>::__div__(size_t f) const {
        if (get(0) < f || get(1) < f || get(2) < f) throw ValueError("Refinement already too small.");
        return py::make_tuple(get(0) / f, get(1) / f, get(2) / f);
    }

    template <> std::string AxisParamProxy<size_t,2,RectangularMeshDivideGenerator<2>>::__str__() const {
        return format("({0}, {1})", get(0), get(1));
    }
    template <> std::string AxisParamProxy<size_t,3,RectangularMeshDivideGenerator<3>>::__str__() const {
        return format("({0}, {1}, {2})", get(0), get(1), get(2));
    }

    template <> py::tuple AxisParamProxy<double,2,RectangularMeshSmoothGenerator<2>>::__mul__(double f) const {
        return py::make_tuple(get(0) * f, get(1) * f);
    }
    template <> py::tuple AxisParamProxy<double,3,RectangularMeshSmoothGenerator<3>>::__mul__(double f) const {
        return py::make_tuple(get(0) * f, get(1) * f, get(2) * f);
    }

    template <> py::tuple AxisParamProxy<double,2,RectangularMeshSmoothGenerator<2>>::__div__(double f) const {
        return py::make_tuple(get(0) / f, get(1) / f);
    }
    template <> py::tuple AxisParamProxy<double,3,RectangularMeshSmoothGenerator<3>>::__div__(double f) const {
        return py::make_tuple(get(0) / f, get(1) / f, get(2) / f);
    }

    template <> std::string AxisParamProxy<double,2,RectangularMeshSmoothGenerator<2>>::__str__() const {
        return format("({0}, {1})", get(0), get(1));
    }
    template <> std::string AxisParamProxy<double,3,RectangularMeshSmoothGenerator<3>>::__str__() const {
        return format("({0}, {1}, {2})", get(0), get(1), get(2));
    }


    template <int dim>
    struct DivideGeneratorDivMethods {

        typedef AxisParamProxy<size_t,dim,RectangularMeshDivideGenerator<dim>> ProxyT;

        static shared_ptr<ProxyT> getPre(RectangularMeshDivideGenerator<dim>& self) {
            return plask::make_shared<ProxyT>(self, &RectangularMeshDivideGenerator<dim>::getPreDivision, &RectangularMeshDivideGenerator<dim>::setPreDivision);
        }

        static shared_ptr<ProxyT> getPost(RectangularMeshDivideGenerator<dim>& self) {
            return plask::make_shared<ProxyT>(self, &RectangularMeshDivideGenerator<dim>::getPostDivision, &RectangularMeshDivideGenerator<dim>::setPostDivision);
        }

        static void setPre(RectangularMeshDivideGenerator<dim>& self, py::object val) {
            // try {
            //     size_t v = py::extract<size_t>(val);
            //     for (int i = 0; i < dim; ++i) self.pre_divisions[i] = v;
            // } catch (py::error_already_set) {
            //     PyErr_Clear();
                if (py::len(val) != dim)
                    throw ValueError("Wrong size of 'prediv' ({0} items provided and {1} required)", py::len(val), dim);
                for (int i = 0; i < dim; ++i) self.pre_divisions[i] = py::extract<size_t>(val[i]);
            // }
            self.fireChanged();
        }

        static void setPost(RectangularMeshDivideGenerator<dim>& self, py::object val) {
            // try {
            //     size_t v = py::extract<size_t>(val);
            //     for (int i = 0; i < dim; ++i) self.post_divisions[i] = v;
            // } catch (py::error_already_set) {
            //     PyErr_Clear();
                if (py::len(val) != dim)
                    throw ValueError("Wrong size of 'postdiv' ({0} items provided and {1} required)", py::len(val), dim);
                for (int i = 0; i < dim; ++i) self.post_divisions[i] = py::extract<size_t>(val[i]);
            // }
            self.fireChanged();
        }

        static void register_proxy(py::object scope) { AxisParamProxy<size_t,dim,RectangularMeshDivideGenerator<dim>>::register_proxy(scope); }
    };

    template <>
    struct DivideGeneratorDivMethods<1> {

        static size_t getPre(RectangularMeshDivideGenerator<1>& self) {
            return self.getPreDivision(Primitive<2>::DIRECTION_TRAN);
        }

        static size_t getPost(RectangularMeshDivideGenerator<1>& self) {
            return self.getPostDivision(Primitive<2>::DIRECTION_TRAN);
        }

        static void setPre(RectangularMeshDivideGenerator<1>& self, py::object val) {
            self.setPreDivision(Primitive<2>::DIRECTION_TRAN, py::extract<size_t>(val));
        }

        static void setPost(RectangularMeshDivideGenerator<1>& self, py::object val) {
            self.setPostDivision(Primitive<2>::DIRECTION_TRAN, py::extract<size_t>(val));
        }

        static void register_proxy(py::object) {}
    };


    template <int dim>
    struct SmoothGeneratorParamMethods {

        typedef AxisParamProxy<double,dim,RectangularMeshSmoothGenerator<dim>> ProxyT;

        static shared_ptr<ProxyT> getSmall(RectangularMeshSmoothGenerator<dim>& self) {
            return plask::make_shared<ProxyT>(self, &RectangularMeshSmoothGenerator<dim>::getFineStep, &RectangularMeshSmoothGenerator<dim>::setFineStep);
        }

        static shared_ptr<ProxyT> getLarge(RectangularMeshSmoothGenerator<dim>& self) {
            return plask::make_shared<ProxyT>(self, &RectangularMeshSmoothGenerator<dim>::getMaxStep, &RectangularMeshSmoothGenerator<dim>::setMaxStep);
        }

        static shared_ptr<ProxyT> getFactor(RectangularMeshSmoothGenerator<dim>& self) {
            return plask::make_shared<ProxyT>(self, &RectangularMeshSmoothGenerator<dim>::getFactor, &RectangularMeshSmoothGenerator<dim>::setFactor);
        }

        static void setSmall(RectangularMeshSmoothGenerator<dim>& self, py::object val) {
            // try {
            //     double v = py::extract<double>(val);
            //     for (int i = 0; i < dim; ++i) self.finestep[i] = v;
            // } catch (py::error_already_set) {
            //     PyErr_Clear();
                if (py::len(val) != dim)
                    throw ValueError("Wrong size of 'small' ({0} items provided and {1} required)", py::len(val), dim);
                for (int i = 0; i < dim; ++i) self.finestep[i] = py::extract<double>(val[i]);
            // }
            self.fireChanged();
        }

        static void setLarge(RectangularMeshSmoothGenerator<dim>& self, py::object val) {
            // try {
            //     double v = py::extract<double>(val);
            //     for (int i = 0; i < dim; ++i) self.finestep[i] = v;
            // } catch (py::error_already_set) {
            //     PyErr_Clear();
                if (py::len(val) != dim)
                    throw ValueError("Wrong size of 'large' ({0} items provided and {1} required)", py::len(val), dim);
                for (int i = 0; i < dim; ++i) self.maxstep[i] = py::extract<double>(val[i]);
            // }
            self.fireChanged();
        }

        static void setFactor(RectangularMeshSmoothGenerator<dim>& self, py::object val) {
            // try {
            //     double v = py::extract<double>(val);
            //     for (int i = 0; i < dim; ++i) self.factor[i] = v;
            // } catch (py::error_already_set) {
            //     PyErr_Clear();
                if (py::len(val) != dim)
                    throw ValueError("Wrong size of 'factor' ({0} items provided and {1} required)", py::len(val), dim);
                for (int i = 0; i < dim; ++i) self.factor[i] = py::extract<double>(val[i]);
            // }
            self.fireChanged();
        }

        static void register_proxy(py::object scope) { AxisParamProxy<double,dim,RectangularMeshSmoothGenerator<dim>>::register_proxy(scope); }
    };

    template <>
    struct SmoothGeneratorParamMethods<1> {

        static double getSmall(RectangularMeshSmoothGenerator<1>& self) {
            return self.getFineStep(Primitive<2>::DIRECTION_TRAN);
        }

        static double getLarge(RectangularMeshSmoothGenerator<1>& self) {
            return self.getMaxStep(Primitive<2>::DIRECTION_TRAN);
        }

        static double getFactor(RectangularMeshSmoothGenerator<1>& self) {
            return self.getFactor(Primitive<2>::DIRECTION_TRAN);
        }

        static void setSmall(RectangularMeshSmoothGenerator<1>& self, py::object val) {
            self.setFineStep(Primitive<2>::DIRECTION_TRAN, py::extract<double>(val));
        }

        static void setLarge(RectangularMeshSmoothGenerator<1>& self, py::object val) {
            self.setMaxStep(Primitive<2>::DIRECTION_TRAN, py::extract<double>(val));
        }

        static void setFactor(RectangularMeshSmoothGenerator<1>& self, py::object val) {
            self.setFactor(Primitive<2>::DIRECTION_TRAN, py::extract<double>(val));
        }

        static void register_proxy(py::object) {}
    };
}

template <int dim>
void RectangularMeshRefinedGenerator_addRefinement1(RectangularMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<DIM>& object, const PathHints& path, double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i), dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), path, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_addRefinement2(RectangularMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<DIM>& object, double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i), dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), position);
}

template <int dim>
void RectangularMeshRefinedGenerator_addRefinement3(RectangularMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObject::Subtree subtree, double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i), subtree, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_addRefinement4(RectangularMeshDivideGenerator<dim>& self, const std::string& axis, Path path, double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i), path, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinement1(RectangularMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<DIM>& object, const PathHints& path, double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i), dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), path, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinement2(RectangularMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<DIM>& object, double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i), dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), position);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinement3(RectangularMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObject::Subtree subtree, double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i), subtree, position);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinement4(RectangularMeshDivideGenerator<dim>& self, const std::string& axis, Path path, double position) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i), path, position);
}


template <int dim>
void RectangularMeshRefinedGenerator_removeRefinements1(RectangularMeshDivideGenerator<dim>& self, GeometryObjectD<DIM>& object, const PathHints& path) {
    self.removeRefinements(dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), path);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinements2(RectangularMeshDivideGenerator<dim>& self, const Path& path) {
    self.removeRefinements(path);
}

template <int dim>
void RectangularMeshRefinedGenerator_removeRefinements3(RectangularMeshDivideGenerator<dim>& self, const GeometryObject::Subtree& subtree) {
    self.removeRefinements(subtree);
}

template <int dim>
py::dict RectangularMeshRefinedGenerator_listRefinements(const RectangularMeshDivideGenerator<dim>& self, const std::string& axis) {
    int i = int(current_axes[axis]) - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name {0}.", axis);
    py::dict refinements;
    for (auto refinement: self.getRefinements(typename Primitive<DIM>::Direction(i))) {
        py::object object { const_pointer_cast<GeometryObjectD<DIM>>(refinement.first.first.lock()) };
        auto pth = refinement.first.second;
        py::object path;
        if (pth.hintFor.size() != 0) path = py::object(pth);
        py::list refs;
        for (auto x: refinement.second) {
            refs.append(x);
        }
        refinements[py::make_tuple(object, path)] = refs;
    }
    return refinements;
}

template <int dim, typename RegisterT>
static void register_refined_generator_base(RegisterT& cls) {
        cls
        .add_property("aspect", &RectangularMeshDivideGenerator<dim>::getAspect, &RectangularMeshDivideGenerator<dim>::setAspect, u8"Maximum aspect ratio for the elements generated by this generator.")
        .def_readwrite("warn_multiple", &RectangularMeshDivideGenerator<dim>::warn_multiple, u8"Warn if refining path points to more than one object")
        .def_readwrite("warn_missing", &RectangularMeshDivideGenerator<dim>::warn_missing, u8"Warn if refining path does not point to any object")
        .def_readwrite("warn_ouside", &RectangularMeshDivideGenerator<dim>::warn_outside, u8"Warn if refining line is outside of its object")
        .def("add_refinement", &RectangularMeshRefinedGenerator_addRefinement1<dim>, u8"Add a refining line inside the object",
            (py::arg("axis"), "object", "path", "at"))
        .def("add_refinement", &RectangularMeshRefinedGenerator_addRefinement2<dim>, u8"Add a refining line inside the object",
            (py::arg("axis"), "object", "at"))
        .def("add_refinement", &RectangularMeshRefinedGenerator_addRefinement3<dim>, u8"Add a refining line inside the object",
            (py::arg("axis"), "subtree", "at"))
        .def("add_refinement", &RectangularMeshRefinedGenerator_addRefinement4<dim>, u8"Add a refining line inside the object",
            (py::arg("axis"), "path", "at"))
        .def("remove_refinement", &RectangularMeshRefinedGenerator_removeRefinement1<dim>, u8"Remove the refining line from the object",
            (py::arg("axis"), "object", "path", "at"))
        .def("remove_refinement", &RectangularMeshRefinedGenerator_removeRefinement2<dim>, u8"Remove the refining line from the object",
            (py::arg("axis"), "object", "at"))
        .def("remove_refinement", &RectangularMeshRefinedGenerator_removeRefinement3<dim>, u8"Remove the refining line from the object",
            (py::arg("axis"), "subtree", "at"))
        .def("remove_refinement", &RectangularMeshRefinedGenerator_removeRefinement4<dim>, u8"Remove the refining line from the object",
            (py::arg("axis"), "path", "at"))
        .def("remove_refinements", &RectangularMeshRefinedGenerator_removeRefinements1<dim>, u8"Remove the all refining lines from the object",
            (py::arg("object"), py::arg("path")=py::object()))
        .def("remove_refinements", &RectangularMeshRefinedGenerator_removeRefinements2<dim>, u8"Remove the all refining lines from the object",
            py::arg("path"))
        .def("remove_refinements", &RectangularMeshRefinedGenerator_removeRefinements3<dim>, u8"Remove the all refining lines from the object",
            py::arg("subtree"))
        .def("clear_refinements", &RectangularMeshDivideGenerator<dim>::clearRefinements, u8"Clear all refining lines",
            py::arg("subtree"))
        .def("get_refinements", &RectangularMeshRefinedGenerator_listRefinements<dim>, py::arg("axis"),
            u8"Get list of all the refinements defined for this generator for specified axis"
        )
    ;
}

template <int dim>
shared_ptr<RectangularMeshDivideGenerator<dim>> RectangularMeshDivideGenerator__init__(py::object prediv, py::object postdiv, double aspect, bool gradual,
                                                                                       bool warn_multiple, bool warn_missing, bool warn_outside) {
    auto result = plask::make_shared<RectangularMeshDivideGenerator<dim>>();
    if (prediv != py::object()) detail::DivideGeneratorDivMethods<dim>::setPre(*result, prediv);
    if (postdiv != py::object()) detail::DivideGeneratorDivMethods<dim>::setPost(*result, postdiv);
    result->gradual = gradual;
    result->aspect = aspect;
    result->warn_multiple = warn_multiple;
    result->warn_missing = warn_missing;
    result->warn_outside = warn_outside;
    return result;
};

template <int dim>
void register_divide_generator() {
     py::class_<RectangularMeshDivideGenerator<dim>, shared_ptr<RectangularMeshDivideGenerator<dim>>,
                   py::bases<MeshGeneratorD<dim>>, boost::noncopyable>
            dividecls("DivideGenerator",
            format(u8"Generator of Rectilinear{0}D mesh by simple division of the geometry.\n\n"
            u8"DivideGenerator()\n"
            u8"    create generator without initial division of geometry objects", dim).c_str(), py::no_init);
            register_refined_generator_base<dim>(dividecls); dividecls
            .def("__init__", py::make_constructor(&RectangularMeshDivideGenerator__init__<dim>, py::default_call_policies(),
                 (py::arg("prediv")=py::object(), py::arg("postdiv")=py::object(), py::arg("aspect")=0, py::arg("gradual")=true,
                  py::arg("warn_multiple")=true, py::arg("warn_missing")=true, py::arg("warn_outside")=true)))
            .add_property("gradual", &RectangularMeshDivideGenerator<dim>::getGradual, &RectangularMeshDivideGenerator<dim>::setGradual, "Limit maximum adjacent objects size change to the factor of two.")
        ;
    py::implicitly_convertible<shared_ptr<RectangularMeshDivideGenerator<dim>>, shared_ptr<const RectangularMeshDivideGenerator<dim>>>();

        if (dim != 1) dividecls
            .add_property("prediv",
                        py::make_function(&detail::DivideGeneratorDivMethods<dim>::getPre, py::with_custodian_and_ward_postcall<0,1>()),
                        &detail::DivideGeneratorDivMethods<dim>::setPre,
                        u8"initial division of all geometry objects")
            .add_property("postdiv",
                        py::make_function(&detail::DivideGeneratorDivMethods<dim>::getPost, py::with_custodian_and_ward_postcall<0,1>()),
                        &detail::DivideGeneratorDivMethods<dim>::setPost,
                        u8"final division of all geometry objects")
        ; else dividecls
            .add_property("prediv",
                        &detail::DivideGeneratorDivMethods<dim>::getPre,
                        &detail::DivideGeneratorDivMethods<dim>::setPre,
                        u8"initial division of all geometry objects")
            .add_property("postdiv",
                        &detail::DivideGeneratorDivMethods<dim>::getPost,
                        &detail::DivideGeneratorDivMethods<dim>::setPost,
                        u8"final division of all geometry objects")
        ;

        detail::DivideGeneratorDivMethods<dim>::register_proxy(dividecls);
}


template <int dim>
shared_ptr<RectangularMeshSmoothGenerator<dim>> RectangularMeshSmoothGenerator__init__(py::object small_, py::object large, py::object factor,
                                                                                       double aspect,
                                                                                       bool warn_multiple, bool warn_missing, bool warn_outside) {
    auto result = plask::make_shared<RectangularMeshSmoothGenerator<dim>>();
    if (small_ != py::object()) detail::SmoothGeneratorParamMethods<dim>::setSmall(*result, small_);
    if (large != py::object()) detail::SmoothGeneratorParamMethods<dim>::setLarge(*result, large);
    if (factor != py::object()) detail::SmoothGeneratorParamMethods<dim>::setFactor(*result, factor);
    result->aspect = aspect;
    result->warn_multiple = warn_multiple;
    result->warn_missing = warn_missing;
    result->warn_outside = warn_outside;
    return result;
}

static shared_ptr<RectangularMesh2DRegularGenerator> RectangularMesh2DRegularGenerator__init__1(double spacing) {
    return make_shared<RectangularMesh2DRegularGenerator>(spacing);
}

static shared_ptr<RectangularMesh2DRegularGenerator> RectangularMesh2DRegularGenerator__init__2(double spacing0, double spacing1) {
    return make_shared<RectangularMesh2DRegularGenerator>(spacing0, spacing1);
}

static shared_ptr<RectangularMesh3DRegularGenerator> RectangularMesh3DRegularGenerator__init__1(double spacing) {
    return make_shared<RectangularMesh3DRegularGenerator>(spacing);
}

static shared_ptr<RectangularMesh3DRegularGenerator> RectangularMesh3DRegularGenerator__init__3(double spacing0, double spacing1, double spacing2) {
    return make_shared<RectangularMesh3DRegularGenerator>(spacing0, spacing1, spacing2);
}


template <int dim>
void register_smooth_generator() {
     py::class_<RectangularMeshSmoothGenerator<dim>, shared_ptr<RectangularMeshSmoothGenerator<dim>>,
                   py::bases<MeshGeneratorD<dim>>, boost::noncopyable>
            dividecls("SmoothGenerator",
            format(u8"Generator of Rectilinear{0}D mesh with dense sampling at edges and smooth change of element size.\n\n"
            u8"SmoothGenerator()\n"
            u8"    create generator without initial division of geometry objects", dim).c_str(), py::no_init);
            register_refined_generator_base<dim>(dividecls); dividecls
            .def("__init__", py::make_constructor(&RectangularMeshSmoothGenerator__init__<dim>, py::default_call_policies(),
                 (py::arg("small")=py::object(), py::arg("large")=py::object(), py::arg("factor")=py::object(), py::arg("aspect")=0,
                  py::arg("warn_multiple")=true, py::arg("warn_missing")=true, py::arg("warn_outside")=true)))
        ;
    py::implicitly_convertible<shared_ptr<RectangularMeshSmoothGenerator<dim>>, shared_ptr<const RectangularMeshSmoothGenerator<dim>>>();

        if (dim != 1) dividecls
            .add_property("small",
                        py::make_function(&detail::SmoothGeneratorParamMethods<dim>::getSmall, py::with_custodian_and_ward_postcall<0,1>()),
                        &detail::SmoothGeneratorParamMethods<dim>::setSmall, u8"small size of mesh elements near object edges along each axis")
            .add_property("large",
                        py::make_function(&detail::SmoothGeneratorParamMethods<dim>::getLarge, py::with_custodian_and_ward_postcall<0,1>()),
                        &detail::SmoothGeneratorParamMethods<dim>::setLarge, u8"maximum size of mesh elements along each axis")
            .add_property("factor",
                        py::make_function(&detail::SmoothGeneratorParamMethods<dim>::getFactor, py::with_custodian_and_ward_postcall<0,1>()),
                        &detail::SmoothGeneratorParamMethods<dim>::setFactor, u8"factor by which element sizes increase along each axis")
        ; else dividecls
            .add_property("small",
                        &detail::SmoothGeneratorParamMethods<dim>::getSmall, &detail::SmoothGeneratorParamMethods<dim>::setSmall,
                        u8"small size of mesh elements near object edges along each axis")
            .add_property("large",
                        &detail::SmoothGeneratorParamMethods<dim>::getLarge, &detail::SmoothGeneratorParamMethods<dim>::setLarge,
                        u8"maximum size of mesh elements along each axis")
            .add_property("factor",
                        &detail::SmoothGeneratorParamMethods<dim>::getFactor, &detail::SmoothGeneratorParamMethods<dim>::setFactor,
                        u8"factor by which element sizes increase along each axis")
        ;

        detail::SmoothGeneratorParamMethods<dim>::register_proxy(dividecls);
}


void register_mesh_rectangular()
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    py::class_<MeshAxis, shared_ptr<MeshAxis>, py::bases<MeshD<1>>, boost::noncopyable>
            ("Axis", u8"Base class for all 1D meshes (used as axes by 2D and 3D rectangular meshes).",
             py::no_init)
            .def("get_midpoints", &MeshAxis::getMidpointsMesh, u8"Get new mesh with points in the middles of elements of this mesh")

    ;

    py::class_<OrderedAxis, shared_ptr<OrderedAxis>, py::bases<MeshAxis>> rectilinear1d("Ordered",
        u8"One-dimesnional rectilinear mesh, used also as rectangular mesh axis\n\n"
        u8"Ordered()\n    create empty mesh\n\n"
        u8"Ordered(points)\n    create mesh filled with points provides in sequence type"
        );
    rectilinear1d.def("__init__", py::make_constructor(&__init__empty<OrderedAxis>))
        .def("__init__", py::make_constructor(&Rectilinear__init__seq<OrderedAxis>, py::default_call_policies(), (py::arg("points"))))
        .def("__getitem__", &OrderedAxis__getitem__)
        .def("__delitem__", &OrderedAxis__delitem__)
        .def("__str__", &__str__<OrderedAxis>)
        .def("__repr__", &OrderedAxis__repr__)
        .def("__array__", &OrderedAxis__array__, py::arg("dtype")=py::object())
        .def("insert", (bool(OrderedAxis::*)(double))&OrderedAxis::addPoint, "Insert point into the mesh", (py::arg("point")))
        .def("extend", &OrderedAxis_extend, "Insert points from the sequence to the mesh", (py::arg("points")))
        .def(py::self == py::self)
        .def("__iter__", py::range(&OrderedAxis::begin, &OrderedAxis::end))
    ;
    detail::OrderedAxis_from_Sequence();
    py::implicitly_convertible<shared_ptr<OrderedAxis>, shared_ptr<const OrderedAxis>>();

    {
        py::scope scope = rectilinear1d;
        (void) scope;   // don't warn about unused variable scope

        py::class_<OrderedMesh1DSimpleGenerator, shared_ptr<OrderedMesh1DSimpleGenerator>,
                   py::bases<MeshGeneratorD<1>>, boost::noncopyable>("SimpleGenerator",
            u8"Generator of ordered 1D mesh with lines at transverse edges of all objects.\n\n"
            u8"SimpleGenerator()\n    create generator")
        ;
        py::implicitly_convertible<shared_ptr<OrderedMesh1DSimpleGenerator>, shared_ptr<const OrderedMesh1DSimpleGenerator>>();

        py::class_<OrderedMesh1DRegularGenerator, shared_ptr<OrderedMesh1DRegularGenerator>,
                   py::bases<MeshGeneratorD<1>>, boost::noncopyable>("RegularGenerator",
            u8"Generator of ordered 1D mesh with lines at transverse edges of all objects\n"
            u8"and fine regular division of each object with spacing approximately equal to\n"
            u8"specified spacing\n\n"
            u8"RegularGenerator(spacing)\n    create generator",
            py::init<double>(py::arg("spacing")))
        ;
        py::implicitly_convertible<shared_ptr<OrderedMesh1DRegularGenerator>, shared_ptr<const OrderedMesh1DRegularGenerator>>();

        register_divide_generator<1>();
        register_smooth_generator<1>();
    }


    py::class_<RegularAxis, shared_ptr<RegularAxis>, py::bases<MeshAxis>>("Regular",
        u8"One-dimesnional regular mesh, used also as rectangular mesh axis\n\n"
        u8"Regular()\n    create empty mesh\n\n"
        u8"Regular(start, stop, num)\n    create mesh of count points equally distributed between start and stop"
        )
        .def("__init__", py::make_constructor(&__init__empty<RegularAxis>))
        .def("__init__", py::make_constructor(&Regular__init__one_param<RegularAxis>, py::default_call_policies(), (py::arg("value"))))
        .def("__init__", py::make_constructor(&Regular__init__params<RegularAxis>, py::default_call_policies(), (py::arg("start"), "stop", "num")))
        .add_property("start", &RegularAxis::first, &RegularAxis_setFirst, u8"Position of the beginning of the mesh")
        .add_property("stop", &RegularAxis::last, &RegularAxis_setLast, u8"Position of the end of the mesh")
        .add_property("step", &RegularAxis::step)
        .def("__getitem__", &RegularAxis__getitem__)
        .def("__str__", &__str__<RegularAxis>)
        .def("__repr__", &RegularAxis__repr__)
        .def("resize", &RegularAxis_resize, u8"Change number of points in this mesh", (py::arg("num")))
        .def(py::self == py::self)
        .def("__iter__", py::range(&RegularAxis::begin, &RegularAxis::end))
    ;
    //detail::RegularAxisFromTupleOrFloat();
    py::implicitly_convertible<RegularAxis, OrderedAxis>();
    py::implicitly_convertible<shared_ptr<RegularAxis>, shared_ptr<const RegularAxis>>();


    py::class_<RectangularMesh<2>, shared_ptr<RectangularMesh<2>>, py::bases<MeshD<2>>> rectangular2D("Rectangular2D",
        u8"Two-dimensional mesh\n\n"
        u8"Rectangular2D(ordering='01')\n    create empty mesh\n\n"
        u8"Rectangular2D(axis0, axis1, ordering='01')\n    create mesh with axes supplied as sequences of numbers\n\n"
        u8"Rectangular2D(geometry, ordering='01')\n    create coarse mesh based on bounding boxes of geometry objects\n\n"
        u8"ordering can be either '01', '10' and specifies ordering of the mesh points (last index changing fastest).",
        py::no_init
        ); rectangular2D
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__empty<RectangularMesh<2>>, py::default_call_policies(), (py::arg("ordering")="01")))
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__axes, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("ordering")="01")))
        .def("__init__", py::make_constructor(&RectilinearMesh2D__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="01")))
        //.def("__init__", py::make_constructor(&Mesh__init__<RectilinearMesh2D,RegularMesh2D>, py::default_call_policies(), py::arg("src")))
        .def("copy", &Mesh__init__<RectangularMesh<2>, RectangularMesh<2>>, u8"Make a copy of this mesh") //TODO should this be a deep copy?
        .add_property("axis0", &RectangularMesh<2>::getAxis0, &RectangularMesh<2>::setAxis0, u8"The first (transverse) axis of the mesh")
        .add_property("axis1", &RectangularMesh<2>::getAxis1, &RectangularMesh<2>::setAxis1, u8"The second (vertical) axis of the mesh")
        .add_property("axis_tran", &RectangularMesh<2>::getAxis0, &RectangularMesh<2>::setAxis0, u8"The first (transverse) axis of the mesh, alias for :attr:`axis0`")
        .add_property("axis_vert", &RectangularMesh<2>::getAxis1, &RectangularMesh<2>::setAxis1, u8"The second (vertical) axis of the mesh, alias for :attr:`axis1`")
        .add_property("major_axis", &RectangularMesh<2>::majorAxis, u8"The slower changing axis")
        .add_property("minor_axis", &RectangularMesh<2>::minorAxis, u8"The quicker changing axis")
        //.def("clear", &RectangularMesh<2>::clear, u8"Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh2D__getitem__)
        .def("index", &RectangularMesh<2>::index, u8"Return single index of the point indexed with index0 and index1", (py::arg("index0"), py::arg("index1")))
        .def("index0", &RectangularMesh<2>::index0, u8"Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectangularMesh<2>::index1, u8"Return index in the second axis of the point with given index", (py::arg("index")))
        .def_readwrite("index_tran", &RectangularMesh<2>::index0, u8"Alias for :attr:`index0`")
        .def_readwrite("index_vert", &RectangularMesh<2>::index1, u8"Alias for :attr:`index1`")
        .def("major_index", &RectangularMesh<2>::majorIndex, u8"Return index in the major axis of the point with given index", (py::arg("index")))
        .def("minor_index", &RectangularMesh<2>::minorIndex, u8"Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("set_optimal_ordering", &RectangularMesh<2>::setOptimalIterationOrder, u8"Set the optimal ordering of the points in this mesh")
        .add_property("ordering", &RectangularMesh2D__getOrdering, &RectangularMesh2D__setOrdering, u8"Ordering of the points in this mesh")
        .def("get_midpoints", &RectangularMesh<2>::getMidpointsMesh, u8"Get new mesh with points in the middles of elements of this mesh")
        .def("Left", &RectangularMesh<2>::getLeftBoundary, u8"Left edge of the mesh for setting boundary conditions").staticmethod("Left")
        .def("Right", &RectangularMesh<2>::getRightBoundary, u8"Right edge of the mesh for setting boundary conditions").staticmethod("Right")
        .def("Top", &RectangularMesh<2>::getTopBoundary, u8"Top edge of the mesh for setting boundary conditions").staticmethod("Top")
        .def("Bottom", &RectangularMesh<2>::getBottomBoundary, u8"Bottom edge of the mesh for setting boundary conditions").staticmethod("Bottom")
        .def("LeftOf", (RectangularMesh<2>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<2>::getLeftOfBoundary,
             u8"Boundary left of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("LeftOf")
        .def("RightOf", (RectangularMesh<2>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<2>::getRightOfBoundary,
             u8"Boundary right of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("RightOf")
        .def("TopOf", (RectangularMesh<2>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<2>::getTopOfBoundary,
             u8"Boundary top of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("TopOf")
        .def("BottomOf", (RectangularMesh<2>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<2>::getBottomOfBoundary,
             u8"Boundary bottom of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("BottomOf")
        .def("Horizontal", (RectangularMesh<2>::Boundary(*)(double,double,double))&RectangularMesh<2>::getHorizontalBoundaryNear,
             u8"Boundary at horizontal line", (py::arg("at"), "start", "stop"))
        .def("Horizontal", (RectangularMesh<2>::Boundary(*)(double))&RectangularMesh<2>::getHorizontalBoundaryNear,
             u8"Boundary at horizontal line", py::arg("at")).staticmethod("Horizontal")
        .def("Vertical", (RectangularMesh<2>::Boundary(*)(double,double,double))&RectangularMesh<2>::getVerticalBoundaryNear,
             u8"Boundary at vertical line", (py::arg("at"), "start", "stop"))
        .def("Vertical", (RectangularMesh<2>::Boundary(*)(double))&RectangularMesh<2>::getVerticalBoundaryNear,
             u8"Boundary at vertical line", py::arg("at")).staticmethod("Vertical")
        .def(py::self == py::self)
    ;
    ExportBoundary<RectangularMesh<2>> { rectangular2D };
    py::implicitly_convertible<shared_ptr<RectangularMesh<2>>, shared_ptr<const RectangularMesh<2>>>();

    {
        py::scope scope = rectangular2D;
        (void) scope;   // don't warn about unused variable scope

        py::class_<RectangularMesh2DSimpleGenerator, shared_ptr<RectangularMesh2DSimpleGenerator>,
                   py::bases<MeshGeneratorD<2>>, boost::noncopyable>("SimpleGenerator",
            u8"Generator of Rectangular2D mesh with lines at edges of all objects.\n\n"
            u8"SimpleGenerator()\n    create generator")
        ;
        py::implicitly_convertible<shared_ptr<RectangularMesh2DSimpleGenerator>, shared_ptr<const RectangularMesh2DSimpleGenerator>>();

        py::class_<RectangularMesh2DRegularGenerator, shared_ptr<RectangularMesh2DRegularGenerator>,
                   py::bases<MeshGeneratorD<2>>, boost::noncopyable>("RegularGenerator",
            u8"Generator of Rectilinear2D mesh with lines at transverse edges of all objects\n"
            u8"and fine regular division of each object with spacing approximately equal to\n"
            u8"specified spacing.\n\n"
            u8"RegularGenerator(spacing)\n"
            u8"    create generator with equal spacing in all directions\n\n"
            u8"RegularGenerator(spacing0, spacing1)\n"
            u8"    create generator with equal spacing\n", py::no_init)
            .def("__init__", py::make_constructor(RectangularMesh2DRegularGenerator__init__1, py::default_call_policies(),
                                                  (py::arg("spacing"))))
            .def("__init__", py::make_constructor(RectangularMesh2DRegularGenerator__init__2, py::default_call_policies(),
                                                  (py::arg("spacing0"), py::arg("spacing1"))))
        ;
        py::implicitly_convertible<shared_ptr<RectangularMesh2DRegularGenerator>, shared_ptr<const RectangularMesh2DRegularGenerator>>();

        register_divide_generator<2>();
        register_smooth_generator<2>();
    }


    py::class_<RectangularMesh<3>, shared_ptr<RectangularMesh<3>>, py::bases<MeshD<3>>> rectangular3D("Rectangular3D",
        u8"Three-dimensional mesh\n\n"
        u8"Rectangular3D(ordering='012')\n    create empty mesh\n\n"
        u8"Rectangular3D(axis0, axis1, axis2, ordering='012')\n    create mesh with axes supplied as mesh.OrderedAxis\n\n"
        u8"Rectangular3D(geometry, ordering='012')\n    create coarse mesh based on bounding boxes of geometry objects\n\n"
        u8"ordering can be any a string containing any permutation of and specifies ordering of the\n"
        u8"mesh points (last index changing fastest).",
        py::no_init
        ); rectangular3D
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__empty<RectangularMesh<3>>, py::default_call_policies(), (py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__axes, py::default_call_policies(), (py::arg("axis0"), "axis1", "axis2", py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&RectilinearMesh3D__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="012")))
        //.def("__init__", py::make_constructor(&Mesh__init__<RectilinearMesh3D, RegularMesh3D>, py::default_call_policies(), py::arg("src")))
        .def("copy", &Mesh__init__<RectangularMesh<3>, RectangularMesh<3>>, "Make a copy of this mesh")
        .add_property("axis0", &RectangularMesh<3>::getAxis0, &RectangularMesh<3>::setAxis0, u8"The first (longitudinal) axis of the mesh")
        .add_property("axis1", &RectangularMesh<3>::getAxis1, &RectangularMesh<3>::setAxis1, u8"The second (transverse) axis of the mesh")
        .add_property("axis2", &RectangularMesh<3>::getAxis2, &RectangularMesh<3>::setAxis2, u8"The third (vertical) axis of the mesh")
        .add_property("axis_long", &RectangularMesh<3>::getAxis0, &RectangularMesh<3>::setAxis0, u8"The first (longitudinal) axis of the mesh, alias for :attr:`axis0`")
        .add_property("axis_tran", &RectangularMesh<3>::getAxis1, &RectangularMesh<3>::setAxis1, u8"The second (transverse) axis of the mesh, alias for :attr:`axis1`")
        .add_property("axis_vert", &RectangularMesh<3>::getAxis2, &RectangularMesh<3>::setAxis2, u8"The third (vertical) axis of the mesh, alias for :attr:`axis2`")
        .add_property("major_axis", &RectangularMesh<3>::majorAxis, u8"The slowest changing axis")
        .add_property("medium_axis", &RectangularMesh<3>::mediumAxis, u8"The middle changing axis")
        .add_property("minor_axis", &RectangularMesh<3>::minorAxis, u8"The quickest changing axis")
        //.def("clear", &RectangularMesh<3>::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh3D__getitem__<RectangularMesh<3>>)
        .def("index", &RectangularMesh<3>::index, (py::arg("index0"), py::arg("index1"), py::arg("index2")),
             "Return single index of the point indexed with index0, index1, and index2")
        .def("index0", &RectangularMesh<3>::index0, u8"Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectangularMesh<3>::index1, u8"Return index in the second axis of the point with given index", (py::arg("index")))
        .def("index2", &RectangularMesh<3>::index2, u8"Return index in the third axis of the point with given index", (py::arg("index")))
        .def_readwrite("index_long", &RectangularMesh<3>::index0, "Alias for :attr:`index0`")
        .def_readwrite("index_tran", &RectangularMesh<3>::index1, "Alias for :attr:`index1`")
        .def_readwrite("index_vert", &RectangularMesh<3>::index2, "Alias for :attr:`index2`")
        .def("major_index", &RectangularMesh<3>::majorIndex, u8"Return index in the major axis of the point with given index", (py::arg("index")))
        .def("middle_index", &RectangularMesh<3>::middleIndex, u8"Return index in the middle axis of the point with given index", (py::arg("index")))
        .def("minor_index", &RectangularMesh<3>::minorIndex, u8"Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("set_optimal_ordering", &RectangularMesh<3>::setOptimalIterationOrder, u8"Set the optimal ordering of the points in this mesh")
        .add_property("ordering", &RectangularMesh3D__getOrdering, &RectangularMesh3D__setOrdering, u8"Ordering of the points in this mesh")
        .def("get_midpoints", &RectangularMesh<3>::getMidpointsMesh, u8"Get new mesh with points in the middles of of elements of this mesh")
        .def("Front", &RectangularMesh<3>::getFrontBoundary, u8"Front side of the mesh for setting boundary conditions").staticmethod("Front")
        .def("Back", &RectangularMesh<3>::getBackBoundary, u8"Back side of the mesh for setting boundary conditions").staticmethod("Back")
        .def("Left", &RectangularMesh<3>::getLeftBoundary, u8"Left side of the mesh for setting boundary conditions").staticmethod("Left")
        .def("Right", &RectangularMesh<3>::getRightBoundary, u8"Right side of the mesh for setting boundary conditions").staticmethod("Right")
        .def("Top", &RectangularMesh<3>::getTopBoundary, u8"Top side of the mesh for setting boundary conditions").staticmethod("Top")
        .def("Bottom", &RectangularMesh<3>::getBottomBoundary, u8"Bottom side of the mesh for setting boundary conditions").staticmethod("Bottom")
        .def("FrontOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getFrontOfBoundary,
             u8"Boundary in front of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("FrontOf")
        .def("BackOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getBackOfBoundary,
             u8"Boundary back of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("BackOf")
        .def("LeftOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getLeftOfBoundary,
             u8"Boundary left of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("LeftOf")
        .def("RightOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getRightOfBoundary,
             u8"Boundary right of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("RightOf")
        .def("TopOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getTopOfBoundary,
             u8"Boundary top of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("TopOf")
        .def("BottomOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getBottomOfBoundary,
             u8"Boundary bottom of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("BottomOf")
        .def(py::self == py::self)
    ;
    ExportBoundary<RectangularMesh<3>> { rectangular3D };
    py::implicitly_convertible<shared_ptr<RectangularMesh<3>>, shared_ptr<const RectangularMesh<3>>>();

    {
        py::scope scope = rectangular3D;
        (void) scope;   // don't warn about unused variable scope

        py::class_<RectangularMesh3DSimpleGenerator, shared_ptr<RectangularMesh3DSimpleGenerator>,
                   py::bases<MeshGeneratorD<3>>, boost::noncopyable>("SimpleGenerator",
            u8"Generator of Rectangular3D mesh with lines at edges of all objects.\n\n"
            u8"SimpleGenerator()\n    create generator")
        ;
        py::implicitly_convertible<shared_ptr<RectangularMesh3DSimpleGenerator>, shared_ptr<const RectangularMesh3DSimpleGenerator>>();

        py::class_<RectangularMesh3DRegularGenerator, shared_ptr<RectangularMesh3DRegularGenerator>,
                   py::bases<MeshGeneratorD<3>>, boost::noncopyable>("RegularGenerator",
            u8"Generator of Rectilinear3D mesh with lines at transverse edges of all objects\n"
            u8"and fine regular division of each object with spacing approximately equal to\n"
            u8"specified spacing\n\n"
            u8"RegularGenerator(spacing)\n"
            u8"    create generator with equal spacing in all directions\n\n"
            u8"RegularGenerator(spacing0, spacing1, spacing2)\n"
            u8"    create generator with equal spacing\n", py::no_init)
            .def("__init__", py::make_constructor(RectangularMesh3DRegularGenerator__init__1, py::default_call_policies(),
                                                  (py::arg("spacing"))))
            .def("__init__", py::make_constructor(RectangularMesh3DRegularGenerator__init__3, py::default_call_policies(),
                                                  (py::arg("spacing0"), py::arg("spacing1"), py::arg("spacing2"))))
        ;
        py::implicitly_convertible<shared_ptr<RectangularMesh3DRegularGenerator>, shared_ptr<const RectangularMesh3DRegularGenerator>>();

        register_divide_generator<3>();
        register_smooth_generator<3>();
    }

}

}} // namespace plask::python
