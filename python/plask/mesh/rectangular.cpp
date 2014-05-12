#include "../python_globals.h"
#include "../python_numpy.h"
#include "../python_mesh.h"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/generator_rectilinear.h>

#if PY_VERSION_HEX >= 0x03000000
#   define NEXT "__next__"
#else
#   define NEXT "next"
#endif

#define DIM RectilinearMeshDivideGenerator<dim>::DIM

namespace plask { namespace python {

extern AxisNames current_axes;


template <typename T>
shared_ptr<T> __init__empty() {
    return make_shared<T>();
}


template <typename T>
static std::string __str__(const T& self) {
    std::stringstream out;
    out << self;
    return out.str();
}


template <typename To, typename From=To>
static shared_ptr<To> Mesh__init__(const From& from) {
    return make_shared<To>(from);
}


namespace detail {
    struct RectilinearAxis_from_Sequence
    {
        RectilinearAxis_from_Sequence() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<RectilinearAxis>());
        }

        static void* convertible(PyObject* obj) {
            if (!PySequence_Check(obj)) return NULL;
            return obj;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<RectilinearAxis>*)data)->storage.bytes;
            py::stl_input_iterator<double> begin(py::object(py::handle<>(py::borrowed(obj)))), end;
            new(storage) RectilinearAxis(std::vector<double>(begin, end));
            data->convertible = storage;
        }
    };
}

static py::object RectilinearAxis__array__(py::object self, py::object dtype) {
    RectilinearAxis* axis = py::extract<RectilinearAxis*>(self);
    npy_intp dims[] = { axis->size() };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)&(*axis->begin()));
    if (arr == nullptr) throw TypeError("cannot create array");
    confirm_array<double>(arr, self, dtype);
    return py::object(py::handle<>(arr));
}

template <typename RectilinearT>
shared_ptr<RectilinearT> Rectilinear__init__seq(py::object seq) {
    py::stl_input_iterator<double> begin(seq), end;
    return make_shared<RectilinearT>(std::vector<double>(begin, end));
}

static std::string RectilinearAxis__repr__(const RectilinearAxis& self) {
    return "Rectilinear(" + __str__(self) + ")";
}

static double RectilinearAxis__getitem__(const RectilinearAxis& self, int i) {
    if (i < 0) i = self.size() + i;
    if (i < 0) throw IndexError("axis/mesh index out of range");
    return self[i];
}

static void RectilinearAxis__delitem__(RectilinearAxis& self, int i) {
    if (i < 0) i = self.size() + i;
    if (i < 0) throw IndexError("axis/mesh index out of range");
    self.removePoint(i);
}

static void RectilinearAxis_extend(RectilinearAxis& self, py::object sequence) {
    py::stl_input_iterator<double> begin(sequence), end;
    std::vector<double> points(begin, end);
    std::sort(points.begin(), points.end());
    self.addOrderedPoints(points.begin(), points.end());
}


namespace detail {
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
}

template <typename RegularT>
shared_ptr<RegularT> Regular__init__one_param(double val) {
    return make_shared<RegularT>(val, val, 1);
}

template <typename RegularT>
shared_ptr<RegularT> Regular__init__params(double first, double last, int count) {
    return make_shared<RegularT>(first, last, count);
}

static std::string RegularAxis__repr__(const RegularAxis& self) {
    return format("Regular(%1%, %2%, %3%)", self.first(), self.last(), self.size());
}

static double RegularAxis__getitem__(const RegularAxis& self, int i) {
    if (i < 0) i = self.size() + i;
    if (i < 0) throw IndexError("axis/mesh index out of range");
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
    return make_shared<MeshT>(axis);
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
    auto mesh = make_shared<MeshT>();
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

static shared_ptr<RectangularMesh<2>> RectangularMesh2D__init__axes(shared_ptr<RectangularAxis> axis0, shared_ptr<RectangularAxis> axis1, std::string order) {
    auto mesh = make_shared<RectangularMesh<2>>(axis0, axis1);
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

static Vec<2,double> RectangularMesh2D__getitem__(const RectangularMesh<2>& self, py::object index) {
    try {
        int indx = py::extract<int>(index);
        if (indx < 0) indx = self.size() + indx;
        if (indx < 0) throw IndexError("mesh index out of range");
        return self[indx];
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 = self.axis0->size() - index0;
    if (index0 < 0 || index0 >= int(self.axis0->size())) {
        throw IndexError("first mesh index (%1%) out of range (0<=index<%2%)", index0, self.axis0->size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 = self.axis1->size() - index1;
    if (index1 < 0 || index1 >= int(self.axis1->size())) {
        throw IndexError("second mesh index (%1%) out of range (0<=index<%2%)", index1, self.axis1->size());
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
    auto mesh = make_shared<MeshT>();
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectangularMesh<3>> RectangularMesh3D__init__axes(shared_ptr<RectangularAxis> axis0, shared_ptr<RectangularAxis> axis1, shared_ptr<RectangularAxis> axis2, std::string order) {
    auto mesh = make_shared<RectangularMesh<3>>(axis0, axis1, axis2);
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}

template <typename MeshT>
Vec<3,double> RectangularMesh3D__getitem__(const MeshT& self, py::object index) {
    try {
        int indx = py::extract<int>(index);
        if (indx < 0) indx = self.size() + indx;
        if (indx < 0) throw IndexError("mesh index out of range");
        return self[indx];
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 = self.axis0->size() - index0;
    if (index0 < 0 || index0 >= int(self.axis0->size())) {
        throw IndexError("first mesh index (%1%) out of range (0<=index<%2%)", index0, self.axis0->size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 = self.axis1->size() - index1;
    if (index1 < 0 || index1 >= int(self.axis1->size())) {
        throw IndexError("second mesh index (%1%) out of range (0<=index<%2%)", index1, self.axis1->size());
    }
    int index2 = py::extract<int>(index[2]);
    if (index2 < 0) index2 = self.axis2->size() - index2;
    if (index2 < 0 || index2 >= int(self.axis2->size())) {
        throw IndexError("third mesh index (%1%) out of range (0<=index<%2%)", index2, self.axis2->size());
    }
    return self(index0, index1, index2);
}





shared_ptr<RectangularMesh<2>> RectilinearMesh2D__init__geometry(const shared_ptr<GeometryObjectD<2>>& geometry, std::string order) {
    auto mesh = RectilinearMesh2DSimpleGenerator().generate_t< RectangularMesh<2> >(geometry);
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectangularMesh<3>> RectilinearMesh3D__init__geometry(const shared_ptr<GeometryObjectD<3>>& geometry, std::string order) {
    auto mesh = RectilinearMesh3DSimpleGenerator().generate_t< RectangularMesh<3> >(geometry);
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}


namespace detail {

    template <int dim>
    struct DivideGeneratorDivProxy {

        typedef DivideGeneratorDivProxy<dim> ThisT;

        typedef size_t (RectilinearMeshDivideGenerator<dim>::*GetF)(typename Primitive<DIM>::Direction)const;
        typedef void (RectilinearMeshDivideGenerator<dim>::*SetF)(typename Primitive<DIM>::Direction,size_t);

        RectilinearMeshDivideGenerator<dim>& obj;
        GetF getter;
        SetF setter;

        DivideGeneratorDivProxy(RectilinearMeshDivideGenerator<dim>& obj, GetF get, SetF set):
            obj(obj), getter(get), setter(set) {}

        size_t get(int i) const { return (obj.*getter)(typename Primitive<DIM>::Direction(i)); }

        size_t __getitem__(int i) const {
            if (i < 0) i += dim; if (i > dim || i < 0) throw IndexError("tuple index out of range");
            return get(i);
        }

        void set(int i, size_t v) { (obj.*setter)(typename Primitive<DIM>::Direction(i), v); }

        void __setitem__(int i, size_t v) {
            if (i < 0) i += dim; if (i > dim || i < 0) throw IndexError("tuple index out of range");
            set(i, v);
        }

        py::tuple __mul__(int f) const;

        py::tuple __div__(int f) const;

        std::string __str__() const;

        struct Iter {
            const ThisT& obj;
            int i;
            Iter(const ThisT& obj): obj(obj), i(-1) {}
            size_t next() {
                ++i; if (i == dim) throw StopIteration(""); return obj.get(i);
            }
        };

        shared_ptr<Iter> __iter__() {
            return make_shared<Iter>(*this);
        }

        static shared_ptr<ThisT> getPre(RectilinearMeshDivideGenerator<dim>& self) {
            return make_shared<ThisT>(self,
                &RectilinearMeshDivideGenerator<dim>::getPreDivision, &RectilinearMeshDivideGenerator<dim>::setPreDivision);
        }

        static shared_ptr<ThisT> getPost(RectilinearMeshDivideGenerator<dim>& self) {
            return make_shared<ThisT>(self,
                &RectilinearMeshDivideGenerator<dim>::getPostDivision, &RectilinearMeshDivideGenerator<dim>::setPostDivision);
        }

        static void setPre(RectilinearMeshDivideGenerator<dim>& self, py::object val) {
            // try {
            //     size_t v = py::extract<size_t>(val);
            //     for (int i = 0; i < dim; ++i) self.pre_divisions[i] = v;
            // } catch (py::error_already_set) {
            //     PyErr_Clear();
                if (py::len(val) != dim)
                    throw ValueError("Wrong size of prediv (%1% elements provided and %2% required)", py::len(val), dim);
                for (int i = 0; i < dim; ++i) self.pre_divisions[i] = py::extract<size_t>(val[i]);
            // }
            self.fireChanged();
        }

        static void setPost(RectilinearMeshDivideGenerator<dim>& self, py::object val) {
            // try {
            //     size_t v = py::extract<size_t>(val);
            //     for (int i = 0; i < dim; ++i) self.post_divisions[i] = v;
            // } catch (py::error_already_set) {
            //     PyErr_Clear();
                if (py::len(val) != dim)
                    throw ValueError("Wrong size of prediv (%1% elements provided and %2% required)", py::len(val), dim);
                for (int i = 0; i < dim; ++i) self.post_divisions[i] = py::extract<size_t>(val[i]);
            // }
            self.fireChanged();
        }

        static void register_proxy(py::scope scope) {
            py::class_<ThisT, shared_ptr<ThisT>, boost::noncopyable> cls("Div", py::no_init); cls
                .def("__getitem__", &ThisT::__getitem__)
                .def("__setitem__", &ThisT::__setitem__)
                .def("__mul__", &ThisT::__mul__)
                .def("__div__", &ThisT::__div__)
                .def("__truediv__", &ThisT::__div__)
                .def("__floordiv__", &ThisT::__div__)
                .def("__iter__", &ThisT::__iter__, py::with_custodian_and_ward_postcall<0,1>())
                .def("__str__", &ThisT::__str__)
            ;

            py::scope scope2 = cls;
            py::class_<Iter, shared_ptr<Iter>, boost::noncopyable>("Iterator", py::no_init)
                .def(NEXT, &Iter::next)
                .def("__iter__", pass_through)
            ;
        }
    };

    template <>
    struct DivideGeneratorDivProxy<1> {

        static size_t getPre(RectilinearMeshDivideGenerator<1>& self) {
            return self.getPreDivision(Primitive<2>::DIRECTION_TRAN);
        }

        static size_t getPost(RectilinearMeshDivideGenerator<1>& self) {
            return self.getPostDivision(Primitive<2>::DIRECTION_TRAN);
        }

        static void setPre(RectilinearMeshDivideGenerator<1>& self, py::object val) {
            self.setPreDivision(Primitive<2>::DIRECTION_TRAN, py::extract<size_t>(val));
        }

        static void setPost(RectilinearMeshDivideGenerator<1>& self, py::object val) {
            self.setPostDivision(Primitive<2>::DIRECTION_TRAN, py::extract<size_t>(val));
        }

        static void register_proxy(py::object) {}
    };



    template <> py::tuple DivideGeneratorDivProxy<2>::__mul__(int f) const {
        return py::make_tuple(get(0) * f, get(1) * f);
    }
    template <> py::tuple DivideGeneratorDivProxy<3>::__mul__(int f) const {
        return py::make_tuple(get(0) * f, get(1) * f, get(2) * f);
    }

    template <> py::tuple DivideGeneratorDivProxy<2>::__div__(int f) const {
        if (get(0) < f || get(1) < f) throw ValueError("Refinement already too small.");
        return py::make_tuple(get(0) / f, get(1) / f);
    }
    template <> py::tuple DivideGeneratorDivProxy<3>::__div__(int f) const {
        if (get(0) < f || get(1) < f || get(2) < f) throw ValueError("Refinement already too small.");
        return py::make_tuple(get(0) / f, get(1) / f, get(2) / f);
    }

    template <> std::string DivideGeneratorDivProxy<2>::__str__() const {
        return format("(%1%, %2%)", get(0), get(1));
    }
    template <> std::string DivideGeneratorDivProxy<3>::__str__() const {
        return format("(%1%, %2%, %3%)", get(0), get(1), get(2));
    }
}

template <int dim>
void RectilinearMeshDivideGenerator_addRefinement1(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<DIM>& object, const PathHints& path, double position) {
    int i = current_axes[axis] - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i), dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), path, position);
}

template <int dim>
void RectilinearMeshDivideGenerator_addRefinement2(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<DIM>& object, double position) {
    int i = current_axes[axis] - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i), dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), position);
}

template <int dim>
void RectilinearMeshDivideGenerator_addRefinement3(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObject::Subtree subtree, double position) {
    int i = current_axes[axis] - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i), subtree, position);
}

template <int dim>
void RectilinearMeshDivideGenerator_addRefinement4(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, Path path, double position) {
    int i = current_axes[axis] - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(typename Primitive<DIM>::Direction(i), path, position);
}

template <int dim>
void RectilinearMeshDivideGenerator_removeRefinement1(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<DIM>& object, const PathHints& path, double position) {
    int i = current_axes[axis] - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i), dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), path, position);
}

template <int dim>
void RectilinearMeshDivideGenerator_removeRefinement2(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<DIM>& object, double position) {
    int i = current_axes[axis] - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i), dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), position);
}

template <int dim>
void RectilinearMeshDivideGenerator_removeRefinement3(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObject::Subtree subtree, double position) {
    int i = current_axes[axis] - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i), subtree, position);
}

template <int dim>
void RectilinearMeshDivideGenerator_removeRefinement4(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, Path path, double position) {
    int i = current_axes[axis] - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(typename Primitive<DIM>::Direction(i), path, position);
}


template <int dim>
void RectilinearMeshDivideGenerator_removeRefinements1(RectilinearMeshDivideGenerator<dim>& self, GeometryObjectD<DIM>& object, const PathHints& path) {
    self.removeRefinements(dynamic_pointer_cast<GeometryObjectD<DIM>>(object.shared_from_this()), path);
}

template <int dim>
void RectilinearMeshDivideGenerator_removeRefinements2(RectilinearMeshDivideGenerator<dim>& self, const Path& path) {
    self.removeRefinements(path);
}

template <int dim>
void RectilinearMeshDivideGenerator_removeRefinements3(RectilinearMeshDivideGenerator<dim>& self, const GeometryObject::Subtree& subtree) {
    self.removeRefinements(subtree);
}

template <int dim>
py::dict RectilinearMeshDivideGenerator_listRefinements(const RectilinearMeshDivideGenerator<dim>& self, const std::string& axis) {
    int i = current_axes[axis] - 3 + DIM;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
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

template <int dim>
shared_ptr<RectilinearMeshDivideGenerator<dim>> RectilinearMeshDivideGenerator__init__(py::object prediv, py::object postdiv, bool gradual,
                                                                                       bool warn_multiple, bool warn_missing, bool warn_outside) {
    auto result = make_shared<RectilinearMeshDivideGenerator<dim>>();
    if (prediv != py::object()) detail::DivideGeneratorDivProxy<dim>::setPre(*result, prediv);
    if (postdiv != py::object()) detail::DivideGeneratorDivProxy<dim>::setPost(*result, postdiv);
    result->gradual = gradual;
    result->warn_multiple = warn_multiple;
    result->warn_missing = warn_missing;
    result->warn_outside = warn_outside;
    return result;
};

template <int dim>
void register_divide_generator() {
     py::class_<RectilinearMeshDivideGenerator<dim>, shared_ptr<RectilinearMeshDivideGenerator<dim>>,
                   py::bases<MeshGeneratorD<dim>>, boost::noncopyable>
            dividecls("DivideGenerator",
            format("Generator of Rectilinear%1%D mesh by simple division of the geometry.\n\n"
            "DivideGenerator()\n"
            "    create generator without initial division of geometry objects", dim).c_str(), py::no_init); dividecls
            .def("__init__", py::make_constructor(&RectilinearMeshDivideGenerator__init__<dim>, py::default_call_policies(),
                 (py::arg("prediv")=py::object(), py::arg("postdiv")=py::object(), py::arg("gradual")=true,
                  py::arg("warn_multiple")=true, py::arg("warn_missing")=true, py::arg("warn_outside")=true)))
            .add_property("gradual", &RectilinearMeshDivideGenerator<dim>::getGradual, &RectilinearMeshDivideGenerator<dim>::setGradual, "Limit maximum adjacent objects size change to the factor of two")
            .def_readwrite("warn_multiple", &RectilinearMeshDivideGenerator<dim>::warn_multiple, "Warn if refining path points to more than one object")
            .def_readwrite("warn_missing", &RectilinearMeshDivideGenerator<dim>::warn_missing, "Warn if refining path does not point to any object")
            .def_readwrite("warn_ouside", &RectilinearMeshDivideGenerator<dim>::warn_outside, "Warn if refining line is outside of its object")
            .def("add_refinement", &RectilinearMeshDivideGenerator_addRefinement1<dim>, "Add a refining line inside the object",
                (py::arg("axis"), "object", "path", "at"))
            .def("add_refinement", &RectilinearMeshDivideGenerator_addRefinement2<dim>, "Add a refining line inside the object",
                (py::arg("axis"), "object", "at"))
            .def("add_refinement", &RectilinearMeshDivideGenerator_addRefinement3<dim>, "Add a refining line inside the object",
                (py::arg("axis"), "subtree", "at"))
            .def("add_refinement", &RectilinearMeshDivideGenerator_addRefinement4<dim>, "Add a refining line inside the object",
                (py::arg("axis"), "path", "at"))
            .def("remove_refinement", &RectilinearMeshDivideGenerator_removeRefinement1<dim>, "Remove the refining line from the object",
                (py::arg("axis"), "object", "path", "at"))
            .def("remove_refinement", &RectilinearMeshDivideGenerator_removeRefinement2<dim>, "Remove the refining line from the object",
                (py::arg("axis"), "object", "at"))
            .def("remove_refinement", &RectilinearMeshDivideGenerator_removeRefinement3<dim>, "Remove the refining line from the object",
                (py::arg("axis"), "subtree", "at"))
            .def("remove_refinement", &RectilinearMeshDivideGenerator_removeRefinement4<dim>, "Remove the refining line from the object",
                (py::arg("axis"), "path", "at"))
            .def("remove_refinements", &RectilinearMeshDivideGenerator_removeRefinements1<dim>, "Remove the all refining lines from the object",
                (py::arg("object"), py::arg("path")=py::object()))
            .def("remove_refinements", &RectilinearMeshDivideGenerator_removeRefinements2<dim>, "Remove the all refining lines from the object",
                py::arg("path"))
            .def("remove_refinements", &RectilinearMeshDivideGenerator_removeRefinements3<dim>, "Remove the all refining lines from the object",
                py::arg("subtree"))
            .def("clear_refinements", &RectilinearMeshDivideGenerator<dim>::clearRefinements, "Clear all refining lines",
                py::arg("subtree"))
            .def("get_refinements", &RectilinearMeshDivideGenerator_listRefinements<dim>, py::arg("axis"),
                "Get list of all the refinements defined for this generator for specified axis"
            )
        ;

        if (dim != 1) dividecls
            .add_property("prediv",
                        py::make_function(&detail::DivideGeneratorDivProxy<dim>::getPre, py::with_custodian_and_ward_postcall<0,1>()),
                        &detail::DivideGeneratorDivProxy<dim>::setPre,
                        "initial division of all geometry objects")
            .add_property("postdiv",
                        py::make_function(&detail::DivideGeneratorDivProxy<dim>::getPost, py::with_custodian_and_ward_postcall<0,1>()),
                        &detail::DivideGeneratorDivProxy<dim>::setPost,
                        "final division of all geometry objects")
        ; else dividecls
            .add_property("prediv",
                        &detail::DivideGeneratorDivProxy<dim>::getPre,
                        &detail::DivideGeneratorDivProxy<dim>::setPre,
                        "initial division of all geometry objects")
            .add_property("postdiv",
                        &detail::DivideGeneratorDivProxy<dim>::getPost,
                        &detail::DivideGeneratorDivProxy<dim>::setPost,
                        "final division of all geometry objects")
        ;

        detail::DivideGeneratorDivProxy<dim>::register_proxy(dividecls);
}





void register_mesh_rectangular()
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    py::class_<RectangularMesh<1>, shared_ptr<RectangularMesh<1>>, py::bases<MeshD<1>>, boost::noncopyable>
            ("Rectangular1D",
             "Base class for all 1D rectangular meshes (used as axes by 2D and 3D rectangular meshes)",
             py::no_init)

    ;

    py::class_<RectilinearAxis, shared_ptr<RectilinearAxis>, py::bases<RectangularMesh<1>>> rectilinear1d("Rectilinear",
        "One-dimesnional rectilinear mesh, used also as rectangular mesh axis\n\n"
        "Rectilinear()\n    create empty mesh\n\n"
        "Rectilinear(points)\n    create mesh filled with points provides in sequence type"
        );
    rectilinear1d.def("__init__", py::make_constructor(&__init__empty<RectilinearAxis>))
        .def("__init__", py::make_constructor(&Rectilinear__init__seq<RectilinearAxis>, py::default_call_policies(), (py::arg("points"))))
        .def("__getitem__", &RectilinearAxis__getitem__)
        .def("__delitem__", &RectilinearAxis__delitem__)
        .def("__str__", &__str__<RectilinearAxis>)
        .def("__repr__", &RectilinearAxis__repr__)
        .def("__array__", &RectilinearAxis__array__, py::arg("dtype")=py::object())
        .def("insert", &RectilinearAxis::addPoint, "Insert point to the mesh", (py::arg("point")))
        .def("extend", &RectilinearAxis_extend, "Insert points from the sequence to the mesh", (py::arg("points")))
        .def(py::self == py::self)
        .def("__iter__", py::range(&RectilinearAxis::begin, &RectilinearAxis::end))
    ;
    detail::RectilinearAxis_from_Sequence();

    {
        py::scope scope = rectilinear1d;

        py::class_<RectilinearMesh1DSimpleGenerator, shared_ptr<RectilinearMesh1DSimpleGenerator>,
                   py::bases<MeshGeneratorD<1>>, boost::noncopyable>("SimpleGenerator",
            "Generator of Rectilinear (1D) mesh with lines at transverse edges of all objects.\n\n"
            "SimpleGenerator()\n    create generator")
        ;

        register_divide_generator<1>();
    }


    py::class_<RegularAxis, shared_ptr<RegularAxis>, py::bases<RectangularMesh<1>>>("Regular",
        "One-dimesnional regular mesh, used also as rectangular mesh axis\n\n"
        "Regular()\n    create empty mesh\n\n"
        "Regular(start, stop, num)\n    create mesh of count points equally distributed between start and stop"
        )
        .def("__init__", py::make_constructor(&__init__empty<RegularAxis>))
        .def("__init__", py::make_constructor(&Regular__init__one_param<RegularAxis>, py::default_call_policies(), (py::arg("value"))))
        .def("__init__", py::make_constructor(&Regular__init__params<RegularAxis>, py::default_call_policies(), (py::arg("start"), "stop", "num")))
        .add_property("start", &RegularAxis::first, &RegularAxis_setFirst, "Position of the beginning of the mesh")
        .add_property("stop", &RegularAxis::last, &RegularAxis_setLast, "Position of the end of the mesh")
        .add_property("step", &RegularAxis::step)
        .def("__getitem__", &RegularAxis__getitem__)
        .def("__str__", &__str__<RegularAxis>)
        .def("__repr__", &RegularAxis__repr__)
        .def("resize", &RegularAxis_resize, "Change number of points in this mesh", (py::arg("num")))
        .def(py::self == py::self)
        .def("__iter__", py::range(&RegularAxis::begin, &RegularAxis::end))
    ;
    detail::RegularAxisFromTupleOrFloat();
    py::implicitly_convertible<RegularAxis, RectilinearAxis>();



    py::class_<RectangularMesh<2>, shared_ptr<RectangularMesh<2>>, py::bases<MeshD<2>>> rectangular2D("Rectangular2D",
        "Two-dimensional mesh\n\n"
        "Rectangular2D(ordering='01')\n    create empty mesh\n\n"
        "Rectangular2D(axis0, axis1, ordering='01')\n    create mesh with axes supplied as sequences of numbers\n\n"
        "Rectangular2D(geometry, ordering='01')\n    create coarse mesh based on bounding boxes of geometry objects\n\n"
        "ordering can be either '01', '10' and specifies ordering of the mesh points (last index changing fastest).",
        py::no_init
        ); rectangular2D
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__empty<RectangularMesh<2>>, py::default_call_policies(), (py::arg("ordering")="01")))
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__axes, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("ordering")="01")))
        .def("__init__", py::make_constructor(&RectilinearMesh2D__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="01")))
        //.def("__init__", py::make_constructor(&Mesh__init__<RectilinearMesh2D,RegularMesh2D>, py::default_call_policies(), py::arg("src")))
        .def("copy", &Mesh__init__<RectangularMesh<2>, RectangularMesh<2>>, "Make a copy of this mesh") //TODO should this be a deep copy?
        .add_property("axis0", &RectangularMesh<2>::getAxis0, &RectangularMesh<2>::setAxis0, "The first (transverse) axis of the mesh")
        .add_property("axis1", &RectangularMesh<2>::getAxis1, &RectangularMesh<2>::setAxis1, "The second (vertical) axis of the mesh")
        .add_property("major_axis", &RectangularMesh<2>::majorAxis, "The slower changing axis")
        .add_property("minor_axis", &RectangularMesh<2>::minorAxis, "The quicker changing axis")
        //.def("clear", &RectangularMesh<2>::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh2D__getitem__)
        .def("index", &RectangularMesh<2>::index, "Return single index of the point indexed with index0 and index1", (py::arg("index0"), py::arg("index1")))
        .def("index0", &RectangularMesh<2>::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectangularMesh<2>::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("major_index", &RectangularMesh<2>::majorIndex, "Return index in the major axis of the point with given index", (py::arg("index")))
        .def("minor_index", &RectangularMesh<2>::minorIndex, "Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("set_optimal_ordering", &RectangularMesh<2>::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .add_property("ordering", &RectangularMesh2D__getOrdering, &RectangularMesh2D__setOrdering, "Ordering of the points in this mesh")
        .def("get_midpoints", &RectangularMesh<2>::getMidpointsMesh, "Get new mesh with points in the middles of objects described by this mesh")
        .def("Left", &RectangularMesh<2>::getLeftBoundary, "Left edge of the mesh for setting boundary conditions").staticmethod("Left")
        .def("Right", &RectangularMesh<2>::getRightBoundary, "Right edge of the mesh for setting boundary conditions").staticmethod("Right")
        .def("Top", &RectangularMesh<2>::getTopBoundary, "Top edge of the mesh for setting boundary conditions").staticmethod("Top")
        .def("Bottom", &RectangularMesh<2>::getBottomBoundary, "Bottom edge of the mesh for setting boundary conditions").staticmethod("Bottom")
        .def("LeftOf", (RectangularMesh<2>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<2>::getLeftOfBoundary,
             "Boundary left of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("LeftOf")
        .def("RightOf", (RectangularMesh<2>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<2>::getRightOfBoundary,
             "Boundary right of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("RightOf")
        .def("TopOf", (RectangularMesh<2>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<2>::getTopOfBoundary,
             "Boundary top of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("TopOf")
        .def("BottomOf", (RectangularMesh<2>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<2>::getBottomOfBoundary,
             "Boundary bottom of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("BottomOf")
        .def("Horizontal", (RectangularMesh<2>::Boundary(*)(double,double,double))&RectangularMesh<2>::getHorizontalBoundaryNear,
             "Boundary at horizontal line", (py::arg("at"), "start", "stop"))
        .def("Horizontal", (RectangularMesh<2>::Boundary(*)(double))&RectangularMesh<2>::getHorizontalBoundaryNear,
             "Boundary at horizontal line", py::arg("at")).staticmethod("Horizontal")
        .def("Vertical", (RectangularMesh<2>::Boundary(*)(double,double,double))&RectangularMesh<2>::getVerticalBoundaryNear,
             "Boundary at vertical line", (py::arg("at"), "start", "stop"))
        .def("Vertical", (RectangularMesh<2>::Boundary(*)(double))&RectangularMesh<2>::getVerticalBoundaryNear,
             "Boundary at vertical line", py::arg("at")).staticmethod("Vertical")
        .def(py::self == py::self)
    ;
    ExportBoundary<RectangularMesh<2>> { rectangular2D };

    {
        py::scope scope = rectangular2D;

        py::class_<RectilinearMesh2DSimpleGenerator, shared_ptr<RectilinearMesh2DSimpleGenerator>,
                   py::bases<MeshGeneratorD<2>>, boost::noncopyable>("SimpleGenerator",
            "Generator of Rectangular2D mesh with lines at edges of all objects.\n\n"
            "SimpleGenerator()\n    create generator")
        ;

        register_divide_generator<2>();
    }


    py::class_<RectangularMesh<3>, shared_ptr<RectangularMesh<3>>, py::bases<MeshD<3>>> rectangular3D("Rectangular3D",
        "Three-dimensional mesh\n\n"
        "Rectangular3D(ordering='012')\n    create empty mesh\n\n"
        "Rectangular3D(axis0, axis1, axis2, ordering='012')\n    create mesh with axes supplied as mesh.RectilinearAxis\n\n"
        "Rectangular3D(geometry, ordering='012')\n    create coarse mesh based on bounding boxes of geometry objects\n\n"
        "ordering can be any a string containing any permutation of and specifies ordering of the\n"
        "mesh points (last index changing fastest).",
        py::no_init
        ); rectangular3D
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__empty<RectangularMesh<3>>, py::default_call_policies(), (py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__axes, py::default_call_policies(), (py::arg("axis0"), "axis1", "axis2", py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&RectilinearMesh3D__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="012")))
        //.def("__init__", py::make_constructor(&Mesh__init__<RectilinearMesh3D, RegularMesh3D>, py::default_call_policies(), py::arg("src")))
        .def("copy", &Mesh__init__<RectangularMesh<3>, RectangularMesh<3>>, "Make a copy of this mesh")
        .add_property("axis0", &RectangularMesh<3>::getAxis0, &RectangularMesh<3>::setAxis0, "The first (longitudinal) axis of the mesh")
        .add_property("axis1", &RectangularMesh<3>::getAxis1, &RectangularMesh<3>::setAxis1, "The second (transverse) axis of the mesh")
        .add_property("axis2", &RectangularMesh<3>::getAxis2, &RectangularMesh<3>::setAxis2, "The third (vertical) axis of the mesh")
        .add_property("major_axis", &RectangularMesh<3>::majorAxis, "The slowest changing axis")
        .add_property("medium_axis", &RectangularMesh<3>::mediumAxis, "The middle changing axis")
        .add_property("minor_axis", &RectangularMesh<3>::minorAxis, "The quickest changing axis")
        //.def("clear", &RectangularMesh<3>::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh3D__getitem__<RectangularMesh<3>>)
        .def("index", &RectangularMesh<3>::index, (py::arg("index0"), py::arg("index1"), py::arg("index2")),
             "Return single index of the point indexed with index0, index1, and index2")
        .def("index0", &RectangularMesh<3>::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectangularMesh<3>::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("index2", &RectangularMesh<3>::index2, "Return index in the third axis of the point with given index", (py::arg("index")))
        .def("major_index", &RectangularMesh<3>::majorIndex, "Return index in the major axis of the point with given index", (py::arg("index")))
        .def("middle_index", &RectangularMesh<3>::middleIndex, "Return index in the middle axis of the point with given index", (py::arg("index")))
        .def("minor_index", &RectangularMesh<3>::minorIndex, "Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("set_optimal_ordering", &RectangularMesh<3>::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .add_property("ordering", &RectangularMesh3D__getOrdering, &RectangularMesh3D__setOrdering, "Ordering of the points in this mesh")
        .def("get_midpoints", &RectangularMesh<3>::getMidpointsMesh, "Get new mesh with points in the middles of objects described by this mesh")
        .def("Front", &RectangularMesh<3>::getFrontBoundary, "Front side of the mesh for setting boundary conditions").staticmethod("Front")
        .def("Back", &RectangularMesh<3>::getBackBoundary, "Back side of the mesh for setting boundary conditions").staticmethod("Back")
        .def("Left", &RectangularMesh<3>::getLeftBoundary, "Left side of the mesh for setting boundary conditions").staticmethod("Left")
        .def("Right", &RectangularMesh<3>::getRightBoundary, "Right side of the mesh for setting boundary conditions").staticmethod("Right")
        .def("Top", &RectangularMesh<3>::getTopBoundary, "Top side of the mesh for setting boundary conditions").staticmethod("Top")
        .def("Bottom", &RectangularMesh<3>::getBottomBoundary, "Bottom side of the mesh for setting boundary conditions").staticmethod("Bottom")
        .def("FrontOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getFrontOfBoundary,
             "Boundary in front of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("FrontOf")
        .def("BackOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getBackOfBoundary,
             "Boundary back of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("BackOf")
        .def("LeftOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getLeftOfBoundary,
             "Boundary left of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("LeftOf")
        .def("RightOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getRightOfBoundary,
             "Boundary right of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("RightOf")
        .def("TopOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getTopOfBoundary,
             "Boundary top of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("TopOf")
        .def("BottomOf", (RectangularMesh<3>::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&RectangularMesh<3>::getBottomOfBoundary,
             "Boundary bottom of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("BottomOf")
        .def(py::self == py::self)
    ;
    ExportBoundary<RectangularMesh<3>> { rectangular3D };

    {
        py::scope scope = rectangular3D;

        py::class_<RectilinearMesh3DSimpleGenerator, shared_ptr<RectilinearMesh3DSimpleGenerator>,
                   py::bases<MeshGeneratorD<3>>, boost::noncopyable>("SimpleGenerator",
            "Generator of Rectangular3D mesh with lines at edges of all objects.\n\n"
            "SimpleGenerator()\n    create generator")
        ;

        register_divide_generator<3>();
    }

}

}} // namespace plask::python
