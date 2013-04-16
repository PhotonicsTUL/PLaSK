#include "../python_globals.h"
#include "../python_numpy.h"
#include "../python_mesh.h"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/rectilinear.h>
#include <plask/mesh/generator_rectilinear.h>
#include <plask/mesh/regular.h>

namespace plask { namespace python {


template <typename T>
static bool __nonempty__(const T& self) { return !self.empty(); }

namespace detail {
    struct Rectilinear1D_from_Sequence
    {
        Rectilinear1D_from_Sequence() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<RectilinearMesh1D>());
        }

        static void* convertible(PyObject* obj) {
            if (!PySequence_Check(obj)) return NULL;
            return obj;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<RectilinearMesh1D>*)data)->storage.bytes;
            py::stl_input_iterator<double> begin(py::object(py::handle<>(py::borrowed(obj)))), end;
            new(storage) RectilinearMesh1D(std::vector<double>(begin, end));
            data->convertible = storage;
        }
    };
}

static py::object Rectilinear1D__array__(py::object self, py::object dtype) {
    RectilinearMesh1D* mesh = py::extract<RectilinearMesh1D*>(self);
    npy_intp dims[] = { mesh->size() };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)&(*mesh->begin()));
    if (arr == nullptr) throw TypeError("cannot create array");
    confirm_array<double>(arr, self, dtype);
    return py::object(py::handle<>(arr));
}

shared_ptr<RectilinearMesh1D> Rectilinear1D__init__empty() {
    return make_shared<RectilinearMesh1D>();
}

shared_ptr<RectilinearMesh1D> Rectilinear1D__init__seq(py::object seq) {
    py::stl_input_iterator<double> begin(seq), end;
    return make_shared<RectilinearMesh1D>(std::vector<double>(begin, end));
}

static std::string Rectilinear1D__str__(const RectilinearMesh1D& self) {
    std::stringstream out;
    out << self;
    return out.str();
}

static std::string Rectilinear1D__repr__(const RectilinearMesh1D& self) {
    return "Rectilinear1D(" + Rectilinear1D__str__(self) + ")";
}

static double Rectilinear1D__getitem__(const RectilinearMesh1D& self, int i) {
    if (i < 0) i = self.size() + i;
    if (i < 0) throw IndexError("mesh.Rectilinear1D index out of range");
    return self[i];
}

static void Rectilinear1D__delitem__(RectilinearMesh1D& self, int i) {
    if (i < 0) i = self.size() + i;
    if (i < 0) throw IndexError("mesh.Rectilinear1D index out of range");
    self.removePoint(i);
}

static void Rectilinear1D_extend(RectilinearMesh1D& self, py::object sequence) {
    py::stl_input_iterator<double> begin(sequence), end;
    std::vector<double> points(begin, end);
    std::sort(points.begin(), points.end());
    self.addOrderedPoints(points.begin(), points.end());
}

namespace detail {
    struct Regular1D_from_Tuple
    {
        Regular1D_from_Tuple() {
            boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<RegularMesh1D>());
        }

        static void* convertible(PyObject* obj) {
            if (PyTuple_Check(obj)) return obj;
            if (PySequence_Check(obj) && PySequence_Length(obj) == 1) return obj;
            return NULL;
        }

        static void construct(PyObject* obj, boost::python::converter::rvalue_from_python_stage1_data* data)
        {
            void* storage = ((boost::python::converter::rvalue_from_python_storage<RegularMesh1D>*)data)->storage.bytes;
            auto tuple = py::object(py::handle<>(py::borrowed(obj)));
            try {
                if (py::len(tuple) == 3)
                    new(storage) RegularMesh1D(py::extract<double>(tuple[0]), py::extract<double>(tuple[1]), py::extract<unsigned>(tuple[2]));
                else if (py::len(tuple) == 1) {
                    double val = py::extract<double>(tuple[0]);
                    new(storage) RegularMesh1D(val, val, 1);
                } else
                    throw py::error_already_set();
                data->convertible = storage;
            } catch (py::error_already_set) {
                throw TypeError("Must provide either mesh.Regular1D or a tuple (first[, last=first, count=1])");
            }
        }
    };
}

shared_ptr<RegularMesh1D> Regular1D__init__empty() {
    return make_shared<RegularMesh1D>();
}

shared_ptr<RegularMesh1D> Regular1D__init__params(double first, double last, int count) {
    return make_shared<RegularMesh1D>(first, last, count);
}

static std::string Regular1D__str__(const RegularMesh1D& self) {
    std::stringstream out;
    out << self;
    return out.str();
}

static std::string Regular1D__repr__(const RegularMesh1D& self) {
    return format("Regular1D(%1%, %2%, %3%)", self.first(), self.last(), self.size());
}

static double Regular1D__getitem__(const RegularMesh1D& self, int i) {
    if (i < 0) i = self.size() + i;
    if (i < 0) throw IndexError("mesh.Regular1D index out of range");
    return self[i];
}

static void RegularMesh1D_resize(RegularMesh1D& self, int count) {
    self.reset(self.first(), self.last(), count);
}

static void RegularMesh1D_setFirst(RegularMesh1D& self, double first) {
    self.reset(first, self.last(), self.size());
}

static void RegularMesh1D_setLast(RegularMesh1D& self, double last) {
    self.reset(self.first(), last, self.size());
}


template <typename To, typename From=To>
static shared_ptr<To> Mesh__init__(const From& from) {
    return make_shared<To>(from);
}


template <typename MeshT>
static void RectangularMesh2D__setOrdering(MeshT& self, std::string order) {
    if (order == "best" || order == "optimal") self.setOptimalIterationOrder();
    else if (order == "10") self.setIterationOrder(MeshT::NORMAL_ORDER);
    else if (order == "01") self.setIterationOrder(MeshT::TRANSPOSED_ORDER);
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

template <typename MeshT, typename AxesT>
static shared_ptr<MeshT> RectangularMesh2D__init__axes(const AxesT& axis0, const AxesT& axis1, std::string order) {
    auto mesh = make_shared<MeshT>(axis0, axis1);
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

template <typename MeshT>
static Vec<2,double> RectangularMesh2D__getitem__(const MeshT& self, py::object index) {
    try {
        int indx = py::extract<int>(index);
        return self[indx];
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 = self.axis0.size() - index0;
    if (index0 < 0 || index0 >= int(self.axis0.size())) {
        throw IndexError("first mesh index (%1%) out of range (0<=index<%2%)", index0, self.axis0.size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 = self.axis1.size() - index1;
    if (index1 < 0 || index1 >= int(self.axis1.size())) {
        throw IndexError("second mesh index (%1%) out of range (0<=index<%2%)", index1, self.axis1.size());
    }
    return self(index0, index1);
}

template <typename MeshT>
static std::string RectangularMesh2D__getOrdering(MeshT& self) {
    return (self.getIterationOrder() == MeshT::NORMAL_ORDER) ? "10" : "01";
}

template <typename MeshT>
void RectangularMesh3D__setOrdering(MeshT& self, std::string order) {
    if (order == "best" || order == "optimal") self.setOptimalIterationOrder();
    else if (order == "012") self.setIterationOrder(MeshT::ORDER_012);
    else if (order == "021") self.setIterationOrder(MeshT::ORDER_021);
    else if (order == "102") self.setIterationOrder(MeshT::ORDER_102);
    else if (order == "120") self.setIterationOrder(MeshT::ORDER_120);
    else if (order == "201") self.setIterationOrder(MeshT::ORDER_201);
    else if (order == "210") self.setIterationOrder(MeshT::ORDER_210);
    else {
        throw ValueError("order must be any permutation of '012' or 'best'");
    }
}

template <typename MeshT>
std::string RectangularMesh3D__getOrdering(MeshT& self) {
    switch (self.getIterationOrder()) {
        case MeshT::ORDER_012: return "012";
        case MeshT::ORDER_021: return "021";
        case MeshT::ORDER_102: return "102";
        case MeshT::ORDER_120: return "120";
        case MeshT::ORDER_201: return "201";
        case MeshT::ORDER_210: return "210";
    }
    return "unknown";
}

template <typename MeshT>
shared_ptr<MeshT> RectangularMesh3D__init__empty(std::string order) {
    auto mesh = make_shared<MeshT>();
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}

template <typename MeshT, typename AxesT>
shared_ptr<MeshT> RectangularMesh3D__init__axes(const AxesT& axis0, const AxesT& axis1, const AxesT& axis2, std::string order) {
    auto mesh = make_shared<MeshT>(axis0, axis1, axis2);
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}

template <typename MeshT>
Vec<3,double> RectangularMesh3D__getitem__(const MeshT& self, py::object index) {
    try {
        int indx = py::extract<int>(index);
        return self[indx];
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 = self.axis0.size() - index0;
    if (index0 < 0 || index0 >= int(self.axis0.size())) {
        throw IndexError("first mesh index (%1%) out of range (0<=index<%2%)", index0, self.axis0.size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 = self.axis1.size() - index1;
    if (index1 < 0 || index1 >= int(self.axis1.size())) {
        throw IndexError("second mesh index (%1%) out of range (0<=index<%2%)", index1, self.axis1.size());
    }
    int index2 = py::extract<int>(index[2]);
    if (index2 < 0) index2 = self.axis2.size() - index2;
    if (index2 < 0 || index2 >= int(self.axis2.size())) {
        throw IndexError("third mesh index (%1%) out of range (0<=index<%2%)", index2, self.axis2.size());
    }
    return self(index0, index1, index2);
}





shared_ptr<RectilinearMesh2D> RectilinearMesh2D__init__geometry(const shared_ptr<GeometryObjectD<2>>& geometry, std::string order) {
    auto mesh = RectilinearMesh2DSimpleGenerator().generate(geometry);
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectilinearMesh3D> RectilinearMesh3D__init__geometry(const shared_ptr<GeometryObjectD<3>>& geometry, std::string order) {
    auto mesh = RectilinearMesh3DSimpleGenerator().generate(geometry);
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}


namespace detail {

    template <int dim>
    struct DivideGeneratorDivProxy {

        typedef DivideGeneratorDivProxy<dim> ThisT;

        typedef size_t (RectilinearMeshDivideGenerator<dim>::*GetF)(typename Primitive<dim>::Direction)const;
        typedef void (RectilinearMeshDivideGenerator<dim>::*SetF)(typename Primitive<dim>::Direction,size_t);

        RectilinearMeshDivideGenerator<dim>& obj;
        GetF getter;
        SetF setter;

        DivideGeneratorDivProxy(RectilinearMeshDivideGenerator<dim>& obj, GetF get, SetF set):
            obj(obj), getter(get), setter(set) {}

        size_t get(int i) const { return (obj.*getter)(typename Primitive<dim>::Direction(i)); }

        size_t __getitem__(int i) const {
            if (i < 0) i += dim; if (i > dim || i < 0) throw IndexError("tuple index out of range");
            return get(i);
        }

        void set(int i, size_t v) { (obj.*setter)(typename Primitive<dim>::Direction(i), v); }

        void __setitem__(int i, size_t v) {
            if (i < 0) i += dim; if (i > dim || i < 0) throw IndexError("tuple index out of range");
            set(i, v);
        }

        py::tuple __mul__(int f) const;

        py::tuple __div__(int f) const;

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
            ;

            py::scope scope2 = cls;
            py::class_<Iter, shared_ptr<Iter>, boost::noncopyable>("Iter", py::no_init)
                .def("next", &Iter::next)
                .def("__iter__", pass_through)
            ;
        }
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
}

template <int dim>
void RectilinearMeshDivideGenerator_addRefinement1(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<dim>& object, const PathHints& path, double position) {
    int i = config.axes[axis] - 3 + dim;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(typename Primitive<dim>::Direction(i), dynamic_pointer_cast<GeometryObjectD<dim>>(object.shared_from_this()), path, position);
}

template <int dim>
void RectilinearMeshDivideGenerator_addRefinement2(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<dim>& object, double position) {
    int i = config.axes[axis] - 3 + dim;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(typename Primitive<dim>::Direction(i), dynamic_pointer_cast<GeometryObjectD<dim>>(object.shared_from_this()), position);
}

template <int dim>
void RectilinearMeshDivideGenerator_addRefinement3(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObject::Subtree subtree, double position) {
    int i = config.axes[axis] - 3 + dim;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(typename Primitive<dim>::Direction(i), subtree, position);
}

template <int dim>
void RectilinearMeshDivideGenerator_addRefinement4(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, Path path, double position) {
    int i = config.axes[axis] - 3 + dim;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(typename Primitive<dim>::Direction(i), path, position);
}

template <int dim>
void RectilinearMeshDivideGenerator_removeRefinement1(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<dim>& object, const PathHints& path, double position) {
    int i = config.axes[axis] - 3 + dim;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(typename Primitive<dim>::Direction(i), dynamic_pointer_cast<GeometryObjectD<dim>>(object.shared_from_this()), path, position);
}

template <int dim>
void RectilinearMeshDivideGenerator_removeRefinement2(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObjectD<dim>& object, double position) {
    int i = config.axes[axis] - 3 + dim;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(typename Primitive<dim>::Direction(i), dynamic_pointer_cast<GeometryObjectD<dim>>(object.shared_from_this()), position);
}

template <int dim>
void RectilinearMeshDivideGenerator_removeRefinement3(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, GeometryObject::Subtree subtree, double position) {
    int i = config.axes[axis] - 3 + dim;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(typename Primitive<dim>::Direction(i), subtree, position);
}

template <int dim>
void RectilinearMeshDivideGenerator_removeRefinement4(RectilinearMeshDivideGenerator<dim>& self, const std::string& axis, Path path, double position) {
    int i = config.axes[axis] - 3 + dim;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(typename Primitive<dim>::Direction(i), path, position);
}


template <int dim>
void RectilinearMeshDivideGenerator_removeRefinements1(RectilinearMeshDivideGenerator<dim>& self, GeometryObjectD<dim>& object, const PathHints& path) {
    self.removeRefinements(dynamic_pointer_cast<GeometryObjectD<dim>>(object.shared_from_this()), path);
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
    int i = config.axes[axis] - 3 + dim;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    py::dict refinements;
    for (auto refinement: self.getRefinements(typename Primitive<dim>::Direction(i))) {
        py::object object { const_pointer_cast<GeometryObjectD<dim>>(refinement.first.first.lock()) };
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
void register_divide_generator() {
     py::class_<RectilinearMeshDivideGenerator<dim>, shared_ptr<RectilinearMeshDivideGenerator<dim>>,
                   py::bases<MeshGeneratorOf<RectangularMesh<dim,RectilinearMesh1D>>>, boost::noncopyable>
            dividecls("DivideGenerator",
            format("Generator of Rectilinear%1%D mesh by simple division of the geometry.\n\n"
            "DivideGenerator()\n"
            "    create generator without initial division of geometry objects", dim).c_str()); dividecls
            .add_property("prediv",
                          py::make_function(&detail::DivideGeneratorDivProxy<dim>::getPre, py::with_custodian_and_ward_postcall<0,1>()),
                          &detail::DivideGeneratorDivProxy<dim>::setPre,
                          "initial division of all geometry objects")
            .add_property("postdiv",
                          py::make_function(&detail::DivideGeneratorDivProxy<dim>::getPost, py::with_custodian_and_ward_postcall<0,1>()),
                          &detail::DivideGeneratorDivProxy<dim>::setPost,
                          "final division of all geometry objects")
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
        detail::DivideGeneratorDivProxy<dim>::register_proxy(dividecls);
}





void register_mesh_rectangular()
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    py::class_<RectilinearMesh1D, shared_ptr<RectilinearMesh1D>, boost::noncopyable>("Rectilinear1D",
        "Rectilinear mesh axis\n\n"
        "Rectilinear1D()\n    create empty mesh\n\n"
        "Rectilinear1D(points)\n    create mesh filled with points provides in sequence type"
        )
        .def("__init__", py::make_constructor(&Rectilinear1D__init__empty))
        .def("__init__", py::make_constructor(&Rectilinear1D__init__seq, py::default_call_policies(), (py::arg("points"))))
        .def("__len__", &RectilinearMesh1D::size)
        .def("__nonzero__", __nonempty__<RectilinearMesh1D>)
        .def("__getitem__", &Rectilinear1D__getitem__)
        .def("__delitem__", &Rectilinear1D__delitem__)
        .def("__str__", &Rectilinear1D__str__)
        .def("__repr__", &Rectilinear1D__repr__)
        .def("__array__", &Rectilinear1D__array__, py::arg("dtype")=py::object())
        .def("insert", &RectilinearMesh1D::addPoint, "Insert point to the mesh", (py::arg("point")))
        .def("extend", &Rectilinear1D_extend, "Insert points from the sequence to the mesh", (py::arg("points")))
        .def(py::self == py::self)
        .def("__iter__", py::range(&RectilinearMesh1D::begin, &RectilinearMesh1D::end))
    ;
    detail::Rectilinear1D_from_Sequence();

    py::class_<RectilinearMesh2D, shared_ptr<RectilinearMesh2D>, py::bases<MeshD<2>>> rectilinear2d("Rectilinear2D",
        "Two-dimensional mesh\n\n"
        "Rectilinear2D(ordering='01')\n    create empty mesh\n\n"
        "Rectilinear2D(axis0, axis1, ordering='01')\n    create mesh with axes supplied as sequences of numbers\n\n"
        "Rectilinear2D(geometry, ordering='01')\n    create coarse mesh based on bounding boxes of geometry objects\n\n"
        "ordering can be either '01', '10' and specifies ordering of the mesh points (last index changing fastest).",
        py::no_init
        ); rectilinear2d
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__empty<RectilinearMesh2D>, py::default_call_policies(), (py::arg("ordering")="01")))
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__axes<RectilinearMesh2D, RectilinearMesh1D>, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("ordering")="01")))
        .def("__init__", py::make_constructor(&RectilinearMesh2D__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="01")))
        .def("__init__", py::make_constructor(&Mesh__init__<RectilinearMesh2D, RegularMesh2D>, py::default_call_policies(), py::arg("src")))
        .def("copy", &Mesh__init__<RectilinearMesh2D, RectilinearMesh2D>, "Make a copy of this mesh")
        .def_readwrite("axis0", &RectilinearMesh2D::axis0, "The first (transverse) axis of the mesh")
        .def_readwrite("axis1", &RectilinearMesh2D::axis1, "The second (vertical) axis of the mesh")
        .add_property("major_axis", py::make_function((RectilinearMesh1D&(RectilinearMesh2D::*)())&RectilinearMesh2D::majorAxis, py::return_internal_reference<>()), "The slower changing axis")
        .add_property("minor_axis", py::make_function((RectilinearMesh1D&(RectilinearMesh2D::*)())&RectilinearMesh2D::minorAxis, py::return_internal_reference<>()), "The quicker changing axis")
        .def("__nonzero__", &__nonempty__<RectilinearMesh2D>, "Return True if the mesh is empty")
        .def("clear", &RectilinearMesh2D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh2D__getitem__<RectilinearMesh2D>)
        .def("index", &RectilinearMesh2D::index, "Return single index of the point indexed with index0 and index1", (py::arg("index0"), py::arg("index1")))
        .def("index0", &RectilinearMesh2D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectilinearMesh2D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("major_index", &RectilinearMesh2D::majorIndex, "Return index in the major axis of the point with given index", (py::arg("index")))
        .def("minor_index", &RectilinearMesh2D::minorIndex, "Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("set_optimal_ordering", &RectilinearMesh2D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .add_property("ordering", &RectangularMesh2D__getOrdering<RectilinearMesh2D>, &RectangularMesh2D__setOrdering<RectilinearMesh2D>, "Ordering of the points in this mesh")
        .def("get_midpoints", &RectilinearMesh2D::getMidpointsMesh, "Get new mesh with points in the middles of objects described by this mesh")
        .add_static_property("left", &RectilinearMesh2D::getLeftBoundary, "Left edge of the mesh for setting boundary conditions")
        .add_static_property("right", &RectilinearMesh2D::getRightBoundary, "Right edge of the mesh for setting boundary conditions")
        .add_static_property("top", &RectilinearMesh2D::getTopBoundary, "Top edge of the mesh for setting boundary conditions")
        .add_static_property("bottom", &RectilinearMesh2D::getBottomBoundary, "Bottom edge of the mesh for setting boundary conditions")
        .def("left_of", (RectilinearMesh2D::Boundary(*)(shared_ptr<const GeometryD<2>>,shared_ptr<const GeometryObject>,const PathHints&))&RectilinearMesh2D::getLeftOfBoundary,
             "Get boundary left of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("left_of")
        .def("right_of", (RectilinearMesh2D::Boundary(*)(shared_ptr<const GeometryD<2>>,shared_ptr<const GeometryObject>,const PathHints&))&RectilinearMesh2D::getLeftOfBoundary,
             "Get boundary right of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("right_of")
        .def("top_of", (RectilinearMesh2D::Boundary(*)(shared_ptr<const GeometryD<2>>,shared_ptr<const GeometryObject>,const PathHints&))&RectilinearMesh2D::getLeftOfBoundary,
             "Get boundary top of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("top_of")
        .def("bottom_of", (RectilinearMesh2D::Boundary(*)(shared_ptr<const GeometryD<2>>,shared_ptr<const GeometryObject>,const PathHints&))&RectilinearMesh2D::getLeftOfBoundary,
             "Get boundary bottom of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("bottom_of")
        .def("horizontal", (RectilinearMesh2D::Boundary(*)(double,double,double))&RectilinearMesh2D::getHorizontalBoundaryNear,
             "Get boundary at horizontal line", (py::arg("at"), "start", "stop"))
        .def("horizontal", (RectilinearMesh2D::Boundary(*)(double))&RectilinearMesh2D::getHorizontalBoundaryNear,
             "Get boundary at horizontal line", py::arg("at")).staticmethod("horizontal")
        .def("vertical", (RectilinearMesh2D::Boundary(*)(double,double,double))&RectilinearMesh2D::getHorizontalBoundaryNear,
             "Get boundary at vertical line", (py::arg("at"), "start", "stop"))
        .def("vertical", (RectilinearMesh2D::Boundary(*)(double))&RectilinearMesh2D::getHorizontalBoundaryNear,
             "Get boundary at vertical line", py::arg("at")).staticmethod("vertical")
        .def(py::self == py::self)
    ;
    ExportBoundary<RectilinearMesh2D> { rectilinear2d };

    py::class_<RectilinearMesh3D, shared_ptr<RectilinearMesh3D>, py::bases<MeshD<3>>> rectilinear3d("Rectilinear3D",
        "Two-dimensional mesh\n\n"
        "Rectilinear3D(ordering='012')\n    create empty mesh\n\n"
        "Rectilinear3D(axis0, axis1, axis2, ordering='012')\n    create mesh with axes supplied as mesh.Rectilinear1D\n\n"
        "Rectilinear3D(geometry, ordering='012')\n    create coarse mesh based on bounding boxes of geometry objects\n\n"
        "ordering can be any a string containing any permutation of and specifies ordering of the\n"
        "mesh points (last index changing fastest).",
        py::no_init
        ); rectilinear3d
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__empty<RectilinearMesh3D>, py::default_call_policies(), (py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__axes<RectilinearMesh3D, RectilinearMesh1D>, py::default_call_policies(), (py::arg("axis0"), "axis1", "axis2", py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&RectilinearMesh3D__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&Mesh__init__<RectilinearMesh3D, RegularMesh3D>, py::default_call_policies(), py::arg("src")))
        .def("copy", &Mesh__init__<RectilinearMesh3D, RectilinearMesh3D>, "Make a copy of this mesh")
        .def_readwrite("axis0", &RectilinearMesh3D::axis0, "The first (longitudinal) axis of the mesh")
        .def_readwrite("axis1", &RectilinearMesh3D::axis1, "The second (transverse) axis of the mesh")
        .def_readwrite("axis2", &RectilinearMesh3D::axis2, "The third (vertical) axis of the mesh")
        .add_property("major_axis", py::make_function((RectilinearMesh1D&(RectilinearMesh3D::*)())&RectilinearMesh3D::majorAxis, py::return_internal_reference<>()), "The slowest changing axis")
        .add_property("medium_axis", py::make_function((RectilinearMesh1D&(RectilinearMesh3D::*)())&RectilinearMesh3D::mediumAxis, py::return_internal_reference<>()), "The middle changing axis")
        .add_property("minor_axis", py::make_function((RectilinearMesh1D&(RectilinearMesh3D::*)())&RectilinearMesh3D::minorAxis, py::return_internal_reference<>()), "The quickest changing axis")
        .def("__nonzero__", &__nonempty__<RectilinearMesh3D>)
        .def("clear", &RectilinearMesh3D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh3D__getitem__<RectilinearMesh3D>)
        .def("index", &RectilinearMesh3D::index, (py::arg("index0"), py::arg("index1"), py::arg("index2")),
             "Return single index of the point indexed with index0, index1, and index2")
        .def("index0", &RectilinearMesh3D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectilinearMesh3D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("index2", &RectilinearMesh3D::index2, "Return index in the third axis of the point with given index", (py::arg("index")))
        .def("major_index", &RectilinearMesh3D::majorIndex, "Return index in the major axis of the point with given index", (py::arg("index")))
        .def("middle_index", &RectilinearMesh3D::middleIndex, "Return index in the middle axis of the point with given index", (py::arg("index")))
        .def("minor_index", &RectilinearMesh3D::minorIndex, "Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("set_optimal_ordering", &RectilinearMesh3D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .add_property("ordering", &RectangularMesh3D__getOrdering<RectilinearMesh3D>, &RectangularMesh3D__setOrdering<RectilinearMesh3D>, "Ordering of the points in this mesh")
        .def("get_midpoints", &RectilinearMesh3D::getMidpointsMesh, "Get new mesh with points in the middles of objects described by this mesh")
        .add_static_property("front", &RectilinearMesh3D::getFrontBoundary, "Front side of the mesh for setting boundary conditions")
        .add_static_property("back", &RectilinearMesh3D::getBackBoundary, "Back side of the mesh for setting boundary conditions")
        .add_static_property("left", &RectilinearMesh3D::getLeftBoundary, "Left side of the mesh for setting boundary conditions")
        .add_static_property("right", &RectilinearMesh3D::getRightBoundary, "Right side of the mesh for setting boundary conditions")
        .add_static_property("top", &RectilinearMesh3D::getTopBoundary, "Top side of the mesh for setting boundary conditions")
        .add_static_property("bottom", &RectilinearMesh3D::getBottomBoundary, "Bottom side of the mesh for setting boundary conditions")
        .def("front_of", (RectilinearMesh3D::Boundary(*)(shared_ptr<const GeometryD<3>>,shared_ptr<const GeometryObject>,const PathHints&))&RectilinearMesh3D::getFrontOfBoundary,
             "Get boundary in front of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("front_of")
        .def("back_of", (RectilinearMesh3D::Boundary(*)(shared_ptr<const GeometryD<3>>,shared_ptr<const GeometryObject>,const PathHints&))&RectilinearMesh3D::getBackOfBoundary,
             "Get boundary back of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("back_of")
        .def("left_of", (RectilinearMesh3D::Boundary(*)(shared_ptr<const GeometryD<3>>,shared_ptr<const GeometryObject>,const PathHints&))&RectilinearMesh3D::getLeftOfBoundary,
             "Get boundary left of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("left_of")
        .def("right_of", (RectilinearMesh3D::Boundary(*)(shared_ptr<const GeometryD<3>>,shared_ptr<const GeometryObject>,const PathHints&))&RectilinearMesh3D::getLeftOfBoundary,
             "Get boundary right of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("right_of")
        .def("top_of", (RectilinearMesh3D::Boundary(*)(shared_ptr<const GeometryD<3>>,shared_ptr<const GeometryObject>,const PathHints&))&RectilinearMesh3D::getLeftOfBoundary,
             "Get boundary top of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("top_of")
        .def("bottom_of", (RectilinearMesh3D::Boundary(*)(shared_ptr<const GeometryD<3>>,shared_ptr<const GeometryObject>,const PathHints&))&RectilinearMesh3D::getLeftOfBoundary,
             "Get boundary bottom of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("bottom_of")
        .def(py::self == py::self)
    ;
    ExportBoundary<RectilinearMesh3D> { rectilinear3d };

    py::class_<RegularMesh1D, shared_ptr<RegularMesh1D>, boost::noncopyable>("Regular1D",
        "Regular mesh axis\n\n"
        "Regular1D()\n    create empty mesh\n\n"
        "Regular1D(start, stop, num)\n    create mesh of count points equally distributed between start and stop"
        )
        .def("__init__", py::make_constructor(&Regular1D__init__empty))
        .def("__init__", py::make_constructor(&Regular1D__init__params, py::default_call_policies(), (py::arg("start"), "stop", "num")))
        .add_property("start", &RegularMesh1D::first, &RegularMesh1D_setFirst, "Position of the beginning of the mesh")
        .add_property("stop", &RegularMesh1D::last, &RegularMesh1D_setLast, "Position of the end of the mesh")
        .add_property("step", &RegularMesh1D::step)
        .def("__len__", &RegularMesh1D::size)
        .def("__nonzero__", __nonempty__<RegularMesh1D>)
        .def("__getitem__", &Regular1D__getitem__)
        .def("__str__", &Regular1D__str__)
        .def("__repr__", &Regular1D__repr__)
        .def("resize", &RegularMesh1D_resize, "Change number of points in this mesh", (py::arg("num")))
        .def(py::self == py::self)
        .def("__iter__", py::range(&RegularMesh1D::begin, &RegularMesh1D::end))
    ;
    detail::Regular1D_from_Tuple();
    py::implicitly_convertible<RegularMesh1D, RectilinearMesh1D>();

    py::class_<RegularMesh2D, shared_ptr<RegularMesh2D>, py::bases<MeshD<2>>> regular2d("Regular2D",
        "Two-dimensional mesh\n\n"
        "Regular2D(ordering='01')\n    create empty mesh\n\n"
        "Regular2D(axis0, axis1, ordering='01')\n    create mesh with axes supplied as sequences of numbers\n\n"
        "ordering can be either '01', '10' and specifies ordering of the mesh points (last index changing fastest).",
        py::no_init
        ); regular2d
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__empty<RegularMesh2D>, py::default_call_policies(), (py::arg("ordering")="01")))
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__axes<RegularMesh2D, RegularMesh1D>, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("ordering")="01")))
        .def("copy", &Mesh__init__<RegularMesh2D, RegularMesh2D>, "Make a copy of this mesh")
        .def_readwrite("axis0", &RegularMesh2D::axis0, "The first (transverse) axis of the mesh")
        .def_readwrite("axis1", &RegularMesh2D::axis1, "The second (vertical) axis of the mesh")
        .add_property("major_axis", py::make_function((RegularMesh1D&(RegularMesh2D::*)())&RegularMesh2D::majorAxis, py::return_internal_reference<>()), "The slower changing axis")
        .add_property("minor_axis", py::make_function((RegularMesh1D&(RegularMesh2D::*)())&RegularMesh2D::minorAxis, py::return_internal_reference<>()), "The quicker changing axis")
        .def("__nonzero__", &__nonempty__<RegularMesh2D>)
        .def("clear", &RegularMesh2D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh2D__getitem__<RegularMesh2D>)
        .def("index", &RegularMesh2D::index, "Return single index of the point indexed with index0 and index1", (py::arg("index0"), py::arg("index1")))
        .def("index0", &RegularMesh2D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RegularMesh2D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("major_index", &RegularMesh2D::majorIndex, "Return index in the major axis of the point with given index", (py::arg("index")))
        .def("minor_index", &RegularMesh2D::minorIndex, "Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("set_optimal_ordering", &RegularMesh2D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .add_property("ordering", &RectangularMesh2D__getOrdering<RegularMesh2D>, &RectangularMesh2D__setOrdering<RegularMesh2D>, "Ordering of the points in this mesh")
        .def("get_midpoints", &RegularMesh2D::getMidpointsMesh, "Get new mesh with points in the middles of objects described by this mesh")
        .add_static_property("left", &RegularMesh2D::getLeftBoundary, "Left edge of the mesh for setting boundary conditions")
        .add_static_property("right", &RegularMesh2D::getRightBoundary, "Right edge of the mesh for setting boundary conditions")
        .add_static_property("top", &RegularMesh2D::getTopBoundary, "Top edge of the mesh for setting boundary conditions")
        .add_static_property("bottom", &RegularMesh2D::getBottomBoundary, "Bottom edge of the mesh for setting boundary conditions")
        .def("left_of", (RegularMesh2D::Boundary(*)(shared_ptr<const GeometryD<2>>,shared_ptr<const GeometryObject>,const PathHints&))&RegularMesh2D::getLeftOfBoundary,
             "Get boundary left of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("left_of")
        .def("right_of", (RegularMesh2D::Boundary(*)(shared_ptr<const GeometryD<2>>,shared_ptr<const GeometryObject>,const PathHints&))&RegularMesh2D::getLeftOfBoundary,
             "Get boundary right of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("right_of")
        .def("top_of", (RegularMesh2D::Boundary(*)(shared_ptr<const GeometryD<2>>,shared_ptr<const GeometryObject>,const PathHints&))&RegularMesh2D::getLeftOfBoundary,
             "Get boundary top of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("top_of")
        .def("bottom_of", (RegularMesh2D::Boundary(*)(shared_ptr<const GeometryD<2>>,shared_ptr<const GeometryObject>,const PathHints&))&RegularMesh2D::getLeftOfBoundary,
             "Get boundary bottom of specified object", (py::arg("geometry"), "object", py::arg("path")=py::object())).staticmethod("bottom_of")
        .def("horizontal", (RegularMesh2D::Boundary(*)(double,double,double))&RegularMesh2D::getHorizontalBoundaryNear,
             "Get boundary at horizontal line", (py::arg("at"), "start", "stop"))
        .def("horizontal", (RegularMesh2D::Boundary(*)(double))&RegularMesh2D::getHorizontalBoundaryNear,
             "Get boundary at horizontal line", py::arg("at")).staticmethod("horizontal")
        .def("vertical", (RegularMesh2D::Boundary(*)(double,double,double))&RegularMesh2D::getHorizontalBoundaryNear,
             "Get boundary at vertical line", (py::arg("at"), "start", "stop"))
        .def("vertical", (RegularMesh2D::Boundary(*)(double))&RegularMesh2D::getHorizontalBoundaryNear,
             "Get boundary at vertical line", py::arg("at")).staticmethod("vertical")
        .def(py::self == py::self)
    ;
    ExportBoundary<RegularMesh2D> { regular2d };
    py::implicitly_convertible<RegularMesh2D, RectilinearMesh2D>();

    py::class_<RegularMesh3D, shared_ptr<RegularMesh3D>, py::bases<MeshD<3>>, boost::noncopyable>regular3d("Regular3D",
        "Two-dimensional mesh\n\n"
        "Regular3D(ordering='012')\n    create empty mesh\n\n"
        "Regular3D(axis0, axis1, axis2, ordering='012')\n    create mesh with axes supplied as mesh.Regular1D\n\n"
        "ordering can be any a string containing any permutation of and specifies ordering of the\n"
        "mesh points (last index changing fastest).",
        py::no_init
        ); regular3d
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__empty<RegularMesh3D>, py::default_call_policies(), (py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__axes<RegularMesh3D, RegularMesh1D>, py::default_call_policies(), (py::arg("axis0"), "axis1", "axis2", py::arg("ordering")="012")))
        .def("copy", &Mesh__init__<RegularMesh3D, RegularMesh3D>, "Make a copy of this mesh")
        .def_readwrite("axis0", &RegularMesh3D::axis0, "The first (longitudinal) axis of the mesh")
        .def_readwrite("axis1", &RegularMesh3D::axis1, "The second (transverse) axis of the mesh")
        .def_readwrite("axis2", &RegularMesh3D::axis2, "The third (vertical) axis of the mesh")
        .add_property("major_axis", py::make_function((RegularMesh1D&(RegularMesh3D::*)())&RegularMesh3D::majorAxis, py::return_internal_reference<>()), "The slowest changing axis")
        .add_property("medium_axis", py::make_function((RegularMesh1D&(RegularMesh3D::*)())&RegularMesh3D::mediumAxis, py::return_internal_reference<>()), "The middle changing axis")
        .add_property("minor_axis", py::make_function((RegularMesh1D&(RegularMesh3D::*)())&RegularMesh3D::minorAxis, py::return_internal_reference<>()), "The quickest changing axis")
        .def("__nonzero__", &__nonempty__<RegularMesh3D>)
        .def("clear", &RegularMesh3D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh3D__getitem__<RegularMesh3D>)
        .def("index", &RegularMesh3D::index, (py::arg("index0"), py::arg("index1"), py::arg("index2")),
             "Return single index of the point indexed with index0, index1, and index2")
        .def("index0", &RegularMesh3D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RegularMesh3D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("index2", &RegularMesh3D::index2, "Return index in the third axis of the point with given index", (py::arg("index")))
        .def("major_index", &RegularMesh3D::majorIndex, "Return index in the major axis of the point with given index", (py::arg("index")))
        .def("middle_index", &RegularMesh3D::middleIndex, "Return index in the middle axis of the point with given index", (py::arg("index")))
        .def("minor_index", &RegularMesh3D::minorIndex, "Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("set_optimal_ordering", &RegularMesh3D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .add_property("ordering", &RectangularMesh3D__getOrdering<RegularMesh3D>, &RectangularMesh3D__setOrdering<RegularMesh3D>, "Ordering of the points in this mesh")
        .def("get_midpoints", &RegularMesh3D::getMidpointsMesh, "Get new mesh with points in the middles of objects described by this mesh")
        .def(py::self == py::self)
    ;
    ExportBoundary<RegularMesh3D> { regular3d };
    py::implicitly_convertible<RegularMesh3D, RectilinearMesh3D>();

    ExportMeshGenerator<RectilinearMesh2D>("Rectilinear2D");
    {
        py::scope scope = rectilinear2d;

        py::class_<RectilinearMesh2DSimpleGenerator, shared_ptr<RectilinearMesh2DSimpleGenerator>,
                py::bases<MeshGeneratorOf<RectilinearMesh2D>>, boost::noncopyable>("SimpleGenerator",
            "Generator of Rectilinear2D mesh with lines at edges of all objects.\n\n"
            "SimpleGenerator()\n    create generator")
        ;

        register_divide_generator<2>();
    }

    ExportMeshGenerator<RectilinearMesh3D>("Rectilinear3D");
    {
        py::scope scope = rectilinear3d;

        py::class_<RectilinearMesh3DSimpleGenerator, shared_ptr<RectilinearMesh3DSimpleGenerator>,
                py::bases<MeshGeneratorOf<RectilinearMesh3D>>, boost::noncopyable>("SimpleGenerator",
            "Generator of Rectilinear3D mesh with lines at edges of all objects.\n\n"
            "SimpleGenerator()\n    create generator")
        ;

        register_divide_generator<3>();
    }

}

}} // namespace plask::python
