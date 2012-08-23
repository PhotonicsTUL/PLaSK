#include "../python_globals.h"
#include "../python_mesh.h"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>
#include <numpy/arrayobject.h>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/rectilinear.h>
#include <plask/mesh/generator_rectilinear.h>
#include <plask/mesh/regular.h>

namespace plask { namespace python {


template <typename T>
static bool __nonempty__(const T& self) { return !self.empty(); }

struct Rectilinear1D_from_Sequence
{
    Rectilinear1D_from_Sequence() {
        boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<RectilinearMesh1D>());
    }

    static void* convertible(PyObject* obj_ptr) {
        if (!PySequence_Check(obj_ptr)) return NULL;
        return obj_ptr;
    }

    static void construct(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        void* storage = ((boost::python::converter::rvalue_from_python_storage<shared_ptr<RectilinearMesh1D>>*)data)->storage.bytes;
        py::stl_input_iterator<double> begin(py::object(py::handle<>(py::borrowed(obj_ptr)))), end;
        new(storage) RectilinearMesh1D(std::vector<double>(begin, end));
        data->convertible = storage;
    }

};

static py::object Rectilinear1D__array__(py::object self) {
    RectilinearMesh1D* mesh = py::extract<RectilinearMesh1D*>(self);
    npy_intp dims[] = { mesh->size() };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)&(*mesh->begin()));
    if (arr == nullptr) throw TypeError("cannot create array");
    py::incref(self.ptr()); PyArray_BASE(arr) = self.ptr(); // Make sure the data vector stays alive as long as the array
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


struct Regular1D_from_Sequence
{
    Regular1D_from_Sequence() {
        boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<RegularMesh1D>());
    }

    static void* convertible(PyObject* obj_ptr) {
        if (!PySequence_Check(obj_ptr)) return NULL;
        return obj_ptr;
    }

    static void construct(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        void* storage = ((boost::python::converter::rvalue_from_python_storage<shared_ptr<RegularMesh1D>>*)data)->storage.bytes;
        auto tuple = py::object(py::handle<>(py::borrowed(obj_ptr)));
        try {
            if (py::len(tuple) != 3) throw py::error_already_set();
            new(storage) RegularMesh1D(py::extract<double>(tuple[0]), py::extract<double>(tuple[1]), py::extract<unsigned>(tuple[2]));
            data->convertible = storage;
        } catch (py::error_already_set) {
            throw TypeError("Must provide either mesh.Regular1D or '[first, last, count]'");
        }
    }

};


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
    return format("Regular1D(%1%, %2%, %3%)", self.getFirst(), self.getLast(), self.size());
}

static double Regular1D__getitem__(const RegularMesh1D& self, int i) {
    if (i < 0) i = self.size() + i;
    if (i < 0) throw IndexError("mesh.Regular1D index out of range");
    return self[i];
}

static void RegularMesh1D_resize(RegularMesh1D& self, int count) {
    self.reset(self.getFirst(), self.getLast(), count);
}

static void RegularMesh1D_setFirst(RegularMesh1D& self, double first) {
    double last = self.getLast();
    self.reset(first, last, self.size());
}

static void RegularMesh1D_setLast(RegularMesh1D& self, double last) {
    double first = self.getFirst();
    self.reset(first, last, self.size());
}






template <typename MeshT>
static void RectangularMesh2D__setOrdering(MeshT& self, std::string order) {
    if (order == "10") self.setIterationOrder(MeshT::NORMAL_ORDER);
    else if (order == "01") self.setIterationOrder(MeshT::TRANSPOSED_ORDER);
    else {
        throw ValueError("order must be either '01' or '10'");
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
    if (index0 < 0) index0 = self.c0.size() - index0;
    if (index0 < 0 || index0 >= int(self.c0.size())) {
        throw IndexError("first mesh index (%1%) out of range (0<=index<%2%)", index0, self.c0.size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 = self.c1.size() - index1;
    if (index1 < 0 || index1 >= int(self.c1.size())) {
        throw IndexError("second mesh index (%1%) out of range (0<=index<%2%)", index1, self.c1.size());
    }
    return self(index0, index1);
}

template <typename MeshT>
static std::string RectangularMesh2D__getOrdering(MeshT& self) {
    return (self.getIterationOrder() == MeshT::NORMAL_ORDER) ? "10" : "01";
}

template <typename MeshT>
void RectangularMesh3D__setOrdering(MeshT& self, std::string order) {
    if (order == "012") self.setIterationOrder(MeshT::ORDER_012);
    else if (order == "021") self.setIterationOrder(MeshT::ORDER_021);
    else if (order == "102") self.setIterationOrder(MeshT::ORDER_102);
    else if (order == "120") self.setIterationOrder(MeshT::ORDER_120);
    else if (order == "201") self.setIterationOrder(MeshT::ORDER_201);
    else if (order == "210") self.setIterationOrder(MeshT::ORDER_210);
    else {
        throw ValueError("order must be any permutation of '012'");
    }
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
    if (index0 < 0) index0 = self.c0.size() - index0;
    if (index0 < 0 || index0 >= int(self.c0.size())) {
        throw IndexError("first mesh index (%1%) out of range (0<=index<%2%)", index0, self.c0.size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 = self.c1.size() - index1;
    if (index1 < 0 || index1 >= int(self.c1.size())) {
        throw IndexError("second mesh index (%1%) out of range (0<=index<%2%)", index1, self.c1.size());
    }
    int index2 = py::extract<int>(index[2]);
    if (index2 < 0) index2 = self.c2.size() - index2;
    if (index2 < 0 || index2 >= int(self.c2.size())) {
        throw IndexError("third mesh index (%1%) out of range (0<=index<%2%)", index2, self.c2.size());
    }
    return self(index0, index1, index2);
}





shared_ptr<RectilinearMesh2D> RectilinearMesh2D__init__geometry(const shared_ptr<GeometryElementD<2>>& geometry, std::string order) {
    auto mesh = RectilinearMesh2DSimpleGenerator().generate(geometry);
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectilinearMesh3D> RectilinearMesh3D__init__geometry(const shared_ptr<GeometryElementD<3>>& geometry, std::string order) {
    auto mesh = RectilinearMesh3DSimpleGenerator().generate(geometry);
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}


py::object RectilinearMesh2DDivideGenerator_getPreDivision(const RectilinearMesh2DDivideGenerator& self) {
    auto division = self.getPreDivision();
    return py::make_tuple(division.first, division.second);
}

void RectilinearMesh2DDivideGenerator_setPreDivision(RectilinearMesh2DDivideGenerator& self, const py::object division) {
    try {
        self.setPreDivision(py::extract<size_t>(division));
    } catch (py::error_already_set) {
        PyErr_Clear();
        try {
            if (!PySequence_Check(division.ptr()) || py::len(division) != 2)
                throw py::error_already_set();
            self.setPreDivision(py::extract<size_t>(division[0]), py::extract<size_t>(division[1]));
        } catch (py::error_already_set) {
            throw TypeError("division must be either a single positive integer or a sequence of two positive integers");
        }
    }
}

py::object RectilinearMesh2DDivideGenerator_getPostDivision(const RectilinearMesh2DDivideGenerator& self) {
    auto division = self.getPostDivision();
    return py::make_tuple(division.first, division.second);
}

void RectilinearMesh2DDivideGenerator_setPostDivision(RectilinearMesh2DDivideGenerator& self, const py::object division) {
    try {
        self.setPostDivision(py::extract<size_t>(division));
    } catch (py::error_already_set) {
        PyErr_Clear();
        try {
            if (!PySequence_Check(division.ptr()) || py::len(division) != 2)
                throw py::error_already_set();
            self.setPostDivision(py::extract<size_t>(division[0]), py::extract<size_t>(division[1]));
        } catch (py::error_already_set) {
            throw TypeError("division must be either a single positive integer or a sequence of two positive integers");
        }
    }
}

void RectilinearMesh2DDivideGenerator_addRefinement1(RectilinearMesh2DDivideGenerator& self, const std::string& axis, GeometryElementD<2>& element, const PathHints& path, double position) {
    int i = config.axes[axis] - 1;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(Primitive<2>::DIRECTION(i), dynamic_pointer_cast<GeometryElementD<2>>(element.shared_from_this()), path, position);
}

void RectilinearMesh2DDivideGenerator_addRefinement2(RectilinearMesh2DDivideGenerator& self, const std::string& axis, GeometryElementD<2>& element, double position) {
    int i = config.axes[axis] - 1;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(Primitive<2>::DIRECTION(i), dynamic_pointer_cast<GeometryElementD<2>>(element.shared_from_this()), position);
}

void RectilinearMesh2DDivideGenerator_addRefinement3(RectilinearMesh2DDivideGenerator& self, const std::string& axis, GeometryElement::Subtree subtree, double position) {
    int i = config.axes[axis] - 1;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(Primitive<2>::DIRECTION(i), subtree, position);
}

void RectilinearMesh2DDivideGenerator_addRefinement4(RectilinearMesh2DDivideGenerator& self, const std::string& axis, Path path, double position) {
    int i = config.axes[axis] - 1;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.addRefinement(Primitive<2>::DIRECTION(i), path, position);
}

void RectilinearMesh2DDivideGenerator_removeRefinement1(RectilinearMesh2DDivideGenerator& self, const std::string& axis, GeometryElementD<2>& element, const PathHints& path, double position) {
    int i = config.axes[axis] - 1;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(Primitive<2>::DIRECTION(i), dynamic_pointer_cast<GeometryElementD<2>>(element.shared_from_this()), path, position);
}

void RectilinearMesh2DDivideGenerator_removeRefinement2(RectilinearMesh2DDivideGenerator& self, const std::string& axis, GeometryElementD<2>& element, double position) {
    int i = config.axes[axis] - 1;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(Primitive<2>::DIRECTION(i), dynamic_pointer_cast<GeometryElementD<2>>(element.shared_from_this()), position);
}

void RectilinearMesh2DDivideGenerator_removeRefinement3(RectilinearMesh2DDivideGenerator& self, const std::string& axis, GeometryElement::Subtree subtree, double position) {
    int i = config.axes[axis] - 1;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(Primitive<2>::DIRECTION(i), subtree, position);
}

void RectilinearMesh2DDivideGenerator_removeRefinement4(RectilinearMesh2DDivideGenerator& self, const std::string& axis, Path path, double position) {
    int i = config.axes[axis] - 1;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    self.removeRefinement(Primitive<2>::DIRECTION(i), path, position);
}


void RectilinearMesh2DDivideGenerator_removeRefinements1(RectilinearMesh2DDivideGenerator& self, GeometryElementD<2>& element, const PathHints& path) {
    self.removeRefinements(dynamic_pointer_cast<GeometryElementD<2>>(element.shared_from_this()), path);
}

void RectilinearMesh2DDivideGenerator_removeRefinements2(RectilinearMesh2DDivideGenerator& self, const Path& path) {
    self.removeRefinements(path);
}

void RectilinearMesh2DDivideGenerator_removeRefinements3(RectilinearMesh2DDivideGenerator& self, const GeometryElement::Subtree& subtree) {
    self.removeRefinements(subtree);
}

py::dict RectilinearMesh2DDivideGenerator_listRefinements(const RectilinearMesh2DDivideGenerator& self, const std::string& axis) {
    int i = config.axes[axis] - 1;
    if (i < 0 || i > 1) throw ValueError("Bad axis name %1%.", axis);
    py::dict refinements;
    for (auto refinement: self.getRefinements(Primitive<2>::DIRECTION(i))) {
        py::object element { const_pointer_cast<GeometryElementD<2>>(refinement.first.first.lock()) };
        auto pth = refinement.first.second;
        py::object path;
        if (pth.hintFor.size() != 0) path = py::object(pth);
        py::list refs;
        for (auto x: refinement.second) {
            refs.append(x);
        }
        refinements[py::make_tuple(element, path)] = refs;
    }
    return refinements;
}


static inline bool plask_import_array() {
    import_array1(false);
    return true;
}

void register_mesh_rectangular()
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    py::class_<RectilinearMesh1D, shared_ptr<RectilinearMesh1D>>("Rectilinear1D",
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
        .def("__array__", &Rectilinear1D__array__)
        .def("insert", &RectilinearMesh1D::addPoint, "Insert point to the mesh", (py::arg("point")))
        .def("extend", &Rectilinear1D_extend, "Insert points from the sequence to the mesh", (py::arg("points")))
        .def(py::self == py::self)
        .def("__iter__", py::range(&RectilinearMesh1D::begin, &RectilinearMesh1D::end))
    ;
    Rectilinear1D_from_Sequence();

    py::class_<RectilinearMesh2D, shared_ptr<RectilinearMesh2D>, py::bases<MeshD<2>>> rectilinear2d("Rectilinear2D",
        "Two-dimensional mesh\n\n"
        "Rectilinear2D(ordering='01')\n    create empty mesh\n\n"
        "Rectilinear2D(axis0, axis1, ordering='01')\n    create mesh with axes supplied as sequences of numbers\n\n"
        "Rectilinear2D(geometry, ordering='01')\n    create coarse mesh based on bounding boxes of geometry elements\n\n"
        "ordering can be either '01', '10' and specifies ordering of the mesh points (last index changing fastest).",
        py::no_init
        ); rectilinear2d
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__empty<RectilinearMesh2D>, py::default_call_policies(), (py::arg("ordering")="10")))
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__axes<RectilinearMesh2D, RectilinearMesh1D>, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("ordering")="10")))
        .def("__init__", py::make_constructor(&RectilinearMesh2D__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="10")))
        .def_readwrite("axis0", &RectilinearMesh2D::c0, "The first (transverse) axis of the mesh")
        .def_readwrite("axis1", &RectilinearMesh2D::c1, "The second (vertical) axis of the mesh")
        .add_property("major_axis", py::make_function((RectilinearMesh1D&(RectilinearMesh2D::*)())&RectilinearMesh2D::majorAxis, py::return_internal_reference<>()), "The slower changing axis")
        .add_property("minor_axis", py::make_function((RectilinearMesh1D&(RectilinearMesh2D::*)())&RectilinearMesh2D::minorAxis, py::return_internal_reference<>()), "The quicker changing axis")
        .def("__nonzero__", &__nonempty__<RectilinearMesh2D>, "Return True if the mesh is empty")
        .def("clear", &RectilinearMesh2D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh2D__getitem__<RectilinearMesh2D>)
        .def("index", &RectilinearMesh2D::index, "Return single index of the point indexed with index0 and index1", (py::arg("index0"), py::arg("index1")))
        .def("index0", &RectilinearMesh2D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectilinearMesh2D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("majorIndex", &RectilinearMesh2D::majorIndex, "Return index in the major axis of the point with given index", (py::arg("index")))
        .def("minorIndex", &RectilinearMesh2D::minorIndex, "Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("setOptimalOrdering", &RectilinearMesh2D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .def("setOrdering", &RectangularMesh2D__setOrdering<RectilinearMesh2D>, "Set desired ordering of the points in this mesh", (py::arg("ordering")))
        .def("getMidpointsMesh", &RectilinearMesh2D::getMidpointsMesh, "Get new mesh with points in the middles of elements described by this mesh")
        .add_static_property("leftBoundary", &RectilinearMesh2D::getLeftBoundary, "Left edge of the mesh for setting boundary conditions")
        .add_static_property("rightBoundary", &RectilinearMesh2D::getRightBoundary, "Right edge of the mesh for setting boundary conditions")
        .add_static_property("topBoundary", &RectilinearMesh2D::getTopBoundary, "Top edge of the mesh for setting boundary conditions")
        .add_static_property("bottomBoundary", &RectilinearMesh2D::getBottomBoundary, "Bottom edge of the mesh for setting boundary conditions")
        .def(py::self == py::self)
    ;
    ExportBoundary<RectilinearMesh2D>("Rectilinear2D");

    py::class_<RectilinearMesh3D, shared_ptr<RectilinearMesh3D>, py::bases<MeshD<3>>> rectilinear3d("Rectilinear3D",
        "Two-dimensional mesh\n\n"
        "Rectilinear3D(ordering='210')\n    create empty mesh\n\n"
        "Rectilinear3D(axis0, axis1, axis2, ordering='210')\n    create mesh with axes supplied as mesh.Rectilinear1D\n\n"
        "Rectilinear3D(geometry, ordering='210')\n    create coarse mesh based on bounding boxes of geometry elements\n\n"
        "ordering can be any a string containing any permutation of and specifies ordering of the\n"
        "mesh points (last index changing fastest).",
        py::no_init
        ); rectilinear3d
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__empty<RectilinearMesh3D>, py::default_call_policies(), (py::arg("ordering")="210")))
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__axes<RectilinearMesh3D, RectilinearMesh1D>, py::default_call_policies(), (py::arg("axis0"), "axis1", "axis2", py::arg("ordering")="210")))
        .def("__init__", py::make_constructor(&RectilinearMesh3D__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="210")))
        .def_readwrite("axis0", &RectilinearMesh3D::c0, "The first (longitudinal) axis of the mesh")
        .def_readwrite("axis1", &RectilinearMesh3D::c1, "The second (transverse) axis of the mesh")
        .def_readwrite("axis2", &RectilinearMesh3D::c2, "The third (vertical) axis of the mesh")
        .add_property("major_axis", py::make_function((RectilinearMesh1D&(RectilinearMesh3D::*)())&RectilinearMesh3D::majorAxis, py::return_internal_reference<>()), "The slowest changing axis")
        .add_property("middle_axis", py::make_function((RectilinearMesh1D&(RectilinearMesh3D::*)())&RectilinearMesh3D::middleAxis, py::return_internal_reference<>()), "The middle changing axis")
        .add_property("minor_axis", py::make_function((RectilinearMesh1D&(RectilinearMesh3D::*)())&RectilinearMesh3D::minorAxis, py::return_internal_reference<>()), "The quickest changing axis")
        .def("__nonzero__", &__nonempty__<RectilinearMesh3D>)
        .def("clear", &RectilinearMesh3D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh3D__getitem__<RectilinearMesh3D>)
        .def("index", &RectilinearMesh3D::index, (py::arg("index0"), py::arg("index1"), py::arg("index2")),
             "Return single index of the point indexed with index0, index1, and index2")
        .def("index0", &RectilinearMesh3D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectilinearMesh3D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("index2", &RectilinearMesh3D::index2, "Return index in the third axis of the point with given index", (py::arg("index")))
        .def("majorIndex", &RectilinearMesh3D::majorIndex, "Return index in the major axis of the point with given index", (py::arg("index")))
        .def("middleIndex", &RectilinearMesh3D::middleIndex, "Return index in the middle axis of the point with given index", (py::arg("index")))
        .def("minorIndex", &RectilinearMesh3D::minorIndex, "Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("setOptimalOrdering", &RectilinearMesh3D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .def("setOrdering", &RectangularMesh3D__setOrdering<RectilinearMesh3D>, "Set desired ordering of the points in this mesh", (py::arg("order")))
        .def("getMidpointsMesh", &RectilinearMesh3D::getMidpointsMesh, "Get new mesh with points in the middles of elements described by this mesh")
        .def(py::self == py::self)
    ;


    py::class_<RegularMesh1D, shared_ptr<RegularMesh1D>>("Regular1D",
        "Regular mesh axis\n\n"
        "Regular1D()\n    create empty mesh\n\n"
        "Regular1D(first, last, count)\n    create mesh of count points equally distributed between first and last"
        )
        .def("__init__", py::make_constructor(&Regular1D__init__empty))
        .def("__init__", py::make_constructor(&Regular1D__init__params, py::default_call_policies(), (py::arg("first"), "last", "count")))
        .add_property("first", &RegularMesh1D::getFirst, &RegularMesh1D_setFirst, "Position of the beginning of the mesh")
        .add_property("last", &RegularMesh1D::getLast, &RegularMesh1D_setLast, "Position of the end of the mesh")
        .add_property("step", &RegularMesh1D::getStep)
        .def("__len__", &RegularMesh1D::size)
        .def("__nonzero__", __nonempty__<RegularMesh1D>)
        .def("__getitem__", &Regular1D__getitem__)
        .def("__str__", &Regular1D__str__)
        .def("__repr__", &Regular1D__repr__)
        .def("resize", &RegularMesh1D_resize, "Change number of points in this mesh", (py::arg("count")))
        .def(py::self == py::self)
        .def("__iter__", py::range(&RegularMesh1D::begin, &RegularMesh1D::end))
    ;
    Regular1D_from_Sequence();

    py::class_<RegularMesh2D, shared_ptr<RegularMesh2D>, py::bases<MeshD<2>>>("Regular2D",
        "Two-dimensional mesh\n\n"
        "Regular2D(ordering='01')\n    create empty mesh\n\n"
        "Regular2D(axis0, axis1, ordering='01')\n    create mesh with axes supplied as sequences of numbers\n\n"
        "ordering can be either '01', '10' and specifies ordering of the mesh points (last index changing fastest).",
        py::no_init
        )
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__empty<RegularMesh2D>, py::default_call_policies(), (py::arg("ordering")="10")))
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__axes<RegularMesh2D, RegularMesh1D>, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("ordering")="10")))
        .def_readwrite("axis0", &RegularMesh2D::c0, "The first (transverse) axis of the mesh")
        .def_readwrite("axis1", &RegularMesh2D::c1, "The second (vertical) axis of the mesh")
        .add_property("major_axis", py::make_function((RegularMesh1D&(RegularMesh2D::*)())&RegularMesh2D::majorAxis, py::return_internal_reference<>()), "The slower changing axis")
        .add_property("minor_axis", py::make_function((RegularMesh1D&(RegularMesh2D::*)())&RegularMesh2D::minorAxis, py::return_internal_reference<>()), "The quicker changing axis")
        .def("__nonzero__", &__nonempty__<RegularMesh2D>)
        .def("clear", &RegularMesh2D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh2D__getitem__<RegularMesh2D>)
        .def("index", &RegularMesh2D::index, "Return single index of the point indexed with index0 and index1", (py::arg("index0"), py::arg("index1")))
        .def("index0", &RegularMesh2D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RegularMesh2D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("majorIndex", &RegularMesh2D::majorIndex, "Return index in the major axis of the point with given index", (py::arg("index")))
        .def("minorIndex", &RegularMesh2D::minorIndex, "Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("setOptimalOrdering", &RegularMesh2D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .def("setOrdering", &RectangularMesh2D__setOrdering<RegularMesh2D>, "Set desired ordering of the points in this mesh", (py::arg("ordering")))
        .def("getMidpointsMesh", &RegularMesh2D::getMidpointsMesh, "Get new mesh with points in the middles of elements described by this mesh")
        .add_static_property("leftBoundary", &RegularMesh2D::getLeftBoundary, "Left edge of the mesh for setting boundary conditions")
        .add_static_property("rightBoundary", &RegularMesh2D::getRightBoundary, "Right edge of the mesh for setting boundary conditions")
        .add_static_property("topBoundary", &RegularMesh2D::getTopBoundary, "Top edge of the mesh for setting boundary conditions")
        .add_static_property("bottomBoundary", &RegularMesh2D::getBottomBoundary, "Bottom edge of the mesh for setting boundary conditions")
        .def(py::self == py::self)
    ;
    ExportBoundary<RegularMesh2D>("Regular2D");

    py::class_<RegularMesh3D, shared_ptr<RegularMesh3D>, py::bases<MeshD<3>>>("Regular3D",
        "Two-dimensional mesh\n\n"
        "Regular3D(ordering='210')\n    create empty mesh\n\n"
        "Regular3D(axis0, axis1, axis2, ordering='210')\n    create mesh with axes supplied as mesh.Regular1D\n\n"
        "ordering can be any a string containing any permutation of and specifies ordering of the\n"
        "mesh points (last index changing fastest).",
        py::no_init
        )
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__empty<RegularMesh3D>, py::default_call_policies(), (py::arg("ordering")="210")))
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__axes<RegularMesh3D, RegularMesh1D>, py::default_call_policies(), (py::arg("axis0"), "axis1", "axis2", py::arg("ordering")="210")))
        .def_readwrite("axis0", &RegularMesh3D::c0, "The first (longitudinal) axis of the mesh")
        .def_readwrite("axis1", &RegularMesh3D::c1, "The second (transverse) axis of the mesh")
        .def_readwrite("axis2", &RegularMesh3D::c2, "The third (vertical) axis of the mesh")
        .add_property("major_axis", py::make_function((RegularMesh1D&(RegularMesh3D::*)())&RegularMesh3D::majorAxis, py::return_internal_reference<>()), "The slowest changing axis")
        .add_property("middle_axis", py::make_function((RegularMesh1D&(RegularMesh3D::*)())&RegularMesh3D::middleAxis, py::return_internal_reference<>()), "The middle changing axis")
        .add_property("minor_axis", py::make_function((RegularMesh1D&(RegularMesh3D::*)())&RegularMesh3D::minorAxis, py::return_internal_reference<>()), "The quickest changing axis")
        .def("__nonzero__", &__nonempty__<RegularMesh3D>)
        .def("clear", &RegularMesh3D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh3D__getitem__<RegularMesh3D>)
        .def("index", &RegularMesh3D::index, (py::arg("index0"), py::arg("index1"), py::arg("index2")),
             "Return single index of the point indexed with index0, index1, and index2")
        .def("index0", &RegularMesh3D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RegularMesh3D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("index2", &RegularMesh3D::index2, "Return index in the third axis of the point with given index", (py::arg("index")))
        .def("majorIndex", &RegularMesh3D::majorIndex, "Return index in the major axis of the point with given index", (py::arg("index")))
        .def("middleIndex", &RegularMesh3D::middleIndex, "Return index in the middle axis of the point with given index", (py::arg("index")))
        .def("minorIndex", &RegularMesh3D::minorIndex, "Return index in the minor axis of the point with given index", (py::arg("index")))
        .def("setOptimalOrdering", &RegularMesh3D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .def("setOrdering", &RectangularMesh3D__setOrdering<RegularMesh3D>, "Set desired ordering of the points in this mesh", (py::arg("order")))
        .def("getMidpointsMesh", &RegularMesh3D::getMidpointsMesh, "Get new mesh with points in the middles of elements described by this mesh")
        .def(py::self == py::self)
    ;

    ExportMeshGenerator<RectilinearMesh2D>("Rectilinear2D");
    {
        py::scope scope = rectilinear2d;

        py::class_<RectilinearMesh2DSimpleGenerator, shared_ptr<RectilinearMesh2DSimpleGenerator>,
                py::bases<MeshGeneratorOf<RectilinearMesh2D>>>("SimpleGenerator",
            "Generator of Rectilinear2D mesh with lines at edges of all elements.\n\n"
            "SimpleGenerator()\n    create generator")
        ;

        py::class_<RectilinearMesh2DDivideGenerator, shared_ptr<RectilinearMesh2DDivideGenerator>,
                py::bases<MeshGeneratorOf<RectilinearMesh2D>>>("DivideGenerator",
            "Generator of Rectilinear2D mesh by simple division of the geometry.\n\n"
            "DivideGenerator(division=1)\n"
            "    create generator with initial division of all geometry elements", py::init<size_t>(py::arg("division")=1))
            .add_property("prediv", &RectilinearMesh2DDivideGenerator_getPreDivision, &RectilinearMesh2DDivideGenerator_setPreDivision,
                        "initial division of all geometry elements")
            .add_property("postdiv", &RectilinearMesh2DDivideGenerator_getPostDivision, &RectilinearMesh2DDivideGenerator_setPostDivision,
                        "final division of all geometry elements")
            .def_readwrite("limit_change", &RectilinearMesh2DDivideGenerator::limit_change, "Limit maximum adjacent elements size change to the factor of two")
            .def_readwrite("warn_multiple", &RectilinearMesh2DDivideGenerator::warn_multiple, "Warn if refining path points to more than one object")
            .def_readwrite("warn_missing", &RectilinearMesh2DDivideGenerator::warn_missing, "Warn if refining path does not point to any object")
            .def_readwrite("warn_ouside", &RectilinearMesh2DDivideGenerator::warn_outside, "Warn if refining line is outside of its object")
            .def("addRefinement", &RectilinearMesh2DDivideGenerator_addRefinement1, "Add a refining line inside the element",
                (py::arg("axis"), "element", "path", "pos"))
            .def("addRefinement", &RectilinearMesh2DDivideGenerator_addRefinement2, "Add a refining line inside the element",
                (py::arg("axis"), "element", "pos"))
            .def("addRefinement", &RectilinearMesh2DDivideGenerator_addRefinement3, "Add a refining line inside the element",
                (py::arg("axis"), "subtree", "pos"))
            .def("addRefinement", &RectilinearMesh2DDivideGenerator_addRefinement4, "Add a refining line inside the element",
                (py::arg("axis"), "path", "pos"))
            .def("removeRefinement", &RectilinearMesh2DDivideGenerator_removeRefinement1, "Remove the refining line from the element",
                (py::arg("axis"), "element", "path", "pos"))
            .def("removeRefinement", &RectilinearMesh2DDivideGenerator_removeRefinement2, "Remove the refining line from the element",
                (py::arg("axis"), "element", "pos"))
            .def("removeRefinement", &RectilinearMesh2DDivideGenerator_removeRefinement3, "Remove the refining line from the element",
                (py::arg("axis"), "subtree", "pos"))
            .def("removeRefinement", &RectilinearMesh2DDivideGenerator_removeRefinement4, "Remove the refining line from the element",
                (py::arg("axis"), "path", "pos"))
            .def("removeRefinements", &RectilinearMesh2DDivideGenerator_removeRefinements1, "Remove the all refining lines from the element",
                (py::arg("element"), py::arg("path")=py::object()))
            .def("removeRefinements", &RectilinearMesh2DDivideGenerator_removeRefinements2, "Remove the all refining lines from the element",
                py::arg("path"))
            .def("removeRefinements", &RectilinearMesh2DDivideGenerator_removeRefinements3, "Remove the all refining lines from the element",
                py::arg("subtree"))
            .def("clearRefinements", &RectilinearMesh2DDivideGenerator::clearRefinements, "Clear all refining lines",
                py::arg("subtree"))
            .def("getRefinements", &RectilinearMesh2DDivideGenerator_listRefinements, py::arg("axis"),
                "Get list of all the refinements defined for this generator for specified axis"
            )
        ;
    }

    ExportMeshGenerator<RectilinearMesh3D>("Rectilinear3D");
    {
        py::scope scope = rectilinear3d;

        py::class_<RectilinearMesh3DSimpleGenerator, shared_ptr<RectilinearMesh3DSimpleGenerator>,
                py::bases<MeshGeneratorOf<RectilinearMesh3D>>>("SimpleGenerator",
            "Generator of Rectilinear3D mesh with lines at edges of all elements.\n\n"
            "SimpleGenerator()\n    create generator")
        ;
    }

}

}} // namespace plask::python
