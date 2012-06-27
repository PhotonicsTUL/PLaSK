#include "../python_globals.h"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>
#include <numpy/arrayobject.h>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/rectilinear.h>
#include <plask/mesh/regular.h>

namespace plask { namespace python {

struct Rectilinear1D_fromto_Sequence
{
    Rectilinear1D_fromto_Sequence() {
        boost::python::to_python_converter<RectilinearMesh1D, Rectilinear1D_fromto_Sequence>();
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

    static PyObject* convert(const RectilinearMesh1D& mesh) {
        npy_intp dims[] = { mesh.size() };
        PyObject* arr = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (arr == nullptr) throw plask::CriticalException("cannot create array from mesh");
        std::copy(mesh.begin(), mesh.end(), (double*)PyArray_DATA(arr));
        return arr;
    }
};



template <typename MeshT>
void RectangularMesh2D__setOrdering(MeshT& self, std::string order) {
    if (order == "10") self.setIterationOrder(MeshT::NORMAL_ORDER);
    else if (order == "01") self.setIterationOrder(MeshT::TRANSPOSED_ORDER);
    else {
        throw ValueError("order must be either '01' or '10'");
    }
}

template <typename MeshT>
shared_ptr<MeshT> RectangularMesh2D__init__empty(std::string order) {
    auto mesh = make_shared<MeshT>();
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

template <typename MeshT, typename AxesT>
shared_ptr<MeshT> RectangularMesh2D__init__axes(const AxesT& axis0, const AxesT& axis1, std::string order) {
    auto mesh = make_shared<MeshT>(axis0, axis1);
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

template <typename MeshT>
Vec<2,double> RectangularMesh2D__getitem__(const MeshT& self, py::object index) {
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





shared_ptr<RectilinearMesh2D> RectilinearMesh2D__init__geometry(const GeometryElementD<2>& geometry, std::string order) {
    auto mesh = make_shared<RectilinearMesh2D>(geometry);
    RectangularMesh2D__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectilinearMesh3D> RectilinearMesh3D__init__geometry(const GeometryElementD<3>& geometry, std::string order) {
    auto mesh = make_shared<RectilinearMesh3D>(geometry);
    RectangularMesh3D__setOrdering(*mesh, order);
    return mesh;
}

template <typename T>
py::object RectilinearMesh__axis(py::object self, T* mesh, RectilinearMesh1D& axis) {
    npy_intp dims[] = { axis.size() };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)&(*axis.begin()));
    if (arr == nullptr) throw plask::CriticalException("cannot create array from mesh");
    py::incref(self.ptr()); PyArray_BASE(arr) = self.ptr(); // Make sure the mesh stays alive as long as the array
    return py::object(py::handle<>(arr));
}

py::object RectilinearMesh2D_axis0(py::object self) {
    RectilinearMesh2D* mesh = py::extract<RectilinearMesh2D*>(self);
    return RectilinearMesh__axis(self, mesh, mesh->c0);
}
py::object RectilinearMesh2D_axis1(py::object self) {
    RectilinearMesh2D* mesh = py::extract<RectilinearMesh2D*>(self);
    return RectilinearMesh__axis(self, mesh, mesh->c1);
}

py::object RectilinearMesh3D_axis0(py::object self) {
    RectilinearMesh3D* mesh = py::extract<RectilinearMesh3D*>(self);
    return RectilinearMesh__axis(self, mesh, mesh->c0);
}
py::object RectilinearMesh3D_axis1(py::object self) {
    RectilinearMesh3D* mesh = py::extract<RectilinearMesh3D*>(self);
    return RectilinearMesh__axis(self, mesh, mesh->c1);
}
py::object RectilinearMesh3D_axis2(py::object self) {
    RectilinearMesh3D* mesh = py::extract<RectilinearMesh3D*>(self);
    return RectilinearMesh__axis(self, mesh, mesh->c2);
}


static inline bool plask_import_array() {
    import_array1(false);
    return true;
}



void register_mesh_rectangular()
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    Rectilinear1D_fromto_Sequence();

    py::class_<RectilinearMesh2D, shared_ptr<RectilinearMesh2D>, py::bases<Mesh<2>>>("Rectilinear2D",
        "Two-dimensional mesh\n\n"
        "Rectilinear2D(ordering='01')\n    create empty mesh\n\n"
        "Rectilinear2D(axis0, axis1, ordering='01')\n    create mesh with axes supplied as sequences of numbers\n\n"
        "Rectilinear2D(geometry, ordering='01')\n    create coarse mesh based on bounding boxes of geometry elements\n\n"
        "ordering can be either '01', '10' and specifies ordering of the mesh points (last index changing fastest).",
        py::no_init
        )
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__empty<RectilinearMesh2D>, py::default_call_policies(), (py::arg("ordering")="10")))
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__axes<RectilinearMesh2D, RectilinearMesh1D>, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("ordering")="10")))
        .def("__init__", py::make_constructor(&RectilinearMesh2D__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="10")))
        .add_property("axis0", &RectilinearMesh2D_axis0, py::make_setter(&RectilinearMesh2D::c0),
                      "List of points along the first (transverse) axis of the mesh")
        .add_property("axis1", &RectilinearMesh2D_axis1, py::make_setter(&RectilinearMesh2D::c1),
                      "List of points along the second (vertical) axis of the mesh")
        .def("empty", &RectilinearMesh2D::empty, "Return True if the mesh is empty")
        .def("clear", &RectilinearMesh2D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh2D__getitem__<RectilinearMesh2D>)
        .def("__iter__", py::range((auto (*)(const plask::RectilinearMesh2D& m)->decltype(m.begin_fast()))&std::begin, (auto (*)(const plask::RectilinearMesh2D& m)->decltype(m.end_fast()))&std::end))
        .def("index", &RectilinearMesh2D::index, "Return single index of the point indexed with index0 and index1", (py::arg("index0"), py::arg("index1")))
        .def("index0", &RectilinearMesh2D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectilinearMesh2D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("setOptimalOrdering", &RectilinearMesh2D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .def("setOrdering", &RectangularMesh2D__setOrdering<RectilinearMesh2D>, "Set desired ordering of the points in this mesh", (py::arg("ordering")))
    ;

    py::class_<RectilinearMesh3D, shared_ptr<RectilinearMesh3D>, py::bases<Mesh<3>>>("Rectilinear3D",
        "Two-dimensional mesh\n\n"
        "Rectilinear3D(ordering='012')\n    create empty mesh\n\n"
        "Rectilinear3D(axis0, axis1, axis2, ordering='012')\n    create mesh with axes supplied as mesh.Rectilinear1D\n\n"
        "Rectilinear3D(geometry, ordering='012')\n    create coarse mesh based on bounding boxes of geometry elements\n\n"
        "ordering can be any a string containing any permutation of and specifies ordering of the\n"
        "mesh points (last index changing fastest).",
        py::no_init
        )
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__empty<RectilinearMesh3D>, py::default_call_policies(), (py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__axes<RectilinearMesh3D, RectilinearMesh1D>, py::default_call_policies(), (py::arg("axis0"), "axis1", "axis2", py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&RectilinearMesh3D__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="012")))
        .add_property("axis0", &RectilinearMesh3D_axis0, py::make_setter(&RectilinearMesh3D::c0),
                      "List of points along the first (longitudinal) axis of the mesh")
        .add_property("axis1", &RectilinearMesh3D_axis1, py::make_setter(&RectilinearMesh3D::c1),
                      "List of points along the second (transverse) axis of the mesh")
        .add_property("axis2", &RectilinearMesh3D_axis2, py::make_setter(&RectilinearMesh3D::c2),
                      "List of points along the third (vertical) axis of the mesh")
        .def("empty", &RectilinearMesh3D::empty, "Return True if the mesh is empty")
        .def("clear", &RectilinearMesh3D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh3D__getitem__<RectilinearMesh3D>)
        .def("__iter__", py::range((auto (*)(const plask::RectilinearMesh3D& m)->decltype(m.begin_fast()))&std::begin, (auto (*)(const plask::RectilinearMesh3D& m)->decltype(m.end_fast()))&std::end))
        .def("index", &RectilinearMesh3D::index, (py::arg("index0"), py::arg("index1"), py::arg("index2")),
             "Return single index of the point indexed with index0, index1, and index2")
        .def("index0", &RectilinearMesh3D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectilinearMesh3D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("index2", &RectilinearMesh3D::index2, "Return index in the third axis of the point with given index", (py::arg("index")))
        .def("setOptimalOrdering", &RectilinearMesh3D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .def("setOrdering", &RectangularMesh3D__setOrdering<RectilinearMesh3D>, "Set desired ordering of the points in this mesh", (py::arg("order")))
    ;

        py::class_<RegularMesh2D, shared_ptr<RegularMesh2D>, py::bases<Mesh<2>>>("Regular2D",
        "Two-dimensional mesh\n\n"
        "Regular2D(ordering='01')\n    create empty mesh\n\n"
        "Regular2D(axis0, axis1, ordering='01')\n    create mesh with axes supplied as sequences of numbers\n\n"
        "ordering can be either '01', '10' and specifies ordering of the mesh points (last index changing fastest).",
        py::no_init
        )
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__empty<RegularMesh2D>, py::default_call_policies(), (py::arg("ordering")="10")))
        .def("__init__", py::make_constructor(&RectangularMesh2D__init__axes<RegularMesh2D, RegularMesh1D>, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("ordering")="10")))
//         .add_property("axis0", &RegularMesh2D_axis0, py::make_setter(&RegularMesh2D::c0),
//                       "List of points along the first (transverse) axis of the mesh")
//         .add_property("axis1", &RegularMesh2D_axis1, py::make_setter(&RegularMesh2D::c1),
//                       "List of points along the second (vertical) axis of the mesh")
        .def("empty", &RegularMesh2D::empty, "Return True if the mesh is empty")
        .def("clear", &RegularMesh2D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh2D__getitem__<RegularMesh2D>)
        .def("__iter__", py::range((auto (*)(const plask::RegularMesh2D& m)->decltype(m.begin_fast()))&std::begin, (auto (*)(const plask::RegularMesh2D& m)->decltype(m.end_fast()))&std::end))
        .def("index", &RegularMesh2D::index, "Return single index of the point indexed with index0 and index1", (py::arg("index0"), py::arg("index1")))
        .def("index0", &RegularMesh2D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RegularMesh2D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("setOptimalOrdering", &RegularMesh2D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .def("setOrdering", &RectangularMesh2D__setOrdering<RegularMesh2D>, "Set desired ordering of the points in this mesh", (py::arg("ordering")))
    ;

    py::class_<RegularMesh3D, shared_ptr<RegularMesh3D>, py::bases<Mesh<3>>>("Regular3D",
        "Two-dimensional mesh\n\n"
        "Regular3D(ordering='012')\n    create empty mesh\n\n"
        "Regular3D(axis0, axis1, axis2, ordering='012')\n    create mesh with axes supplied as mesh.Regular1D\n\n"
        "ordering can be any a string containing any permutation of and specifies ordering of the\n"
        "mesh points (last index changing fastest).",
        py::no_init
        )
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__empty<RegularMesh3D>, py::default_call_policies(), (py::arg("ordering")="012")))
        .def("__init__", py::make_constructor(&RectangularMesh3D__init__axes<RegularMesh3D, RegularMesh1D>, py::default_call_policies(), (py::arg("axis0"), "axis1", "axis2", py::arg("ordering")="012")))
//         .add_property("axis0", &RegularMesh3D_axis0, py::make_setter(&RegularMesh3D::c0),
//                       "List of points along the first (longitudinal) axis of the mesh")
//         .add_property("axis1", &RegularMesh3D_axis1, py::make_setter(&RegularMesh3D::c1),
//                       "List of points along the second (transverse) axis of the mesh")
//         .add_property("axis2", &RegularMesh3D_axis2, py::make_setter(&RegularMesh3D::c2),
//                       "List of points along the third (vertical) axis of the mesh")
        .def("empty", &RegularMesh3D::empty, "Return True if the mesh is empty")
        .def("clear", &RegularMesh3D::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectangularMesh3D__getitem__<RegularMesh3D>)
        .def("__iter__", py::range((auto (*)(const plask::RegularMesh3D& m)->decltype(m.begin_fast()))&std::begin, (auto (*)(const plask::RegularMesh3D& m)->decltype(m.end_fast()))&std::end))
        .def("index", &RegularMesh3D::index, (py::arg("index0"), py::arg("index1"), py::arg("index2")),
             "Return single index of the point indexed with index0, index1, and index2")
        .def("index0", &RegularMesh3D::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RegularMesh3D::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("index2", &RegularMesh3D::index2, "Return index in the third axis of the point with given index", (py::arg("index")))
        .def("setOptimalOrdering", &RegularMesh3D::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .def("setOrdering", &RectangularMesh3D__setOrdering<RegularMesh3D>, "Set desired ordering of the points in this mesh", (py::arg("order")))
    ;

}

}} // namespace plask::python
