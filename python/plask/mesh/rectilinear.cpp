#include "../python_globals.h"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>
#include <numpy/arrayobject.h>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/rectilinear1d.h>
#include <plask/mesh/rectilinear2d.h>
#include <plask/mesh/rectilinear3d.h>

namespace plask { namespace python {

struct Rectilinear1D_fromto_Sequence
{
    Rectilinear1D_fromto_Sequence() {
        boost::python::to_python_converter<RectilinearMesh1d, Rectilinear1D_fromto_Sequence>();
        boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<RectilinearMesh1d>());
    }

    static void* convertible(PyObject* obj_ptr) {
        if (!PySequence_Check(obj_ptr)) return NULL;
        return obj_ptr;
    }

    static void construct(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        void* storage = ((boost::python::converter::rvalue_from_python_storage<shared_ptr<RectilinearMesh1d>>*)data)->storage.bytes;
        py::stl_input_iterator<double> begin(py::object(py::handle<>(py::borrowed(obj_ptr)))), end;
        new(storage) RectilinearMesh1d(std::vector<double>(begin, end));
        data->convertible = storage;
    }

    static PyObject* convert(const RectilinearMesh1d& mesh) {
        npy_intp dims[] = { mesh.size() };
        PyObject* arr = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
        if (arr == nullptr) throw plask::CriticalException("cannot create array from mesh");
        std::copy(mesh.begin(), mesh.end(), (double*)PyArray_DATA(arr));
        return arr;
    }
};


template <typename T>
py::object RectilinearMesh__axis(py::object self, T* mesh, RectilinearMesh1d& axis) {
    npy_intp dims[] = { axis.size() };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void*)&(*axis.begin()));
    if (arr == nullptr) throw plask::CriticalException("cannot create array from mesh");
    py::incref(self.ptr()); PyArray_BASE(arr) = self.ptr(); // Make sure the mesh stays alive as long as the array
    return py::object(py::handle<>(arr));
}

py::object RectilinearMesh2d_axis0(py::object self) {
    RectilinearMesh2d* mesh = py::extract<RectilinearMesh2d*>(self);
    return RectilinearMesh__axis(self, mesh, mesh->c0);
}
py::object RectilinearMesh2d_axis1(py::object self) {
    RectilinearMesh2d* mesh = py::extract<RectilinearMesh2d*>(self);
    return RectilinearMesh__axis(self, mesh, mesh->c1);
}

py::object RectilinearMesh3d_axis0(py::object self) {
    RectilinearMesh3d* mesh = py::extract<RectilinearMesh3d*>(self);
    return RectilinearMesh__axis(self, mesh, mesh->c0);
}
py::object RectilinearMesh3d_axis1(py::object self) {
    RectilinearMesh3d* mesh = py::extract<RectilinearMesh3d*>(self);
    return RectilinearMesh__axis(self, mesh, mesh->c1);
}
py::object RectilinearMesh3d_axis2(py::object self) {
    RectilinearMesh3d* mesh = py::extract<RectilinearMesh3d*>(self);
    return RectilinearMesh__axis(self, mesh, mesh->c2);
}


static inline bool plask_import_array() {
    import_array1(false);
    return true;
}





void RectilinearMesh2d__setOrdering(RectilinearMesh2d& self, std::string order) {
    if (order == "01") self.setIterationOrder(RectilinearMesh2d::NORMAL_ORDER);
    else if (order == "10") self.setIterationOrder(RectilinearMesh2d::TRANSPOSED_ORDER);
    else {
        throw ValueError("order must be either '01' or '10'");
    }
}

shared_ptr<RectilinearMesh2d> RectilinearMesh2d__init__empty(std::string order) {
    auto mesh = make_shared<RectilinearMesh2d>();
    RectilinearMesh2d__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectilinearMesh2d> RectilinearMesh2d__init__axes(const RectilinearMesh1d& axis0, const RectilinearMesh1d& axis1, std::string order) {
    auto mesh = make_shared<RectilinearMesh2d>(axis0, axis1);
    RectilinearMesh2d__setOrdering(*mesh, order);
    return mesh;
}

shared_ptr<RectilinearMesh2d> RectilinearMesh2d__init__geometry(const GeometryElementD<2>& geometry, std::string order) {
    auto mesh = make_shared<RectilinearMesh2d>(geometry);
    RectilinearMesh2d__setOrdering(*mesh, order);
    return mesh;
}

Vec<2,double> RectilinearMesh2d__getitem__(const RectilinearMesh2d& self, py::object index) {
    try {
        int indx = py::extract<int>(index);
        return self[indx];
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 = self.c0.size() - index0;
    if (index0 < 0 || index0 >= self.c0.size()) {
        throw IndexError("first mesh index (%1%) out of range (0<=index<%2%)", index0, self.c0.size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 = self.c1.size() - index1;
    if (index1 < 0 || index1 >= self.c1.size()) {
        throw IndexError("second mesh index (%1%) out of range (0<=index<%2%)", index1, self.c1.size());
    }
    return self(index0, index1);
}

shared_ptr<RectilinearMesh3d> RectilinearMesh3d__init__empty() {
    return make_shared<RectilinearMesh3d>();
}

shared_ptr<RectilinearMesh3d> RectilinearMesh3d__init__axes(const RectilinearMesh1d& axis0, const RectilinearMesh1d& axis1, const RectilinearMesh1d& axis2) {
    return make_shared<RectilinearMesh3d>(axis0, axis1, axis2);
}

shared_ptr<RectilinearMesh3d> RectilinearMesh3d__init__geometry(const GeometryElementD<3>& geometry) {
    return make_shared<RectilinearMesh3d>(geometry);
}

Vec<3,double> RectilinearMesh3d__getitem__(const RectilinearMesh3d& self, py::object index) {
    try {
        int indx = py::extract<int>(index);
        return self[indx];
    } catch (py::error_already_set) {
        PyErr_Clear();
    }
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 = self.c0.size() - index0;
    if (index0 < 0 || index0 >= self.c0.size()) {
        throw IndexError("first mesh index (%1%) out of range (0<=index<%2%)", index0, self.c0.size());
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 = self.c1.size() - index1;
    if (index1 < 0 || index1 >= self.c1.size()) {
        throw IndexError("second mesh index (%1%) out of range (0<=index<%2%)", index1, self.c1.size());
    }
    int index2 = py::extract<int>(index[2]);
    if (index2 < 0) index2 = self.c2.size() - index2;
    if (index2 < 0 || index2 >= self.c2.size()) {
        throw IndexError("third mesh index (%1%) out of range (0<=index<%2%)", index2, self.c2.size());
    }
    return self(index0, index1, index2);
}

void RectilinearMesh3d__setOrdering(RectilinearMesh3d& self, std::string order) {
    if (order == "012") self.setIterationOrder(RectilinearMesh3d::ORDER_012);
    else if (order == "021") self.setIterationOrder(RectilinearMesh3d::ORDER_021);
    else if (order == "102") self.setIterationOrder(RectilinearMesh3d::ORDER_102);
    else if (order == "120") self.setIterationOrder(RectilinearMesh3d::ORDER_120);
    else if (order == "201") self.setIterationOrder(RectilinearMesh3d::ORDER_201);
    else if (order == "210") self.setIterationOrder(RectilinearMesh3d::ORDER_210);
    else {
        throw ValueError("order must be any permutation of '012'");
    }
}



void register_mesh_rectilinear()
{
    // Initialize numpy
    if (!plask_import_array()) throw(py::error_already_set());

    Rectilinear1D_fromto_Sequence();

    py::class_<RectilinearMesh2d, shared_ptr<RectilinearMesh2d>, py::bases<Mesh<2>>>("Rectilinear2D",
        "Two-dimensional mesh\n\n"
        "Rectilinear2D(ordering='01')\n    create empty mesh\n\n"
        "Rectilinear2D(axis0, axis1, ordering='01')\n    create mesh with axes supplied as sequences of numbers\n\n"
        "Rectilinear2D(geometry, ordering='01')\n    create coarse mesh based on bounding boxes of geometry elements\n\n"
        "ordering can be either '01', '10' and specifies initial ordering of the mesh points",
        py::no_init
        )
        .def("__init__", py::make_constructor(&RectilinearMesh2d__init__empty, py::default_call_policies(), (py::arg("ordering")="01")))
        .def("__init__", py::make_constructor(&RectilinearMesh2d__init__axes, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("ordering")="01")))
        .def("__init__", py::make_constructor(&RectilinearMesh2d__init__geometry, py::default_call_policies(), (py::arg("geometry"), py::arg("ordering")="01")))
        .add_property("axis0", &RectilinearMesh2d_axis0, py::make_setter(&RectilinearMesh2d::c0),
                      "List of points along the first (transverse) axis of the mesh")
        .add_property("axis1", &RectilinearMesh2d_axis1, py::make_setter(&RectilinearMesh2d::c1),
                      "List of points along the second (vertical) axis of the mesh")
        .def("empty", &RectilinearMesh2d::empty, "Return True if the mesh is empty")
        .def("clear", &RectilinearMesh2d::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectilinearMesh2d__getitem__)
        .def("__iter__", py::range(&RectilinearMesh2d::begin_fast, &RectilinearMesh2d::end_fast))
        .def("index", &RectilinearMesh2d::index, "Return single index of the point indexed with index0 and index1", (py::arg("index0"), py::arg("index1")))
        .def("index0", &RectilinearMesh2d::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectilinearMesh2d::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("setOptimalOrdering", &RectilinearMesh2d::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .def("setOrdering", &RectilinearMesh2d__setOrdering, "Set desired ordering of the points in this mesh", (py::arg("ordering")))
    ;

    py::class_<RectilinearMesh3d, shared_ptr<RectilinearMesh3d>, py::bases<Mesh<3>>>("Rectilinear3D",
        "Two-dimensional mesh\n\n"
        "Rectilinear3D()\n    create empty mesh\n\n"
        "Rectilinear3D(axis0,axis1,axis2)\n    create mesh with axes supplied as mesh.Rectilinear1D\n\n"
        "Rectilinear3D(geometry)\n    create coarse mesh based on bounding boxes of geometry elements\n\n",
        py::no_init
        )
        .def("__init__", py::make_constructor(&RectilinearMesh3d__init__empty))
        .def("__init__", py::make_constructor(&RectilinearMesh3d__init__axes, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("axis2"))))
        .def("__init__", py::make_constructor(&RectilinearMesh3d__init__geometry, py::default_call_policies(), (py::arg("geometry"))))
        .add_property("axis0", &RectilinearMesh3d_axis0, py::make_setter(&RectilinearMesh3d::c0),
                      "List of points along the first (longitudinal) axis of the mesh")
        .add_property("axis1", &RectilinearMesh3d_axis1, py::make_setter(&RectilinearMesh3d::c1),
                      "List of points along the second (transverse) axis of the mesh")
        .add_property("axis2", &RectilinearMesh3d_axis2, py::make_setter(&RectilinearMesh3d::c2),
                      "List of points along the third (vertical) axis of the mesh")
        .def("empty", &RectilinearMesh3d::empty, "Return True if the mesh is empty")
        .def("clear", &RectilinearMesh3d::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectilinearMesh3d__getitem__)
        .def("__iter__", py::range(&RectilinearMesh3d::begin, &RectilinearMesh3d::end))
        .def("index", &RectilinearMesh3d::index, (py::arg("index0"), py::arg("index1"), py::arg("index2")),
             "Return single index of the point indexed with index0, index1, and index2")
        .def("index0", &RectilinearMesh3d::index0, "Return index in the first axis of the point with given index", (py::arg("index")))
        .def("index1", &RectilinearMesh3d::index1, "Return index in the second axis of the point with given index", (py::arg("index")))
        .def("index2", &RectilinearMesh3d::index2, "Return index in the third axis of the point with given index", (py::arg("index")))
        .def("setOptimalOrdering", &RectilinearMesh3d::setOptimalIterationOrder, "Set the optimal ordering of the points in this mesh")
        .def("setOrdering", &RectilinearMesh3d__setOrdering, "Set desired ordering of the points in this mesh", (py::arg("order")))
    ;
}

}} // namespace plask::python
