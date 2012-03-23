#include "../globals.h"
#include <algorithm>
#include <boost/python/stl_iterator.hpp>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/rectilinear1d.h>
#include <plask/mesh/rectilinear2d.h>
#include <plask/mesh/rectilinear3d.h>

namespace plask { namespace python {

void RectilinearMesh1d_addPoints(RectilinearMesh1d& self, py::object points) {
    py::stl_input_iterator<double> begin(points), end;
    std::vector<double> data(begin, end);
    std::sort(data.begin(), data.end());
    self.addOrderedPoints(data.begin(), data.end(), data.size());
}

shared_ptr<RectilinearMesh1d> RectilinearMesh1d__init__(py::object points) {
    shared_ptr<RectilinearMesh1d> mesh(new RectilinearMesh1d);
    if (points != py::object()) RectilinearMesh1d_addPoints(*mesh, points);
    return mesh;
}

double RectilinearMesh1d__getitem__(const RectilinearMesh1d& self, int index) {
    if (index < 0) index = self.size() - index;
    if (index < 0 || index >= self.size()) {
        PyErr_SetString(PyExc_IndexError, format("mesh index (%1%) out of range (0<=index<%2%)", index, self.size()).c_str());
        throw py::error_already_set();
    }
    return self[index];
}

std::string RectilinearMesh1d__str__(const RectilinearMesh1d& self) {
    std::stringstream s;
    s << self;
    return s.str();
}

std::string RectilinearMesh1d__repr__(const RectilinearMesh1d& self) {
    return "meshes.Rectilinear1D(" + RectilinearMesh1d__str__(self) + ")";
}



shared_ptr<RectilinearMesh2d> RectilinearMesh2d__init__empty() {
    return make_shared<RectilinearMesh2d>();
}

shared_ptr<RectilinearMesh2d> RectilinearMesh2d__init__axes(const RectilinearMesh1d& axis0, const RectilinearMesh1d& axis1) {
    return make_shared<RectilinearMesh2d>(axis0, axis1);
}

shared_ptr<RectilinearMesh2d> RectilinearMesh2d__init__geometry(const GeometryElementD<2>& geometry) {
    return make_shared<RectilinearMesh2d>(geometry);
}

Vec<2,double> RectilinearMesh2d__getitem__(const RectilinearMesh2d& self, py::object index) {
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 = self.c0.size() - index0;
    if (index0 < 0 || index0 >= self.c0.size()) {
        PyErr_SetString(PyExc_IndexError, format("first mesh index (%1%) out of range (0<=index<%2%)", index0, self.c0.size()).c_str());
        throw py::error_already_set();
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 = self.c1.size() - index1;
    if (index1 < 0 || index1 >= self.c1.size()) {
        PyErr_SetString(PyExc_IndexError, format("second mesh index (%1%) out of range (0<=index<%2%)", index1, self.c1.size()).c_str());
        throw py::error_already_set();
    }
    return self(index0, index1);
}

void RectilinearMesh2d_setaxis0(RectilinearMesh2d& self, py::object points) {
    py::stl_input_iterator<double> begin(points), end;
    std::vector<double> data(begin, end);
    std::sort(data.begin(), data.end());
    self.c0.addOrderedPoints(data.begin(), data.end(), data.size());
}

void RectilinearMesh2d_setaxis1(RectilinearMesh2d& self, py::object points) {
    py::stl_input_iterator<double> begin(points), end;
    std::vector<double> data(begin, end);
    std::sort(data.begin(), data.end());
    self.c1.addOrderedPoints(data.begin(), data.end(), data.size());
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
    int index0 = py::extract<int>(index[0]);
    if (index0 < 0) index0 = self.c0.size() - index0;
    if (index0 < 0 || index0 >= self.c0.size()) {
        PyErr_SetString(PyExc_IndexError, format("first mesh index (%1%) out of range (0<=index<%2%)", index0, self.c0.size()).c_str());
        throw py::error_already_set();
    }
    int index1 = py::extract<int>(index[1]);
    if (index1 < 0) index1 = self.c1.size() - index1;
    if (index1 < 0 || index1 >= self.c1.size()) {
        PyErr_SetString(PyExc_IndexError, format("second mesh index (%1%) out of range (0<=index<%2%)", index1, self.c1.size()).c_str());
        throw py::error_already_set();
    }
    int index2 = py::extract<int>(index[2]);
    if (index2 < 0) index2 = self.c2.size() - index2;
    if (index2 < 0 || index2 >= self.c2.size()) {
        PyErr_SetString(PyExc_IndexError, format("third mesh index (%1%) out of range (0<=index<%2%)", index2, self.c2.size()).c_str());
        throw py::error_already_set();
    }
    return self(index0, index1, index2);
}

void RectilinearMesh3d_setaxis0(RectilinearMesh3d& self, py::object points) {
    py::stl_input_iterator<double> begin(points), end;
    std::vector<double> data(begin, end);
    std::sort(data.begin(), data.end());
    self.c0.addOrderedPoints(data.begin(), data.end(), data.size());
}

void RectilinearMesh3d_setaxis1(RectilinearMesh3d& self, py::object points) {
    py::stl_input_iterator<double> begin(points), end;
    std::vector<double> data(begin, end);
    std::sort(data.begin(), data.end());
    self.c1.addOrderedPoints(data.begin(), data.end(), data.size());
}

void RectilinearMesh3d_setaxis2(RectilinearMesh3d& self, py::object points) {
    py::stl_input_iterator<double> begin(points), end;
    std::vector<double> data(begin, end);
    std::sort(data.begin(), data.end());
    self.c2.addOrderedPoints(data.begin(), data.end(), data.size());
}



void register_mesh_rectilinear()
{
    py::class_<RectilinearMesh1d, shared_ptr<RectilinearMesh1d>>("Rectilinear1D",
        "One-dimensional mesh\n\n"
        "Rectilinear1D()\n    create empty mesh\n\n"
        "Rectilinear1D(points)\n    create mesh filled with sequence of points\n\n",
        py::no_init)
        .def("__init__", py::make_constructor(&RectilinearMesh1d__init__, py::default_call_policies(), (py::arg("points")=py::object())))
        .def("index", &RectilinearMesh1d::findIndex, "Find index of the point with specified value", (py::arg("value")))
        .def(py::self == py::other<RectilinearMesh1d>())
        .def("size", &RectilinearMesh1d::size, "Return the size of the mesh")
        .def("empty", &RectilinearMesh1d::empty, "Return True if the mesh is empty")
        .def("addPoint", &RectilinearMesh1d::addPoint, "Add point to the mesh", (py::arg("value")))
        .def("addPoints", &RectilinearMesh1d_addPoints, "Add sequence of the points to the mesh", (py::arg("points")))
        .def("addPointLinear", &RectilinearMesh1d::addPointsLinear, "Add equally distributed points",
             (py::arg("first"), py::arg("last"), py::arg("count")))
        .def("clear", &RectilinearMesh1d::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectilinearMesh1d__getitem__)
        .def("__str__", &RectilinearMesh1d__str__)
        .def("__repr__", &RectilinearMesh1d__repr__)
        .def("__iter__", py::range(&RectilinearMesh1d::begin, &RectilinearMesh1d::end))
    ;


    py::class_<RectilinearMesh2d, shared_ptr<RectilinearMesh2d>, py::bases<Mesh<2>>>("Rectilinear2D",
        "Two-dimensional mesh\n\n"
        "Rectilinear2D()\n    create empty mesh\n\n"
        "Rectilinear2D(axis0,axis1)\n    create mesh with axes supplied as meshes.Rectilinear1D\n\n"
        "Rectilinear2D(geometry)\n    create coarse mesh based on bounding boxes of geometry elements\n\n",
        py::no_init
        )
        .def("__init__", py::make_constructor(&RectilinearMesh2d__init__empty))
        .def("__init__", py::make_constructor(&RectilinearMesh2d__init__axes, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"))))
        .def("__init__", py::make_constructor(&RectilinearMesh2d__init__geometry, py::default_call_policies(), (py::arg("geometry"))))
        .add_property("axis0", py::make_getter(&RectilinearMesh2d::c0), &RectilinearMesh2d_setaxis0,
                      "Rectilinear1D mesh containing first (transverse) axis of the mesh")
        .add_property("axis1", py::make_getter(&RectilinearMesh2d::c1), &RectilinearMesh2d_setaxis1,
                      "Rectilinear1D mesh containing second (vertical) axis of the mesh")
        .def("empty", &RectilinearMesh2d::empty, "Return True if the mesh is empty")
        .def("clear", &RectilinearMesh2d::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectilinearMesh2d__getitem__)
        .def("__iter__", py::range(&RectilinearMesh2d::begin_fast, &RectilinearMesh2d::end_fast))
        //.def("makeOptimized", &RectilinearMesh2d::makeOptimized, py::return_value_policy<py::manage_new_object>(), "Return copy of this mesh with optimal ordering points")
    ;

    py::class_<RectilinearMesh3d, shared_ptr<RectilinearMesh3d>, py::bases<Mesh<3>>>("Rectilinear3D",
        "Two-dimensional mesh\n\n"
        "Rectilinear3D()\n    create empty mesh\n\n"
        "Rectilinear3D(axis0,axis1,axis2)\n    create mesh with axes supplied as meshes.Rectilinear1D\n\n"
        "Rectilinear3D(geometry)\n    create coarse mesh based on bounding boxes of geometry elements\n\n",
        py::no_init
        )
        .def("__init__", py::make_constructor(&RectilinearMesh3d__init__empty))
        .def("__init__", py::make_constructor(&RectilinearMesh3d__init__axes, py::default_call_policies(), (py::arg("axis0"), py::arg("axis1"), py::arg("axis2"))))
        .def("__init__", py::make_constructor(&RectilinearMesh3d__init__geometry, py::default_call_policies(), (py::arg("geometry"))))
        .add_property("axis0", py::make_getter(&RectilinearMesh3d::c0), &RectilinearMesh2d_setaxis0,
                      "Rectilinear1D mesh containing first (longitudinal) axis of the mesh")
        .add_property("axis0", py::make_getter(&RectilinearMesh3d::c1), &RectilinearMesh2d_setaxis0,
                      "Rectilinear1D mesh containing second (transverse) axis of the mesh")
        .add_property("axis2", py::make_getter(&RectilinearMesh3d::c2), &RectilinearMesh2d_setaxis1,
                      "Rectilinear1D mesh containing third (vertical) axis of the mesh")
        .def("empty", &RectilinearMesh3d::empty, "Return True if the mesh is empty")
        .def("clear", &RectilinearMesh3d::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectilinearMesh3d__getitem__)
        .def("__iter__", py::range(&RectilinearMesh3d::begin, &RectilinearMesh3d::end))
    ;
}

}} // namespace plask::python
