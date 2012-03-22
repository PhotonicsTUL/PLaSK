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
    return self[index];
}

std::string RectilinearMesh1d__str__(const RectilinearMesh1d& self) {
    std::stringstream s;
    s << self;
    return s.str();
}

std::string RectilinearMesh1d__repr__(const RectilinearMesh1d& self) {
    return "Rectilinear1D(" + RectilinearMesh1d__str__(self) + ")";
}



Vec<2,double> RectilinearMesh2d__getitem__(const RectilinearMesh2d& self, py::object index) {
    return self(py::extract<int>(index[0]), py::extract<int>(index[1]));
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


Vec<3,double> RectilinearMesh3d__getitem__(const RectilinearMesh3d& self, py::object index) {
    return self(py::extract<int>(index[0]), py::extract<int>(index[1]), py::extract<int>(index[2]));
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


    py::class_<RectilinearMesh2d, shared_ptr<RectilinearMesh2d>>("Rectilinear2D",
        "Two-dimensional mesh\n\n"
        "Rectilinear2D()\n    create empty mesh\n\n"
        //TODO constructors
        )
        .add_property("axis0", py::make_getter(&RectilinearMesh2d::c0), &RectilinearMesh2d_setaxis0,
                      "Rectilinear1D mesh containing first (transverse) axis of the mesh")
        .add_property("axis1", py::make_getter(&RectilinearMesh2d::c1), &RectilinearMesh2d_setaxis1,
                      "Rectilinear1D mesh containing second (vertical) axis of the mesh")
        .def("size", &RectilinearMesh2d::size, "Return the size of the mesh")
        .def("empty", &RectilinearMesh2d::empty, "Return True if the mesh is empty")
        .def("clear", &RectilinearMesh2d::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectilinearMesh2d__getitem__)
        .def("__iter__", py::range(&RectilinearMesh2d::begin, &RectilinearMesh2d::end))
    ;


    py::class_<RectilinearMesh3d, shared_ptr<RectilinearMesh3d>>("Rectilinear3D",
        "Two-dimensional mesh\n\n"
        "Rectilinear2D()\n    create empty mesh\n\n"
        //TODO constructors
        )
        .add_property("axis0", py::make_getter(&RectilinearMesh3d::c0), &RectilinearMesh2d_setaxis0,
                      "Rectilinear1D mesh containing first (longitudat) axis of the mesh")
        .add_property("axis0", py::make_getter(&RectilinearMesh3d::c1), &RectilinearMesh2d_setaxis0,
                      "Rectilinear1D mesh containing second (transverse) axis of the mesh")
        .add_property("axis2", py::make_getter(&RectilinearMesh3d::c2), &RectilinearMesh2d_setaxis1,
                      "Rectilinear1D mesh containing third (vertical) axis of the mesh")
        .def("size", &RectilinearMesh3d::size, "Return the size of the mesh")
        .def("empty", &RectilinearMesh3d::empty, "Return True if the mesh is empty")
        .def("clear", &RectilinearMesh3d::clear, "Remove all points from the mesh")
        .def("__getitem__", &RectilinearMesh3d__getitem__)
        .def("__iter__", py::range(&RectilinearMesh3d::begin, &RectilinearMesh3d::end))
    ;
    py::implicitly_convertible<RectilinearMesh3d, RectilinearMesh3d::External>();
}

}} // namespace plask::python
