#ifndef PLASK__PYTHON_GEOMETRY_PRIMITIVE_H
#define PLASK__PYTHON_GEOMETRY_PRIMITIVE_H

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
namespace py = boost::python;

#include <plask/geometry/primitives.h>
#include "geometry.h"

namespace plask { namespace python {

// Default constructor wrap
static shared_ptr<Rect2d> Rect2d_constructor_default() {
    return shared_ptr<Rect2d> { new Rect2d() };
}
// Init constructor wrap
static shared_ptr<Rect2d> Rect2d_constructor_2vec(const Vec<2,double>& lower, const Vec<2,double>& upper) {
    return shared_ptr<Rect2d> { new Rect2d(lower, upper) };
}
// Init constructor wrap
static shared_ptr<Rect2d> Rect2d_constructor_4numbers(double x1, double y1, double x2, double y2) {
    //TODO: check plask.axes to set components properly
    shared_ptr<Rect2d> R { new Rect2d(Vec<2,double>(x1,y1), Vec<2,double>(x2,y2)) };
    R->fix();
    return R;
}
// __str__(v)
static std::string Rect2d__str__(const Rect2d& to_print) {
    std::stringstream out;
    out << to_print;
    return out.str();
}
// __repr__(v)
static std::string Rect2d__repr__(const Rect2d& to_print) {
    std::stringstream out;
    out << "Rect2D(" << to_print << ")";
    return out.str();
}


// Default constructor wrap
static shared_ptr<Rect3d> Rect3d_constructor_default() {
    return shared_ptr<Rect3d> { new Rect3d() };
}
// Init constructor wrap
static shared_ptr<Rect3d> Rect3d_constructor_2vec(const Vec<3,double>& lower, const Vec<3,double>& upper) {
    return shared_ptr<Rect3d> { new Rect3d(lower, upper) };
}
// Init constructor wrap
static shared_ptr<Rect3d> Rect3d_constructor_4numbers(double x1, double y1, double z1, double x2, double y2, double z2) {
    //TODO: check plask.axes to set components properly
    shared_ptr<Rect3d> R { new Rect3d(Vec<3,double>(x1,y1,z1), Vec<3,double>(x2,y2,z2)) };
    R->fix();
    return R;
}
// __str__(v)
static std::string Rect3d__str__(const Rect3d& to_print) {
    std::stringstream out;
    out << to_print;
    return out.str();
}
// __repr__(v)
static std::string Rect3d__repr__(const Rect3d& to_print) {
    std::stringstream out;
    out << "Rect3D(" << to_print << ")";
    return out.str();
}




/// Register primitives to Python
inline static void register_geometry_primitive()
{
    void (Rect2d::*includeR2p)(const Vec<2,double>&) = &Rect2d::include;
    void (Rect2d::*includeR2R)(const Rect2d&)       = &Rect2d::include;

    py::class_<Rect2d,shared_ptr<Rect2d>>("Rect2D", "Rectangle class. Allows for some basic operation on rectangles.")
        .def("__init__", py::make_constructor(&Rect2d_constructor_default))
        .def("__init__", py::make_constructor(&Rect2d_constructor_2vec))
        .def("__init__", py::make_constructor(&Rect2d_constructor_4numbers))
        .def_readwrite("lower", &Rect2d::lower, "Lower left corner of the rectangle")
        .def_readwrite("upper", &Rect2d::upper, "Upper right corner of the rectangle")
        .def("fix", &Rect2d::fix, "Ensure that lower[0] <= upper[0] and lower[1] <= upper[1]. Exchange components of lower and upper if necessary.")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("inside", &Rect2d::inside, "Check if the point is inside rectangle.")
        .def("intersect", &Rect2d::intersect, "Check if this and other rectangles have common points.")
        .def("include", includeR2p, "Make this rectangle, the minimal one which include this and given point")
        .def("include", includeR2R, "Make this rectangle, the minimal one which include this and the other rectangle.")
        .def("translated", &Rect2d::translated)
        .def("translate", &Rect2d::translate)
        .def("__str__", &Rect2d__str__)
        .def("__repr__", &Rect2d__repr__)
    ;


    void (Rect3d::*includeR3p)(const Vec<3,double>&) = &Rect3d::include;
    void (Rect3d::*includeR3R)(const Rect3d&)       = &Rect3d::include;

    py::class_<Rect3d,shared_ptr<Rect3d>>("Rect3D", "Cuboid class. Allows for some basic operation on rectangles.")
        .def("__init__", py::make_constructor(&Rect3d_constructor_default))
        .def("__init__", py::make_constructor(&Rect3d_constructor_2vec))
        .def("__init__", py::make_constructor(&Rect3d_constructor_4numbers))
        .def_readwrite("lower", &Rect3d::lower, "Closer lower left corner of the rectangle")
        .def_readwrite("upper", &Rect3d::upper, "Farer upper right corner of the rectangle")
        .def("fix", &Rect3d::fix, "Ensure that lower[0] <= upper.c0, lower[1] <= upper[1], and lower[2] <= upper[3].  Exchange components of lower and upper if necessary.")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("inside", &Rect3d::inside, "Check if the point is inside cuboid.")
        .def("intersect", &Rect3d::intersect, "Check if this and other cuboid have common points.")
        .def("include", includeR3p, "Make this cuboid, the minimal one which include this and given point")
        .def("include", includeR3R, "Make this cuboid, the minimal one which include this and the other cuboid.")
        .def("translated", &Rect3d::translated)
        .def("translate", &Rect3d::translate)
        .def("__str__", &Rect3d__str__)
        .def("__repr__", &Rect3d__repr__)
    ;


    py::class_<std::vector<Rect2d>>("ListRect2D")
        .def(py::vector_indexing_suite<std::vector<Rect2d>>())
    ;

    py::class_<std::vector<Rect3d>>("ListRect3D")
        .def(py::vector_indexing_suite<std::vector<Rect3d>>())
    ;

}

}} // namespace plask::python
#endif // PLASK__PYTHON_GEOMETRY_PRIMITIVE_H
