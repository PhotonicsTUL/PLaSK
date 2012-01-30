#include "geometry.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <plask/geometry/primitives.h>

namespace plask { namespace python {

// Default constructor wrap
static shared_ptr<Rect2d> Rect2d_constructor_default() {
    return shared_ptr<Rect2d> { new Rect2d() };
}
// Init constructor wrap
static shared_ptr<Rect2d> Rect2d_constructor_2vec(const Vec<2,double>& lower, const Vec<2,double>& upper) {
    shared_ptr<Rect2d> R { new Rect2d(lower, upper) };
    R->fix();
    return R;
}
// Init constructor wrap
static shared_ptr<Rect2d> Rect2d_constructor_4numbers(double x1, double y1, double x2, double y2) {
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
    out << "Box2D(" << to_print.lower.c0 << ", " << to_print.lower.c1 << ", "
                    << to_print.upper.c0 << ", " << to_print.upper.c1 << ")";
    return out.str();
}


// Default constructor wrap
static shared_ptr<Rect3d> Rect3d_constructor_default() {
    return shared_ptr<Rect3d> { new Rect3d() };
}
// Init constructor wrap
static shared_ptr<Rect3d> Rect3d_constructor_2vec(const Vec<3,double>& lower, const Vec<3,double>& upper) {
    shared_ptr<Rect3d> R { new Rect3d(lower, upper) };
    R->fix();
    return R;
}
// Init constructor wrap
static shared_ptr<Rect3d> Rect3d_constructor_4numbers(double x1, double y1, double z1, double x2, double y2, double z2) {
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
    out << "Box3D(" << to_print.lower.c0 << ", " << to_print.lower.c1 << ", " << to_print.lower.c2 << ", "
                    << to_print.upper.c0 << ", " << to_print.upper.c1 << ", " << to_print.upper.c2 << ")";
    return out.str();
}

/// Register primitives to Python
void register_geometry_primitive()
{
    void (Rect2d::*includeR2p)(const Vec<2,double>&) = &Rect2d::include;
    void (Rect2d::*includeR2R)(const Rect2d&)       = &Rect2d::include;

    py::class_<Rect2d, shared_ptr<Rect2d>>("Box2D",
        "Rectangular two-dimensional box. Provides some basic operation on boxes.\n\n"
        "Box2D() -> create empty box\n\n"
        "Box2D(lower, upper) -> create box with opposite corners described by 2D vectors\n\n"
        "Box2D(l1, l2, u1, u2) -> create box with opposite corners described by two coordinates\n\n"
        )
        .def("__init__", py::make_constructor(&Rect2d_constructor_default))
        .def("__init__", py::make_constructor(&Rect2d_constructor_2vec))
        .def("__init__", py::make_constructor(&Rect2d_constructor_4numbers))
        .def_readwrite("lower", &Rect2d::lower, "Lower left corner of the box")
        .def_readwrite("upper", &Rect2d::upper, "Upper right corner of the box")
        .def("fix", &Rect2d::fix, "Ensure that lower[0] <= upper[0] and lower[1] <= upper[1]. Exchange components of lower and upper if necessary.")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("inside", &Rect2d::inside, py::args("point"), "Check if the point is inside the box.")
        .def("intersect", &Rect2d::intersect, py::args("other"), "Check if this and the other box have common points.")
        .def("include", includeR2p, py::args("point"), "Make this box, the minimal one which include this and given point")
        .def("include", includeR2R, py::args("other"), "Make this box, the minimal one which include this and the other box.")
        .def("translated", &Rect2d::translated, py::args("trans"), "Get translated copy of this box")
        .def("translate", &Rect2d::translate, py::args("trans"), "Translate this box")
        .def("__str__", &Rect2d__str__)
        .def("__repr__", &Rect2d__repr__)
    ;


    void (Rect3d::*includeR3p)(const Vec<3,double>&) = &Rect3d::include;
    void (Rect3d::*includeR3R)(const Rect3d&)       = &Rect3d::include;

    py::class_<Rect3d, shared_ptr<Rect3d>>("Box3D",
        "Cuboidal three-dimensional box. Provides some basic operation on boxes.\n\n"
        "Box3D() -> create empty box\n"
        "Box3D(lower, upper) -> create box with opposite corners described by 3D vectors\n"
        "Box3D(l1, l2, l3, u1, u2, u3) -> create box with opposite corners described by three coordinates"
        )
        .def("__init__", py::make_constructor(&Rect3d_constructor_default))
        .def("__init__", py::make_constructor(&Rect3d_constructor_2vec))
        .def("__init__", py::make_constructor(&Rect3d_constructor_4numbers))
        .def_readwrite("lower", &Rect3d::lower, "Closer lower left corner of the box")
        .def_readwrite("upper", &Rect3d::upper, "Farther upper right corner of the box")
        .def("fix", &Rect3d::fix, "Ensure that lower[0] <= upper.c0, lower[1] <= upper[1], and lower[2] <= upper[3].  Exchange components of lower and upper if necessary.")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("inside", &Rect3d::inside, "Check if the point is inside the box.")
        .def("intersect", &Rect3d::intersect, "Check if this and the other box have common points.")
        .def("include", includeR3p, "Make this box, the minimal one which include this and given point")
        .def("include", includeR3R, "Make this box, the minimal one which include this and the other box.")
        .def("translated", &Rect3d::translated)
        .def("translate", &Rect3d::translate)
        .def("__str__", &Rect3d__str__)
        .def("__repr__", &Rect3d__repr__)
    ;

    py::class_< std::vector<Rect2d>, shared_ptr<std::vector<Rect2d>> >("Box2D_list")
        .def(py::vector_indexing_suite<std::vector<Rect2d>>())
    ;

    py::class_< std::vector<Rect3d>, shared_ptr<std::vector<Rect3d>> >("Box3D_list")
        .def(py::vector_indexing_suite<std::vector<Rect3d>>())
    ;
}

}} // namespace plask::python
