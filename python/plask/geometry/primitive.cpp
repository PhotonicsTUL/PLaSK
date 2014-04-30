#include "geometry.h"

#include <plask/geometry/primitives.h>

namespace plask { namespace python {

// Default constructor wrap
static shared_ptr<Box2D> Box2D_constructor_default() {
    return shared_ptr<Box2D> { new Box2D() };
}
// Init constructor wrap
static shared_ptr<Box2D> Box2D_constructor_2vec(const Vec<2,double>& lower, const Vec<2,double>& upper) {
    shared_ptr<Box2D> R { new Box2D(lower, upper) };
    R->fix();
    return R;
}
// Init constructor wrap
static shared_ptr<Box2D> Box2D_constructor_4numbers(double x1, double y1, double x2, double y2) {
    shared_ptr<Box2D> R { new Box2D(Vec<2,double>(x1,y1), Vec<2,double>(x2,y2)) };
    R->fix();
    return R;
}
// __str__(v)
static std::string Box2D__str__(const Box2D& to_print) {
    std::stringstream out;
    out << to_print;
    return out.str();
}
// __repr__(v)
static std::string Box2D__repr__(const Box2D& to_print) {
    std::stringstream out;
    out << "plask.geometry.Box2D(" << to_print.lower.c0 << ", " << to_print.lower.c1 << ", "
                                   << to_print.upper.c0 << ", " << to_print.upper.c1 << ")";
    return out.str();
}


// Default constructor wrap
static shared_ptr<Box3D> Box3D_constructor_default() {
    return shared_ptr<Box3D> { new Box3D() };
}
// Init constructor wrap
static shared_ptr<Box3D> Box3D_constructor_2vec(const Vec<3,double>& lower, const Vec<3,double>& upper) {
    shared_ptr<Box3D> R { new Box3D(lower, upper) };
    R->fix();
    return R;
}
// Init constructor wrap
static shared_ptr<Box3D> Box3D_constructor_4numbers(double x1, double y1, double z1, double x2, double y2, double z2) {
    shared_ptr<Box3D> R { new Box3D(Vec<3,double>(x1,y1,z1), Vec<3,double>(x2,y2,z2)) };
    R->fix();
    return R;
}
// __str__(v)
static std::string Box3D__str__(const Box3D& self) {
    std::stringstream out;
    out << self;
    return out.str();
}
// __repr__(v)
static std::string Box3D__repr__(const Box3D& self) {
    std::stringstream out;
    out << "plask.geometry.Box3D(" << self.lower.c0 << ", " << self.lower.c1 << ", " << self.lower.c2 << ", "
                                   << self.upper.c0 << ", " << self.upper.c1 << ", " << self.upper.c2 << ")";
    return out.str();
}

/// Register primitives to Python
void register_geometry_primitive()
{
    void (Box2D::*includeR2p)(const Vec<2,double>&) = &Box2D::makeInclude;
    void (Box2D::*includeR2R)(const Box2D&)       = &Box2D::makeInclude;

    py::class_<Box2D, shared_ptr<Box2D>>("Box2D",
        "Rectangular two-dimensional box. Provides some basic operation on boxes.\n\n"
        "Box2D()\n    create empty box\n\n"
        "Box2D(lower, upper)\n    create box with opposite corners described by 2D vectors\n\n"
        "Box2D(l1, l2, u1, u2)\n    create box with opposite corners described by two coordinates\n",
        py::no_init)
        .def("__init__", py::make_constructor(&Box2D_constructor_default))
        .def("__init__", py::make_constructor(&Box2D_constructor_2vec, py::default_call_policies(), (py::arg("lower"), py::arg("upper"))))
        .def("__init__", py::make_constructor(&Box2D_constructor_4numbers, py::default_call_policies(), (py::arg("l0"), py::arg("l1"), py::arg("u0"), py::arg("u1"))))
        .def_readwrite("lower", &Box2D::lower, "Lower left corner of the box")
        .def_readwrite("upper", &Box2D::upper, "Upper right corner of the box")
        .add_property("left", &Box2D::getLeft, &Box2D::setLeft, "Left edge of the box")
        .add_property("right",  &Box2D::getRight, &Box2D::setRight, "Right edge of the box")
        .add_property("top",  &Box2D::getTop, &Box2D::setTop, "Top edge of the box")
        .add_property("bottom",  &Box2D::getBottom, &Box2D::setBottom, "Bottom edge of the box")
        .add_property("width", &Box2D::width, "Width of the box")
        .add_property("height", &Box2D::height, "Height of the box")
        .def("__nonzero__", &Box2D::isValid, "Return True if the box is valid.")
        .def("fix", &Box2D::fix, "Ensure that lower[0] <= upper[0] and lower[1] <= upper[1]. Exchange components of lower and upper if necessary.")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("contains", &Box2D::contains, py::args("point"), "Check if the point is inside the box.")
        .def("__contains__", &Box2D::contains, py::args("point"), "Check if the point is inside the box.")
        .def("intersects", &Box2D::intersects, py::args("other"), "Check if this and the other box have common points.")
        .def("extend", includeR2p, py::args("point"), "")
        .def("extend", includeR2R, py::args("box"),
             "Extend the box to include the given point or box.\n\n"
             "Args:\n"
             "    point (plask.vector): The point to include.\n"
             "    box (Box2D): The box to include.\n"
            )
        .def("translated", &Box2D::translated, py::args("trans"), "Get translated copy of this box")
        .def("translate", &Box2D::translate, py::args("trans"), "Translate this box")
        .def("__str__", &Box2D__str__)
        .def("__repr__", &Box2D__repr__)
    ;

    register_vector_of<Box2D>("Box2D");


    void (Box3D::*includeR3p)(const Vec<3,double>&) = &Box3D::makeInclude;
    void (Box3D::*includeR3R)(const Box3D&)       = &Box3D::makeInclude;

    py::class_<Box3D, shared_ptr<Box3D>>("Box3D",
        "Cuboidal three-dimensional box. Provides some basic operation on boxes.\n\n"
        "Box3D()\n    create empty box\n\n"
        "Box3D(lower, upper)\n    create box with opposite corners described by 3D vectors\n\n"
        "Box3D(l1, l2, l3, u1, u2, u3)\n    create box with opposite corners described by three coordinates\n",
        py::no_init)
        .def("__init__", py::make_constructor(&Box3D_constructor_default))
        .def("__init__", py::make_constructor(&Box3D_constructor_2vec, py::default_call_policies(), (py::arg("lower"), py::arg("upper"))))
        .def("__init__", py::make_constructor(&Box3D_constructor_4numbers, py::default_call_policies(), (py::arg("l0"), py::arg("l1"), py::arg("l2"), py::arg("u0"), py::arg("u1"), py::arg("u2"))))
        .def_readwrite("lower", &Box3D::lower, "Closer lower left corner of the box")
        .def_readwrite("upper", &Box3D::upper, "Farther upper right corner of the box")
        .add_property("front", &Box3D::getFront, &Box3D::setFront, "Front edge of the box")
        .add_property("back", &Box3D::getBack, &Box3D::setBack, "Back edge of the box")
        .add_property("left", &Box3D::getLeft, &Box3D::setLeft, "Left edge of the box")
        .add_property("right",  &Box3D::getRight, &Box3D::setRight, "Right edge of the box")
        .add_property("top",  &Box3D::getTop, &Box3D::setTop, "Top edge of the box")
        .add_property("bottom",  &Box3D::getBottom, &Box3D::setBottom, "Bottom edge of the box")
        .add_property("depth", &Box3D::depth, "Depth of the box")
        .add_property("width", &Box3D::width, "Width of the box")
        .add_property("height", &Box3D::height, "Height of the box")
        .def("__nonzero__", &Box3D::isValid, "Return True if the box is valid.")
        .def("fix", &Box3D::fix, "Ensure that lower[0] <= upper[0], lower[1] <= upper[1], and lower[2] <= upper[3]. Exchange components of lower and upper if necessary.")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("contains", &Box3D::contains, "Check if the point is inside the box.")
        .def("__contains__", &Box2D::contains, py::args("point"), "Check if the point is inside the box.")
        .def("intersects", &Box3D::intersects, "Check if this and the other box have common points.")
        .def("extend", includeR3p, (py::arg("point")), "")
        .def("extend", includeR3R, (py::arg("box")),
             "Extend the box to include the given point or box.\n\n"
             "Args:\n"
             "    point (plask.vector): The point to include.\n"
             "    box (Box3D): The box to include.\n"
            )
        .def("translated", &Box2D::translated, py::args("trans"), "Get translated copy of this box")
        .def("translate", &Box2D::translate, py::args("trans"), "Translate this box")
        .def("__str__", &Box3D__str__)
        .def("__repr__", &Box3D__repr__)
    ;

    register_vector_of<Box3D>("Box3D");
}

}} // namespace plask::python
