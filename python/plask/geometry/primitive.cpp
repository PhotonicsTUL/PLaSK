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
    py::class_<Box2D, shared_ptr<Box2D>>("Box2D",
        u8"Box2D()\n"
        u8"Box2D(lower, upper)\n"
        u8"Box2D(left, bottom, right, top)\n\n"
        u8"Rectangular two-dimensional box.\n\n"
        u8"This class holds a rectangular box with its sides along the axes. It provides\n"
        u8"some basic geometric operations and is used mainly to represent 2D geometry\n"
        u8"bounding boxes.\n\n"
        u8"Args:\n"
        u8"    lower (plask.vec): Lower left corner of the box.\n"
        u8"    upper (plask.ver): Upper right corner of the box.\n"
        u8"    left (float): Left edge of the box.\n"
        u8"    bottom (float): Bottom edge of the box.\n"
        u8"    right (float): Right edge of the box.\n"
        u8"    top (float): Top edge of the box.\n\n",
        py::no_init)
        .def("__init__", py::make_constructor(&Box2D_constructor_default))
        .def("__init__", py::make_constructor(&Box2D_constructor_2vec, py::default_call_policies(), (py::arg("lower"), py::arg("upper"))))
        .def("__init__", py::make_constructor(&Box2D_constructor_4numbers, py::default_call_policies(), (py::arg("left"), py::arg("bottom"), py::arg("right"), py::arg("top"))))
        .def_readonly("lower", &Box2D::lower, u8"Lower left corner of the box.")
        .def_readonly("upper", &Box2D::upper, u8"Upper right corner of the box.")
        .add_property("left", &Box2D::getLeft, u8"Left edge of the box.")
        .add_property("right",  &Box2D::getRight, u8"Right edge of the box.")
        .add_property("top",  &Box2D::getTop, u8"Top edge of the box.")
        .add_property("bottom",  &Box2D::getBottom, u8"Bottom edge of the box.")
        .add_property("width", &Box2D::width, u8"Width of the box.")
        .add_property("height", &Box2D::height, u8"Height of the box.")
        .add_property("size", &Box2D::size, u8"Size of the box.")
        .add_property("center", &Box2D::center, u8"Center of the box.")
        .def("__nonzero__", &Box2D::isValid, u8"Return True if the box is valid.")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("contains", &Box2D::contains, py::args("point"),
            u8"Check if the point is inside the box.\n\n"
            u8"Args:\n"
            u8"    point (plask.vec): Point to test.\n")
        .def("__contains__", &Box2D::contains, py::args("point"), u8"Check if the point is inside the box.")
        .def("intersects", &Box2D::intersects, py::args("other"),
            u8"Check if this and the other box have common points.\n\n"
            u8"Args:\n"
            u8"    other (plask.geometry.Box2D): Box to check common points with.\n")
        .def("intersection", &Box2D::intersection, py::args("other"),
            u8"Get the biggest box which is included in both this and the other box.\n\n"
            u8"Args:\n"
            u8"    other (plask.geometry.Box2D): Box to make intersection with.\n")
        .def("__mult__", &Box2D::intersection)
        .def("extension", &Box2D::extension, py::args("other"),
             u8"Get the minimal box which include both this and other box.\n\n"
             u8"Args:\n"
             u8"    other (plask.geometry.Box2D): Box.\n")
        .def("__add__", &Box2D::extension)
        .def("translated", &Box2D::translated, py::args("trans"),
            u8"Get translated copy of this box.\n\n"
            u8"Args:\n"
            u8"   trans (plask.vec): Translation vector.")
        .def("__str__", &Box2D__str__)
        .def("__repr__", &Box2D__repr__)
    ;

    register_vector_of<Box2D>("Box2D");


    py::class_<Box3D, shared_ptr<Box3D>>("Box3D",
        u8"Box3D()\n"
        u8"Box3D(lower, upper)\n"
        u8"Box3D(back, left, bottom, front, right, top)\n\n"
        u8"Cuboidal three-dimensional box.\n\n"
        u8"This class holds a cuboidal box with its sides along the axes. It provides\n"
        u8"some basic geometric operations and is used mainly to represent 3D geometry\n"
        u8"bounding boxes.\n\n"
        u8"Args:\n"
        u8"    lower (plask.vec): Back lower left corner of the box.\n"
        u8"    upper (plask.ver): Front upper right corner of the box.\n"
        u8"    back (float): Back edge of the box.\n"
        u8"    left (float): Left edge of the box.\n"
        u8"    bottom (float): Bottom edge of the box.\n"
        u8"    front (float): Front edge of the box.\n"
        u8"    right (float): Right edge of the box.\n"
        u8"    top (float): Top edge of the box.\n",
        py::no_init)
        .def("__init__", py::make_constructor(&Box3D_constructor_default))
        .def("__init__", py::make_constructor(&Box3D_constructor_2vec, py::default_call_policies(), (py::arg("lower"), py::arg("upper"))))
        .def("__init__", py::make_constructor(&Box3D_constructor_4numbers, py::default_call_policies(), (py::arg("back"), py::arg("left"), py::arg("bottom"), py::arg("front"), py::arg("right"), py::arg("top"))))
        .def_readonly("lower", &Box3D::lower, u8"Closer lower left corner of the box.")
        .def_readonly("upper", &Box3D::upper, u8"Farther upper right corner of the box.")
        .add_property("front", &Box3D::getFront, u8"Front edge of the box.")
        .add_property("back", &Box3D::getBack, u8"Back edge of the box.")
        .add_property("left", &Box3D::getLeft, u8"Left edge of the box.")
        .add_property("right",  &Box3D::getRight, u8"Right edge of the box.")
        .add_property("top",  &Box3D::getTop, u8"Top edge of the box.")
        .add_property("bottom",  &Box3D::getBottom, u8"Bottom edge of the box.")
        .add_property("depth", &Box3D::depth, u8"Depth of the box.")
        .add_property("width", &Box3D::width, u8"Width of the box.")
        .add_property("height", &Box3D::height, u8"Height of the box.")
        .add_property("size", &Box3D::size, u8"Size of the box.")
        .add_property("center", &Box3D::center, u8"Center of the box.")
        .def("__nonzero__", &Box3D::isValid, u8"Return True if the box is valid.")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("contains", &Box3D::contains, py::args("point"),
            u8"Check if the point is inside the box.\n\n"
            u8"Args:\n"
            u8"    point (plask.vec): Point to test.\n")
        .def("__contains__", &Box3D::contains, py::args("point"), "Check if the point is inside the box.")
        .def("intersects", &Box3D::intersects,
            u8"Check if this and the other box have common points."
            u8"Args:\n"
            u8"    other (plask.geometry.Box3D): Box to check common points with.\n")
        .def("intersection", &Box3D::intersection, py::args("other"),
            u8"Get the biggest box which is included in both this and the other box.\n\n"
            u8"Args:\n"
            u8"    other (plask.geometry.Box3D): Box to make intersection with.\n")
        .def("extension", &Box3D::extension, py::args("other"),
            u8"Get the minimal box which include both this and other box.\n\n"
            u8"Args:\n"
            u8"    other (plask.geometry.Box3D): Box.\n")
        .def("translated", &Box3D::translated, py::args("trans"),
            u8"Get translated copy of this box.\n\n"
            u8"Args:\n"
            u8"   trans (plask.vec): Translation vector.")
        .def("__str__", &Box3D__str__)
        .def("__repr__", &Box3D__repr__)
    ;

    register_vector_of<Box3D>("Box3D");
}

}} // namespace plask::python
