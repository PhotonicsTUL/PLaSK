#include "geometry.h"

#include <plask/geometry/leaf.h>

namespace plask { namespace python {


// Rectangle constructor wraps
static shared_ptr<Block<2>> Rectangle_constructor_wh(double w, double h, shared_ptr<Material> material) {
    return shared_ptr<Block<2>> ( new Block<2>(Vec<2,double>(w,h), material) );
}
static shared_ptr<Block<2>> Rectangle_constructor_vec(const Vec<2,double>& size, shared_ptr<Material> material) {
    return shared_ptr<Block<2>> ( new Block<2>(size, material) );
}

// Cuboid constructor wraps
static shared_ptr<Block<3>> Cuboid_constructor_dwh(double d, double w, double h, shared_ptr<Material> material) {
    return shared_ptr<Block<3>> ( new Block<3>(Vec<3,double>(d,w,h), material) );
}
static shared_ptr<Block<3>> Cuboid_constructor_vec(const Vec<3,double>& size, shared_ptr<Material> material) {
    return shared_ptr<Block<3>> ( new Block<3>(size, material) );
}





void register_geometry_leafs()
{
    py::class_<Block<2>, shared_ptr<Block<2>>, py::bases<GeometryElementLeaf<2>>>("Rectangle",
        "Geometry object (2D): a rectangle filled with one material\n\n"
        "Rectangle(size) -> initialize rectangle with size given in two-dimensional vector\n\n"
        "Rectangle(width, height) -> initialize rectangle with given width and height\n"
        )
        .def("__init__", py::make_constructor(&Rectangle_constructor_wh))
        .def("__init__", py::make_constructor(&Rectangle_constructor_vec))
    ;

    py::class_<Block<3>, shared_ptr<Block<3>>, py::bases<GeometryElementLeaf<3>>>("Cuboid",
        "Geometry object (3D): a cuboid filled with one material\n\n"
        "Cuboid(size) -> initialize rectangle with size given in three-dimensional vector\n\n"
        "Cuboid(depth, width, height) -> initialize rectangle with given depth, width, and heigh\n"
        )
        .def("__init__", py::make_constructor(&Cuboid_constructor_dwh))
        .def("__init__", py::make_constructor(&Cuboid_constructor_vec))
    ;


}




}} // namespace plask::python
