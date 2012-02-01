#include "geometry.h"

#include <plask/geometry/leaf.h>

namespace plask { namespace python {


// Block2D constructor wraps
static shared_ptr<Block<2>> Block2D_constructor_wh(double w, double h, shared_ptr<Material> material) {
    return shared_ptr<Block<2>> ( new Block<2>(Vec<2,double>(w,h), material) );
}
static shared_ptr<Block<2>> Block2D_constructor_vec(const Vec<2,double>& size, shared_ptr<Material> material) {
    return shared_ptr<Block<2>> ( new Block<2>(size, material) );
}

// Block3D constructor wraps
static shared_ptr<Block<3>> Block3D_constructor_dwh(double d, double w, double h, shared_ptr<Material> material) {
    return shared_ptr<Block<3>> ( new Block<3>(Vec<3,double>(d,w,h), material) );
}
static shared_ptr<Block<3>> Block3D_constructor_vec(const Vec<3,double>& size, shared_ptr<Material> material) {
    return shared_ptr<Block<3>> ( new Block<3>(size, material) );
}





void register_geometry_leafs()
{
    py::class_<Block<2>, shared_ptr<Block<2>>, py::bases<GeometryElementLeaf<2>>>("Block2D",
        "Geometry object (2D): a rectangular block filled with one material\n\n"
        "Block2D(size) -> initialize block with size given in two-dimensional vector\n\n"
        "Block2D(width, height) -> initialize block with given width and height\n"
        )
        .def("__init__", py::make_constructor(&Block2D_constructor_wh))
        .def("__init__", py::make_constructor(&Block2D_constructor_vec))
    ;

    py::class_<Block<3>, shared_ptr<Block<3>>, py::bases<GeometryElementLeaf<3>>>("Block3D",
        "Geometry object (3D): a cuboidal block filled with one material\n\n"
        "Block3D(size) -> initialize block with size given in three-dimensional vector\n\n"
        "Block3D(depth, width, height) -> initialize block with given depth, width, and heigh\n"
        )
        .def("__init__", py::make_constructor(&Block3D_constructor_dwh))
        .def("__init__", py::make_constructor(&Block3D_constructor_vec))
    ;


}




}} // namespace plask::python
