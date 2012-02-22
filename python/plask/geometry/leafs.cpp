#include "geometry.h"

#include <plask/geometry/leaf.h>

namespace plask { namespace python {


// Rectangle constructor wraps
static shared_ptr<Rectangle> Rectangle_constructor_wh(double w, double h, shared_ptr<Material> material) {
    return shared_ptr<Rectangle> ( new Rectangle(Vec<2,double>(w,h), material) );
}
static shared_ptr<Rectangle> Rectangle_constructor_vec(const Vec<2,double>& size, shared_ptr<Material> material) {
    return shared_ptr<Rectangle> ( new Rectangle(size, material) );
}

// Cuboid constructor wraps
static shared_ptr<Cuboid> Cuboid_constructor_dwh(double d, double w, double h, shared_ptr<Material> material) {
    return shared_ptr<Cuboid> ( new Block<3>(Vec<3,double>(d,w,h), material) );
}
static shared_ptr<Cuboid> Cuboid_constructor_vec(const Vec<3,double>& size, shared_ptr<Material> material) {
    return shared_ptr<Cuboid> ( new Block<3>(size, material) );
}





void register_geometry_leafs()
{
    py::scope scope;

    py::class_<Rectangle, shared_ptr<Rectangle>, py::bases<GeometryElementLeaf<2>>> block2d("Rectangle",
        "Geometry object (2D): a rectangular block filled with one material\n\n"
        "Rectangle(size) -> initialize block with size given in two-dimensional vector\n\n"
        "Rectangle(width, height) -> initialize block with given width and height\n",
        py::no_init
        ); block2d
        .def("__init__", py::make_constructor(&Rectangle_constructor_wh, py::default_call_policies(), (py::arg("width"), py::arg("height"), py::arg("material"))))
        .def("__init__", py::make_constructor(&Rectangle_constructor_vec, py::default_call_policies(), (py::arg("dimensions"), py::arg("material"))))
    ;
    scope.attr("Block2D") = block2d;

    py::class_<Cuboid, shared_ptr<Cuboid>, py::bases<GeometryElementLeaf<3>>> block3d("Cuboid",
        "Geometry object (3D): a cuboidal block filled with one material\n\n"
        "Cuboid(size) -> initialize block with size given in three-dimensional vector\n\n"
        "Cuboid(depth, width, height) -> initialize block with given depth, width, and heigh\n",
        py::no_init
        ); block3d
        .def("__init__", py::make_constructor(&Cuboid_constructor_dwh, py::default_call_policies(), (py::arg("depth"), py::arg("width"), py::arg("height"), py::arg("material"))))
        .def("__init__", py::make_constructor(&Cuboid_constructor_vec, py::default_call_policies(), (py::arg("dimensions"), py::arg("material"))))
    ;
    scope.attr("Block3D") = block3d;


}




}} // namespace plask::python
