#include "geometry.h"

#include <plask/geometry/leaf.h>

namespace plask { namespace python {

/// Initialize class GeometryObjectLeaf for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectLeaf, "GeometryObjectLeaf", "Base class for all "," leaves") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectLeaf, GeometryObjectD<dim>)
        .def_readwrite("material", &GeometryObjectLeaf<dim>::material, "material of the geometry object")
    ;
}



// Rectangle constructor wraps
static shared_ptr<Rectangle> Rectangle_constructor_wh(double w, double h, shared_ptr<Material> material) {
    return make_shared<Rectangle>(Vec<2,double>(w,h), material);
}
static shared_ptr<Rectangle> Rectangle_constructor_vec(const Vec<2,double>& size, shared_ptr<Material> material) {
    return make_shared<Rectangle>(size, material);
}

// Cuboid constructor wraps
static shared_ptr<Cuboid> Cuboid_constructor_dwh(double d, double w, double h, shared_ptr<Material> material) {
    return make_shared<Cuboid>(Vec<3,double>(d,w,h), material);
}
static shared_ptr<Cuboid> Cuboid_constructor_vec(const Vec<3,double>& size, shared_ptr<Material> material) {
    return make_shared<Cuboid>(size, material);
}





void register_geometry_leafs()
{
    py::scope scope;

    init_GeometryObjectLeaf<2>();
    init_GeometryObjectLeaf<3>();

    py::class_<Rectangle, shared_ptr<Rectangle>, py::bases<GeometryObjectLeaf<2>>, boost::noncopyable> block2D("Rectangle",
        "Geometry object (2D) : a rectangular block filled with one material\n\n"
        "Rectangle(size)\n    initialize block with size given in two-dimensional vector\n\n"
        "Rectangle(width, height)\n    initialize block with given width and height\n",
        py::no_init
        ); block2D
        .def("__init__", py::make_constructor(&Rectangle_constructor_wh, py::default_call_policies(), (py::arg("width"), py::arg("height"), py::arg("material"))))
        .def("__init__", py::make_constructor(&Rectangle_constructor_vec, py::default_call_policies(), (py::arg("dimensions"), py::arg("material"))))
        .add_property("dimensions", py::make_getter(&Block<2>::size, py::return_value_policy<py::return_by_value>()), (void(Block<2>::*)(const Vec<2>&))&Block<2>::setSize, "Dimensions of the rectangle")
    ;
    scope.attr("Block2D") = block2D;

    py::class_<Cuboid, shared_ptr<Cuboid>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable> block3D("Cuboid",
        "Geometry object (3D) : a cuboidal block filled with one material\n\n"
        "Cuboid(size)\n    initialize block with size given in three-dimensional vector\n\n"
        "Cuboid(depth, width, height)\n    initialize block with given depth, width, and heigh\n",
        py::no_init
        ); block3D
        .def("__init__", py::make_constructor(&Cuboid_constructor_dwh, py::default_call_policies(), (py::arg("depth"), py::arg("width"), py::arg("height"), py::arg("material"))))
        .def("__init__", py::make_constructor(&Cuboid_constructor_vec, py::default_call_policies(), (py::arg("dimensions"), py::arg("material"))))
        .add_property("dimensions", py::make_getter(&Block<3>::size, py::return_value_policy<py::return_by_value>()), (void(Block<3>::*)(const Vec<3>&))&Block<3>::setSize, "Dimensions of the rectangle")
    ;
    scope.attr("Block3D") = block3D;


}




}} // namespace plask::python
