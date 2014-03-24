#include "geometry.h"

#include <plask/geometry/leaf.h>
#include <plask/geometry/cylinder.h>

namespace plask { namespace python {

extern AxisNames current_axes;

/// Initialize class GeometryObjectLeaf for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectLeaf, "GeometryObjectLeaf", "Base class for all "," leaves") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectLeaf, GeometryObjectD<dim>)
       .add_property("material", &GeometryObjectLeaf<dim>::singleMaterial, &GeometryObjectLeaf<dim>::setMaterial,
                     "material of the geometry object (or None if there is no single material for the object)")
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

// Access to attributes
template <size_t dim>
static double Block__getattr__(const Block<dim>& self, const std::string& name) {
    if (name[0] == 'd') {
        size_t axis = current_axes[name.substr(1)] + dim-3;
        if (axis < dim) return self.size[axis];
    }
    throw AttributeError("'Block%1%D' object has no attribute '%2%'", dim, name);
}

template <size_t dim>
static void Block__setattr__(py::object self, const std::string& name, const py::object& value) {
    if (name[0] == 'd') {
        size_t axis = current_axes[name.substr(1)] + dim-3;
        if (axis < dim) {
            Block<dim>& block = py::extract<Block<dim>&>(self);
            auto dims = block.size;
            dims[axis] = py::extract<double>(value);
            block.setSize(dims);
            return;
        }
    }
    self.attr("__class__").attr("__base__").attr("__setattr__")(self, name, value);
}

template <int dim, int axis>
static double Block__getdim(const Block<dim>& self) {
    return self.size[axis];
}

template <int dim, int axis>
static void Block__setdim(Block<dim>& self, double value) {
    auto dims = self.size;
    dims[axis] = value;
    self.setSize(dims);
}

void register_geometry_leafs()
{
    py::scope scope;

    init_GeometryObjectLeaf<2>();
    init_GeometryObjectLeaf<3>();

    py::class_<Rectangle, shared_ptr<Rectangle>, py::bases<GeometryObjectLeaf<2>>, boost::noncopyable> block2D("Rectangle",
        "Rectangular block filled with one material.\n\n"
        "Geometry object 2D.\n\n"
        "Args:\n"
        "    width (float): Rectangle width.\n"
        "    height (float): Rectangle height.\n"
        "    material (Material): Cylinder material.\n"
        "    dims (plask.vector): 2D vector representing dimensions of the rectangle.\n",
        py::no_init
        ); block2D
        .def("__init__", py::make_constructor(&Rectangle_constructor_wh, py::default_call_policies(), (py::arg("width"), py::arg("height"), py::arg("material"))))
        .def("__init__", py::make_constructor(&Rectangle_constructor_vec, py::default_call_policies(), (py::arg("dims"), py::arg("material"))))
        .add_property("dims", py::make_getter(&Block<2>::size, py::return_value_policy<py::return_by_value>()), (void(Block<2>::*)(const Vec<2>&))&Block<2>::setSize, "Dimensions of the rectangle.")
        .add_property("width", &Block__getdim<2,0>, &Block__setdim<2,0>, "Width of the rectangle.")
        .add_property("height", &Block__getdim<2,1>, &Block__setdim<2,1>, "Height of the rectangle.")
        .def("__getattr__", &Block__getattr__<2>)
        .def("__setattr__", &Block__setattr__<2>)
    ;
    scope.attr("Block2D") = block2D;

    py::class_<Cuboid, shared_ptr<Cuboid>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable> block3D("Cuboid",
        "Cuboidal block filled with one material.\n\n"
        "Geometry object 3D.\n\n"
        "Args:\n"
        "    depth (float): Cuboid depth.\n"
        "    width (float): Cuboid width.\n"
        "    height (float): Cuboid height.\n"
        "    material (Material): Cylinder material.\n"
        "    dims (plask.vector): 3D vector representing dimensions of the cuboid.\n",
        py::no_init
        ); block3D
        .def("__init__", py::make_constructor(&Cuboid_constructor_dwh, py::default_call_policies(), (py::arg("depth"), py::arg("width"), py::arg("height"), py::arg("material"))))
        .def("__init__", py::make_constructor(&Cuboid_constructor_vec, py::default_call_policies(), (py::arg("dims"), py::arg("material"))))
        .add_property("dims", py::make_getter(&Block<3>::size, py::return_value_policy<py::return_by_value>()), (void(Block<3>::*)(const Vec<3>&))&Block<3>::setSize, "Dimensions of the cuboid.")
        .add_property("depth", &Block__getdim<3,0>, &Block__setdim<3,0>, "Depth of the cuboid.")
        .add_property("width", &Block__getdim<3,1>, &Block__setdim<3,1>, "Width of the cuboid.")
        .add_property("height", &Block__getdim<3,2>, &Block__setdim<3,2>, "Height of the cuboid.")
        .def("__getattr__", &Block__getattr__<3>)
        .def("__setattr__", &Block__setattr__<3>)
    ;
    scope.attr("Block3D") = block3D;

    py::class_<Cylinder, shared_ptr<Cylinder>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable> ("Cylinder",
        "Vertical cylinder filled with one material.\n\n"
        "Geometry object 3D.\n\n"
        "Args:\n"
        "    radius (float): Cylinder radius.\n"
        "    height (float): Cylinder height.\n"
        "    material (Material): Cylinder material.\n",
        py::init<double, double, const shared_ptr<Material>&>((py::arg("radius"), "height", "material"))
        )
        .add_property("radius", py::make_getter(&Cylinder::radius), &Cylinder::setRadius, "Radius of the cylinder.")
        .add_property("height", py::make_getter(&Cylinder::height), &Cylinder::setHeight, "Height of the cylinder.")
    ;
    scope.attr("Block3D") = block3D;


}




}} // namespace plask::python
