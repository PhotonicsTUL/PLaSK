#include "geometry.h"

#include <plask/geometry/leaf.h>
#include <plask/geometry/cylinder.h>
#include <plask/geometry/triangle.h>
#include <plask/geometry/circle.h>

namespace plask { namespace python {

extern AxisNames current_axes;

/// Initialize class GeometryObjectLeaf for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectLeaf, "GeometryObjectLeaf", "Base class for all "," leaves") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectLeaf, GeometryObjectD<dim>)
       .add_property("material", &GeometryObjectLeaf<dim>::singleMaterial, &GeometryObjectLeaf<dim>::setMaterial,
                     u8"material of the geometry object (or None if there is no single material for the object)")
       .add_property("representative_material", &GeometryObjectLeaf<dim>::getRepresentativeMaterial)
    ;
}

template <int dim>
void setLeafMaterial(shared_ptr<GeometryObjectLeaf<dim>> self, py::object omaterial) {
    if (PyTuple_Check(omaterial.ptr()) || PyList_Check(omaterial.ptr())) {
        auto l = py::len(omaterial);
        double shape = 1.;
        if (l != 2 && l != 3)
            throw TypeError("Argument is not a proper material");
        std::string mat1 = py::extract<std::string>(omaterial[0]);
        std::string mat2 = py::extract<std::string>(omaterial[1]);
        if (l == 3) shape = py::extract<double>(omaterial[2]);
        self->setMaterialTopBottomComposition(MaterialsDB::getDefault().getFactory(mat2, mat1, shape));
    } else {
        shared_ptr<Material> material = py::extract<shared_ptr<Material>>(omaterial);
        self->setMaterial(material);
    }
}

// Rectangle constructor wraps
static shared_ptr<Rectangle> Rectangle_constructor_wh(double w, double h, const py::object& material) {
    auto result = plask::make_shared<Rectangle>(Vec<2,double>(w,h));
    setLeafMaterial<2>(result, material);
    return result;
}
static shared_ptr<Rectangle> Rectangle_constructor_vec(const Vec<2,double>& size, const py::object& material) {
    auto result = plask::make_shared<Rectangle>(size);
    setLeafMaterial<2>(result, material);
    return result;
}

// Cuboid constructor wraps
static shared_ptr<Cuboid> Cuboid_constructor_dwh(double d, double w, double h, const py::object& material) {
    auto result =  plask::make_shared<Cuboid>(Vec<3,double>(d,w,h));
    setLeafMaterial<3>(result, material);
    return result;
}
static shared_ptr<Cuboid> Cuboid_constructor_vec(const Vec<3,double>& size, const py::object& material) {
    auto result =  plask::make_shared<Cuboid>(size);
    setLeafMaterial<3>(result, material);
    return result;
}

static shared_ptr<Triangle> Triangle_constructor_vec(const Vec<2, double>& p0, const Vec<2,double>& p1, const py::object& material) {
    auto result = plask::make_shared<Triangle>(p0, p1);
    setLeafMaterial<2>(result, material);
    return result;
}

static shared_ptr<Triangle> Triangle_constructor_pts(double x0, double y0, double x1, double y1, const py::object& material) {
    auto result = plask::make_shared<Triangle>(vec(x0,y0), vec(x1,y1));
    setLeafMaterial<2>(result, material);
    return result;
}

// Cylinder constructor wraps
static shared_ptr<Cylinder> Cylinder_constructor(double radius, double height, const py::object& material) {
    auto result =  plask::make_shared<Cylinder>(radius, height);
    setLeafMaterial<3>(result, material);
    return result;
}

// Circle constructor wraps
template <int dim>
static shared_ptr<Circle<dim>> Circle_constructor(double radius, const py::object& material) {
    auto result =  plask::make_shared<Circle<dim>>(radius);
    setLeafMaterial<dim>(result, material);
    return result;
}

// Access to attributes
template <size_t dim>
static double Block__getattr__(const Block<dim>& self, const std::string& name) {
    if (name[0] == 'd') {
        size_t axis = current_axes[name.substr(1)] + dim-3;
        if (axis < dim) return self.size[axis];
    }
    throw AttributeError("'Block{0}D' object has no attribute '{1}'", dim, name);
}

template <size_t dim>
static void Block__setattr__(py::object self, const std::string& name, const py::object& value) {
    if (name[0] == 'd') {
        size_t axis = current_axes[name.substr(1)] + dim-3;
        if (axis < dim) {
            Block<dim>* block = py::extract<Block<dim>*>(self);
            auto dims = block->size;
            dims[axis] = py::extract<double>(value);
            block->setSize(dims);
            return;
        }
    }
    self.attr("__class__").attr("__base__").attr("__setattr__")(self, name, value);
}

static double Triangle__getattr__(const Triangle& self, const std::string& name) {
    if (name.front() == 'a' || name.front() == 'b') {
        size_t axis = current_axes[name.substr(1)] - 1;
        if (axis < 2) return (name.front() == 'a') ? self.p0[axis] : self.p1[axis];
    }
    throw AttributeError(u8"'Triangle' object has no attribute '{0}'", name);
}

static void Triangle__setattr__(py::object self, const std::string& name, const py::object& value) {
    const bool zero = (name.front() == 'a');
    if (zero || name.front() == 'b') {
        size_t axis = current_axes[name.substr(1)] - 1;
        if (axis < 2) {
            Triangle& t = py::extract<Triangle&>(self);
            Vec<2, double> v = zero ? t.p0 : t.p1;
            v[axis] = py::extract<double>(value);
            if (zero) t.setP0(v); else t.setP1(v);
            return;
        }
    }
    self.attr("__class__").attr("__base__").attr("__setattr__")(self, name, value);
}

// This wrappers are necessary so the attributes show in documentation
template <int i> static double Triangle_get_a(const Triangle& self) { return self.p0[i]; }
template <int i> static double Triangle_get_b(const Triangle& self) { return self.p1[i]; }
template <int i> static void Triangle_set_a(Triangle& self, double c) { auto v = self.p0; v[i] = c; self.setP0(v); }
template <int i> static void Triangle_set_b(Triangle& self, double c) { auto v = self.p1; v[i] = c; self.setP1(v); }


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
        u8"Rectangle(width, height, material)\n"
        u8"Rectangle(dims, material)\n\n"
        u8"Rectangular block (2D geometry object).\n\n"
        u8"Sides of the rectangle are always parallel to the axes and its origin is located\n"
        u8"in the lower left corner.\n\n"
        u8"Args:\n"
        u8"    width (float): Rectangle width.\n"
        u8"    height (float): Rectangle height.\n"
        u8"    material (Material): Rectangle material.\n"
        u8"    dims (plask.vec): 2D vector representing dimensions of the rectangle.\n",
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
        u8"Cuboid(depth, width, height, material)\n"
        u8"Cuboid(dims, material)\n\n"
        u8"Cuboidal block (3D geometry object).\n\n"
        u8"Sides of the cuboid are always parallel to the axes and its origin is located\n"
        u8"in the lower left back corner.\n\n"
        u8"Args:\n"
        u8"    depth (float): Cuboid depth.\n"
        u8"    width (float): Cuboid width.\n"
        u8"    height (float): Cuboid height.\n"
        u8"    material (Material): Cuboid material.\n"
        u8"    dims (plask.vec): 3D vector representing dimensions of the cuboid.\n",
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

    py::class_<Triangle, shared_ptr<Triangle>, py::bases<GeometryObjectLeaf<2>>, boost::noncopyable> triangle("Triangle",
        u8"Triangle(a0, a1, b0, b1, material)\n"
        u8"Triangle(a, b, material)\n"
        u8"Triangle (2D geometry object).\n\n"
        u8"Three triangle vertices are located at points (0, 0), *a*, and *b*.\n\n"
        u8"Args:\n"
        u8"    plask.vec a: Local coordinates of the first triangle vertex.\n"
        u8"    plask.vec b: Local coordinates of the second triangle vertex.\n"
        u8"    a0 (double): Horizontal component of the local coordinates of the first\n"
        u8"                 triangle vertex.\n"
        u8"    a1 (double): Vertical component of the local coordinates of the first\n"
        u8"                 triangle vertex.\n"
        u8"    b0 (double): Horizontal component of the local coordinates of the second\n"
        u8"                 triangle vertex.\n"
        u8"    b1 (double): Vertical component of the local coordinates of the second\n"
        u8"                 triangle vertex.\n"
        u8"    material (Material): Triangle material.\n",
        py::no_init
        ); triangle
        .def("__init__", py::make_constructor(&Triangle_constructor_vec, py::default_call_policies(), (py::arg("a"), "b", "material")))
        .def("__init__", py::make_constructor(&Triangle_constructor_pts, py::default_call_policies(), (py::arg("a0"), "a1", "b0", "b1", "material")))
        .add_property("a", py::make_getter(&Triangle::p0, py::return_value_policy<py::return_by_value>()), (void(Triangle::*)(const Vec<2>&))&Triangle::setP0, "Coordinates of the first vertex.")
        .add_property("b", py::make_getter(&Triangle::p1, py::return_value_policy<py::return_by_value>()), (void(Triangle::*)(const Vec<2>&))&Triangle::setP1, "Coordinates of the second vertex.")
        .add_property("a0", &Triangle_get_a<0>, &Triangle_set_a<0>, "Horizontal coordinate of the first vertex.\n\nInstead of 0 you can use transverse axis name.")
        .add_property("a1", &Triangle_get_a<1>, &Triangle_set_a<1>, "Vertical coordinate of the first vertex.\n\nInstead of 1 you can use vertical axis name.")
        .add_property("b0", &Triangle_get_b<0>, &Triangle_set_b<0>, "Horizontal coordinate of the second vertex.\n\nInstead of 0 you can use transverse axis name.")
        .add_property("b1", &Triangle_get_b<1>, &Triangle_set_b<1>, "Vertical coordinate of the second vertex.\n\nInstead of 1 you can use vertical axis name.")
        .def("__getattr__", &Triangle__getattr__)
        .def("__setattr__", &Triangle__setattr__)
    ;

    py::class_<Circle<2>, shared_ptr<Circle<2>>, py::bases<GeometryObjectLeaf<2>>, boost::noncopyable> ("Circle",
        u8"Circle(radius, material)\n\n"
        u8"Circle (2D geometry object).\n\n"
        u8"Args:\n"
        u8"    radius (float): Circle radius.\n"
        u8"    material (Material): Circle material.\n",
        py::no_init
        )
        .def("__init__", py::make_constructor(&Circle_constructor<2>, py::default_call_policies(), (py::arg("radius"), "material")))
        .add_property("radius", py::make_getter(&Circle<2>::radius), &Circle<2>::setRadius, u8"Radius of the circle.")
    ;

    py::class_<Circle<3>, shared_ptr<Circle<3>>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable> ("Sphere",
        u8"Sphere(radius, material)\n\n"
        u8"Sphere (3D geometry object).\n\n"
        u8"Args:\n"
        u8"    radius (float): Sphere radius.\n"
        u8"    material (Material): Sphere material.\n",
        py::no_init
        )
        .def("__init__", py::make_constructor(&Circle_constructor<3>, py::default_call_policies(), (py::arg("radius"), "material")))
        .add_property("radius", py::make_getter(&Circle<3>::radius), &Circle<3>::setRadius, u8"Radius of the circle.")
    ;

    py::class_<Cylinder, shared_ptr<Cylinder>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable> ("Cylinder",
        u8"Cylinder(radius, height, material)\n\n"
        u8"Vertical cylinder (3D geometry object).\n\n"
        u8"The cylinder base always lies in the horizontal (longitudinal-transverse)\n"
        u8"plane and it height spans in the vertical\n"
        u8"direction.\n\n"
        u8"Args:\n"
        u8"    radius (float): Cylinder radius.\n"
        u8"    height (float): Cylinder height.\n"
        u8"    material (Material): Cylinder material.\n",
        py::no_init
        )
        .def("__init__", py::make_constructor(&Cylinder_constructor, py::default_call_policies(), (py::arg("radius"), "height", "material")))
        .add_property("radius", py::make_getter(&Cylinder::radius), &Cylinder::setRadius, u8"Radius of the cylinder.")
        .add_property("height", py::make_getter(&Cylinder::height), &Cylinder::setHeight, u8"Height of the cylinder.")
    ;


}




}} // namespace plask::python
