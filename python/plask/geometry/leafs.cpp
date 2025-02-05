/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include "../python/util/raw_constructor.hpp"
#include "geometry.hpp"

#include <plask/geometry/circle.hpp>
#include <plask/geometry/cuboid.hpp>
#include <plask/geometry/cylinder.hpp>
#include <plask/geometry/ellipse.hpp>
#include <plask/geometry/elliptic_cylinder.hpp>
#include <plask/geometry/leaf.hpp>
#include <plask/geometry/polygon.hpp>
#include <plask/geometry/prism.hpp>
#include <plask/geometry/triangle.hpp>

namespace plask { namespace python {

extern AxisNames current_axes;

template <int dim> void setLeafMaterial(shared_ptr<GeometryObjectLeaf<dim>> self, py::object omaterial);

template <int dim> py::object getLeafMaterial(shared_ptr<GeometryObjectLeaf<dim>> self);

/// Initialize class GeometryObjectLeaf for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectLeaf, "GeometryObjectLeaf", "Base class for all ", " leaves") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectLeaf, GeometryObjectD<dim>)
        .add_property("material", &getLeafMaterial<dim>, &setLeafMaterial<dim>,
                      u8"material of the geometry object (or None if there is no single material for the object)")
        .add_property("representative_material", &GeometryObjectLeaf<dim>::getRepresentativeMaterial);
}

/// Class providing custom material from relative position and Python callable
template <int dim> struct PythonMaterialProvider : public GeometryObjectLeaf<dim>::MaterialProvider {
    typedef typename GeometryObjectLeaf<dim>::DVec DVec;
    typedef typename GeometryObjectLeaf<dim>::Box Box;

    py::object callable;

    PythonMaterialProvider(const py::object& callable) : callable(callable) {}

    shared_ptr<Material> getMaterial(const GeometryObjectLeaf<dim>& thisObj, const DVec& p) const override;

    shared_ptr<Material> singleMaterial() const override { return shared_ptr<Material>(); }

    PythonMaterialProvider* clone() const override { return new PythonMaterialProvider(callable); }

    shared_ptr<Material> getRepresentativeMaterial() const override;

    bool isUniform(Primitive<3>::Direction /*direction*/) const override { return false; }

    XMLWriter::Element& writeXML(XMLWriter::Element& /*dest_xml_object*/, const AxisNames& /*axes*/) const override {
        throw NotImplemented("writing Python callable material to XML");
    }
};

template <> shared_ptr<Material> PythonMaterialProvider<2>::getMaterial(const GeometryObjectLeaf<2>& thisObj, const DVec& p) const {
    Box b = thisObj.getBoundingBox();
    DVec rp((p.tran() - b.lower.tran()) / b.width(), (p.vert() - b.lower.vert()) / b.height());
    return py::extract<shared_ptr<Material>>(callable(rp));
}

template <> shared_ptr<Material> PythonMaterialProvider<3>::getMaterial(const GeometryObjectLeaf<3>& thisObj, const DVec& p) const {
    Box b = thisObj.getBoundingBox();
    DVec rp((p.lon() - b.lower.lon()) / b.depth(), (p.tran() - b.lower.tran()) / b.width(),
            (p.vert() - b.lower.vert()) / b.height());
    return py::extract<shared_ptr<Material>>(callable(rp));
}

template <> shared_ptr<Material> PythonMaterialProvider<2>::getRepresentativeMaterial() const {
    return py::extract<shared_ptr<Material>>(callable(DVec(0.5, 0.5)));
}

template <> shared_ptr<Material> PythonMaterialProvider<3>::getRepresentativeMaterial() const {
    return py::extract<shared_ptr<Material>>(callable(DVec(0.5, 0.5, 0.5)));
}

template <int dim> py::object getLeafMaterial(shared_ptr<GeometryObjectLeaf<dim>> self) {
    const typename GeometryObjectLeaf<dim>::MaterialProvider* material = self->getMaterialProvider();
    if (auto solid_material = dynamic_cast<const typename GeometryObjectLeaf<dim>::SolidMaterial*>(material)) {
        return py::object(material->singleMaterial());
    } else if (auto gradient_material = dynamic_cast<const typename GeometryObjectLeaf<dim>::GradientMaterial*>(material)) {
        return py::make_tuple((*gradient_material->materialFactory)(0.), (*gradient_material->materialFactory)(1.));
    } else if (auto python_material = dynamic_cast<const PythonMaterialProvider<dim>*>(material)) {
        return python_material->callable;
    } else {
        return py::object();
    }
}

template <int dim> void setLeafMaterialFast(shared_ptr<GeometryObjectLeaf<dim>> self, py::object omaterial) {
    if (PyTuple_Check(omaterial.ptr()) || PyList_Check(omaterial.ptr())) {
        auto l = py::len(omaterial);
        double shape = 1.;
        if (l != 2 && l != 3) throw TypeError("argument is not a proper material");
        std::string mat1 = py::extract<std::string>(omaterial[0]);
        std::string mat2 = py::extract<std::string>(omaterial[1]);
        if (l == 3) shape = py::extract<double>(omaterial[2]);
        self->setMaterialTopBottomCompositionFast(MaterialsDB::getDefault().getFactory(mat2, mat1, shape));
    } else if (PyCallable_Check(omaterial.ptr())) {
        self->setMaterialProviderFast(new PythonMaterialProvider<dim>(omaterial));
    } else {
        shared_ptr<Material> material = py::extract<shared_ptr<Material>>(omaterial);
        self->setMaterialFast(material);
    }
}

template <int dim> void setLeafMaterial(shared_ptr<GeometryObjectLeaf<dim>> self, py::object omaterial) {
    setLeafMaterialFast(self, omaterial);
    self->fireChanged();
}

// Rectangle constructor wraps
static shared_ptr<Rectangle> Rectangle_constructor_wh(double w, double h, const py::object& material) {
    auto result = plask::make_shared<Rectangle>(Vec<2, double>(w, h));
    setLeafMaterialFast<2>(result, material);
    return result;
}
static shared_ptr<Rectangle> Rectangle_constructor_vec(const Vec<2, double>& size, const py::object& material) {
    auto result = plask::make_shared<Rectangle>(size);
    setLeafMaterialFast<2>(result, material);
    return result;
}

// Cuboid constructor wraps
static shared_ptr<Cuboid> Cuboid_constructor_dwh(double d,
                                                 double w,
                                                 double h,
                                                 const py::object& material,
                                                 const py::object& angle) {
    shared_ptr<Cuboid> result;
    if (angle.is_none()) {
        result.reset(new Cuboid(Vec<3, double>(d, w, h)));
    } else {
        py::extract<double> angl(angle);
        if (!angl.check()) throw TypeError("'angle' must be either None or float in Cuboid.__init__");
        result.reset(new RotatedCuboid(Vec<3, double>(d, w, h), angl));
    }
    setLeafMaterialFast<3>(result, material);
    return result;
}
static shared_ptr<Cuboid> Cuboid_constructor_vec(const Vec<3, double>& size, const py::object& material, const py::object& angle) {
    shared_ptr<Cuboid> result;
    if (angle.is_none()) {
        result.reset(new Cuboid(size));
    } else {
        py::extract<double> angl(angle);
        if (!angl.check()) throw TypeError("'angle' must be either None or float in Cuboid.__init__");
        result.reset(new RotatedCuboid(size, angl));
    }
    setLeafMaterialFast<3>(result, material);
    return result;
}

py::object Cuboid_getAngle(const Cuboid* self) {
    if (auto rotated_cuboid = dynamic_cast<const RotatedCuboid*>(self))
        return py::object(rotated_cuboid->getAngle());
    else
        return py::object();
}

void Cuboid_setAngle(Cuboid* self, double angle) {
    if (auto rotated_cuboid = dynamic_cast<RotatedCuboid*>(self))
        rotated_cuboid->setAngle(angle);
    else
        throw TypeError("specify angle value (possibly 0) at cuboid creation to be able to change it");
}

// Triangle constructor wraps
static shared_ptr<Triangle> Triangle_constructor_vec(const Vec<2, double>& p0,
                                                     const Vec<2, double>& p1,
                                                     const py::object& material) {
    auto result = plask::make_shared<Triangle>(p0, p1);
    setLeafMaterialFast<2>(result, material);
    return result;
}

static shared_ptr<Triangle> Triangle_constructor_pts(double x0, double y0, double x1, double y1, const py::object& material) {
    auto result = plask::make_shared<Triangle>(vec(x0, y0), vec(x1, y1));
    setLeafMaterialFast<2>(result, material);
    return result;
}

// Cylinder constructor wraps
static shared_ptr<Cylinder> Cylinder_constructor(double radius, double height, const py::object& material) {
    auto result = plask::make_shared<Cylinder>(radius, height);
    setLeafMaterialFast<3>(result, material);
    return result;
}

// Hollow cylinder constructor wraps
static shared_ptr<HollowCylinder> HollowCylinder_constructor(double inner_radius,
                                                             double outer_radius,
                                                             double height,
                                                             const py::object& material) {
    auto result = plask::make_shared<HollowCylinder>(inner_radius, outer_radius, height);
    setLeafMaterialFast<3>(result, material);
    return result;
}

// Circle constructor wraps
template <int dim> static shared_ptr<Circle<dim>> Circle_constructor(double radius, const py::object& material) {
    auto result = plask::make_shared<Circle<dim>>(radius);
    setLeafMaterialFast<dim>(result, material);
    return result;
}

// Ellipse wraps
static shared_ptr<Ellipse> Ellipse_constructor(double radius0, double radius1, const py::object& material) {
    auto result = plask::make_shared<Ellipse>(radius0, radius1);
    setLeafMaterialFast<2>(result, material);
    return result;
}

static shared_ptr<Ellipse> Ellipse_constructor_1(py::object radii, const py::object& material) {
    if (py::len(radii) != 2) throw TypeError("radii must be a tuple of two floats");
    double rx = py::extract<double>(radii[0]);
    double ry = py::extract<double>(radii[1]);
    auto result = plask::make_shared<Ellipse>(rx, ry);
    setLeafMaterialFast<2>(result, material);
    return result;
}

template <typename T> static py::object Ellipse_getRadii(const T* self) {
    auto radii = self->getRadii();
    return py::make_tuple(radii.first, radii.second);
}

template <typename T> static void Ellipse_setRadii(T* self, py::object radii) {
    if (py::len(radii) != 2) throw TypeError("radii must be a tuple of two floats");
    double rx = py::extract<double>(radii[0]);
    double ry = py::extract<double>(radii[1]);
    self->setRadii(rx, ry);
}

static shared_ptr<EllipticCylinder> EllipticCylinder_constructor(const py::tuple& args, const py::dict& kwargs) {
    double r0, r1, h, angle = 0.;

    PyObject* self;
    PyObject* pymaterial;

    if ((py::len(args) >= 2 && PySequence_Check(py::object(args[1]).ptr())) || kwargs.contains("radii")) {
        PyObject* radii;
        static const char* kwlist[] = {"self", "radii", "height", "material", "angle", NULL};
        if (!PyArg_ParseTupleAndKeywords(args.ptr(), kwargs.ptr(), "OOdO|$d:EllipticCylinder.__init__", (char**)kwlist,  //
                                         &self, &radii, &h, &pymaterial, &angle))
            throw py::error_already_set();
        if (PySequence_Length(radii) != 2) throw TypeError("radii must be a tuple of two floats");
        r0 = py::extract<double>(PySequence_GetItem(radii, 0));
        r1 = py::extract<double>(PySequence_GetItem(radii, 1));
    } else {
        static const char* kwlist[] = {"self",
                                       "radius0",
                                       "radius1",
                                       "height",
                                       "material",
                                       "angle",
                                       NULL};
        if (!PyArg_ParseTupleAndKeywords(args.ptr(), kwargs.ptr(), "OdddO|$d:EllipticCylinder.__init__", (char**)kwlist,  //
                                         &self, &r0, &r1, &h, &pymaterial, &angle))
            throw py::error_already_set();
    }

    py::object material = py::object(py::borrowed(pymaterial));

    angle *= M_PI / 180.;

    auto result = plask::make_shared<EllipticCylinder>(r0, r1, angle, h);
    setLeafMaterialFast<3>(result, material);
    return result;
}

static double EllipticCylinder_getAngle(const EllipticCylinder* self) { return self->getAngle() * 180. / M_PI; }

static void EllipticCylinder_setAngle(EllipticCylinder* self, double angle) { self->setAngle(angle * M_PI / 180.); }

// Prism constructor wraps
static shared_ptr<TriangularPrism> TriangularPrism_constructor_vec(const LateralVec<double>& p0,
                                                                   const LateralVec<double>& p1,
                                                                   double height,
                                                                   const py::object& material) {
    auto result = plask::make_shared<TriangularPrism>(p0, p1, height);
    setLeafMaterialFast<3>(result, material);
    return result;
}

static shared_ptr<TriangularPrism> TriangularPrism_constructor_pts(double x0,
                                                                   double y0,
                                                                   double x1,
                                                                   double y1,
                                                                   double height,
                                                                   const py::object& material) {
    auto result = plask::make_shared<TriangularPrism>(vec(x0, y0), vec(x1, y1), height);
    setLeafMaterialFast<3>(result, material);
    return result;
}

// Access to attributes
template <size_t dim> static double Block__getattr__(const Block<int(dim)>& self, const std::string& name) {
    if (name[0] == 'd') {
        size_t axis = current_axes[name.substr(1)] + dim - 3;
        if (axis < dim) return self.size[axis];
    }
    throw AttributeError("'Block{0}D' object has no attribute '{1}'", dim, name);
}

template <size_t dim> static void Block__setattr__(py::object self, const std::string& name, const py::object& value) {
    if (name[0] == 'd') {
        size_t axis = current_axes[name.substr(1)] + dim - 3;
        if (axis < dim) {
            Block<int(dim)>* block = py::extract<Block<int(dim)>*>(self);
            auto dims = block->size;
            dims[axis] = py::extract<double>(value);
            block->setSize(dims);
            return;
        }
    }
    self.attr("__class__").attr("__base__").attr("__setattr__")(self, name, value);
}

template <typename T> static double Triangle__getattr__(py::object self, const std::string& name) {
    T& t = py::extract<T&>(self);
    if (name.front() == 'a' || name.front() == 'b') {
        size_t axis = current_axes[name.substr(1)] - 1;
        if (axis < 2) return (name.front() == 'a') ? t.p0[axis] : t.p1[axis];
    }
    throw AttributeError(u8"'{0}' object has no attribute '{1}'",
                         py::extract<std::string>(self.attr("__class__").attr("__name__"))(), name);
}

template <typename T> static void Triangle__setattr__(py::object self, const std::string& name, const py::object& value) {
    const bool zero = (name.front() == 'a');
    if (zero || name.front() == 'b') {
        size_t axis = current_axes[name.substr(1)] - 1;
        if (axis < 2) {
            T& t = py::extract<T&>(self);
            decltype(t.p0) v = zero ? t.p0 : t.p1;
            v[axis] = py::extract<double>(value);
            if (zero)
                t.setP0(v);
            else
                t.setP1(v);
            return;
        }
    }
    self.attr("__class__").attr("__base__").attr("__setattr__")(self, name, value);
}

// This wrappers are necessary so the attributes show in documentation
template <typename T, int i> static double Triangle_get_a(const T& self) { return self.p0[i]; }
template <typename T, int i> static double Triangle_get_b(const T& self) { return self.p1[i]; }
template <typename T, int i> static void Triangle_set_a(T& self, double c) {
    auto v = self.p0;
    v[i] = c;
    self.setP0(v);
}
template <typename T, int i> static void Triangle_set_b(T& self, double c) {
    auto v = self.p1;
    v[i] = c;
    self.setP1(v);
}

template <int dim, int axis> static double Block__getdim(const Block<dim>& self) { return self.size[axis]; }

template <int dim, int axis> static void Block__setdim(Block<dim>& self, double value) {
    auto dims = self.size;
    dims[axis] = value;
    self.setSize(dims);
}

static shared_ptr<Polygon> Polygon_constructor(const py::object& vetrices, const py::object& material) {
    std::vector<Vec<2, double>> verts;
    for (size_t i = 0; i < py::len(vetrices); ++i) {
        verts.push_back(py::extract<Vec<2, double>>(vetrices[i]));
    }
    auto result = plask::make_shared<Polygon>(verts);
    setLeafMaterialFast<2>(result, material);
    return result;
}

static shared_ptr<Prism> Prism_constructor(const py::object& vetrices, double height, const py::object& material) {
    std::vector<LateralVec<double>> verts;
    for (size_t i = 0; i < py::len(vetrices); ++i) {
        verts.push_back(py::extract<LateralVec<double>>(vetrices[i]));
    }
    auto result = plask::make_shared<Prism>(height, verts);
    setLeafMaterialFast<3>(result, material);
    return result;
}

template <typename PolyT, typename VecT> class Vertices2D {
    shared_ptr<PolyT> polygon;

    size_t index(int i) const {
        size_t n = polygon->getVertices().size();
        if (i < 0) i += n;
        if (i < 0 || i >= n) throw IndexError("vertex index out of range");
        return i;
    }

  public:
    Vertices2D(shared_ptr<PolyT> polygon) : polygon(polygon) {}

    static Vertices2D fromPolygon(shared_ptr<PolyT> polygon) { return Vertices2D(polygon); }

    size_t __len__() const { return polygon->getVertices().size(); }

    VecT __getitem__(int i) const { return polygon->getVertices()[index(i)]; }

    void __setitem__(int i, const Vec<2, double>& v) { polygon->setVertex(index(i), v); }

    void append(const Vec<2, double>& v) { polygon->addVertex(v); }

    void insert(int i, const Vec<2, double>& v) {
        size_t n = polygon->getVertices().size();
        if (i < 0) i += n;
        if (i < 0 || i > n) throw IndexError("vertex index out of range");
        if (i == n)
            polygon->addVertex(v);
        else
            polygon->insertVertex(i, v);
    }

    void __delitem__(int i) { polygon->removeVertex(index(i)); }

    std::string __str__() const {
        std::string result = "[";
        for (size_t i = 0; i < polygon->getVertices().size(); ++i) {
            result += str(polygon->getVertices()[i]);
            result += (i != polygon->getVertices().size() - 1) ? ", " : "]";
        }
        return result;
    }

    std::string __repr__() const {
        std::string result = "[";
        for (size_t i = 0; i < polygon->getVertices().size(); ++i) {
            result += format("plask.vec({}, {})", polygon->getVertices()[i].c0, polygon->getVertices()[i].c1);
            result += (i != polygon->getVertices().size() - 1) ? ", " : "]";
        }
        return result;
    }

    struct Iterator {
        shared_ptr<PolyT> polygon;
        size_t i;

        Iterator(const Vertices2D* pv) : polygon(pv->polygon), i(0) {}

        VecT __next__() {
            if (i >= polygon->getVertices().size()) throw StopIteration();
            return polygon->getVertex(i++);
        }

        Iterator* __iter__() { return this; }
    };

    Iterator __iter__() const { return Iterator(this); }

    static void register_class() {
        py::class_<Vertices2D> vertices("Vertices", py::no_init);
        vertices  //
            .def("__len__", &Vertices2D::__len__)
            .def("__getitem__", &Vertices2D::__getitem__)
            .def("__setitem__", &Vertices2D::__setitem__)
            .def("append", &Vertices2D::append)
            .def("insert", &Vertices2D::insert)
            .def("__delitem__", &Vertices2D::__delitem__)
            .def("__str__", &Vertices2D::__str__)
            .def("__repr__", &Vertices2D::__repr__)
            .def("__iter__", &Vertices2D::__iter__);

        py::scope vertices_scope = vertices;

        py::class_<Vertices2D::Iterator>("Iterator", py::no_init)  //
            .def("__iter__", &Vertices2D::Iterator::__iter__, py::return_self<>())
            .def("__next__", &Vertices2D::Iterator::__next__);
    }

    static void setVertices(PolyT& self, const py::object& vertices) {
        std::vector<VecT> verts;
        for (size_t i = 0; i < py::len(vertices); ++i) {
            verts.push_back(py::extract<VecT>(vertices[i]));
        }
        self.setVertices(verts);
    }
};

void register_geometry_leafs() {
    py::scope scope;

    init_GeometryObjectLeaf<2>();
    init_GeometryObjectLeaf<3>();

    py::class_<Rectangle, shared_ptr<Rectangle>, py::bases<GeometryObjectLeaf<2>>, boost::noncopyable> block2D(
        "Rectangle",
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
        py::no_init);
    block2D
        .def("__init__", py::make_constructor(&Rectangle_constructor_wh, py::default_call_policies(),
                                              (py::arg("width"), py::arg("height"), py::arg("material"))))
        .def("__init__",
             py::make_constructor(&Rectangle_constructor_vec, py::default_call_policies(), (py::arg("dims"), py::arg("material"))))
        .add_property("dims", py::make_getter(&Block<2>::size, py::return_value_policy<py::return_by_value>()),
                      (void(Block<2>::*)(const Vec<2>&)) & Block<2>::setSize, "Dimensions of the rectangle.")
        .add_property("width", &Block__getdim<2, 0>, &Block__setdim<2, 0>, "Width of the rectangle.")
        .add_property("height", &Block__getdim<2, 1>, &Block__setdim<2, 1>, "Height of the rectangle.")
        .def("__getattr__", &Block__getattr__<2>)
        .def("__setattr__", &Block__setattr__<2>);
    scope.attr("Block2D") = block2D;

    py::class_<Cuboid, shared_ptr<Cuboid>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable> block3D(
        "Cuboid",
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
        u8"    dims (plask.vec): 3D vector representing dimensions of the cuboid.\n"
        u8"    angle (None or float): rotation angle in the horizontal (long-vert) plane [deg].\n",
        py::no_init);
    block3D
        .def("__init__", py::make_constructor(&Cuboid_constructor_dwh, py::default_call_policies(),
                                              (py::arg("depth"), py::arg("width"), py::arg("height"), py::arg("material"),
                                               py::arg("angle") = py::object())))
        .def("__init__", py::make_constructor(&Cuboid_constructor_vec, py::default_call_policies(),
                                              (py::arg("dims"), py::arg("material"), py::arg("angle") = py::object())))
        .add_property("dims", py::make_getter(&Block<3>::size, py::return_value_policy<py::return_by_value>()),
                      (void(Block<3>::*)(const Vec<3>&)) & Block<3>::setSize, "Dimensions of the cuboid.")
        .add_property("depth", &Block__getdim<3, 0>, &Block__setdim<3, 0>, "Depth of the cuboid.")
        .add_property("width", &Block__getdim<3, 1>, &Block__setdim<3, 1>, "Width of the cuboid.")
        .add_property("height", &Block__getdim<3, 2>, &Block__setdim<3, 2>, "Height of the cuboid.")
        .add_property("angle", &Cuboid_getAngle, &Cuboid_setAngle, "Rotation angle in the horizontal (long-vert) plane (deg).")
        .def("__getattr__", &Block__getattr__<3>)
        .def("__setattr__", &Block__setattr__<3>);
    scope.attr("Block3D") = block3D;

    py::class_<RotatedCuboid, shared_ptr<RotatedCuboid>, py::bases<Block<3>>, boost::noncopyable> cuboid("Cuboid");
    scope.attr("_CuboidRotated") = cuboid;
    scope.attr("Cuboid") = block3D;

    py::class_<Triangle, shared_ptr<Triangle>, py::bases<GeometryObjectLeaf<2>>, boost::noncopyable> triangle(
        "Triangle",
        u8"Triangle(a0, a1, b0, b1, material)\n"
        u8"Triangle(a, b, material)\n\n"
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
        py::no_init);
    triangle
        .def("__init__",
             py::make_constructor(&Triangle_constructor_vec, py::default_call_policies(), (py::arg("a"), "b", "material")))
        .def("__init__", py::make_constructor(&Triangle_constructor_pts, py::default_call_policies(),
                                              (py::arg("a0"), "a1", "b0", "b1", "material")))
        .add_property("a", py::make_getter(&Triangle::p0, py::return_value_policy<py::return_by_value>()),
                      (void(Triangle::*)(const Vec<2>&)) & Triangle::setP0, "Coordinates of the first vertex.")
        .add_property("b", py::make_getter(&Triangle::p1, py::return_value_policy<py::return_by_value>()),
                      (void(Triangle::*)(const Vec<2>&)) & Triangle::setP1, "Coordinates of the second vertex.")
        .add_property("a0", &Triangle_get_a<Triangle, 0>, &Triangle_set_a<Triangle, 0>,
                      "Horizontal coordinate of the first vertex.\n\nInstead of 0 you can use transverse axis name.")
        .add_property("a1", &Triangle_get_a<Triangle, 1>, &Triangle_set_a<Triangle, 1>,
                      "Vertical coordinate of the first vertex.\n\nInstead of 1 you can use vertical axis name.")
        .add_property("b0", &Triangle_get_b<Triangle, 0>, &Triangle_set_b<Triangle, 0>,
                      "Horizontal coordinate of the second vertex.\n\nInstead of 0 you can use transverse axis name.")
        .add_property("b1", &Triangle_get_b<Triangle, 1>, &Triangle_set_b<Triangle, 1>,
                      "Vertical coordinate of the second vertex.\n\nInstead of 1 you can use vertical axis name.")
        .def("__getattr__", &Triangle__getattr__<Triangle>)
        .def("__setattr__", &Triangle__setattr__<Triangle>);

    py::class_<Circle<2>, shared_ptr<Circle<2>>, py::bases<GeometryObjectLeaf<2>>, boost::noncopyable>(
        "Circle",
        u8"Circle(radius, material)\n\n"
        u8"Circle (2D geometry object).\n\n"
        u8"Args:\n"
        u8"    radius (float): Circle radius.\n"
        u8"    material (Material): Circle material.\n",
        py::no_init)
        .def("__init__", py::make_constructor(&Circle_constructor<2>, py::default_call_policies(), (py::arg("radius"), "material")))
        .add_property("radius", py::make_getter(&Circle<2>::radius), &Circle<2>::setRadius, u8"Radius of the circle.");

    py::class_<Circle<3>, shared_ptr<Circle<3>>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable>(
        "Sphere",
        u8"Sphere(radius, material)\n\n"
        u8"Sphere (3D geometry object).\n\n"
        u8"Args:\n"
        u8"    radius (float): Sphere radius.\n"
        u8"    material (Material): Sphere material.\n",
        py::no_init)
        .def("__init__", py::make_constructor(&Circle_constructor<3>, py::default_call_policies(), (py::arg("radius"), "material")))
        .add_property("radius", py::make_getter(&Circle<3>::radius), &Circle<3>::setRadius, u8"Radius of the circle.");

    py::class_<Ellipse, shared_ptr<Ellipse>, py::bases<GeometryObjectLeaf<2>>, boost::noncopyable>(
        "Ellipse",
        u8"Ellipse(radius0, radius1, material)\n\n"
        u8"Ellipse (2D geometry object).\n\n"
        u8"Args:\n"
        u8"    radius0 (float): Ellipse radius along transverse direction.\n"
        u8"    radius1 (float): Ellipse radius along vertical direction.\n"
        u8"    radii (tuple): Ellipse radii.\n"
        u8"    material (Material): Ellipse material.\n",
        py::no_init)
        .def("__init__",
             py::make_constructor(&Ellipse_constructor, py::default_call_policies(), (py::arg("radius0"), "radius1", "material")))
        .def("__init__", py::make_constructor(&Ellipse_constructor_1, py::default_call_policies(), (py::arg("radii"), "material")))
        .add_property("radius0", py::make_getter(&Ellipse::radius0), &Ellipse::setRadius0, u8"Transverse radius of the ellipse.")
        .add_property("radius1", py::make_getter(&Ellipse::radius1), &Ellipse::setRadius1, u8"Vertical radius of the ellipse.")
        .add_property("radii", &Ellipse_getRadii<Ellipse>, &Ellipse_setRadii<Ellipse>, u8"Radii of the ellipse.");

    py::class_<Cylinder, shared_ptr<Cylinder>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable>(
        "Cylinder",
        u8"Cylinder(radius, height, material)\n\n"
        u8"Vertical cylinder (3D geometry object).\n\n"
        u8"The cylinder base always lies in the horizontal (longitudinal-transverse)\n"
        u8"plane and it height spans in the vertical direction.\n\n"
        u8"Args:\n"
        u8"    radius (float): Cylinder radius.\n"
        u8"    height (float): Cylinder height.\n"
        u8"    material (Material): Cylinder material.\n",
        py::no_init)
        .def("__init__",
             py::make_constructor(&Cylinder_constructor, py::default_call_policies(), (py::arg("radius"), "height", "material")))
        .add_property("radius", py::make_getter(&Cylinder::radius), &Cylinder::setRadius, u8"Radius of the cylinder.")
        .add_property("height", py::make_getter(&Cylinder::height), &Cylinder::setHeight, u8"Height of the cylinder.");

    py::class_<HollowCylinder, shared_ptr<HollowCylinder>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable>(
        "Tube",
        u8"Tube(inner_radius, outer_radius, height, material)\n\n"
        u8"Vertically oriented tube i.e. hollow cylinder (3D geometry object).\n\n"
        u8"The tube base always lies in the horizontal (longitudinal-transverse)\n"
        u8"plane and it height spans in the vertical direction.\n\n"
        u8"Args:\n"
        u8"    inner_radius (float): Tube inner radius.\n"
        u8"    outer_radius (float): Tube outer radius.\n"
        u8"    height (float): Cylinder height.\n"
        u8"    material (Material): Cylinder material.\n",
        py::no_init)
        .def("__init__", py::make_constructor(&HollowCylinder_constructor, py::default_call_policies(),
                                              (py::arg("inner_radius"), "outer_radius", "height", "material")))
        .add_property("inner_radius", py::make_getter(&HollowCylinder::inner_radius), &HollowCylinder::setInnerRadius,
                      u8"Inner tube radius.")
        .add_property("outer_radius", py::make_getter(&HollowCylinder::outer_radius), &HollowCylinder::setOuterRadius,
                      u8"Outer tube radius.")
        .add_property("height", py::make_getter(&HollowCylinder::height), &HollowCylinder::setHeight, u8"Height of the Tube.");

    py::class_<EllipticCylinder, shared_ptr<EllipticCylinder>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable>(
        "EllipticCylinder",
        u8"EllipticCylinder(radius0, radius1, height, material, *, angle=0.)\n"
        u8"EllipticCylinder(radii, height, material, *, angle=0.)\n\n"
        u8"Vertical elliptic cylinder (3D geometry object).\n\n"
        u8"The cylinder base always lies in the horizontal (longitudinal-transverse)\n"
        u8"plane and it height spans in the vertical direction.\n\n"
        u8"Args:\n"
        u8"    radius0 (float): Ellipse radius along the first direction (longitudinal).\n"
        u8"    radius1 (float): Ellipse radius along the second direction (transverse).\n"
        u8"    angle (float): Rotation angle in the horizontal plane [deg].\n"
        u8"    height (float): Cylinder height.\n"
        u8"    material (Material): Cylinder material.\n",
        py::no_init)
        .def("__init__", raw_constructor(&EllipticCylinder_constructor))
        .add_property("radius0", py::make_getter(&EllipticCylinder::radius0), &EllipticCylinder::setRadius0,
                      u8"Longitudinal radius of the base ellipse.")
        .add_property("radius1", py::make_getter(&EllipticCylinder::radius1), &EllipticCylinder::setRadius1,
                      u8"Transverse radius of the base circle.")
        .add_property("radii", &Ellipse_getRadii<EllipticCylinder>, &Ellipse_setRadii<EllipticCylinder>, u8"Radii of the ellipse.")
        .add_property("height", py::make_getter(&EllipticCylinder::height), &EllipticCylinder::setHeight,
                      u8"Height of the cylinder.")
        .add_property("angle", &EllipticCylinder_getAngle, &EllipticCylinder_setAngle,
                      u8"Rotation angle in the horizontal plane.\n\n"
                      u8"This rotates the ellipse horizontally, so the longitudinal and transverse\n"
                      u8"dimensions are no longer along the axes, but correspond to the object sides.\n");

    py::class_<TriangularPrism, shared_ptr<TriangularPrism>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable> triangular_prism(
        "TriangularPrism",
        u8"TriangularPrism(a0, a1, b0, b1, height, material)\n"
        u8"TriangularPrism(a, b, height, material)\n"
        u8"TriangularPrism (3D geometry object).\n\n"
        u8"Three triangle vertices are located at points (0, 0, 0), (*a*, 0), and (*b*, 0).\n\n"
        u8"Args:\n"
        u8"    plask.vec a: Local coordinates of the first triangle vertex.\n"
        u8"    plask.vec b: Local coordinates of the second triangle vertex.\n"
        u8"    a0 (float): Longitudinal component of the local coordinates of the first\n"
        u8"                triangle vertex.\n"
        u8"    a1 (float): Transverse component of the local coordinates of the first\n"
        u8"                triangle vertex.\n"
        u8"    b0 (float): Longitudinal component of the local coordinates of the second\n"
        u8"                triangle vertex.\n"
        u8"    b1 (float): Transverse component of the local coordinates of the second\n"
        u8"                triangle vertex.\n"
        u8"    height (float): Prism height\n"
        u8"    material (Material): Prism material.\n",
        py::no_init);
    triangular_prism
        .def("__init__", py::make_constructor(&TriangularPrism_constructor_vec, py::default_call_policies(),
                                              (py::arg("a"), "b", "height", "material")))
        .def("__init__", py::make_constructor(&TriangularPrism_constructor_pts, py::default_call_policies(),
                                              (py::arg("a0"), "a1", "b0", "b1", "height", "material")))
        .add_property("a", py::make_getter(&TriangularPrism::p0, py::return_value_policy<py::return_by_value>()),
                      (void(TriangularPrism::*)(const Vec<2>&)) & TriangularPrism::setP0,
                      "Horizontal coordinates of the first base vertex.")
        .add_property("b", py::make_getter(&TriangularPrism::p1, py::return_value_policy<py::return_by_value>()),
                      (void(TriangularPrism::*)(const Vec<2>&)) & TriangularPrism::setP1,
                      "Horizontal coordinates of the second base vertex.")
        .add_property("a0", &Triangle_get_a<TriangularPrism, 0>, &Triangle_set_a<TriangularPrism, 0>,
                      "Horizontal coordinate of the first base vertex.\n\nInstead of 0 you can use longitudinal axis name.")
        .add_property("a1", &Triangle_get_a<TriangularPrism, 1>, &Triangle_set_a<TriangularPrism, 1>,
                      "Vertical coordinate of the first base vertex.\n\nInstead of 1 you can use transverse axis name.")
        .add_property("b0", &Triangle_get_b<TriangularPrism, 0>, &Triangle_set_b<TriangularPrism, 0>,
                      "Horizontal coordinate of the second base vertex.\n\nInstead of 0 you can use longitudinal axis name.")
        .add_property("b1", &Triangle_get_b<TriangularPrism, 1>, &Triangle_set_b<TriangularPrism, 1>,
                      "Vertical coordinate of the second base vertex.\n\nInstead of 1 you can use transverse axis name.")
        .add_property("height", py::make_getter(&TriangularPrism::height), &TriangularPrism::setHeight, "Prism height.")
        .def("__getattr__", &Triangle__getattr__<TriangularPrism>)
        .def("__setattr__", &Triangle__setattr__<TriangularPrism>);

    py::class_<Polygon, shared_ptr<Polygon>, py::bases<GeometryObjectLeaf<2>>, boost::noncopyable> polygon(
        "Polygon",
        u8"Polygon(vertices, material)\n\n"
        u8"Polygon (2D geometry object).\n\n"
        u8"Polygon with specified vertices.\n\n"
        u8"Args:\n"
        u8"    vertices (list of plask.vec): List of polygon vertices.\n"
        u8"    material (Material): Polygon material.\n",
        py::no_init);
    polygon
        .def("__init__", py::make_constructor(&Polygon_constructor, py::default_call_policies(), (py::arg("vertices"), "material")))
        .add_property("vertices", &Vertices2D<Polygon, Vec<2, double>>::fromPolygon,
                      &Vertices2D<Polygon, Vec<2, double>>::setVertices, "List of polygon vertices.");

    {
        py::scope polygon_scope = polygon;
        Vertices2D<Polygon, Vec<2, double>>::register_class();
    }

    py::class_<Prism, shared_ptr<Prism>, py::bases<GeometryObjectLeaf<3>>, boost::noncopyable> prism(
        "Prism",
        "Prism with base defined by polygon with specified vertices\n\n"
        u8"Prism(vertices, material)\n\n"
        u8"Prism (3D geometry object).\n\n"
        u8"Prism with specified vertices.\n\n"
        u8"Args:\n"
        u8"    vertices (list of plask.vec): List of polygon vertices.\n"
        u8"    height (float): Prism height.\n"
        u8"    material (Material): Prism material.\n",
        py::no_init);
    prism
        .def("__init__",
             py::make_constructor(&Prism_constructor, py::default_call_policies(), (py::arg("vertices"), "height", "material")))
        .add_property("height", &Prism::getHeight, &Prism::setHeight, "Prism height.")
        .add_property("vertices", &Vertices2D<Prism, LateralVec<double>>::fromPolygon,
                      &Vertices2D<Prism, LateralVec<double>>::setVertices, "List of polygon vertices.");

    {
        py::scope prism_scope = prism;
        Vertices2D<Prism, LateralVec<double>>::register_class();
    }
}
}}  // namespace plask::python
