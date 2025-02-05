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
#include "../python/globals.hpp"
#include "../python/numpy.hpp"
#include "../python/mesh.hpp"

#include <plask/mesh/extruded_triangular3d.hpp>

namespace plask { namespace python {

static py::list top_element_nodes(const ExtrudedTriangularMesh3D::Element& self) {
    py::list result;
    result.append(self.getTopNode(0));
    result.append(self.getTopNode(1));
    result.append(self.getTopNode(2));
    return result;
}

static py::list bottom_element_nodes(const ExtrudedTriangularMesh3D::Element& self) {
    py::list result;
    result.append(self.getBottomNode(0));
    result.append(self.getBottomNode(1));
    result.append(self.getBottomNode(2));
    return result;
}

void register_mesh_extruded_triangular() {

    py::class_<ExtrudedTriangularMesh3D, shared_ptr<ExtrudedTriangularMesh3D>, py::bases<MeshD<3>>> extrudedTriangularMesh3D("ExtrudedTriangular",
        u8"3D mesh which is a cartesian product of 2D triangular mesh at long-tran and 1D mesh at vert axis");
    extrudedTriangularMesh3D
            //.def("__iter__", py::range(&ExtrudedTriangularMesh3D::begin, &ExtrudedTriangularMesh3D::end)) // we will use native iterators which perform better
            .add_property("elements", py::make_function(&ExtrudedTriangularMesh3D::elements, py::with_custodian_and_ward_postcall<0,1>()), u8"Element list in the mesh")
            .def_readwrite("long_tran", &ExtrudedTriangularMesh3D::longTranMesh, u8"Two-dimensional triangular mesh (Triangular2D) with long-tran coordinates of nodes.")
            .def_readonly("axis_vert", &ExtrudedTriangularMesh3D::vertAxis, u8"The third (vertical) axis of the mesh.")
            //.add_property("long_tran")
            .def("Front", &ExtrudedTriangularMesh3D::getFrontBoundary, u8"Front side of the mesh for setting boundary conditions").staticmethod("Front")
            .def("Back", &ExtrudedTriangularMesh3D::getBackBoundary, u8"Back side of the mesh for setting boundary conditions").staticmethod("Back")
            .def("Left", &ExtrudedTriangularMesh3D::getLeftBoundary, u8"Left edge of the mesh for setting boundary conditions").staticmethod("Left")
            .def("Right", &ExtrudedTriangularMesh3D::getRightBoundary, u8"Right edge of the mesh for setting boundary conditions").staticmethod("Right")
            .def("Top", &ExtrudedTriangularMesh3D::getTopBoundary, u8"Top edge of the mesh for setting boundary conditions").staticmethod("Top")
            .def("Bottom", &ExtrudedTriangularMesh3D::getBottomBoundary, u8"Bottom edge of the mesh for setting boundary conditions").staticmethod("Bottom")
            //.def("Edge", &ExtrudedTriangularMesh3D::getAllBoundary, u8"Whole edge (outer and inner) of the mesh for setting boundary conditions").staticmethod("Edge")
            .def("FrontOf", (ExtrudedTriangularMesh3D::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&ExtrudedTriangularMesh3D::getFrontOfBoundary,
                 u8"Boundary in front of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("FrontOf")
            .def("BackOf", (ExtrudedTriangularMesh3D::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&ExtrudedTriangularMesh3D::getBackOfBoundary,
                 u8"Boundary back of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("BackOf")
            .def("LeftOf", (ExtrudedTriangularMesh3D::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&ExtrudedTriangularMesh3D::getLeftOfBoundary,
                 u8"Boundary left of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("LeftOf")
            .def("RightOf", (ExtrudedTriangularMesh3D::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&ExtrudedTriangularMesh3D::getRightOfBoundary,
                 u8"Boundary right of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("RightOf")
            .def("TopOf", (ExtrudedTriangularMesh3D::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&ExtrudedTriangularMesh3D::getTopOfBoundary,
                 u8"Boundary top of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("TopOf")
            .def("BottomOf", (ExtrudedTriangularMesh3D::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&ExtrudedTriangularMesh3D::getBottomOfBoundary,
                 u8"Boundary bottom of specified object", (py::arg("object"), py::arg("path")=py::object())).staticmethod("BottomOf")
            /*.def("EdgeOf", (ExtrudedTriangularMesh3D::Boundary(*)(shared_ptr<const GeometryObject>,const PathHints&))&ExtrudedTriangularMesh3D::getAllBoundaryIn,
                 u8"Edge of specified object (and edge of mesh holes inside the object)", (py::arg("object"), py::arg("path")=py::object())).staticmethod("EdgeOf")*/
            .def(py::self == py::self)
            ;

    py::implicitly_convertible<shared_ptr<ExtrudedTriangularMesh3D>, shared_ptr<const ExtrudedTriangularMesh3D>>();

    {
        py::scope scope = extrudedTriangularMesh3D;
        (void) scope;   // don't warn about unused variable scope

        py::class_<ExtrudedTriangularMesh3D::Element>("Element", u8"Element (FEM-like, triangle) of the :py:class:`mesh.ExtrudedTriangular", py::no_init)
            .add_property("area", /*double*/ &ExtrudedTriangularMesh3D::Element::getArea, u8"Volume of the element")
            .add_property("volume", /*double*/ &ExtrudedTriangularMesh3D::Element::getArea, u8"Alias for :attr:`area`")
            .add_property("center", /*Vec<2,double>*/ &ExtrudedTriangularMesh3D::Element::getMidpoint, u8"Position of the element center")
            //.add_property("top_node_indexes", &ExtrudedTriangularMesh3D::Element::triangleNodes, u8"Indices of the element (triangle) vertices on the original mesh.")
            //.add_property("bottom_node_indexes", &ExtrudedTriangularMesh3D::Element::triangleNodes, u8"Indices of the element (triangle) vertices on the original mesh.")
            .add_property("top_nodes", top_element_nodes, "coordinates of the top base (triangle) vertices")
            .add_property("bottom_nodes", bottom_element_nodes, "coordinates of the bottom base (triangle) vertices")
            .def("top_node", &ExtrudedTriangularMesh3D::Element::getTopNode, py::arg("index"), py::return_value_policy<py::return_by_value>(), "coordinates of the top base (triangle) vertex")
            .def("bottom_node", &ExtrudedTriangularMesh3D::Element::getBottomNode, py::arg("index"), py::return_value_policy<py::return_by_value>(), "coordinates of the bottom base (triangle) vertex")
            .add_property("box", /*Box2D*/ &ExtrudedTriangularMesh3D::Element::getBoundingBox, u8"bounding box of the element")
            //.def("barycentric", &ExtrudedTriangularMesh3D::Element::barycentric, py::return_value_policy<py::return_by_value>(), "barycentric (area) coordinates of given point")
            .def("__contains__", &ExtrudedTriangularMesh3D::Element::contains, "check if given point is included in triangle represented by this element")
            //.add_property("index", /*size_t*/ &TriangularMesh2D::Element::getIndex, u8"element index")
        ;

        py::implicitly_convertible<shared_ptr<ExtrudedTriangularMesh3D::Element>, shared_ptr<const ExtrudedTriangularMesh3D::Element>>();

        py::class_<ExtrudedTriangularMesh3D::Elements>("Elements", u8"Element list in the :py:class:`mesh.ExtrudedTriangular", py::no_init)
            .def("__len__", &ExtrudedTriangularMesh3D::Elements::size)
            .def("__getitem__", &ExtrudedTriangularMesh3D::Elements::at, py::with_custodian_and_ward_postcall<0,1>())
            .def("__iter__", py::range<py::with_custodian_and_ward_postcall<0,1>>(&ExtrudedTriangularMesh3D::Elements::begin, &ExtrudedTriangularMesh3D::Elements::end))
            //.add_property("mesh", &RectangularMesh_ElementMesh<RectangularMesh2D>, "Mesh with element centers")
        ;

        py::implicitly_convertible<shared_ptr<ExtrudedTriangularMesh3D::Elements>, shared_ptr<const ExtrudedTriangularMesh3D::Elements>>();


        /*py::class_<TriangularMesh2D::Builder>("Builder",
                                              u8"Allows for adding triangles to the :py:class:`mesh.TriangularMesh2D effectively",
                                              py::init<TriangularMesh2D&>()[py::with_custodian_and_ward<1,2>()])
            .def("add",
                 (TriangularMesh2D::Builder&(TriangularMesh2D::Builder::*)(Vec<2,double>, Vec<2,double>, Vec<2,double>)) &TriangularMesh2D::Builder::add,
                 (py::arg("p1"), py::arg("p2"), py::arg("p3")), py::return_self<>(),
                 "add a triangle (with given vertices: p1, p2, p3) to the mesh")
            .def("add",
                 (TriangularMesh2D::Builder&(TriangularMesh2D::Builder::*)(const TriangularMesh2D::Element&)) &TriangularMesh2D::Builder::add,
                 (py::arg("element")), py::return_self<>(),
                 "add a triangle represented by the given element to the mesh")
        ;

        py::implicitly_convertible<shared_ptr<TriangularMesh2D::Builder>, shared_ptr<const TriangularMesh2D::Builder>>();*/
    }

    /*py::class_<TriangleGenerator, shared_ptr<TriangleGenerator>,
               py::bases<MeshGeneratorD<2>>, boost::noncopyable>("TriangleGenerator",
       u8"Generator which creates triangular mesh by Triangle library authored by Jonathan Richard Shewchuk.\n"
       u8"\n"
       u8"Triangle generates exact Delaunay triangulations, constrained Delaunay triangulations,"
       u8"conforming Delaunay triangulations, Voronoi diagrams, and high-quality triangular meshes.\n"
       u8"The latter can be generated with no small or large angles,"
       u8"and are thus suitable for finite element analysis.\n"
       u8"\n"
       u8"See: https://www.cs.cmu.edu/~quake/triangle.html",
       py::init<>()
    );
    py::implicitly_convertible<shared_ptr<TriangleGenerator>, shared_ptr<const TriangleGenerator>>();*/
}

} } // namespace plask::python
