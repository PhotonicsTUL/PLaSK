#include "../python_globals.h"
#include "../python_numpy.h"
#include "../python_mesh.h"

//#include <boost/python/stl_iterator.hpp>
#include <boost/python/iterator.hpp>

#include <plask/mesh/mesh.h>
#include <plask/mesh/interpolation.h>
#include <plask/mesh/triangular2d.h>
#include <plask/mesh/generator_triangular.h>

namespace plask { namespace python {

namespace py = boost::python;


void register_mesh_triangular() {

    py::class_<TriangularMesh2D, shared_ptr<TriangularMesh2D>, py::bases<MeshD<2>>> triangularMesh2D("TriangularMesh2D",
        u8"Two-dimensional triangular mesh\n\n",
        py::no_init);
    triangularMesh2D
            .def("__iter__", py::range(&TriangularMesh2D::begin, &TriangularMesh2D::end)) // we will use native iterators which perform better
            .add_property("elements", py::make_function(&TriangularMesh2D::elements, py::with_custodian_and_ward_postcall<0,1>()), u8"Element list in the mesh")
            .def(py::self == py::self)
            ;

    py::implicitly_convertible<shared_ptr<TriangularMesh2D>, shared_ptr<const TriangularMesh2D>>();

    {
        py::scope scope = triangularMesh2D;
        (void) scope;   // don't warn about unused variable scope

        py::class_<TriangularMesh2D::Element>("Element", u8"Element (FEM-like, triangle) of the :py:class:`mesh.Rectangular2D", py::no_init)
            .add_property("area", /*double*/ &TriangularMesh2D::Element::getArea, u8"Area of the element")
            .add_property("volume", /*double*/ &TriangularMesh2D::Element::getArea, u8"Alias for :attr:`area`")
            .add_property("center", /*Vec<2,double>*/ &TriangularMesh2D::Element::getMidpoint, u8"Position of the element center")
            .add_property("node_indexes", &TriangularMesh2D::Element::triangleNodes, u8"Indices of the element (triangle) vertices on the orignal mesh.")
            .add_property("nodes", &TriangularMesh2D::Element::getNodes, "coordinates of the element (triangle) vertices")
            .def("node", &TriangularMesh2D::Element::getNode, py::arg("index"), py::return_value_policy<py::return_by_value>(), "coordinate of the element (triangle) vertex")
            .add_property("box", /*Box2D*/ &TriangularMesh2D::Element::getBoundingBox, u8"bounding box of the element")
            .def("barycentric", &TriangularMesh2D::Element::barycentric, py::return_value_policy<py::return_by_value>(), "barycentric (area) coordinates of given point")
            //.add_property("index", /*size_t*/ &TriangularMesh2D::Element::getIndex, u8"element index")
        ;

        py::implicitly_convertible<shared_ptr<TriangularMesh2D::Element>, shared_ptr<const TriangularMesh2D::Element>>();

        py::class_<TriangularMesh2D::Elements>("Elements", u8"Element list in the :py:class:`mesh.TriangularMesh2D", py::no_init)
            .def("__len__", &TriangularMesh2D::Elements::size)
            .def("__getitem__", &TriangularMesh2D::Elements::operator[], py::with_custodian_and_ward_postcall<0,1>())
            .def("__iter__", py::range<py::with_custodian_and_ward_postcall<0,1>>(&TriangularMesh2D::Elements::begin, &TriangularMesh2D::Elements::end))
            //.add_property("mesh", &RectangularMesh_ElementMesh<RectangularMesh2D>, "Mesh with element centers")
        ;

        py::implicitly_convertible<shared_ptr<TriangularMesh2D::Elements>, shared_ptr<const TriangularMesh2D::Elements>>();


        py::class_<TriangularMesh2D::Builder>("Builder",
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

        py::implicitly_convertible<shared_ptr<TriangularMesh2D::Builder>, shared_ptr<const TriangularMesh2D::Builder>>();
    }


}

} } // namespace plask::python
