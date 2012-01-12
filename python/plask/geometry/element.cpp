#include "geometry.h"
#include <plask/geometry/element.h>

namespace plask { namespace python {

/// Initialize class GeometryElementD for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementD, "GeometryElement", "Base class for "," geometry elements") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementD, GeometryElement)
        .def("inside", &GeometryElementD<dim>::inside, "Return True if the geometry element includes a point (in local coordinates)")
        .def("intersect", &GeometryElementD<dim>::intersect, "Return True if the geometry element has common points (in local coordinates) with an area")
        .def("getMaterial", &GeometryElementD<dim>::getMaterial, "Return material at given point, provided that it is inside the bounding box (in local coordinates) and None otherwise")
        .add_property("boundingBox", &GeometryElementD<dim>::getBoundingBox, "Minimal rectangle which includes all points of the geometry element (in local coordinates)")
        .add_property("boundingBoxSize", &GeometryElementD<dim>::getBoundingBoxSize, "Size of the bounding box")
        .add_property("leafsBoundigBoxes", &GeometryElementD<dim>::getLeafsBoundingBoxes, "Calculate bounding boxes of all leafs (in local coordinates)")
    ;
}


/// Initialize class GeometryElementLeaf for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementLeaf, "GeometryElementLeaf", "Base class for all "," leaves") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementLeaf, GeometryElementD<dim>)
        .def_readwrite("material", &GeometryElementLeaf<dim>::material, "material of the geometry object")
    ;
}


/// Initialize class GeometryElementTransform for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementTransform, "GeometryElementTransform", "Base class for all "," transform nodes") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementTransform, GeometryElementD<dim>)
        .add_property("child",
                      (shared_ptr<typename GeometryElementTransform<dim>::ChildType> (GeometryElementTransform<dim>::*)()) &GeometryElementTransform<dim>::getChild,
                      &GeometryElementTransform<dim>::setChild)
        .def("hasChild", &GeometryElementTransform<dim>::hasChild)
    ;
}


DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementContainer, "GeometryElementContainer", "Base class for all "," containers") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementContainer, GeometryElementD<dim>)
    ;
}


void register_geometry_element()
{
    py::enum_<GeometryElementType>("ElementType")
        .value("LEAF", GE_TYPE_LEAF)
        .value("TRANSFORM", GE_TYPE_TRANSFORM)
        .value("SPACE_CHANGER", GE_TYPE_SPACE_CHANGER)
        .value("CONTAINER", GE_TYPE_CONTAINER)
    ;

    py::class_<GeometryElement, boost::noncopyable>("GeometryElement",
        "Base class for all geometry elements.", py::no_init)
        .add_property("type", &GeometryElement::getType)
        .def("validate", &GeometryElement::validate)
    ;

    init_GeometryElementD<2>();
    init_GeometryElementD<3>();

    init_GeometryElementLeaf<2>();
    init_GeometryElementLeaf<3>();

    init_GeometryElementTransform<2>();
    init_GeometryElementTransform<3>();

    init_GeometryElementContainer<2>();
    init_GeometryElementContainer<3>();

    // Space changer
    py::class_<GeometryElementChangeSpace<3,2>, shared_ptr<GeometryElementChangeSpace<3,2>>, py::bases<GeometryElementTransform<3>>, boost::noncopyable>
    ("GeometryElementChangeSpace2Dto3D", "Base class for elements changing space 2D to 3D", py::no_init);

    py::class_<GeometryElementChangeSpace<2,3>, shared_ptr<GeometryElementChangeSpace<2,3>>, py::bases<GeometryElementTransform<2>>, boost::noncopyable>
    ("GeometryElementChangeSpace3Dto2D", "Base class for elements changing space 3D to 2D using some averaging or cross-section", py::no_init);

}


}} // namespace plask::python
