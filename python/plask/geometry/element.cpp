#include "geometry.h"
#include <plask/geometry/element.h>

namespace plask { namespace python {

/// Initialize class GeometryElementD for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementD, "GeometryElement", "Base class for "," geometry elements") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementD, GeometryElement)
        .def("inside", &GeometryElementD<dim>::inside)
        .def("intersect", &GeometryElementD<dim>::intersect)
        .add_property("boundingBox", &GeometryElementD<dim>::getBoundingBox)
        .add_property("boundingBoxSize", &GeometryElementD<dim>::getBoundingBoxSize)
        .add_property("material", &GeometryElementD<dim>::getMaterial)
        .add_property("leafsBoundigBoxes", &GeometryElementD<dim>::getLeafsBoundingBoxes)
    ;
}


/// Initialize class GeometryElementLeaf for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementLeaf, "GeometryElementLeaf", "Base class for all "," leaves") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementLeaf, GeometryElementD<dim>)
        .add_property("material", &GeometryElementLeaf<dim>::getMaterial, &GeometryElementLeaf<dim>::material)
    ;
}


/// Wrapper for GeometryElementTransform::getChild (required because of the overloading)
template <int dim>
static const GeometryElementD<dim>* GeometryElementTransform_getChild(const GeometryElementTransform<dim>& self) {
    return &self.getChild();
}
/// Wrapper for GeometryElementTransform::setChild (required because of the overloading)
template <int dim>
static void GeometryElementTransform_setChild(GeometryElementTransform<dim>& self, GeometryElementD<dim>* child) {
    self.setChild(*child);
}
/// Initialize class GeometryElementTransform for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementTransform, "GeometryElementTransform", "Base class for all "," transform nodes") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementTransform, GeometryElementD<dim>)
        .add_property("child",
            py::make_function(&GeometryElementTransform_getChild<dim>, py::return_internal_reference<>()),
            py::make_function(&GeometryElementTransform_setChild<dim>, py::with_custodian_and_ward<1, 2>()))
        .def("hasChild", &GeometryElementTransform<2>::hasChild)
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
