#ifndef PLASK__PYTHON_GEOMETRY_ELEMENT_H
#define PLASK__PYTHON_GEOMETRY_ELEMENT_H

#include "geometry.h"

namespace plask { namespace python {

/// Initialize class GeometryElementD for Python
DECLARE_GEOMETRY_ELEMENT(GeometryElementD, "GeometryElement", "Base class for "," geometry elements") {
    ABSTRACT_GEOMETRY_ELEMENT(GeometryElementD)
        .def("inside", &GeometryElementD<dim>::inside)
        .def("intersect", &GeometryElementD<dim>::intersect)
        .add_property("boundingBox", &GeometryElementD<dim>::getBoundingBox)
        .add_property("boundingBoxSize", &GeometryElementD<dim>::getBoundingBoxSize)
        .add_property("material", &GeometryElementD<dim>::getMaterial)
        .add_property("leafsBoundigBoxes", &GeometryElementD<dim>::getLeafsBoundingBoxes)
    ;
}

/// Wrapper for GeometryElementTransform::getChild (required because of the overloading)
template <int dim>
const GeometryElementD<dim>* GeometryElementTransform_getChild(const GeometryElementTransform<dim>& self) {
    return &self.getChild();
}
/// Wrapper for GeometryElementTransform::setChild
template <int dim>
void GeometryElementTransform_setChild(GeometryElementTransform<dim>& self, GeometryElementD<dim>* child) {
    self.setChild(*child);
}
/// Initialize class GeometryElementTransform for Python
DECLARE_GEOMETRY_ELEMENT(GeometryElementTransform, "GeometryElementTransform", "Base class for all "," transform nodes") {
    ABSTRACT_GEOMETRY_ELEMENT(GeometryElementTransform)
        .add_property("child",
            py::make_function(&GeometryElementTransform_getChild<dim>, py::return_internal_reference<1>()),
            py::make_function(&GeometryElementTransform_setChild<dim>, py::with_custodian_and_ward<1, 2>()))
        .def("hasChild", &GeometryElementTransform<2>::hasChild)
    ;
}

DECLARE_GEOMETRY_ELEMENT(GeometryElementContainer, "GeometryElementContainer", "Base class for all "," containers") {
    ABSTRACT_GEOMETRY_ELEMENT(GeometryElementContainer)
    ;
}

inline static void init_geometry_element_h()
{
    py::enum_<GeometryElementType>("ElementType")
        .value("LEAF", GE_TYPE_LEAF)
        .value("TRANSFORM", GE_TYPE_TRANSFORM)
        .value("SPACE_CHANGER", GE_TYPE_SPACE_CHANGER)
        .value("CONTAINER", GE_TYPE_CONTAINER)
    ;

    py::class_<GeometryElement, shared_ptr<GeometryElement>, boost::noncopyable>("GeometryElement",
        "Base class for all geometry elements.", py::no_init)
        .add_property("type", &GeometryElement::getType)
        .def("validate", &GeometryElement::validate)
    ;

    init_GeometryElementD<2>();
    init_GeometryElementD<3>();

    init_GeometryElementTransform<2>();
    init_GeometryElementTransform<3>();

    //TODO: GeometryElementChangeSpace

    init_GeometryElementContainer<2>();
    init_GeometryElementContainer<3>();
}


}} // namespace plask::python
#endif // PLASK__PYTHON_GEOMETRY_ELEMENT_H
