#include "geometry.h"
#include <plask/geometry/element.h>
#include <plask/geometry/path.h>

namespace plask { namespace python {

// Some helpful wrappers
template <int dim> struct GeometryElementD_inside {};
template <> struct GeometryElementD_inside<2> {
    static inline bool call(const GeometryElementD<2>& self, double c0, double c1) {
        return self.inside(Vec<2,double>(c0, c1));
    }
};
template <> struct GeometryElementD_inside<3> {
    static inline bool call(const GeometryElementD<3>& self, double c0, double c1, double c2) {
        return self.inside(Vec<3,double>(c0, c1, c2));
    }
};

template <int dim> struct GeometryElementD_getMaterial {};
template <> struct GeometryElementD_getMaterial<2> {
    static inline shared_ptr<Material> call(const GeometryElementD<2>& self, double c0, double c1) {
        return self.getMaterial(Vec<2,double>(c0, c1));
    }
};
template <> struct GeometryElementD_getMaterial<3> {
    static inline shared_ptr<Material> call(const GeometryElementD<3>& self, double c0, double c1, double c2) {
        return self.getMaterial(Vec<3,double>(c0, c1, c2));
    }
};

/// Initialize class GeometryElementD for Python
template <int dim> struct GeometryElementD_vector_args { static const py::detail::keywords<dim> args; };
template<> const py::detail::keywords<2> GeometryElementD_vector_args<2>::args = (py::arg("c0"), py::arg("c1"));
template<> const py::detail::keywords<3> GeometryElementD_vector_args<3>::args = (py::arg("c0"), py::arg("c1"), py::arg("c2"));
DECLARE_GEOMETRY_ELEMENT_23D(GeometryElementD, "GeometryElement", "Base class for "," geometry elements") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryElementD, GeometryElement)
        .def("inside", &GeometryElementD<dim>::inside, (py::arg("point")),
             "Return True if the geometry element includes a point (in local coordinates)")
        .def("inside", &GeometryElementD_inside<dim>::call, GeometryElementD_vector_args<dim>::args,
             "Return True if the geometry element includes a point (in local coordinates)")
        .def("intersect", &GeometryElementD<dim>::intersect, (py::arg("area")),
             "Return True if the geometry element has common points (in local coordinates) with an area")
        .def("getMaterial", &GeometryElementD<dim>::getMaterial, (py::arg("point")),
             "Return material at given point, provided that it is inside the bounding box (in local coordinates) and None otherwise")
        .def("getMaterial", &GeometryElementD_getMaterial<dim>::call, GeometryElementD_vector_args<dim>::args,
             "Return material at given point, provided that it is inside the bounding box (in local coordinates) and None otherwise")
        .add_property("boundingBox", &GeometryElementD<dim>::getBoundingBox,
                      "Minimal rectangle which includes all points of the geometry element (in local coordinates)")
        .add_property("boundingBoxSize", &GeometryElementD<dim>::getBoundingBoxSize,
                      "Size of the bounding box")
        .add_property("leafsBoundigBoxes", &GeometryElementD<dim>::getLeafsBoundingBoxes,
                      "Calculate bounding boxes of all leafs (in local coordinates)")
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
                      &GeometryElementTransform<dim>::setChild, "Child of the transform object")
        .def("hasChild", &GeometryElementTransform<dim>::hasChild, "Return true if the transform object has a set child")
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

    py::class_<GeometryElement, shared_ptr<GeometryElement>, boost::noncopyable>("GeometryElement",
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
