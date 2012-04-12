#include "geometry.h"
#include <plask/geometry/transform.h>
#include <plask/geometry/leaf.h>
#include <plask/geometry/container.h>
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

template <int dim>
static py::list GeometryElementD_leafsAsTranslations(const GeometryElementD<dim>& self, const PathHints* path=0) {
    py::list result;
    auto leafs = self.getLeafs(path);
    auto translations = self.getLeafsPositions(path);
    auto l = leafs.begin();
    auto t = translations.begin();
    for (; l != leafs.end(); ++l, ++t) {
        result.append(make_shared<Translation<dim>>(const_pointer_cast<GeometryElementD<dim>>(static_pointer_cast<const GeometryElementD<dim>>(*l)), *t));
    }
    return result;
}

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
        .add_property("bbox", &GeometryElementD<dim>::getBoundingBox,
                      "Minimal rectangle which includes all points of the geometry element (in local coordinates)")
        .add_property("bbox_size", &GeometryElementD<dim>::getBoundingBoxSize,
                      "Size of the bounding box")
        .def("getLeafsPositions", &GeometryElementD<dim>::getLeafsPositions, (py::arg("path")=py::object()),
             "Calculate positions of all leafs (in local coordinates)")
        .def("getLeafsBBoxes", (std::vector<typename GeometryElementD<dim>::Box> (GeometryElementD<dim>::*)(const PathHints*) const) &GeometryElementD<dim>::getLeafsBoundingBoxes,
                     (py::arg("path")=py::object()), "Calculate bounding boxes of all leafs (in local coordinates)")
        .def("getLeafsAsTranslations", &GeometryElementD_leafsAsTranslations<dim>, (py::arg("path")=py::object()),
             "Return list of Translation objects holding all leafs")
    ;
}

std::string GeometryElement__repr__(const shared_ptr<GeometryElement>& self) {
    std::stringstream out;
    try {
        py::object obj(self);
        py::object cls = obj.attr("__class__");
        std::string module = py::extract<std::string>(cls.attr("__module__"));
        std::string name = py::extract<std::string>(cls.attr("__name__"));
        out << "<" << module << "." << name << " object at (" << self << ")>";
    } catch (py::error_already_set) {
        PyErr_Clear();
        out << "<Unrecognized plask.geometry.GeometryElement subclass object at (" << self << ")>";
    }
    return out.str();
}

static std::vector<shared_ptr<GeometryElement>> GeometryElement_getLeafs(const shared_ptr<GeometryElement>& self, const PathHints* path=0) {
    std::vector<shared_ptr<const GeometryElement>> leafs = self->getLeafs();
    std::vector<shared_ptr<GeometryElement>> result;
    result.reserve(leafs.size());
    for (auto i: leafs) result.push_back(const_pointer_cast<GeometryElement>(i));
    return result;
}

void register_geometry_element()
{
    py::enum_<GeometryElement::Type>("ElementType")
        .value("LEAF", GeometryElement::TYPE_LEAF)
        .value("TRANSFORM", GeometryElement::TYPE_TRANSFORM)
        .value("SPACE_CHANGER", GeometryElement::TYPE_SPACE_CHANGER)
        .value("CONTAINER", GeometryElement::TYPE_CONTAINER)
    ;

    py::class_<GeometryElement, shared_ptr<GeometryElement>, boost::noncopyable>("GeometryElement",
        "Base class for all geometry elements.", py::no_init)
        .add_property("type", &GeometryElement::getType)
        .def("validate", &GeometryElement::validate)
        .def("getLeafs", &GeometryElement_getLeafs, (py::arg("path")=py::object()),
             "Return list of all leafs in the subtree originating from this element")
        .def("__repr__", &GeometryElement__repr__)
    ;

    register_vector_of<shared_ptr<GeometryElement>>("GeometryElement");

    init_GeometryElementD<2>();
    init_GeometryElementD<3>();

}


}} // namespace plask::python
