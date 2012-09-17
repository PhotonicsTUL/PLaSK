#include "geometry.h"
#include <plask/geometry/transform.h>
#include <plask/geometry/leaf.h>
#include <plask/geometry/container.h>
#include <plask/geometry/path.h>

namespace plask { namespace python {

// Some helpful wrappers
template <int dim> struct GeometryObjectD_includes {};
template <> struct GeometryObjectD_includes<2> {
    static inline bool call(const GeometryObjectD<2>& self, double c0, double c1) {
        return self.includes(Vec<2,double>(c0, c1));
    }
};
template <> struct GeometryObjectD_includes<3> {
    static inline bool call(const GeometryObjectD<3>& self, double c0, double c1, double c2) {
        return self.includes(Vec<3,double>(c0, c1, c2));
    }
};

template <int dim> struct GeometryObjectD_getMaterial {};
template <> struct GeometryObjectD_getMaterial<2> {
    static inline shared_ptr<Material> call(const GeometryObjectD<2>& self, double c0, double c1) {
        return self.getMaterial(Vec<2,double>(c0, c1));
    }
};
template <> struct GeometryObjectD_getMaterial<3> {
    static inline shared_ptr<Material> call(const GeometryObjectD<3>& self, double c0, double c1, double c2) {
        return self.getMaterial(Vec<3,double>(c0, c1, c2));
    }
};

template <int dim> struct GeometryObjectD_getPathsTo;
template <> struct GeometryObjectD_getPathsTo<2> {
    static inline GeometryObject::Subtree call(const GeometryObjectD<2>& self, double c0, double c1, bool all) {
        return self.getPathsTo(Vec<2,double>(c0, c1), all);
    }
};
template <> struct GeometryObjectD_getPathsTo<3> {
    static inline GeometryObject::Subtree call(const GeometryObjectD<3>& self, double c0, double c1, double c2, bool all) {
        return self.getPathsTo(Vec<3,double>(c0, c1, c2), all);
    }
};

template <int dim>
static py::list GeometryObjectD_getLeafsAsTranslations(const GeometryObjectD<dim>& self, const PathHints& path) {
    py::list result;
    auto leafs = self.getLeafs(&path);
    auto translations = self.getLeafsPositions(&path);
    auto l = leafs.begin();
    auto t = translations.begin();
    for (; l != leafs.end(); ++l, ++t) {
        result.append(make_shared<Translation<dim>>(const_pointer_cast<GeometryObjectD<dim>>(static_pointer_cast<const GeometryObjectD<dim>>(*l)), *t));
    }
    return result;
}

template <int dim>
static py::list GeometryObjectD_getObjectAsTranslations(const shared_ptr<GeometryObjectD<dim>>& self, const shared_ptr<GeometryObjectD<dim>>& object, const PathHints& path) {
    auto translations = self->extractObject(*object, path);
    py::list result;
    for (auto i: translations) result.append(const_pointer_cast<GeometryObjectD<dim>>(i));
    return result;
}

static py::list GeometryObject_getLeafs(const shared_ptr<GeometryObject>& self, const PathHints& path) {
    std::vector<shared_ptr<const GeometryObject>> leafs = self->getLeafs(&path);
    py::list result;
    for (auto i: leafs) result.append(const_pointer_cast<GeometryObject>(i));
    return result;
}

std::string GeometryObject__repr__(const shared_ptr<GeometryObject>& self) {
    std::stringstream out;
    try {
        py::object obj(self);
        py::object cls = obj.attr("__class__");
        std::string solver = py::extract<std::string>(cls.attr("__module__"));
        std::string name = py::extract<std::string>(cls.attr("__name__"));
        out << "<" << solver << "." << name << " object at (" << self << ")>";
    } catch (py::error_already_set) {
        PyErr_Clear();
        out << "<Unrecognized plask.geometry.GeometryObject subclass object at (" << self << ")>";
    }
    return out.str();
}

/// Initialize class GeometryObjectD for Python
template <int dim> struct GeometryObjectD_vector_args { static const py::detail::keywords<dim> args; };
template<> const py::detail::keywords<2> GeometryObjectD_vector_args<2>::args = (py::arg("c0"), py::arg("c1"));
template<> const py::detail::keywords<3> GeometryObjectD_vector_args<3>::args = (py::arg("c0"), py::arg("c1"), py::arg("c2"));
DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectD, "GeometryObject", "Base class for "," geometry objects") {
    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectD, GeometryObject)
        .def("includes", &GeometryObjectD<dim>::includes, (py::arg("point")),
             "Return True if the geometry object includes a point (in local coordinates)")
        .def("includes", &GeometryObjectD_includes<dim>::call, GeometryObjectD_vector_args<dim>::args,
             "Return True if the geometry object includes a point (in local coordinates)")
        .def("intersects", &GeometryObjectD<dim>::intersects, (py::arg("area")),
             "Return True if the geometry object has common points (in local coordinates) with an area")
        .def("getMaterial", &GeometryObjectD<dim>::getMaterial, (py::arg("point")),
             "Return material at given point, provided that it is inside the bounding box (in local coordinates) and None otherwise")
        .def("getMaterial", &GeometryObjectD_getMaterial<dim>::call, GeometryObjectD_vector_args<dim>::args,
             "Return material at given point, provided that it is inside the bounding box (in local coordinates) and None otherwise")
        .add_property("bbox", &GeometryObjectD<dim>::getBoundingBox,
                      "Minimal rectangle which includes all points of the geometry object (in local coordinates)")
        .add_property("bbox_size", &GeometryObjectD<dim>::getBoundingBoxSize,
                      "Size of the bounding box")
        .def("getLeafsPositions", (std::vector<typename Primitive<dim>::DVec>(GeometryObjectD<dim>::*)(const PathHints&)const) &GeometryObjectD<dim>::getLeafsPositions,
             (py::arg("path")=py::object()), "Calculate positions of all leafs (in local coordinates)")
        .def("getLeafsBBoxes", (std::vector<typename Primitive<dim>::Box>(GeometryObjectD<dim>::*)(const PathHints&)const) &GeometryObjectD<dim>::getLeafsBoundingBoxes,
             (py::arg("path")=py::object()), "Calculate bounding boxes of all leafs (in local coordinates)")
        .def("getLeafsAsTranslations", &GeometryObjectD_getLeafsAsTranslations<dim>, (py::arg("path")=py::object()),
             "Return list of Translation objects holding all leafs")
        .def("getLeafs", &GeometryObject_getLeafs, (py::arg("path")=py::object()),
             "Return list of all leafs in the subtree originating from this object")
        .def("getObjectPositions", (std::vector<typename Primitive<dim>::DVec>(GeometryObjectD<dim>::*)(const GeometryObject&, const PathHints&)const) &GeometryObjectD<dim>::getObjectPositions,
             (py::arg("object"), py::arg("path")=py::object()), "Calculate positions of all all instances of specified object (in local coordinates)")
        .def("getObjectBBoxes", (std::vector<typename Primitive<dim>::Box>(GeometryObjectD<dim>::*)(const GeometryObject&, const PathHints&)const) &GeometryObjectD<dim>::getObjectBoundingBoxes,
             (py::arg("object"), py::arg("path")=py::object()), "Calculate bounding boxes of all instances of specified object (in local coordinates)")
        .def("getObjectAsTranslations", &GeometryObjectD_getObjectAsTranslations<dim>,
             (py::arg("object"), py::arg("path")=py::object()), "Return Translations holding specified object")
        .def("getPathsTo", (GeometryObject::Subtree(GeometryObjectD<dim>::*)(const typename GeometryObjectD<dim>::DVec&,bool)const) &GeometryObjectD<dim>::getPathsTo, (py::arg("point"), py::arg("all")=false),
             "Return subtree containing paths to all leafs covering specified point")
        .def("getPathsTo", &GeometryObjectD_getPathsTo<dim>::call, (GeometryObjectD_vector_args<dim>::args, py::arg("all")=false),
             "Return subtree containing paths to all leafs covering specified point")
    ;
}

void register_geometry_object()
{
    py_enum<GeometryObject::Type>("ObjectType")
        .value("LEAF", GeometryObject::TYPE_LEAF)
        .value("TRANSFORM", GeometryObject::TYPE_TRANSFORM)
        .value("SPACE_CHANGER", GeometryObject::TYPE_SPACE_CHANGER)
        .value("CONTAINER", GeometryObject::TYPE_CONTAINER)
    ;

    py::class_<GeometryObject, shared_ptr<GeometryObject>, boost::noncopyable>("GeometryObject",
        "Base class for all geometry objects.", py::no_init)
        .add_property("type", &GeometryObject::getType)
        .def("validate", &GeometryObject::validate, "Check if the object is compete and ready for calculations")
        .def("__repr__", &GeometryObject__repr__)
        .def("__eq__", __is__<GeometryObject>)
    ;

    register_vector_of<shared_ptr<GeometryObject>>("GeometryObject");

    init_GeometryObjectD<2>();
    init_GeometryObjectD<3>();

}


}} // namespace plask::python
