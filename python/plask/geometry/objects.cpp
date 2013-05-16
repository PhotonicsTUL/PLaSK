#include "geometry.h"
#include <plask/geometry/transform.h>
#include <plask/geometry/leaf.h>
#include <plask/geometry/container.h>
#include <plask/geometry/path.h>
#include "../../util/py_set.h"

namespace plask { namespace python {

// Some helpful wrappers
template <int dim> struct MethodsD;

template <> struct MethodsD<2> {

    static bool includes(const GeometryObjectD<2>& self, double c0, double c1) {
        return self.includes(Vec<2,double>(c0, c1));
    }

    static shared_ptr<Material> getMaterial(const GeometryObjectD<2>& self, double c0, double c1) {
        return self.getMaterial(Vec<2,double>(c0, c1));
    }

    static GeometryObject::Subtree getPathsAt(const GeometryObjectD<2>& self, double c0, double c1, bool all) {
        return self.getPathsAt(Vec<2,double>(c0, c1), all);
    }

    static bool objectIncludes1(const GeometryObjectD<2>& self, const GeometryObject& object, const PathHints& path, double c0, double c1) {
        return self.objectIncludes(object, path, Vec<2,double>(c0, c1));
    }

    static bool objectIncludes2(const GeometryObjectD<2>& self, const GeometryObject& object, double c0, double c1) {
        return self.objectIncludes(object, Vec<2,double>(c0, c1));
    }

    static py::list getRolesAt(const GeometryObjectD<2>& self, double c0, double c1) {
        py::list result;
        for (auto role: self.getRolesAt(Vec<2,double>(c0, c1))) result.append(py::object(role));
        return result;
    }

    static bool hasRoleAt(const GeometryObjectD<2>& self, const std::string& role, double c0, double c1) {
        return self.hasRoleAt(role, Vec<2,double>(c0, c1)) != nullptr;
    }

};

template <> struct MethodsD<3> {

    static bool includes(const GeometryObjectD<3>& self, double c0, double c1, double c2) {
        return self.includes(Vec<3,double>(c0, c1, c2));
    }

    static shared_ptr<Material> getMaterial(const GeometryObjectD<3>& self, double c0, double c1, double c2) {
        return self.getMaterial(Vec<3,double>(c0, c1, c2));
    }

    static GeometryObject::Subtree getPathsAt(const GeometryObjectD<3>& self, double c0, double c1, double c2, bool all) {
        return self.getPathsAt(Vec<3,double>(c0, c1, c2), all);
    }

    static bool objectIncludes1(const GeometryObjectD<3>& self, const GeometryObject& object, const PathHints& path, double c0, double c1, double c2) {
        return self.objectIncludes(object, path, Vec<3,double>(c0, c1, c2));
    }

    static bool objectIncludes2(const GeometryObjectD<3>& self, const GeometryObject& object, double c0, double c1, double c2) {
        return self.objectIncludes(object, Vec<3,double>(c0, c1, c2));
    }

    static py::list getRolesAt(const GeometryObjectD<3>& self, double c0, double c1, double c2) {
        py::list result;
        for (auto role: self.getRolesAt(Vec<3,double>(c0, c1, c2))) result.append(py::object(role));
        return result;
    }

    static bool hasRoleAt(const GeometryObjectD<3>& self, const std::string& role, double c0, double c1, double c2) {
        return self.hasRoleAt(role, Vec<3,double>(c0, c1, c2)) != nullptr;
    }

};

template <int dim>
static py::list GeometryObjectD_getRolesAt(const GeometryObjectD<dim>& self, const typename GeometryObjectD<dim>::DVec& point) {
    py::list result;
    for (auto role: self.getRolesAt(point)) result.append(py::object(role));
    return result;
}

template <int dim>
static bool GeometryObjectD_hasRoleAt(const GeometryObjectD<dim>& self, const std::string& role, const typename GeometryObjectD<dim>::DVec& point) {
    return self.hasRoleAt(role, point) != nullptr;
}


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

// template <int dim>
// static py::list GeometryObjectD_getObjectAsTranslations(const shared_ptr<GeometryObjectD<dim>>& self, const shared_ptr<GeometryObjectD<dim>>& object, const PathHints& path) {
//     auto translations = self->extractObject(*object, path);
//     py::list result;
//     for (auto i: translations) result.append(const_pointer_cast<GeometryObjectD<dim>>(i));
//     return result;
// }

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


py::object GeometryObject_getRole(const GeometryObject& self) {
    if (self.roles.size() == 0) return py::object();
    if (self.roles.size() != 1) throw TypeError("Object has more than one role, use .roles instead");
    return py::object(*(self.roles.begin()));
}

void GeometryObject_setRole(GeometryObject& self, const std::string& role) {
    self.clearRoles();
    self.addRole(role);
}

void GeometryObject_setRoles(GeometryObject& self, py::object roles) {
    py::stl_input_iterator<std::string> begin(roles), end;
    self.clearRoles();
    for (auto role = begin; role != end; ++role) self.addRole(*role);
}


template <int dim> struct GeometryObjectD_vector_args;

template<> struct GeometryObjectD_vector_args<2> {
    static inline const py::detail::keywords<2> args() {
        return py::arg("c0"), py::arg("c1");
    }
    template <size_t nargs>
    static inline py::detail::keywords<nargs+2> args(const py::detail::keywords<nargs>& other) {
        return other, py::arg("c0"), py::arg("c1");
    }
};

template<> struct GeometryObjectD_vector_args<3> {
    static inline const py::detail::keywords<3> args() {
        return py::arg("c0"), py::arg("c1"), py::arg("c2");
    }
    template <size_t nargs>
    static inline const py::detail::keywords<nargs+3> args(const py::detail::keywords<nargs>& other) {
        return other, py::arg("c0"), py::arg("c1"), py::arg("c2");
    }
};


/// Initialize class GeometryObjectD for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectD, "GeometryObject", "Base class for "," geometry objects") {

    typedef typename Primitive<dim>::DVec DVec;
    typedef typename Primitive<dim>::Box Box;

    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectD, GeometryObject)

        .def("includes", &GeometryObjectD<dim>::includes, (py::arg("point")),
             "Return True if the geometry object includes a point (in local coordinates)")
        .def("includes", &MethodsD<dim>::includes, GeometryObjectD_vector_args<dim>::args(),
             "Return True if the geometry object includes a point (in local coordinates)")

        /*.def("intersects", &GeometryObjectD<dim>::intersects, (py::arg("area")),
             "Return True if the geometry object has common points (in local coordinates) with an area")*/  //TODO unused now, to uncomment ot remove

        .def("get_material", &GeometryObjectD<dim>::getMaterial, (py::arg("point")),
             "Return material at given point, provided that it is inside the bounding box (in local coordinates) and None otherwise")
        .def("get_material", &MethodsD<dim>::getMaterial, GeometryObjectD_vector_args<dim>::args(),
             "Return material at given point, provided that it is inside the bounding box (in local coordinates) and None otherwise")

        .add_property("bbox", &GeometryObjectD<dim>::getBoundingBox,
                      "Minimal rectangle which includes all points of the geometry object (in local coordinates)")

        .add_property("bbox_size", &GeometryObjectD<dim>::getBoundingBoxSize,
                      "Size of the bounding box")

        .def("get_leafs_positions", (std::vector<DVec>(GeometryObjectD<dim>::*)(const PathHints&)const) &GeometryObjectD<dim>::getLeafsPositions,
             (py::arg("path")=py::object()), "Calculate positions of all leafs (in local coordinates)")

        .def("get_leafs_bboxes", (std::vector<Box>(GeometryObjectD<dim>::*)(const PathHints&)const) &GeometryObjectD<dim>::getLeafsBoundingBoxes,
             (py::arg("path")=py::object()), "Calculate bounding boxes of all leafs (in local coordinates)")

        .def("get_leafs_translations", &GeometryObjectD_getLeafsAsTranslations<dim>, (py::arg("path")=py::object()),
                "Return list of Translation objects holding all leafs")

        .def("get_leafs", &GeometryObject_getLeafs, (py::arg("path")=py::object()),
             "Return list of all leafs in the subtree originating from this object")

        .def("get_object_positions", (std::vector<DVec>(GeometryObjectD<dim>::*)(const GeometryObject&, const PathHints&)const) &GeometryObjectD<dim>::getObjectPositions,
             (py::arg("object"), py::arg("path")=py::object()), "Calculate positions of all all instances of specified object (in local coordinates)")

        .def("get_object_bboxes", (std::vector<Box>(GeometryObjectD<dim>::*)(const GeometryObject&, const PathHints&)const) &GeometryObjectD<dim>::getObjectBoundingBoxes,
             (py::arg("object"), py::arg("path")=py::object()), "Calculate bounding boxes of all instances of specified object (in local coordinates)")

        // .def("getObjectAsTranslations", &GeometryObjectD_getObjectAsTranslations<dim>,
        //         (py::arg("object"), py::arg("path")=py::object()), "Return Translations holding specified object")

        .def("get_paths", (GeometryObject::Subtree(GeometryObjectD<dim>::*)(const DVec&,bool)const) &GeometryObjectD<dim>::getPathsAt, (py::arg("point"), py::arg("all")=false),
             "Return subtree containing paths to all leafs covering specified point")

        .def("get_paths", &MethodsD<dim>::getPathsAt, (GeometryObjectD_vector_args<dim>::args(), py::arg("all")=false),
             "Return subtree containing paths to all leafs covering specified point")

        .def("object_includes", (bool(GeometryObjectD<dim>::*)(const GeometryObject&,const PathHints&,const DVec&)const)&GeometryObjectD<dim>::objectIncludes,
             (py::arg("object"), "path", "point"), "Return true if the specified object includes given point")
        .def("object_includes", (bool(GeometryObjectD<dim>::*)(const GeometryObject&,const DVec&)const)&GeometryObjectD<dim>::objectIncludes,
             (py::arg("object"), "point"), "Return true if the specified object includes given point")
        .def("object_includes", &MethodsD<dim>::objectIncludes1, GeometryObjectD_vector_args<dim>::args((py::arg("object"), "path")),
             "Return true if the specified child includes given point")
        .def("object_includes", &MethodsD<dim>::objectIncludes2, (GeometryObjectD_vector_args<dim>::args(py::arg("object"))),
             "Return true if the specified child includes given point")

        .add_property("role", &GeometryObject_getRole, &GeometryObject_setRole, "Role of the object")
        .add_property("roles", py::make_getter(&GeometryObject::roles), &GeometryObject_setRoles, "Roles of the object")

        .def("get_roles", &GeometryObjectD_getRolesAt<dim>, py::arg("point"), "Return roles of objects at specified point")
        .def("get_roles", &MethodsD<dim>::getRolesAt, GeometryObjectD_vector_args<dim>::args(), "Return roles of objects at specified point")

        .def("has_role", &GeometryObjectD_hasRoleAt<dim>, (py::arg("role"), "point"), "Return true if the specified point has given role")
        .def("has_role", &MethodsD<dim>::hasRoleAt, GeometryObjectD_vector_args<dim>::args(py::arg("role")),
             "Return true if the specified point has given role")
    ;
}

void register_geometry_object()
{
    export_set<std::string>("string_set");
    py::delattr(py::scope(), "string_set");

    py_enum<GeometryObject::Type>("ObjectType")
        .value("LEAF", GeometryObject::TYPE_LEAF)
        .value("TRANSFORM", GeometryObject::TYPE_TRANSFORM)
        .value("SPACE_CHANGER", GeometryObject::TYPE_SPACE_CHANGER)
        .value("CONTAINER", GeometryObject::TYPE_CONTAINER)
    ;

    py::class_<GeometryObject, shared_ptr<GeometryObject>, boost::noncopyable>("GeometryObject",
        "Base class for all geometry objects.", py::no_init)
        .add_property("type", &GeometryObject::getType)
        .def("validate", &GeometryObject::validate, "Check if the object is complete and ready for calculations")
        .def("__repr__", &GeometryObject__repr__)
        .def("__eq__", __is__<GeometryObject>)
    ;

    py::implicitly_convertible<shared_ptr<GeometryObject>, shared_ptr<const GeometryObject>>();
    py::implicitly_convertible<shared_ptr<GeometryObjectD<2>>, shared_ptr<const GeometryObjectD<2>>>();
    py::implicitly_convertible<shared_ptr<GeometryObjectD<3>>, shared_ptr<const GeometryObjectD<3>>>();

    register_vector_of<shared_ptr<GeometryObject>>("GeometryObject");

    init_GeometryObjectD<2>();
    init_GeometryObjectD<3>();

}


}} // namespace plask::python
