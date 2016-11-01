#include "geometry.h"
#include <plask/geometry/transform.h>
#include <plask/geometry/leaf.h>
#include <plask/geometry/container.h>
#include <plask/geometry/path.h>
#include "../python_util/py_set.h"

#if PY_VERSION_HEX >= 0x03000000
#   define NEXT "__next__"
#else
#   define NEXT "next"
#endif

namespace plask { namespace python {

// Some helpful wrappers
template <int dim> struct MethodsD;

template <> struct MethodsD<2> {

    static bool contains(const GeometryObjectD<2>& self, double c0, double c1) {
        return self.contains(Vec<2,double>(c0,c1));
    }

    static shared_ptr<Material> getMaterial(const GeometryObjectD<2>& self, double c0, double c1) {
        return self.getMaterial(Vec<2,double>(c0,c1));
    }

    static GeometryObject::Subtree getPathsAt(const GeometryObjectD<2>& self, double c0, double c1, bool all) {
        return self.getPathsAt(Vec<2,double>(c0,c1), all);
    }

    static bool objectIncludes1(const GeometryObjectD<2>& self, const GeometryObject& object, const PathHints& path, double c0, double c1) {
        return self.objectIncludes(object, path, Vec<2,double>(c0,c1));
    }

    static bool objectIncludes2(const GeometryObjectD<2>& self, const GeometryObject& object, double c0, double c1) {
        return self.objectIncludes(object, Vec<2,double>(c0,c1));
    }

    static py::list getRolesAt(const GeometryObjectD<2>& self, double c0, double c1) {
        py::list result;
        for (auto role: self.getRolesAt(Vec<2,double>(c0,c1))) result.append(py::object(role));
        return result;
    }

    static bool hasRoleAt(const GeometryObjectD<2>& self, const std::string& role, double c0, double c1) {
        return self.hasRoleAt(role, Vec<2,double>(c0,c1)) != nullptr;
    }

};

template <> struct MethodsD<3> {

    static bool contains(const GeometryObjectD<3>& self, double c0, double c1, double c2) {
        return self.contains(Vec<3,double>(c0,c1,c2));
    }

    static shared_ptr<Material> getMaterial(const GeometryObjectD<3>& self, double c0, double c1, double c2) {
        return self.getMaterial(Vec<3,double>(c0,c1,c2));
    }

    static GeometryObject::Subtree getPathsAt(const GeometryObjectD<3>& self, double c0, double c1, double c2, bool all) {
        return self.getPathsAt(Vec<3,double>(c0,c1,c2), all);
    }

    static bool objectIncludes1(const GeometryObjectD<3>& self, const GeometryObject& object, const PathHints& path, double c0, double c1, double c2) {
        return self.objectIncludes(object, path, Vec<3,double>(c0,c1,c2));
    }

    static bool objectIncludes2(const GeometryObjectD<3>& self, const GeometryObject& object, double c0, double c1, double c2) {
        return self.objectIncludes(object, Vec<3,double>(c0,c1,c2));
    }

    static py::list getRolesAt(const GeometryObjectD<3>& self, double c0, double c1, double c2) {
        py::list result;
        for (auto role: self.getRolesAt(Vec<3,double>(c0,c1,c2))) result.append(py::object(role));
        return result;
    }

    static bool hasRoleAt(const GeometryObjectD<3>& self, const std::string& role, double c0, double c1, double c2) {
        return self.hasRoleAt(role, Vec<3,double>(c0,c1,c2)) != nullptr;
    }

};

shared_ptr<GeometryObject> GeometryObject__getitem__(py::object oself, int i) {
    GeometryObject* self = py::extract<GeometryObject*>(oself);
    int n = self->getChildrenCount();
    if (n == 0) {
        if (self->isLeaf())
            throw TypeError("{0} object has no items", std::string(py::extract<std::string>(oself.attr("__class__").attr("__name__"))));
        else
            throw IndexError("{0} object has no items", std::string(py::extract<std::string>(oself.attr("__class__").attr("__name__"))));
    }
    if (i < 0) i = n + i;
    if (i < 0 || i >= n) {
        throw IndexError("{0} index {1} out of range (0 <= index < {2})",
            std::string(py::extract<std::string>(oself.attr("__class__").attr("__name__"))), i, n);
    }
    return self->getChildNo(i);
}

struct GeometryObjectIter {
    shared_ptr<GeometryObject> parent;
    int i;

    GeometryObjectIter(const shared_ptr<GeometryObject>& parent): parent(parent), i(-1) {}
    shared_ptr<GeometryObject> next() {
        ++i; if (i == parent->getChildrenCount()) throw StopIteration("");
        return parent->getChildNo(i);
    }

    static GeometryObjectIter __iter__(const shared_ptr<GeometryObject>& self) {
        return GeometryObjectIter(self);
    }
};

static py::list GeometryObject_getMatching(const shared_ptr<GeometryObject>& self, const py::object& callable) {
    std::vector<shared_ptr<const GeometryObject>> objs = self->getObjects(PredicatePythonCallable(callable));
    py::list result;
    for (auto i: objs) result.append(const_pointer_cast<GeometryObject>(i));
    return result;
}

static py::list GeometryObject_getWithRole(const shared_ptr<GeometryObject>& self, const std::string& role) {
    std::vector<shared_ptr<const GeometryObject>> objs = self->getObjectsWithRole(role);
    py::list result;
    for (auto i: objs) result.append(const_pointer_cast<GeometryObject>(i));
    return result;
}


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
        result.append(plask::make_shared<Translation<dim>>(const_pointer_cast<GeometryObjectD<dim>>(static_pointer_cast<const GeometryObjectD<dim>>(*l)), *t));
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

py::object GeometryObject_deepCopy(const py::object& oself, py::object omemo) {
    py::dict memo = (omemo == py::object())? py::dict() : py::dict(omemo);
    unsigned long id = (unsigned long)(oself.ptr());
    if (memo.has_key(id)) return memo[id];
    shared_ptr<GeometryObject> self = py::extract<shared_ptr<GeometryObject>>(oself);
    py::object result = py::object(self->deepCopy());
    memo[id] = result;
    return result;
}

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, contains,
    u8"Test if the geometry object contains a point.\n\n"
    u8"Args:\n"
    u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
    u8"    c0 (float): Horizontal local coordinate of the tested point.\n"
    u8"    c1 (float): Vertical local coordinate of the tested point.\n"
    u8"Returns:\n"
    u8"    bool: True if the geometry object contains the given point.",

    u8"Test if the geometry object contains a point.\n\n"
    u8"Args:\n"
    u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
    u8"    c0 (float): Longitudinal local coordinate of the tested point.\n"
    u8"    c1 (float): Transverse local coordinate of the tested point.\n"
    u8"    c2 (float): Vertical local coordinate of the tested point.\n"
    u8"Returns:\n"
    u8"    bool: True if the geometry object contains the given point.\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, object_contains,
    u8"Test if the specified geometry object contains a point.\n\n"
    u8"The given geometry object must be located somewhere within the *self*\n"
    u8"geometry tree.\n\n"
    u8"Args:\n"
    u8"    object: Object to test.\n"
    u8"    path: Path specifying a particular object instance.\n"
    u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
    u8"    c0 (float): Horizontal local coordinate of the tested point.\n"
    u8"    c1 (float): Vertical local coordinate of the tested point.\n"
    u8"Returns:\n"
    u8"    bool: True if the specified geometry object contains the given point.\n",

    u8"Test if the specified geometry object contains a point.\n\n"
    u8"The given geometry object must be located somewhere within the *self*\n"
    u8"geometry tree.\n\n"
    u8"Args:\n"
    u8"    object: Object to test.\n"
    u8"    path: Path specifying a particular object instance.\n"
    u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
    u8"    c0 (float): Longitudinal local coordinate of the tested point.\n"
    u8"    c1 (float): Transverse local coordinate of the tested point.\n"
    u8"    c2 (float): Vertical local coordinate of the tested point.\n"
    u8"Returns:\n"
    u8"    bool: True if the specified geometry object contains the given point.\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, get_material,
    "Get material at the given point.\n\n"
    "This method returns a material object with the material at the given point if\n"
    "this point is located within the geometry object *self*. Otherwise the method\n"
    "returns None.\n\n"
    "Args:\n"
    "    point (plask.vector): Vector with local coordinates of the tested point.\n"
    "    c0 (float): Horizontal local coordinate of the tested point.\n"
    "    c1 (float): Vertical local coordinate of the tested point.\n"
    "Returns:\n"
    "    Material at the specified point or None.\n",

    "Get material at the given point.\n\n"
    "This method returns a material object with the material at the given point if\n"
    "this point is located within the geometry object *self*. Otherwise the method\n"
    "returns None.\n\n"
    "Args:\n"
    "    point (plask.vector): Vector with local coordinates of the tested point.\n"
    "    c0 (float): Longitudinal local coordinate of the tested point.\n"
    "    c1 (float): Transverse local coordinate of the tested point.\n"
    "    c2 (float): Vertical local coordinate of the tested point.\n"
    "Returns:\n"
    "    Material at the specified point or None.\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, get_roles,
    "Get roles of objects at specified point.\n\n"
    "This method returns a set of all the roles given to the every object\n"
    "intersecting the specified point.\n\n"
    "Args:\n"
    "    point (plask.vector): Vector with local coordinates of the tested point.\n"
    "    c0 (float): Horizontal local coordinate of the tested point.\n"
    "    c1 (float): Vertical local coordinate of the tested point.\n"
    "Returns:\n"
    "    set: Set of the roles at given point.\n",

    "Get roles of objects at specified point.\n\n"
    "This method returns a set of all the roles given to the every object\n"
    "intersecting the specified point.\n\n"
    "Args:\n"
    "    point (plask.vector): Vector with local coordinates of the tested point.\n"
    "    c0 (float): Longitudinal local coordinate of the tested point.\n"
    "    c1 (float): Transverse local coordinate of the tested point.\n"
    "    c2 (float): Vertical local coordinate of the tested point.\n"
    "Returns:\n"
    "    set: Set of the roles at given point.\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, has_role,
    "Test if the specified point has a given role.\n\n"
    "This method checks if any object intersecting the specified point has the role\n"
    "*role*.\n\n"
    "Args:\n"
    "    point (plask.vector): Vector with local coordinates of the tested point.\n"
    "    c0 (float): Horizontal local coordinate of the tested point.\n"
    "    c1 (float): Vertical local coordinate of the tested point.\n"
    "Returns:\n"
    "    bool: True if the point has the role *role*.\n",

    "Test if the specified point has a given role.\n\n"
    "This method checks if any object intersecting the specified point has the role\n"
    "*role*.\n\n"
    "Args:\n"
    "    point (plask.vector): Vector with local coordinates of the tested point.\n"
    "    c0 (float): Longitudinal local coordinate of the tested point.\n"
    "    c1 (float): Transverse local coordinate of the tested point.\n"
    "    c2 (float): Vertical local coordinate of the tested point.\n"
    "Returns:\n"
    "    bool: True if the point has the role *role*.\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, get_paths,
    "Get subtree containing paths to all leafs covering the specified point.\n\n"
    "Args:\n"
    "    point (plask.vector): Vector with local coordinates of the tested point.\n"
    "    c0 (float): Horizontal local coordinate of the tested point.\n"
    "    c1 (float): Vertical local coordinate of the tested point.\n"
    "    all (bool): If *True* then all the leafs intersecting the point are\n"
    "              considered. Otherwise, only the path to the topmost (i.e. visible)\n"
    "              object is returned.\n"
    "Returns:\n"
    "    Subtree with the path to the specified point.\n\n"
    "See also:\n"
    "    :class:`plask.geometry.Subtree`\n",

    "Get subtree containing paths to all leafs covering the specified point.\n\n"
    "Args:\n"
    "    point (plask.vector): Vector with local coordinates of the tested point.\n"
    "    c0 (float): Longitudinal local coordinate of the tested point.\n"
    "    c1 (float): Transverse local coordinate of the tested point.\n"
    "    c2 (float): Vertical local coordinate of the tested point.\n"
    "    all (bool): If *True* then all the leafs intersecting the point are\n"
    "              considered. Otherwise, only the path to the topmost (i.e. visible)\n"
    "              object is returned.\n"
    "Returns:\n"
    "    Subtree with the path to the specified point.\n\n"
    "See also:\n"
    "    :class:`plask.geometry.Subtree`\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, bbox,
    "Minimal rectangle which contains all points of the geometry object\n"
    "(in local coordinates).\n\n"
    "See also:\n"
    "    :class:`plask.geometry.Box2D`\n",

    "Minimal rectangle which contains all points of the geometry object\n"
    "(in local coordinates).\n\n"
    "See also:\n"
    "    :class:`plask.geometry.Box3D`\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, get_leafs_positions,
    "Calculate positions of all the geometry tree leafs.\n\n"
    "This method computes position of all the geometry tree leafs located under this\n"
    "geometry object. By *leaf* we understand a proper geometry object, in contrast\n"
    "to any container or transformation.\n\n"
    "The position are computed in the local coordinates of this object.\n\n"
    "Args:\n"
    "    path: Path that can be used to select only some leafs.\n"
    "Returns:\n"
    "    sequence: List of vectors containing the position of the leafs.\n\n"
    "All these methods are guaranteed to return their sequences in the same order:\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_positions`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_translations`.\n",

    "Calculate positions of all the geometry tree leafs.\n\n"
    "This method computes position of all the geometry tree leafs located under this\n"
    "geometry object. By *leaf* we understand a proper geometry object, in contrast\n"
    "to any container or transformation.\n\n"
    "The position are computed in the local coordinates of this object.\n\n"
    "Args:\n"
    "    path: Path that can be used to select only some leafs.\n"
    "Returns:\n"
    "    sequence: List of vectors containing the position of the leafs.\n\n"
    "All these methods are guaranteed to return their sequences in the same order:\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_positions`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_translations`.\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, get_leafs_bboxes,
    "Calculate bounding boxes of all the geometry tree leafs.\n\n"
    "This method computes the bounding boxes of all the geometry tree leafs located\n"
    "under this geometry object. By *leaf* we understand a proper geometry object,\n"
    "in contrast to any container or transformation.\n\n"
    "The bounding boxes are computed in the local coordinates of this object.\n\n"
    "Args:\n"
    "    path: Path that can be used to select only some leafs.\n"
    "Returns:\n"
    "    sequence: List of vectors containing the position of the leafs.\n\n"
    "All these methods are guaranteed to return their sequences in the same order:\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_positions`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_translations`.\n",

    "Calculate bounding boxes of all the geometry tree leafs.\n\n"
    "This method computes the bounding boxes of all the geometry tree leafs located\n"
    "under this geometry object. By *leaf* we understand a proper geometry object,\n"
    "in contrast to any container or transformation.\n\n"
    "The bounding boxes are computed in the local coordinates of this object.\n\n"
    "Args:\n"
    "    path: Path that can be used to select only some leafs.\n"
    "Returns:\n"
    "    sequence: List of vectors containing the position of the leafs.\n\n"
    "All these methods are guaranteed to return their sequences in the same order:\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_positions`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_translations`.\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, get_leafs_translations,
    "Get list of :class:`Translation` objects holding all the geometry tree leafs.\n\n"
    "This method computes the :class:`Translation` objects of all the geometry tree\n"
    "leafs located under this geometry object. By *leaf* we understand a proper\n"
    "geometry object, in contrast to any container or transformation.\n\n"
    "The translations are computed in the local coordinates of this object.\n\n"
    "Args:\n"
    "    path: Path that can be used to select only some leafs.\n"
    "Returns:\n"
    "    sequence: List of translations of the leafs.\n\n"
    "All these methods are guaranteed to return their sequences in the same order:\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_positions`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_translations`.\n",

    "Get list of :class:`Translation` objects holding all the geometry tree leafs.\n\n"
    "This method computes the :class:`Translation` objects of all the geometry tree\n"
    "leafs located under this geometry object. By *leaf* we understand a proper\n"
    "geometry object, in contrast to any container or transformation.\n\n"
    "The translations are computed in the local coordinates of this object.\n\n"
    "Args:\n"
    "    path: Path that can be used to select only some leafs.\n"
    "Returns:\n"
    "    sequence: List of translations of the leafs.\n\n"
    "All these methods are guaranteed to return their sequences in the same order:\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_positions`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_translations`.\n"

)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, get_leafs,
    "Get list of the geometry tree leafs.\n\n"
    "This method returns all the geometry tree leafs located under this geometry\n"
    "object. By *leaf* we understand a proper geometry object, in contrast to any\n"
    "container or transformation.\n\n"
    "Args:\n"
    "    path: Path that can be used to select only some leafs.\n"
    "Returns:\n"
    "    sequence: List of translations of the leafs.\n\n"
    "All these methods are guaranteed to return their sequences in the same order:\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_positions`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_leafs_translations`.\n",

    "Get list of the geometry tree leafs.\n\n"
    "This method returns all the geometry tree leafs located under this geometry\n"
    "object. By *leaf* we understand a proper geometry object, in contrast to any\n"
    "container or transformation.\n\n"
    "Args:\n"
    "    path: Path that can be used to select only some leafs.\n"
    "Returns:\n"
    "    sequence: List of translations of the leafs.\n\n"
    "All these methods are guaranteed to return their sequences in the same order:\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_positions`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_leafs_translations`.\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, get_object_positions,
    "Calculate positions of all instances of the specified object.\n\n"
    "The position are computed in the local coordinates of *self*.\n\n"
    "Args:\n"
    "    object: Object to test.\n"
    "    path: Path specifying a particular object instance.\n"
    "Returns:\n"
    "    sequence: List of vectors containing the position of the instances of\n"
    "    the object.\n\n"
    "All these methods are guaranteed to return their sequences in the same order,\n"
    "provided they are called with the same arguments:\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_object_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_object_positions`\n",

    "Calculate positions of all instances of the specified object.\n\n"
    "The position are computed in the local coordinates of *self*.\n\n"
    "Args:\n"
    "    object: Object to test.\n"
    "    path: Path specifying a particular object instance.\n"
    "Returns:\n"
    "    sequence: List of vectors containing the position of the instances of\n"
    "    the object.\n\n"
    "All these methods are guaranteed to return their sequences in the same order,\n"
    "provided they are called with the same arguments:\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_object_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_object_positions`\n"
)

GEOMETRY_ELEMENT_23D_DOC(GeometryObjectD, get_object_bboxes,
    "Calculate bounding boxes of all instances of specified object.\n\n"
    "The bounding boxes are computed in the local coordinates of *self*.\n\n"
    "Args:\n"
    "    object: Object to test.\n"
    "    path: Path specifying a particular object instance.\n"
    "Returns:\n"
    "    sequence: List of bounding boxes of the instances of the object.\n\n"
    "All these methods are guaranteed to return their sequences in the same order,\n"
    "provided they are called with the same arguments:\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_object_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject2D.get_object_positions`\n",

    "Calculate bounding boxes of all instances of specified object.\n\n"
    "The bounding boxes are computed in the local coordinates of *self*.\n\n"
    "Args:\n"
    "    object: Object to test.\n"
    "    path: Path specifying a particular object instance.\n"
    "Returns:\n"
    "    sequence: List of bounding boxes of the instances of the object.\n\n"
    "All these methods are guaranteed to return their sequences in the same order,\n"
    "provided they are called with the same arguments:\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_object_bboxes`,\n"
    ":meth:`~plask.geometry.GeometryObject3D.get_object_positions`\n"
)

/// Initialize class GeometryObjectD for Python
DECLARE_GEOMETRY_ELEMENT_23D(GeometryObjectD, "GeometryObject", "Base class for "," geometry objects.") {

    typedef typename Primitive<dim>::DVec DVec;
    typedef typename Primitive<dim>::Box Box;

    ABSTRACT_GEOMETRY_ELEMENT_23D(GeometryObjectD, GeometryObject)

        .def("__contains__", &GeometryObjectD<dim>::contains, (py::arg("point")),
             "Return True if the geometry object contains a point (in local coordinates).")
        .def("contains", &GeometryObjectD<dim>::contains, (py::arg("point")), USE_23D_DOC(GeometryObjectD, contains))
        .def("contains", &MethodsD<dim>::contains, GeometryObjectD_vector_args<dim>::args())

        /*.def("intersects", &GeometryObjectD<dim>::intersects, (py::arg("area")),
             "Return True if the geometry object has common points (in local coordinates) with an area")*/  //TODO unused now, to uncomment ot remove

        .def("get_material", &GeometryObjectD<dim>::getMaterial, (py::arg("point")), USE_23D_DOC(GeometryObjectD, get_material))
        .def("get_material", &MethodsD<dim>::getMaterial, GeometryObjectD_vector_args<dim>::args())

        .add_property("bbox", &GeometryObjectD<dim>::getBoundingBox, USE_23D_DOC(GeometryObjectD, bbox))

        // // replaced by bbox.size
        // .add_property("bbox_size", &GeometryObjectD<dim>::getBoundingBoxSize,
        //                 "Dimensions of the bounding box.\n\n"
        //                 )

        .def("get_leafs_positions", (std::vector<DVec>(GeometryObjectD<dim>::*)(const PathHints&)const) &GeometryObjectD<dim>::getLeafsPositions,
             (py::arg("path")=py::object()), USE_23D_DOC(GeometryObjectD, get_leafs_positions))

        .def("get_leafs_bboxes", (std::vector<Box>(GeometryObjectD<dim>::*)(const PathHints&)const) &GeometryObjectD<dim>::getLeafsBoundingBoxes,
             (py::arg("path")=py::object()), USE_23D_DOC(GeometryObjectD, get_leafs_bboxes))

        .def("get_leafs_translations", &GeometryObjectD_getLeafsAsTranslations<dim>, (py::arg("path")=py::object()),
             USE_23D_DOC(GeometryObjectD, get_leafs_translations))

        .def("get_leafs", &GeometryObject_getLeafs, (py::arg("path")=py::object()), USE_23D_DOC(GeometryObjectD, get_leafs))

        .def("get_object_positions", (std::vector<DVec>(GeometryObjectD<dim>::*)(const GeometryObject&, const PathHints&)const) &GeometryObjectD<dim>::getObjectPositions,
             (py::arg("object"), py::arg("path")=py::object()), USE_23D_DOC(GeometryObjectD, get_object_positions))

        .def("get_object_bboxes", (std::vector<Box>(GeometryObjectD<dim>::*)(const GeometryObject&, const PathHints&)const) &GeometryObjectD<dim>::getObjectBoundingBoxes,
             (py::arg("object"), py::arg("path")=py::object()), USE_23D_DOC(GeometryObjectD, get_object_bboxes))

        // .def("getObjectAsTranslations", &GeometryObjectD_getObjectAsTranslations<dim>,
        //         (py::arg("object"), py::arg("path")=py::object()), "Return Translations holding specified object")

        .def("get_paths", (GeometryObject::Subtree(GeometryObjectD<dim>::*)(const DVec&,bool)const) &GeometryObjectD<dim>::getPathsAt,
             (py::arg("point"), py::arg("all")=false), USE_23D_DOC(GeometryObjectD, get_paths))

        .def("get_paths", &MethodsD<dim>::getPathsAt, (GeometryObjectD_vector_args<dim>::args(), py::arg("all")=false))

        .def("object_contains", (bool(GeometryObjectD<dim>::*)(const GeometryObject&,const PathHints&,const DVec&)const)&GeometryObjectD<dim>::objectIncludes,
             (py::arg("object"), "path", "point"), USE_23D_DOC(GeometryObjectD, object_contains))
        .def("object_contains", (bool(GeometryObjectD<dim>::*)(const GeometryObject&,const DVec&)const)&GeometryObjectD<dim>::objectIncludes, (py::arg("object"), "point"))
        .def("object_contains", &MethodsD<dim>::objectIncludes1, GeometryObjectD_vector_args<dim>::args((py::arg("object"), "path")))
        .def("object_contains", &MethodsD<dim>::objectIncludes2, (GeometryObjectD_vector_args<dim>::args(py::arg("object"))))

        .add_property("role", &GeometryObject_getRole, &GeometryObject_setRole, "Role of the object. Valid only if the object has only one role.")
        .add_property("roles", py::make_getter(&GeometryObject::roles), &GeometryObject_setRoles, "List of all the roles of the object.")

        .def("get_roles", &GeometryObjectD_getRolesAt<dim>, py::arg("point"), USE_23D_DOC(GeometryObjectD, get_roles))
        .def("get_roles", &MethodsD<dim>::getRolesAt, GeometryObjectD_vector_args<dim>::args())

        .def("has_role", &GeometryObjectD_hasRoleAt<dim>, (py::arg("role"), "point"), USE_23D_DOC(GeometryObjectD, has_role))
        .def("has_role", &MethodsD<dim>::hasRoleAt, GeometryObjectD_vector_args<dim>::args(py::arg("role")))
    ;
}

struct GeometryObjectSteps {

    shared_ptr<GeometryObject> obj;

    GeometryObjectSteps(const shared_ptr<GeometryObject>& obj): obj(obj) {}

    double get_ply() const { return obj->min_ply; }
    void set_ply(double val) { obj->setMinPly(abs(val)); }

    unsigned get_max_points() const { return obj->max_points; }
    void set_max_points(unsigned val) { obj->setMaxPoints(val); }

    std::string str() { return format("<dist={0}, num={1}>", obj->min_ply, obj->max_points);  }

    static GeometryObjectSteps get(const shared_ptr<GeometryObject>& obj) { return GeometryObjectSteps(obj); }

    static void set(const shared_ptr<GeometryObject>& obj, unsigned num) { obj->setMaxPoints(num); }

};

void register_geometry_object()
{
    export_set<std::string>("string_set");
    py::delattr(py::scope(), "string_set");

    // py_enum<GeometryObject::Type>()
    //     .value("LEAF", GeometryObject::TYPE_LEAF)
    //     .value("TRANSFORM", GeometryObject::TYPE_TRANSFORM)
    //     .value("SPACE_CHANGER", GeometryObject::TYPE_SPACE_CHANGER)
    //     .value("CONTAINER", GeometryObject::TYPE_CONTAINER)
    // ;

    py::class_<GeometryObject, shared_ptr<GeometryObject>, boost::noncopyable> geometry_object("GeometryObject",
        "Base class for all geometry objects.", py::no_init); geometry_object
        // .add_property("type", &GeometryObject::getType)
        .add_property("steps", &GeometryObjectSteps::get, &GeometryObjectSteps::set,
                      "Step info for mesh generation for non-uniform objects.\n\n"
                      "This parameter is considered only for the non-uniform leafs in the geometry\n"
                      "tree. It has two attributes that can be changed:\n\n"
                      ".. autosummary::\n"
                      "   ~plask.geometry.GeometryObject._Steps.num\n"
                      "   ~plask.geometry.GeometryObject._Steps.dist\n\n"
                      "The exact meaning of these attributes depend on the mesh generator, however in\n"
                      "general they indicate how densely should the non-uniform object be subdivided.\n\n"
                      "It is possible to assign simply an integer number to this parameter, in which\n"
                      "case it changes its ``num`` attribute.\n"
                     )
        .add_property("dims", &GeometryObject::getDimensionsCount, "Number of object's dimensions (int, 2 or 3).")
        .def("validate", &GeometryObject::validate,
             "Check if the object is complete and ready for calculations.\n\n"
             "This method is specific for a particular object. It raises an exception if\n"
             "the object definition is somehow incomplete.\n"
            )
        .def("get_matching_objects", &GeometryObject_getMatching, py::arg("cond"),
             "Get list of the geometry tree objects matching condition.\n\n"
             "This method returns all the objects in the geometry tree that match the specified\n"
             "condition.\n\n"
             "Args:\n"
             "    cond: Python callable that accepts a geometry object and returns Boolean\n"
             "          indicating whether the object should be returned by this method or not.\n"
             "Returns:\n"
             "    sequence: List of objects matching your condition.\n\n"
            )
        .def("get_role_objects", &GeometryObject_getWithRole, py::arg("role"),
             "Get list of the geometry tree objects that have the specified role.\n\n"
             "This method returns all the objects in the geometry tree that have the specified\n"
             "role.\n\n"
             "Warning:\n\n"
             "    This method will return the very object with the role specified and not its\n"
             "    items, which is against the normal behavior of the roles.\n\n"
             "Args:\n"
             "    role (str): Role to search objects with.\n"
             "Returns:\n"
             "    sequence: List of objects matching your condition.\n\n"
            )
        .def("__getitem__", &GeometryObject__getitem__)
        .def("__iter__", GeometryObjectIter::__iter__)
        .def("__repr__", &GeometryObject__repr__)
        .def("__eq__", __is__<GeometryObject>)
        .def("__hash__", __hash__<GeometryObject>)
        .def("__copy__", &GeometryObject::shallowCopy)
        .def("__deepcopy__", &GeometryObject_deepCopy, (py::arg("x"), py::arg("memo")=py::object()))
    ;

    {py::scope scope = geometry_object;
    py::class_<GeometryObjectSteps>("_Steps", py::no_init)
        .add_property("dist", &GeometryObjectSteps::get_ply, &GeometryObjectSteps::set_ply,
                      "Minimum step size.")
        .add_property("num", &GeometryObjectSteps::get_max_points, &GeometryObjectSteps::set_max_points,
                      "Maximum number of the mesh steps in each direction the object is divided into.")
        .def("__str__", &GeometryObjectSteps::str)
        .def("__repr__", &GeometryObjectSteps::str)
    ;
    py::class_<GeometryObjectIter>("_Iterator", "Items iterator.", py::no_init)
        .def(NEXT, &GeometryObjectIter::next)
        .def("__iter__", pass_through)
    ;}

    py::implicitly_convertible<shared_ptr<GeometryObject>, shared_ptr<const GeometryObject>>();
    py::implicitly_convertible<shared_ptr<GeometryObjectD<2>>, shared_ptr<const GeometryObjectD<2>>>();
    py::implicitly_convertible<shared_ptr<GeometryObjectD<3>>, shared_ptr<const GeometryObjectD<3>>>();

    register_vector_of<shared_ptr<GeometryObject>>("GeometryObject");

    init_GeometryObjectD<2>();
    init_GeometryObjectD<3>();
}


}} // namespace plask::python
