#include "../python_globals.h"
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include "../python_util/raw_constructor.h"

#include <plask/geometry/space.h>
#include <plask/geometry/path.h>
#include <plask/mesh/mesh.h>

// see: http://stackoverflow.com/questions/19987448/is-there-an-easy-way-to-write-utf-8-octets-in-visual-studio
#ifndef _MSC_VER
    #pragma execution_character_set("utf-8")
#endif

namespace plask { namespace python {

extern AxisNames current_axes;

std::string Geometry_getAxes(const Geometry& self) {
    return self.axisNames.str();
}

void Geometry_setAxes(Geometry& self, const std::string& axis) {
    self.axisNames = AxisNames::axisNamesRegister.get(axis);
}

template <typename S> struct Space_getMaterial {
    static inline shared_ptr<Material> call(const S& self, double c0, double c1) {
        return self.getMaterial(Vec<2,double>(c0, c1));
    }
};
template <> struct Space_getMaterial<Geometry3D> {
    static inline shared_ptr<Material> call(const Geometry3D& self, double c0, double c1, double c2) {
        return self.getMaterial(Vec<3,double>(c0, c1, c2));
    }
};

template <typename SpaceType>
static py::list Space_getMatching(const shared_ptr<SpaceType>& self, const py::object& callable) {
    std::vector<shared_ptr<const GeometryObject>> objs = self->getChild()->getObjects(PredicatePythonCallable(callable));
    py::list result;
    for (auto i: objs) result.append(const_pointer_cast<GeometryObject>(i));
    return result;
}

template <typename SpaceType>
static py::list Space_getWithRole(const shared_ptr<SpaceType>& self, const std::string& role) {
    std::vector<shared_ptr<const GeometryObject>> objs = self->getObjectsWithRole(role);
    py::list result;
    for (auto i: objs) result.append(const_pointer_cast<GeometryObject>(i));
    return result;
}

template <typename S> struct Space_getPathsTo {
    static inline GeometryObject::Subtree call(const S& self, double c0, double c1, bool all) {
        return self.getPathsAt(Vec<2,double>(c0, c1), all);
    }
};
template <> struct Space_getPathsTo<Geometry3D> {
    static inline GeometryObject::Subtree call(const Geometry3D& self, double c0, double c1, double c2, bool all) {
        return self.getPathsAt(Vec<3,double>(c0, c1, c2), all);
    }
};

template <typename S>
static py::list Space_leafsAsTranslations(const S& self, const PathHints& path) {
    py::list result;
    auto leafs = self.getLeafs(&path);
    auto translations = self.getLeafsPositions(&path);
    auto l = leafs.begin();
    auto t = translations.begin();
    for (; l != leafs.end(); ++l, ++t) {
        result.append(plask::make_shared<Translation<S::DIM>>(const_pointer_cast<GeometryObjectD<S::DIM>>(static_pointer_cast<const GeometryObjectD<S::DIM>>(*l)), *t));
    }
    return result;
}

template <typename S>
static std::vector<shared_ptr<GeometryObject>> Space_getLeafs(S& self, const PathHints& path) {
    std::vector<shared_ptr<const GeometryObject>> leafs = self.getLeafs(&path);
    std::vector<shared_ptr<GeometryObject>> result;
    result.reserve(leafs.size());
    for (auto i: leafs) result.push_back(const_pointer_cast<GeometryObject>(i));
    return result;
}

template <typename S>
static bool objectIncludes_2D(const S& self, const GeometryObject& object, const PathHints& path, double c0, double c1) {
    return self.objectIncludes(object, path, Vec<2,double>(c0, c1));
}
template <typename S>
static bool objectIncludes0_2D(const S& self, const GeometryObject& object, double c0, double c1) {
    return self.objectIncludes(object, Vec<2,double>(c0, c1));
}

static bool objectIncludes_3D(const Geometry3D& self, const GeometryObject& object, const PathHints& path, double c0, double c1, double c2) {
    return self.objectIncludes(object, path, Vec<3,double>(c0, c1, c2));
}
static bool objectIncludes0_3D(const Geometry3D& self, const GeometryObject& object, double c0, double c1, double c2) {
    return self.objectIncludes(object, Vec<3,double>(c0, c1, c2));
}

template <int dim>
PyObject* GeometryObjectIncludesPoints(const shared_ptr<GeometryObjectD<dim>>& self, const GeometryObject& obj, const PathHints* pth, const MeshD<dim>& mesh);

template <typename S>
PyObject* SpaceObjectIncludesPoints(const S& self, const GeometryObject& obj, const PathHints* pth, const MeshD<S::DIM>& mesh) {
    return GeometryObjectIncludesPoints(self.getChild(), obj, pth, mesh);
}

template <typename S>
PyObject* SpaceObjectIncludesPoints0(const S& self, const GeometryObject& obj, const MeshD<S::DIM>& mesh) {
    return GeometryObjectIncludesPoints(self.getChild(), obj, nullptr, mesh);
}



static void _Space_setEdges(Geometry& self, py::dict edges, std::set<std::string>& parsed, const std::string& err_msg) {
   self.setEdges(
        [&](const std::string& s)->boost::optional<std::string> {
            std::string str = s;
            std::replace(str.begin(), str.end(), '-', '_');
            parsed.insert(str);
            return edges.has_key(str) ?
                boost::optional<std::string>( (edges[str]==py::object()) ? std::string("null") : py::extract<std::string>(edges[str]) ) :
                boost::optional<std::string>();
        },
    current_axes);

    // Test if we have any spurious edges
    py::stl_input_iterator<std::string> begin(edges), end;
    for (auto item = begin; item != end; item++)
        if (parsed.find(*item) == parsed.end())
            throw ValueError(err_msg, *item);
}


static shared_ptr<Geometry2DCartesian> Geometry2DCartesian__init__(py::tuple args, py::dict kwargs) {
    boost::python::ssize_t na = py::len(args);

    shared_ptr <Geometry2DCartesian> space;

    if (na == 3) {
        if (kwargs.has_key("geometry")) throw TypeError("got multiple values for keyword argument 'geometry'");
        if (kwargs.has_key("length")) throw TypeError("got multiple values for keyword argument 'length'");
        shared_ptr<GeometryObjectD<2>> object = py::extract<shared_ptr<GeometryObjectD<2>>>(args[1]);
        double length = py::extract<double>(args[2]);
        space = plask::make_shared<Geometry2DCartesian>(object, length);
    } else if (na == 2) {
        if (kwargs.has_key("geometry")) throw TypeError("got multiple values for keyword argument 'geometry'");
        try {
            shared_ptr<Extrusion> extrusion = py::extract<shared_ptr<Extrusion>>(args[1]);
            if (kwargs.has_key("length")) throw TypeError("keyword argument 'length' not allowed if 'geometry' is of type Extrusion");
            space = plask::make_shared<Geometry2DCartesian>(extrusion);
        } catch (py::error_already_set) {
            PyErr_Clear();
            shared_ptr<GeometryObjectD<2>> object;
            try {
                object = py::extract<shared_ptr<GeometryObjectD<2>>>(args[1]);
            } catch (py::error_already_set) {
                PyErr_Clear();
                throw TypeError("'geometry' argument type must be either Extrusion or GeometryObject2D");
            }
            double length = kwargs.has_key("length")? py::extract<double>(kwargs["length"]) : INFINITY;
            space = plask::make_shared<Geometry2DCartesian>(object, length);
        }
    } else if (na == 1 && kwargs.has_key("geometry")) {
        try {
            shared_ptr<Extrusion> extrusion = py::extract<shared_ptr<Extrusion>>(kwargs["geometry"]);
            if (kwargs.has_key("length")) throw TypeError("keyword argument 'length' not allowed if 'geometry' is of type Extrusion");
            space = plask::make_shared<Geometry2DCartesian>(extrusion);
        } catch (py::error_already_set) {
            PyErr_Clear();
            shared_ptr<GeometryObjectD<2>> object;
            try {
                object = py::extract<shared_ptr<GeometryObjectD<2>>>(kwargs["geometry"]);
            } catch (py::error_already_set) {
                PyErr_Clear();
                throw TypeError("'geometry' argument type must be either Extrusion or GeometryObject2D");
            }
            double length = kwargs.has_key("length")? py::extract<double>(kwargs["length"]) : INFINITY;
            space = plask::make_shared<Geometry2DCartesian>(object, length);
        }
    } else {
        throw TypeError("__init__() takes 2 or 3 non-keyword arguments ({0}) given", na);
    }

    std::set<std::string> parsed_kwargs;
    parsed_kwargs.insert("geometry");
    parsed_kwargs.insert("length");

    _Space_setEdges(*space, kwargs, parsed_kwargs, "__init__() got an unexpected keyword argument '{}'");

    space->axisNames = current_axes;

    return space;
}

static shared_ptr<Geometry2DCylindrical> Geometry2DCylindrical__init__(py::tuple args, py::dict kwargs) {
	boost::python::ssize_t na = py::len(args);

    shared_ptr<Geometry2DCylindrical> space;
    py::object geometry;

    if (na == 2) geometry = args[1];
    else if (na == 1 && kwargs.has_key("geometry")) geometry = kwargs["geometry"];
    else throw TypeError("__init__() takes 1 or 2 non-keyword arguments ({0} given)", na);

    try {
        shared_ptr<Revolution> revolution = py::extract<shared_ptr<Revolution>>(geometry);
        space = plask::make_shared<Geometry2DCylindrical>(revolution);
    } catch (py::error_already_set) {
        PyErr_Clear();
        shared_ptr<GeometryObjectD<2>> object;
        try {
            object = py::extract<shared_ptr<GeometryObjectD<2>>>(geometry);
        } catch (py::error_already_set) {
            PyErr_Clear();
            throw TypeError("'geometry' argument type must be either Extrusion or GeometryObject2D");
        }
        space = plask::make_shared<Geometry2DCylindrical>(object);
    }

    std::set<std::string> parsed_kwargs;
    parsed_kwargs.insert("geometry");

    _Space_setEdges(*space, kwargs, parsed_kwargs, "__init__() got an unexpected keyword argument '{}'");

    space->axisNames = current_axes;

    return space;
}

static shared_ptr<Geometry3D> Geometry3D__init__(py::tuple args, py::dict kwargs) {
	boost::python::ssize_t na = py::len(args);

    shared_ptr <Geometry3D> space;

    if (na == 2) {
        if (kwargs.has_key("geometry")) throw TypeError("got multiple values for keyword argument 'geometry'");
        shared_ptr<GeometryObjectD<3>> object = py::extract<shared_ptr<GeometryObjectD<3>>>(args[1]);
        space = plask::make_shared<Geometry3D>(object);
    } else if (na == 1 && kwargs.has_key("geometry")) {
        shared_ptr<GeometryObjectD<3>> object = py::extract<shared_ptr<GeometryObjectD<3>>>(kwargs["geometry"]);
        space = plask::make_shared<Geometry3D>(object);
    } else {
        throw TypeError("__init__() exactly 2 non-keyword arguments ({0}) given", na);
    }

    std::set<std::string> parsed_kwargs;
    parsed_kwargs.insert("geometry");

    _Space_setEdges(*space, kwargs, parsed_kwargs, "__init__() got an unexpected keyword argument '{}'");

    space->axisNames = current_axes;

    return space;
}

template <typename S>
static typename Primitive<S::DIM>::Box Space_childBoundingBox(const S& self) {
    return self.getChildBoundingBox();
}

static void Space_setEdges(Geometry& self, py::dict edges) {
    std::set<std::string> parsed;
    _Space_setEdges(self, edges, parsed, "unexpected edge name '{}'");
}

struct EdgesProxy : public std::map<std::string, py::object> {
    void __setitem__(const std::string&, const py::object&) {
        throw TypeError("Edges do not support item assignment");
    }
    std::string __repr__() const {
        std::string result;
        std::string sep = "{'";
        for(auto i: *this) {
            result += sep; result += i.first; result += "': ";
            if (i.second != py::object()) result += "'";
            result += py::extract<std::string>(py::str(i.second));
            if (i.second != py::object()) result += "'";
            sep = ", '";
        }
        result += "}";
        return result;
    }
    py::list keys() const {
        py::list result;
        for(auto i: *this) {
            result.append(i.first);
        }
        return result;
    }
    py::list items() const {
        py::list result;
        for(auto i: *this) {
            result.append(py::make_tuple(i.first, i.second));
        }
        return result;
    }
};


inline static py::object _border(const Geometry& self, Geometry::Direction direction, bool higher) {
    auto str = self.getEdge(direction, higher).str();
    return (str=="null") ? py::object() : py::object(str);
}

static EdgesProxy Geometry2DCartesian_getEdges(const Geometry2DCartesian& self) {
    EdgesProxy edges;
    edges["left"] = _border(self, Geometry::DIRECTION_TRAN, false);
    edges["right"] = _border(self, Geometry::DIRECTION_TRAN, true);
    edges["top"] = _border(self, Geometry::DIRECTION_VERT, true);
    edges["bottom"] = _border(self, Geometry::DIRECTION_VERT, false);
    return edges;
}

static EdgesProxy Geometry2DCylindrical_getEdges(const Geometry2DCylindrical& self) {
    EdgesProxy edges;
    edges["inner"] = _border(self, Geometry::DIRECTION_TRAN, false);
    edges["outer"] = _border(self, Geometry::DIRECTION_TRAN, true);
    edges["top"] = _border(self, Geometry::DIRECTION_VERT, true);
    edges["bottom"] = _border(self, Geometry::DIRECTION_VERT, false);
    return edges;
}

static EdgesProxy Geometry3D_getEdges(const Geometry3D& self) {
    EdgesProxy edges;
    edges["back"] = _border(self, Geometry::DIRECTION_LONG, false);
    edges["front"] = _border(self, Geometry::DIRECTION_LONG, true);
    edges["left"] = _border(self, Geometry::DIRECTION_TRAN, false);
    edges["right"] = _border(self, Geometry::DIRECTION_TRAN, true);
    edges["top"] = _border(self, Geometry::DIRECTION_VERT, true);
    edges["bottom"] = _border(self, Geometry::DIRECTION_VERT, false);
    return edges;
}

template <typename GeometryT>
static py::list Geometry_getRolesAt(const GeometryT& self, const typename GeometryObjectD<GeometryT::DIM>::DVec& point) {
    py::list result;
    for (auto role: self.getRolesAt(point)) result.append(py::object(role));
    return result;
}

template <typename GeometryT>
static py::list Geometry2D_getRolesAt(const GeometryT& self, double c0, double c1) {
    py::list result;
    for (auto role: self.getRolesAt(vec(c0,c1))) result.append(py::object(role));
    return result;
}

static py::list Geometry3D_getRolesAt(const Geometry3D& self, double c0, double c1, double c2) {
    py::list result;
    for (auto role: self.getRolesAt(vec(c0,c1,c2))) result.append(py::object(role));
    return result;
}

template <typename GeometryT>
static bool Geometry_hasRoleAt(const GeometryT& self, const std::string& role, const typename GeometryObjectD<GeometryT::DIM>::DVec& point) {
    return self.hasRoleAt(role, point) != nullptr;
}

template <typename GeometryT>
static bool Geometry2D_hasRoleAt(const GeometryT& self, const std::string& role, double c0, double c1) {
    return self.hasRoleAt(role, vec(c0,c1)) != nullptr;
}

static bool Geometry3D_hasRoleAt(const Geometry3D& self, const std::string& role, double c0, double c1, double c2) {
    return self.hasRoleAt(role, vec(c0,c1,c2)) != nullptr;
}

// template <typename S>
// static shared_ptr<S> Space_getSubspace(py::tuple args, py::dict kwargs) {
//     const S* self = py::extract<S*>(args[0]);
//
//     py::object arg;
//     py::ssize_t n = py::len(args);
//     if (n >= 2) {
//         if (kwargs.has_key("object")) throw TypeError("got multiple values for keyword argument 'object'");
//         arg = args[1];
//     } else
//         arg = kwargs["object"];
//     shared_ptr<GeometryObjectD<S::DIMS>> object = py::extract<shared_ptr<GeometryObjectD<S::DIMS>>>(arg);
//
//     PathHints* path = nullptr;
//     if (n >= 3) {
//         if (kwargs.has_key("path")) throw TypeError("got multiple values for keyword argument 'path'");
//         path = py::extract<PathHints*>(args[2]);
//     } else if (kwargs.has_key("path"))
//         path = py::extract<PathHints*>(kwargs["path"]);
//
//     if (n >= 4) throw TypeError("getSubspace() takes 2 or 3 non-keyword arguments ({0}) given", n);
//
//     S* space = self->getSubspace(object, path, false);
//
//     std::set<std::string> parsed;
//     parsed.insert("object");
//     parsed.insert("path");
//     _Space_setEdges(*space, kwargs, parsed, "unexpected edge name '{}'");
//
//     return shared_ptr<S>(space);
// }

void register_calculation_spaces() {

    py::class_<Geometry, shared_ptr<Geometry>, py::bases<GeometryObject>, boost::noncopyable>("Geometry",
        "Base class for all geometries", py::no_init)
        .def("__eq__", __is__<Geometry>)
        .add_property<>("axes", Geometry_getAxes, Geometry_getAxes, "Names of axes for this geometry.")
    ;

    py::class_<EdgesProxy>("_EdgesProxy")
        .def(py::map_indexing_suite<EdgesProxy, true>())
        .def("__setitem__", &EdgesProxy::__setitem__)
        .def("__repr__", &EdgesProxy::__repr__)
        .def("keys", &EdgesProxy::keys)
        .def("items", &EdgesProxy::items)
    ;

    py::class_<Geometry2DCartesian, shared_ptr<Geometry2DCartesian>, py::bases<Geometry>>("Cartesian2D",
        u8"Geometry in 2D Cartesian space.\n\n"
        u8"Cartesian2D(root, length=infty, **edges)\n\n"
        u8"Create a space around a two-dimensional geometry object with a given length.\n\n"
        u8"Args:\n"
        u8"    root (GeometryObject2D Extrusion): Root object of the geometry.\n"
        u8"        If this parameters is an extrusion, the `length` should be skipped,\n"
        u8"        as it is read directly from extrusion.\n"
        u8"    length (float): Length of the geometry.\n"
        u8"        This information is required by some solvers. Furthermore it is\n"
        u8"        necessary if you want to use :mod:`plask.filters` to translate the\n"
        u8"        data between this geometry and the :class:`Cartesian3D` geometry.\n\n"
        u8"    edges (dict): Optional edges specification.\n"
        u8"        Edges are given as additional constructor keyword arguments. Available\n"
        u8"        keys are *left*, *right*, *top*, and *bottom* and their values must be\n"
        u8"        strings specifying the edge (either a material name or *mirror*,\n"
        u8"        *periodic*, or *extend*).\n\n"
        u8"Example:\n"
        u8"    >>> block = geometry.Block2D(4, 2, 'GaAs')\n"
        u8"    >>> geometry.Cartesian2D(block, length=10, left='mirror', bottom='AlAs')\n"
        u8"    <plask.geometry.Cartesian2D object at (0x3dd6c70)>",
        py::no_init)
        .def("__init__", raw_constructor(Geometry2DCartesian__init__, 1))
        .add_property("item", &Geometry2DCartesian::getChild,
                      u8":class:`~plask.geometry.GeometryObject2D` at the root of the geometry tree."
                     )
        .add_property("extrusion", &Geometry2DCartesian::getExtrusion,
                      u8":class:`~plask.geometry.Extrusion` object at the very root of the tree."
                     )
        .add_property("bbox", &Space_childBoundingBox<Geometry2DCartesian>,
                      u8"Minimal rectangle which contains all points of the geometry object.\n\n"
                      u8"See also:\n"
                      u8"    :class:`plask.geometry.Box2D`\n"
                     )
        .def_readwrite("default_material", &Geometry2DCartesian::defaultMaterial,
                       u8"Material of the *empty* regions of the geometry.\n\n"
                       u8"This material is returned by :meth:`~plask.geometry.Cartesian2D.get_material`\n"
                       u8"for the points that do not belong to any object in the geometry tree.\n"
                      )
        .add_property("front_material", &Geometry2DCartesian::getFrontMaterial, &Geometry2DCartesian::setFrontMaterial,
                      u8"Material at the positive side of the axis along the extrusion.")
        .add_property("back_material", &Geometry2DCartesian::getBackMaterial, &Geometry2DCartesian::setBackMaterial,
                      u8"Material at the negative side of the axis along the extrusion.")
        .add_property("edges", &Geometry2DCartesian_getEdges, &Space_setEdges,
                      u8"Dictionary specifying the geometry edges.")
        .def("get_material", &Geometry2DCartesian::getMaterial, (py::arg("point")))
        .def("get_material", &Space_getMaterial<Geometry2DCartesian>::call, (py::arg("c0"), "c1"),
             u8"Get material at the given point.\n\n"
             u8"This method returns a material object with the material at the given point if\n"
             u8"this point is located within the geometry object *self*. Otherwise the method\n"
             u8"returns :attr:`~plask.geometry.Cartesian2D.default_material`.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Horizontal coordinate of the tested point.\n"
             u8"    c1 (float): Vertical coordinate of the tested point.\n"
             u8"Returns:\n"
             u8"    Material at the specified point."
            )
        .def("get_matching_objects", &Space_getMatching<Geometry2DCartesian>, py::arg("cond"),
             u8"Get list of the geometry tree objects matching condition.\n\n"
             u8"This method returns all the objects in the geometry tree that match the specified\n"
             u8"condition.\n\n"
             u8"Args:\n"
             u8"    cond: Python callable that accepts a geometry object and returns Boolean\n"
             u8"          indicating whether the object should be returned by this method or not.\n"
             u8"Returns:\n"
             u8"    sequence: List of objects matching your condition.\n\n"
            )
        .def("get_role_objects", &Space_getWithRole<Geometry2DCartesian>, py::arg("role"),
             u8"Get list of the geometry tree objects that have the specified role.\n\n"
             u8"This method returns all the objects in the geometry tree that have the specified\n"
             u8"role.\n\n"
             u8".. rubric:: Warning!\n\n"
             u8"This method will return the very object with the role specified and not its items,\n"
             u8"which is against the normal behavior of the roles.\n\n"
             u8"Args:\n"
             u8"    role (str): Role to search objects with.\n"
             u8"Returns:\n"
             u8"    sequence: List of objects matching your condition.\n\n"
            )
        .def("get_leafs", &Space_getLeafs<Geometry2DCartesian>, (py::arg("path")=py::object()),
             u8"Get list of the geometry tree leafs.\n\n"
             u8"This method returns all the geometry tree leafs located under this geometry\n"
             u8"object. By *leaf* we understand a proper geometry object, in contrast to any\n"
             u8"container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of translations of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_translations`.\n"
            )
        .def("get_leafs_positions", (std::vector<Vec<2>>(Geometry2DCartesian::*)(const PathHints&)const) &Geometry2DCartesian::getLeafsPositions,
             (py::arg("path")=py::object()),
             u8"Calculate positions of all the geometry tree leafs.\n\n"
             u8"This method computes position of all the geometry tree leafs located under this\n"
             u8"geometry object. By *leaf* we understand a proper geometry object, in contrast\n"
             u8"to any container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of vectors containing the position of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_translations`.\n"
            )
        .def("get_leafs_bboxes", (std::vector<Box2D>(Geometry2DCartesian::*)(const PathHints&)const) &Geometry2DCartesian::getLeafsBoundingBoxes,
             (py::arg("path")=py::object()),
             u8"Calculate bounding boxes of all the geometry tree leafs.\n\n"
             u8"This method computes the bounding boxes of all the geometry tree leafs located\n"
             u8"under this geometry object. By *leaf* we understand a proper geometry object,\n"
             u8"in contrast to any container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of vectors containing the position of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_translations`.\n"
            )
        .def("get_leafs_translations", &Space_leafsAsTranslations<Geometry2DCartesian>, (py::arg("path")=py::object()),
             u8"Get list of :class:`Translation` objects holding all the geometry tree leafs.\n\n"
             u8"This method computes the :class:`Translation` objects of all the geometry tree\n"
             u8"leafs located under this geometry object. By *leaf* we understand a proper\n"
             u8"geometry object, in contrast to any container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of translations of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_leafs_translations`.\n"
            )
        .def("get_object_positions", (std::vector<Vec<2>>(Geometry2DCartesian::*)(const shared_ptr<const GeometryObject>&, const PathHints&)const) &Geometry2DCartesian::getObjectPositions,
             (py::arg("object"), py::arg("path")=py::object()),
             u8"Calculate positions of all instances of the specified object.\n\n"
             u8"Args:\n"
             u8"    object: Object to test.\n"
             u8"    path: Path specifying a particular object instance.\n"
             u8"Returns:\n"
             u8"    sequence: List of vectors containing the position of the instances of\n"
             u8"    the object.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order,\n"
             u8"provided they are called with the same arguments:\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_object_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_object_positions`\n"
            )
        .def("get_object_bboxes", (std::vector<Box2D>(Geometry2DCartesian::*)(const shared_ptr<const GeometryObject>&, const PathHints&)const) &Geometry2DCartesian::getObjectBoundingBoxes,
             (py::arg("object"), py::arg("path")=py::object()),
             u8"Calculate bounding boxes of all instances of specified object.\n\n"
             u8"The bounding boxes are computed in the local coordinates of *self*.\n\n"
             u8"Args:\n"
             u8"    object: Object to test.\n"
             u8"    path: Path specifying a particular object instance.\n"
             u8"Returns:\n"
             u8"    sequence: List of bounding boxes of the instances of the object.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order,\n"
             u8"provided they are called with the same arguments:\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_object_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian2D.get_object_positions`\n"
            )
        .def("get_paths", &Geometry2DCartesian::getPathsAt, (py::arg("point"), py::arg("all")=false),
             u8"Get subtree containing paths to all leafs covering the specified point.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Horizontal coordinate of the tested point.\n"
             u8"    c1 (float): Vertical coordinate of the tested point.\n"
             u8"    all (bool): If *True* then all the leafs intersecting the point are\n"
             u8"              considered. Otherwise, only the path to the topmost (i.e. visible)\n"
             u8"              object is returned.\n"
             u8"Returns:\n"
             u8"    Subtree with the path to the specified point.\n\n"
             u8"See also:\n"
             u8"    :class:`plask.geometry.Subtree`\n"
            )
        .def("get_paths", &Space_getPathsTo<Geometry2DCartesian>::call, (py::arg("c0"), "c1", py::arg("all")=false))
        .def("get_roles", &Geometry_getRolesAt<Geometry2DCartesian>, py::arg("point"),
             u8"Get roles of objects at specified point.\n\n"
             u8"This method returns a set of all the roles given to the every object\n"
             u8"intersecting the specified point.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Horizontal coordinate of the tested point.\n"
             u8"    c1 (float): Vertical coordinate of the tested point.\n"
             u8"Returns:\n"
             u8"    set: Set of the roles at given point.\n"
            )
        .def("get_roles", &Geometry2D_getRolesAt<Geometry2DCartesian>, (py::arg("c0"), "c1"))
        .def("has_role", &Geometry_hasRoleAt<Geometry2DCartesian>, (py::arg("role"), "point"),
             u8"Test if the specified point has a given role.\n\n"
             u8"This method checks if any object intersecting the specified point has the role\n"
             u8"*role*.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Horizontal coordinate of the tested point.\n"
             u8"    c1 (float): Vertical coordinate of the tested point.\n"
             u8"Returns:\n"
             u8"    bool: True if the point has the role *role*."
            )
        .def("has_role", &Geometry2D_hasRoleAt<Geometry2DCartesian>, (py::arg("role"), "c0", "c1"))
        .def("object_contains", (bool(Geometry2DCartesian::*)(const GeometryObject&,const PathHints&,const Vec<2>&)const)&Geometry2DCartesian::objectIncludes,
             (py::arg("object"), "path", "point")
            )
        .def("object_contains", (bool(Geometry2DCartesian::*)(const GeometryObject&,const Vec<2>&)const)&Geometry2DCartesian::objectIncludes,
             (py::arg("object"), "point"))
        .def("object_contains", &objectIncludes_2D<Geometry2DCartesian>, (py::arg("object"), "path", "c0", "c1"),
             u8"Test if the specified geometry object contains a point.\n\n"
             u8"The given geometry object must be located somewhere within the *self*\n"
             u8"geometry tree.\n\n"
             u8"Args:\n"
             u8"    object: Object to test.\n"
             u8"    path: Path specifying a particular object instance.\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Horizontal coordinate of the tested point.\n"
             u8"    c1 (float): Vertical coordinate of the tested point.\n"
             u8"    mesh (plask.mesh.Mesh): Mech, which points are tested.\n"
             u8"Returns:\n"
             u8"    bool: True if the specified geometry object contains the given point.\n"
             u8"          If a mesh is tested, the return value is an array of bools.\n"
            )
        .def("object_contains", &objectIncludes0_2D<Geometry2DCartesian>, (py::arg("object"), "c0", "c1"))
        .def("object_contains", &SpaceObjectIncludesPoints<Geometry2DCartesian>, (py::arg("object"), "path", "mesh"))
        .def("object_contains", &SpaceObjectIncludesPoints0<Geometry2DCartesian>, (py::arg("object"), "mesh"))
//         .def("get_subspace", py::raw_function(&Space_getSubspace<Geometry2DCartesian>, 2),
//              u8"Return sub- or super-space originating from provided object.\nOptionally specify 'path' to the unique instance of this object and edges of the new space")
    ;

    py::class_<Geometry2DCylindrical, shared_ptr<Geometry2DCylindrical>, py::bases<Geometry>>("Cylindrical2D",
        u8"Geometry in 2D cylindrical space.\n\n"
        u8"Cylindrical2D(root, **edges)\n"
        u8"Create a cylindrical space around a two-dimensional geometry object.\n\n"
        u8"Args:\n"
        u8"    root (GeometryObject2D or Revolution): Root object of the geometry.\n"
        u8"    edges (dict): Optional edges specification.\n"
        u8"        Edges are given as additional constructor keyword arguments. Available\n"
        u8"        keys are *inner*, *outer*, *top*, and *bottom* and their values must be\n"
        u8"        strings specifying the edge (either a material name or *mirror*,\n"
        u8"        *periodic*, or *extend*).\n\n"
        u8"Example:\n"
        u8"    >>> block = geometry.Block2D(4, 2, 'GaAs')\n"
        u8"    >>> geometry.Cylindrical2D(block, bottom='AlAs', outer='extend')\n"
        u8"    <plask.geometry.Cylindrical2D object at (0x3dd6c70)>",
        py::no_init)
        .def("__init__", raw_constructor(Geometry2DCylindrical__init__, 1))
        .add_property("item", &Geometry2DCylindrical::getChild,
                      ":class:`~plask.geometry.GeometryObject2D` at the root of the geometry tree."
                     )
        .add_property("revolution", &Geometry2DCylindrical::getRevolution,
                      ":class:`~plask.geometry.Revolution` object at the very root of the tree."
                     )
        .add_property("bbox", &Space_childBoundingBox<Geometry2DCylindrical>,
                      "Minimal rectangle which contains all points of the geometry object.\n\n"
                      "See also:\n"
                      "    :class:`plask.geometry.Box2D`\n"
                     )
        .def_readwrite("default_material", &Geometry2DCylindrical::defaultMaterial,
                       u8"This material is returned by :meth:`~plask.geometry.Cylindrical2D.get_material`\n"
                       u8"for the points that do not belong to any object in the geometry tree.\n"
                       u8"any object in the geometry tree.\n"
                      )
        .add_property("edges", &Geometry2DCylindrical_getEdges, &Space_setEdges,
                      "Dictionary specifying the geometry edges.")
        .def("get_material", &Geometry2DCylindrical::getMaterial, (py::arg("point")))
        .def("get_material", &Space_getMaterial<Geometry2DCylindrical>::call, (py::arg("c0"), "c1"),
             u8"Get material at the given point.\n\n"
             u8"This method returns a material object with the material at the given point if\n"
             u8"this point is located within the geometry object *self*. Otherwise the method\n"
             u8"returns :attr:`~plask.geometry.Cylindrical2D.default_material`.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Horizontal coordinate of the tested point.\n"
             u8"    c1 (float): Vertical coordinate of the tested point.\n"
             u8"Returns:\n"
             u8"    Material at the specified point.\n"
            )
        .def("get_matching_objects", &Space_getMatching<Geometry2DCylindrical>, py::arg("cond"),
             u8"Get list of the geometry tree objects matching condition.\n\n"
             u8"This method returns all the objects in the geometry tree that match the specified\n"
             u8"condition.\n\n"
             u8"Args:\n"
             u8"    cond: Python callable that accepts a geometry object and returns Boolean\n"
             u8"          indicating whether the object should be returned by this method or not.\n"
             u8"Returns:\n"
             u8"    sequence: List of objects matching your condition.\n\n"
            )
        .def("get_role_objects", &Space_getWithRole<Geometry2DCylindrical>, py::arg("role"),
             u8"Get list of the geometry tree objects that have the specified role.\n\n"
             u8"This method returns all the objects in the geometry tree that have the specified\n"
             u8"role.\n\n"
             u8".. rubric:: Warning!\n\n"
             u8"This method will return the very object with the role specified and not its items,\n"
             u8"which is against the normal behavior of the roles.\n\n"
             u8"Args:\n"
             u8"    role (str): Role to search objects with.\n"
             u8"Returns:\n"
             u8"    sequence: List of objects matching your condition.\n\n"
            )
        .def("get_leafs", &Space_getLeafs<Geometry2DCylindrical>, (py::arg("path")=py::object()),
             u8"Get list of the geometry tree leafs.\n\n"
             u8"This method returns all the geometry tree leafs located under this geometry\n"
             u8"object. By *leaf* we understand a proper geometry object, in contrast to any\n"
             u8"container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of translations of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_translations`.\n"
            )
        .def("get_leafs_positions", (std::vector<Vec<2>>(Geometry2DCylindrical::*)(const PathHints&)const) &Geometry2DCylindrical::getLeafsPositions,
             (py::arg("path")=py::object()),
             u8"Calculate positions of all the geometry tree leafs.\n\n"
             u8"This method computes position of all the geometry tree leafs located under this\n"
             u8"geometry object. By *leaf* we understand a proper geometry object, in contrast\n"
             u8"to any container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of vectors containing the position of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_translations`.\n"
            )
        .def("get_leafs_bboxes", (std::vector<Box2D>(Geometry2DCylindrical::*)(const PathHints&)const) &Geometry2DCylindrical::getLeafsBoundingBoxes,
             (py::arg("path")=py::object()),
             u8"Calculate bounding boxes of all the geometry tree leafs.\n\n"
             u8"This method computes the bounding boxes of all the geometry tree leafs located\n"
             u8"under this geometry object. By *leaf* we understand a proper geometry object,\n"
             u8"in contrast to any container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of vectors containing the position of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_translations`.\n"
            )
        .def("get_leafs_translations", &Space_leafsAsTranslations<Geometry2DCylindrical>, (py::arg("path")=py::object()),
             u8"Get list of :class:`Translation` objects holding all the geometry tree leafs.\n\n"
             u8"This method computes the :class:`Translation` objects of all the geometry tree\n"
             u8"leafs located under this geometry object. By *leaf* we understand a proper\n"
             u8"geometry object, in contrast to any container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of translations of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_leafs_translations`.\n"
            )
        .def("get_object_positions", (std::vector<Vec<2>>(Geometry2DCylindrical::*)(const shared_ptr<const GeometryObject>&, const PathHints&)const) &Geometry2DCylindrical::getObjectPositions,
             (py::arg("object"), py::arg("path")=py::object()),
             u8"Calculate positions of all instances of the specified object.\n\n"
             u8"Args:\n"
             u8"    object: Object to test.\n"
             u8"    path: Path specifying a particular object instance.\n"
             u8"Returns:\n"
             u8"    sequence: List of vectors containing the position of the instances of\n"
             u8"    the object.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order,\n"
             u8"provided they are called with the same arguments:\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_object_bboxes`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_object_positions`\n"
            )
        .def("get_object_bboxes", (std::vector<Box2D>(Geometry2DCylindrical::*)(const shared_ptr<const GeometryObject>&, const PathHints&)const) &Geometry2DCylindrical::getObjectBoundingBoxes,
             (py::arg("object"), py::arg("path")=py::object()),
             u8"Calculate bounding boxes of all instances of specified object.\n\n"
             u8"The bounding boxes are computed in the local coordinates of *self*.\n\n"
             u8"Args:\n"
             u8"    object: Object to test.\n"
             u8"    path: Path specifying a particular object instance.\n"
             u8"Returns:\n"
             u8"    sequence: List of bounding boxes of the instances of the object.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order,\n"
             u8"provided they are called with the same arguments:\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_object_bboxes`,\n"
             u8":meth:`~plask.geometry.Cylindrical2D.get_object_positions`\n"
            )
        .def("get_paths", &Geometry2DCylindrical::getPathsAt, (py::arg("point"), py::arg("all")=false),
             u8"Get subtree containing paths to all leafs covering the specified point.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Horizontal coordinate of the tested point.\n"
             u8"    c1 (float): Vertical coordinate of the tested point.\n"
             u8"    all (bool): If *True* then all the leafs intersecting the point are\n"
             u8"              considered. Otherwise, only the path to the topmost (i.e. visible)\n"
             u8"              object is returned.\n"
             u8"Returns:\n"
             u8"    Subtree with the path to the specified point.\n\n"
             u8"See also:\n"
             u8"    :class:`plask.geometry.Subtree`\n"
            )
        .def("get_paths", &Space_getPathsTo<Geometry2DCylindrical>::call, (py::arg("c0"), "c1", py::arg("all")=false))
        .def("get_roles", &Geometry_getRolesAt<Geometry2DCylindrical>, py::arg("point"),
             u8"Get roles of objects at specified point.\n\n"
             u8"This method returns a set of all the roles given to the every object\n"
             u8"intersecting the specified point.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Horizontal coordinate of the tested point.\n"
             u8"    c1 (float): Vertical coordinate of the tested point.\n"
             u8"Returns:\n"
             u8"    set: Set of the roles at given point.\n"
            )
        .def("get_roles", &Geometry2D_getRolesAt<Geometry2DCylindrical>, (py::arg("c0"), "c1"))
        .def("has_role", &Geometry_hasRoleAt<Geometry2DCylindrical>, (py::arg("role"), "point"),
             u8"Test if the specified point has a given role.\n\n"
             u8"This method checks if any object intersecting the specified point has the role\n"
             u8"*role*.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Horizontal coordinate of the tested point.\n"
             u8"    c1 (float): Vertical coordinate of the tested point.\n"
             u8"    mesh (plask.mesh.Mesh): Mech, which points are tested.\n"
             u8"Returns:\n"
             u8"    bool: True if the point has the role *role*.\n"
            )
        .def("has_role", &Geometry2D_hasRoleAt<Geometry2DCylindrical>, (py::arg("role"), "c0", "c1"))
        .def("object_contains", (bool(Geometry2DCylindrical::*)(const GeometryObject&,const PathHints&,const Vec<2>&)const)&Geometry2DCylindrical::objectIncludes,
             (py::arg("object"), "path", "point")
            )
        .def("object_contains", (bool(Geometry2DCylindrical::*)(const GeometryObject&,const Vec<2>&)const)&Geometry2DCylindrical::objectIncludes,
             (py::arg("object"), "point"))
        .def("object_contains", &objectIncludes_2D<Geometry2DCylindrical>, (py::arg("object"), "path", "c0", "c1"),
             u8"Test if the specified geometry object contains a point.\n\n"
             u8"The given geometry object must be located somewhere within the *self*\n"
             u8"geometry tree.\n\n"
             u8"Args:\n"
             u8"    object: Object to test.\n"
             u8"    path: Path specifying a particular object instance.\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Horizontal coordinate of the tested point.\n"
             u8"    c1 (float): Vertical coordinate of the tested point.\n"
             u8"Returns:\n"
             u8"    bool: True if the specified geometry object contains the given point.\n"
             u8"          If a mesh is tested, the return value is an array of bools.\n"
            )
        .def("object_contains", &objectIncludes0_2D<Geometry2DCylindrical>, (py::arg("object"), "c0", "c1"))
        .def("object_contains", &SpaceObjectIncludesPoints<Geometry2DCylindrical>, (py::arg("object"), "path", "mesh"))
        .def("object_contains", &SpaceObjectIncludesPoints0<Geometry2DCylindrical>, (py::arg("object"), "mesh"))
//         .def("get_subspace", py::raw_function(&Space_getSubspace<Geometry2DCylindrical>, 2),
//              u8"Return sub- or super-space originating from provided object.\nOptionally specify 'path' to the unique instance of this object and edges of the new space")
    ;

    py::class_<Geometry3D, shared_ptr<Geometry3D>, py::bases<Geometry>>("Cartesian3D",
        u8"Geometry in 3D space.\n\n"
        u8"Cartesian3D(geometry, **edges)\n"
        u8"    Create a space around a two-dimensional geometry object.\n\n"
        u8"Args:\n"
        u8"    root (GeometryObject3D): Root object of the geometry.\n"
        u8"    edges (dict): Optional edges specification.\n"
        u8"        Edges are given as additional constructor keyword arguments. Available\n"
        u8"        keys are *back*, *front*, *left*, *right*, *top*, and *bottom* and their\n"
        u8"        values must be strings specifying the edge (either a material name or\n"
        u8"        *mirror*, *periodic*, or *extend*).\n\n"
        u8"Example:\n"
        u8"    >>> block = geometry.Block3D(4, 2, 1, 'GaAs')\n"
        u8"    >>> geometry.Cartesian3D(block, left='mirror', bottom='GaAs',\n"
        u8"    ...                      front='periodic', back='periodic')\n"
        u8"    <plask.geometry.Cartesian3D object at (0x3dd6c70)>",
        py::no_init)
        .def("__init__", raw_constructor(Geometry3D__init__, 1))
        .add_property("item", &Geometry3D::getChild,
                      ":class:`~plask.geometry.GeometryObject3D` at the root of the geometry tree."
                     )
        .add_property("bbox", &Space_childBoundingBox<Geometry3D>,
                      u8"Minimal rectangle which contains all points of the geometry object.\n\n"
                      u8"See also:\n"
                      u8"    :class:`plask.geometry.Box3D`\n"
                     )
        .def_readwrite("default_material", &Geometry3D::defaultMaterial,
                       u8"Material of the *empty* regions of the geometry.\n\n"
                       u8"This material is returned by :meth:`~plask.geometry.Cartesian3D.get_material`\n"
                       u8"for the points that do not belong to any object in the geometry tree.\n"
                      )
        .add_property("edges", &Geometry3D_getEdges, &Space_setEdges,
                      "Dictionary specifying the geometry edges.")
        .def("get_material", &Geometry3D::getMaterial, (py::arg("point")))
        .def("get_material", &Space_getMaterial<Geometry3D>::call, (py::arg("c0"), "c1", "c2"),
             u8"Get material at the given point.\n\n"
             u8"This method returns a material object with the material at the given point if\n"
             u8"this point is located within the geometry object *self*. Otherwise the method\n"
             u8"returns :attr:`~plask.geometry.Cartesian3D.default_material`.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Longitudinal coordinate of the tested point.\n"
             u8"    c1 (float): Transverse coordinate of the tested point.\n"
             u8"    c2 (float): Vertical coordinate of the tested point.\n"
             u8"Returns:\n"
             u8"    Material at the specified point.\n"
            )
        .def("get_matching_objects", &Space_getMatching<Geometry3D>, py::arg("cond"),
             u8"Get list of the geometry tree objects matching condition.\n\n"
             u8"This method returns all the objects in the geometry tree that match the specified\n"
             u8"condition.\n\n"
             u8"Args:\n"
             u8"    cond: Python callable that accepts a geometry object and returns Boolean\n"
             u8"          indicating whether the object should be returned by this method or not.\n"
             u8"Returns:\n"
             u8"    sequence: List of objects matching your condition.\n\n"
            )
        .def("get_role_objects", &Space_getWithRole<Geometry3D>, py::arg("role"),
             u8"Get list of the geometry tree objects that have the specified role.\n\n"
             u8"This method returns all the objects in the geometry tree that have the specified\n"
             u8"role.\n\n"
             u8".. rubric:: Warning!\n\n"
             u8"This method will return the very object with the role specified and not its items,\n"
             u8"which is against the normal behavior of the roles.\n\n"
             u8"Args:\n"
             u8"    role (str): Role to search objects with.\n"
             u8"Returns:\n"
             u8"    sequence: List of objects matching your condition.\n\n"
            )
        .def("get_leafs", &Space_getLeafs<Geometry3D>, (py::arg("path")=py::object()),
             u8"Get list of the geometry tree leafs.\n\n"
             u8"This method returns all the geometry tree leafs located under this geometry\n"
             u8"object. By *leaf* we understand a proper geometry object, in contrast to any\n"
             u8"container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of translations of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_translations`.\n"
            )
        .def("get_leafs_positions", (std::vector<Vec<3>>(Geometry3D::*)(const PathHints&)const) &Geometry3D::getLeafsPositions,
             (py::arg("path")=py::object()),
             u8"Calculate positions of all the geometry tree leafs.\n\n"
             u8"This method computes position of all the geometry tree leafs located under this\n"
             u8"geometry object. By *leaf* we understand a proper geometry object, in contrast\n"
             u8"to any container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of vectors containing the position of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_translations`.\n"
            )
        .def("get_leafs_bboxes", (std::vector<Box3D>(Geometry3D::*)(const PathHints&)const) &Geometry3D::getLeafsBoundingBoxes,
             (py::arg("path")=py::object()),
             u8"Calculate bounding boxes of all the geometry tree leafs.\n\n"
             u8"This method computes the bounding boxes of all the geometry tree leafs located\n"
             u8"under this geometry object. By *leaf* we understand a proper geometry object,\n"
             u8"in contrast to any container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of vectors containing the position of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_translations`.\n"
            )
        .def("get_leafs_translations", &Space_leafsAsTranslations<Geometry3D>, (py::arg("path")=py::object()),
             u8"Get list of :class:`Translation` objects holding all the geometry tree leafs.\n\n"
             u8"This method computes the :class:`Translation` objects of all the geometry tree\n"
             u8"leafs located under this geometry object. By *leaf* we understand a proper\n"
             u8"geometry object, in contrast to any container or transformation.\n\n"
             u8"Args:\n"
             u8"    path: Path that can be used to select only some leafs.\n"
             u8"Returns:\n"
             u8"    sequence: List of translations of the leafs.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order:\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_positions`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_leafs_translations`.\n"
            )
        .def("get_object_positions", (std::vector<Vec<3>>(Geometry3D::*)(const shared_ptr<const GeometryObject>&, const PathHints&)const) &Geometry3D::getObjectPositions,
             (py::arg("object"), py::arg("path")=py::object()),
             u8"Calculate positions of all instances of the specified object.\n\n"
             u8"Args:\n"
             u8"    object: Object to test.\n"
             u8"    path: Path specifying a particular object instance.\n"
             u8"Returns:\n"
             u8"    sequence: List of vectors containing the position of the instances of\n"
             u8"    the object.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order,\n"
             u8"provided they are called with the same arguments:\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_object_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_object_positions`\n"
            )
        .def("get_object_bboxes", (std::vector<Box3D>(Geometry3D::*)(const shared_ptr<const GeometryObject>&, const PathHints&)const) &Geometry3D::getObjectBoundingBoxes,
             (py::arg("object"), py::arg("path")=py::object()),
             u8"Calculate bounding boxes of all instances of specified object.\n\n"
             u8"The bounding boxes are computed in the local coordinates of *self*.\n\n"
             u8"Args:\n"
             u8"    object: Object to test.\n"
             u8"    path: Path specifying a particular object instance.\n"
             u8"Returns:\n"
             u8"    sequence: List of bounding boxes of the instances of the object.\n\n"
             u8"All these methods are guaranteed to return their sequences in the same order,\n"
             u8"provided they are called with the same arguments:\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_object_bboxes`,\n"
             u8":meth:`~plask.geometry.Cartesian3D.get_object_positions`\n"
            )
        .def("get_paths", &Geometry3D::getPathsAt, (py::arg("point"), py::arg("all")=false),
             u8"Get subtree containing paths to all leafs covering the specified point.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Longitudinal coordinate of the tested point.\n"
             u8"    c1 (float): Transverse coordinate of the tested point.\n"
             u8"    c2 (float): Vertical coordinate of the tested point.\n"
             u8"    all (bool): If *True* then all the leafs intersecting the point are\n"
             u8"              considered. Otherwise, only the path to the topmost (i.e. visible)\n"
             u8"              object is returned.\n"
             u8"Returns:\n"
             u8"    Subtree with the path to the specified point.\n\n"
             u8"See also:\n"
             u8"    :class:`plask.geometry.Subtree`\n"
            )
        .def("get_paths", &Space_getPathsTo<Geometry3D>::call, (py::arg("c0"), "c1", "c2", py::arg("all")=false))
        .def("get_roles", &Geometry_getRolesAt<Geometry3D>, py::arg("point"),
             u8"Get roles of objects at specified point.\n\n"
             u8"This method returns a set of all the roles given to the every object\n"
             u8"intersecting the specified point.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Longitudinal coordinate of the tested point.\n"
             u8"    c1 (float): Transverse coordinate of the tested point.\n"
             u8"    c2 (float): Vertical coordinate of the tested point.\n"
             u8"Returns:\n"
             u8"    set: Set of the roles at given point.\n"
            )
        .def("get_roles", &Geometry3D_getRolesAt, (py::arg("c0"), "c1", "c2"))
        .def("has_role", &Geometry_hasRoleAt<Geometry3D>, (py::arg("role"), "point"),
             u8"Test if the specified point has a given role.\n\n"
             u8"This method checks if any object intersecting the specified point has the role\n"
             u8"*role*.\n\n"
             u8"Args:\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Longitudinal coordinate of the tested point.\n"
             u8"    c1 (float): Transverse coordinate of the tested point.\n"
             u8"    c2 (float): Vertical coordinate of the tested point.\n"
             u8"    mesh (plask.mesh.Mesh): Mech, which points are tested.\n"
             u8"Returns:\n"
             u8"    bool: True if the point has the role *role*.\n"
             u8"          If a mesh is tested, the return value is an array of bools.\n"
            )
        .def("has_role", &Geometry3D_hasRoleAt, (py::arg("role"), "c0", "c1", "c2"))
        .def("object_contains", (bool(Geometry3D::*)(const GeometryObject&,const PathHints&,const Vec<3>&)const)&Geometry3D::objectIncludes,
             (py::arg("object"), "path", "point")
            )
        .def("object_contains", (bool(Geometry3D::*)(const GeometryObject&,const Vec<3>&)const)&Geometry3D::objectIncludes,
             (py::arg("object"), "point"))
        .def("object_contains", &objectIncludes_3D, (py::arg("object"), "path", "c0", "c1", "c2"),
             u8"Test if the specified geometry object contains a point.\n\n"
             u8"The given geometry object must be located somewhere within the *self*\n"
             u8"geometry tree.\n\n"
             u8"Args:\n"
             u8"    object: Object to test.\n"
             u8"    path: Path specifying a particular object instance.\n"
             u8"    point (plask.vector): Vector with local coordinates of the tested point.\n"
             u8"    c0 (float): Longitudinal coordinate of the tested point.\n"
             u8"    c1 (float): Transverse coordinate of the tested point.\n"
             u8"    c2 (float): Vertical coordinate of the tested point.\n"
             u8"Returns:\n"
             u8"    bool: True if the specified geometry object contains the given point.\n"
            )
        .def("object_contains", &objectIncludes0_3D, (py::arg("object"), "c0", "c1", "c2"))
        .def("object_contains", &SpaceObjectIncludesPoints<Geometry3D>, (py::arg("object"), "path", "mesh"))
        .def("object_contains", &SpaceObjectIncludesPoints0<Geometry3D>, (py::arg("object"), "mesh"))
//         .def("getSubspace", py::raw_function(&Space_getSubspace<Geometry3D>, 2),
//              u8"Return sub- or super-space originating from provided object.\nOptionally specify 'path' to the unique instance of this object and edges of the new space")
    ;

    py::implicitly_convertible<shared_ptr<GeometryD<2>>, shared_ptr<Geometry>>();
    py::implicitly_convertible<shared_ptr<GeometryD<3>>, shared_ptr<Geometry>>();

    py::implicitly_convertible<shared_ptr<Geometry2DCartesian>, shared_ptr<const Geometry2DCartesian>>();
    py::implicitly_convertible<shared_ptr<Geometry2DCartesian>, shared_ptr<const GeometryD<2>>>();

    py::implicitly_convertible<shared_ptr<Geometry2DCylindrical>, shared_ptr<const Geometry2DCylindrical>>();
    py::implicitly_convertible<shared_ptr<Geometry2DCylindrical>, shared_ptr<const GeometryD<2>>>();

    py::implicitly_convertible<shared_ptr<Geometry3D>, shared_ptr<const Geometry3D>>();
    py::implicitly_convertible<shared_ptr<Geometry3D>, shared_ptr<const GeometryD<3>>>();

}


}} // namespace plask::python
