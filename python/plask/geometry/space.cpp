#include "../python_globals.h"
#include <boost/python/stl_iterator.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include "../../util/raw_constructor.h"

#include <plask/geometry/space.h>
#include <plask/geometry/path.h>

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
        result.append(make_shared<Translation<S::DIM>>(const_pointer_cast<GeometryObjectD<S::DIM>>(static_pointer_cast<const GeometryObjectD<S::DIM>>(*l)), *t));
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


static void _Space_setBorders(Geometry& self, py::dict borders, std::set<std::string>& parsed, const std::string& err_msg) {
   self.setBorders(
        [&](const std::string& s)->boost::optional<std::string> {
            std::string str = s;
            std::replace(str.begin(), str.end(), '-', '_');
            parsed.insert(str);
            return borders.has_key(str) ?
                boost::optional<std::string>( (borders[str]==py::object()) ? std::string("null") : py::extract<std::string>(borders[str]) ) :
                boost::optional<std::string>();
        },
    current_axes);

    // Test if we have any spurious borders
    py::stl_input_iterator<std::string> begin(borders), end;
    for (auto item = begin; item != end; item++)
        if (parsed.find(*item) == parsed.end())
            throw ValueError(err_msg, *item);
}


static shared_ptr<Geometry2DCartesian> Geometry2DCartesian__init__(py::tuple args, py::dict kwargs) {
    int na = py::len(args);

    shared_ptr <Geometry2DCartesian> space;

    if (na == 3) {
        if (kwargs.has_key("geometry")) throw TypeError("got multiple values for keyword argument 'geometry'");
        if (kwargs.has_key("length")) throw TypeError("got multiple values for keyword argument 'length'");
        shared_ptr<GeometryObjectD<2>> object = py::extract<shared_ptr<GeometryObjectD<2>>>(args[1]);
        double length = py::extract<double>(args[2]);
        space = make_shared<Geometry2DCartesian>(object, length);
    } else if (na == 2) {
        if (kwargs.has_key("geometry")) throw TypeError("got multiple values for keyword argument 'geometry'");
        try {
            shared_ptr<Extrusion> extrusion = py::extract<shared_ptr<Extrusion>>(args[1]);
            if (kwargs.has_key("length")) throw TypeError("keyword argument 'length' not allowed if 'geometry' is of type Extrusion");
            space = make_shared<Geometry2DCartesian>(extrusion);
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
            space = make_shared<Geometry2DCartesian>(object, length);
        }
    } else if (na == 1 && kwargs.has_key("geometry")) {
        try {
            shared_ptr<Extrusion> extrusion = py::extract<shared_ptr<Extrusion>>(kwargs["geometry"]);
            if (kwargs.has_key("length")) throw TypeError("keyword argument 'length' not allowed if 'geometry' is of type Extrusion");
            space = make_shared<Geometry2DCartesian>(extrusion);
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
            space = make_shared<Geometry2DCartesian>(object, length);
        }
    } else {
        throw TypeError("__init__() takes 2 or 3 non-keyword arguments (%1%) given", na);
    }

    std::set<std::string> parsed_kwargs;
    parsed_kwargs.insert("geometry");
    parsed_kwargs.insert("length");

    _Space_setBorders(*space, kwargs, parsed_kwargs, "__init__() got an unexpected keyword argument '%s'");

    return space;
}

static shared_ptr<Geometry2DCylindrical> Geometry2DCylindrical__init__(py::tuple args, py::dict kwargs) {
    int na = py::len(args);

    shared_ptr<Geometry2DCylindrical> space;
    py::object geometry;

    if (na == 2) geometry = args[1];
    else if (na == 1 && kwargs.has_key("geometry")) geometry = kwargs["geometry"];
    else throw TypeError("__init__() takes 1 or 2 non-keyword arguments (%1% given)", na);

    try {
        shared_ptr<Revolution> revolution = py::extract<shared_ptr<Revolution>>(geometry);
        space = make_shared<Geometry2DCylindrical>(revolution);
    } catch (py::error_already_set) {
        PyErr_Clear();
        shared_ptr<GeometryObjectD<2>> object;
        try {
            object = py::extract<shared_ptr<GeometryObjectD<2>>>(geometry);
        } catch (py::error_already_set) {
            PyErr_Clear();
            throw TypeError("'geometry' argument type must be either Extrusion or GeometryObject2D");
        }
        space = make_shared<Geometry2DCylindrical>(object);
    }

    std::set<std::string> parsed_kwargs;
    parsed_kwargs.insert("geometry");

    _Space_setBorders(*space, kwargs, parsed_kwargs, "__init__() got an unexpected keyword argument '%s'");

    return space;
}

static shared_ptr<Geometry3D> Geometry3D__init__(py::tuple args, py::dict kwargs) {
    int na = py::len(args);

    shared_ptr <Geometry3D> space;

    if (na == 2) {
        if (kwargs.has_key("geometry")) throw TypeError("got multiple values for keyword argument 'geometry'");
        shared_ptr<GeometryObjectD<3>> object = py::extract<shared_ptr<GeometryObjectD<3>>>(args[1]);
        space = make_shared<Geometry3D>(object);
    } else if (na == 1 && kwargs.has_key("geometry")) {
        shared_ptr<GeometryObjectD<3>> object = py::extract<shared_ptr<GeometryObjectD<3>>>(kwargs["geometry"]);
        space = make_shared<Geometry3D>(object);
    } else {
        throw TypeError("__init__() exactly 2 non-keyword arguments (%1%) given", na);
    }

    std::set<std::string> parsed_kwargs;
    parsed_kwargs.insert("geometry");

    _Space_setBorders(*space, kwargs, parsed_kwargs, "__init__() got an unexpected keyword argument '%s'");

    return space;
}

template <typename S>
static typename Primitive<S::DIM>::Box Space_childBoundingBox(const S& self) {
    return self.getChildBoundingBox();
}

static void Space_setBorders(Geometry& self, py::dict borders) {
    std::set<std::string> parsed;
    _Space_setBorders(self, borders, parsed, "unexpected border name '%s'");
}

struct BordersProxy : public std::map<std::string, py::object> {
    void __setitem__(const std::string&, const py::object&) {
        throw TypeError("Borders do not support item assignment");
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
    auto str = self.getBorder(direction, higher).str();
    return (str=="null") ? py::object() : py::object(str);
}

static BordersProxy Geometry2DCartesian_getBorders(const Geometry2DCartesian& self) {
    BordersProxy borders;
    borders["left"] = _border(self, Geometry::DIRECTION_TRAN, false);
    borders["right"] = _border(self, Geometry::DIRECTION_TRAN, true);
    borders["top"] = _border(self, Geometry::DIRECTION_VERT, true);
    borders["bottom"] = _border(self, Geometry::DIRECTION_VERT, false);
    return borders;
}

static BordersProxy Geometry2DCylindrical_getBorders(const Geometry2DCylindrical& self) {
    BordersProxy borders;
    borders["inner"] = _border(self, Geometry::DIRECTION_TRAN, false);
    borders["outer"] = _border(self, Geometry::DIRECTION_TRAN, true);
    borders["top"] = _border(self, Geometry::DIRECTION_VERT, true);
    borders["bottom"] = _border(self, Geometry::DIRECTION_VERT, false);
    return borders;
}

static BordersProxy Geometry3D_getBorders(const Geometry3D& self) {
    BordersProxy borders;
    borders["back"] = _border(self, Geometry::DIRECTION_LONG, false);
    borders["front"] = _border(self, Geometry::DIRECTION_LONG, true);
    borders["left"] = _border(self, Geometry::DIRECTION_TRAN, false);
    borders["right"] = _border(self, Geometry::DIRECTION_TRAN, true);
    borders["top"] = _border(self, Geometry::DIRECTION_VERT, true);
    borders["bottom"] = _border(self, Geometry::DIRECTION_VERT, false);
    return borders;
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
//     if (n >= 4) throw TypeError("getSubspace() takes 2 or 3 non-keyword arguments (%1%) given", n);
//
//     S* space = self->getSubspace(object, path, false);
//
//     std::set<std::string> parsed;
//     parsed.insert("object");
//     parsed.insert("path");
//     _Space_setBorders(*space, kwargs, parsed, "unexpected border name '%s'");
//
//     return shared_ptr<S>(space);
// }

void register_calculation_spaces() {

    py::class_<Geometry, shared_ptr<Geometry>, py::bases<GeometryObject>, boost::noncopyable>("Geometry",
        "Base class for all geometries", py::no_init)
        .def("__eq__", __is__<Geometry>)
        .add_property<>("axes", Geometry_getAxes, Geometry_getAxes, "Names of axes for this geometry")
    ;

    py::class_<BordersProxy>("BordersProxy")
        .def(py::map_indexing_suite<BordersProxy, true>())
        .def("__setitem__", &BordersProxy::__setitem__)
        .def("__repr__", &BordersProxy::__repr__)
        .def("keys", &BordersProxy::keys)
        .def("items", &BordersProxy::items)
    ;

    py::class_<Geometry2DCartesian, shared_ptr<Geometry2DCartesian>, py::bases<Geometry>>("Cartesian2D",
        "Geometry in 2D Cartesian space\n\n"
        "Cartesian2D(geometry, length=infty, **borders)\n"
        "    Create a space around the two-dimensional geometry object with given length.\n\n"
        "    'geometry' can be either a 2D geometry object or plask.geometry.Extrusion, in which case\n"
        "    the 'length' parameter should be skipped, as it is read directly from extrusion.\n"
        "    'borders' is a dictionary specifying the type of the surroundings around the structure.", //TODO
        py::no_init)
        .def("__init__", raw_constructor(Geometry2DCartesian__init__, 1))
        .add_property("item", &Geometry2DCartesian::getChild, "GeometryObject2D at the root of the tree")
        .add_property("extrusion", &Geometry2DCartesian::getExtrusion, "Extrusion object at the very root of the tree")
        .add_property("bbox", &Space_childBoundingBox<Geometry2DCartesian>, "Minimal rectangle which contains all points of the geometry object")
        .def_readwrite("default_material", &Geometry2DCartesian::defaultMaterial, "Material of the 'empty' regions of the geometry")
        .add_property("front_material", &Geometry2DCartesian::getFrontMaterial, &Geometry2DCartesian::setFrontMaterial,
                      "Material on the positive side of the axis along the extrusion")
        .add_property("back_material", &Geometry2DCartesian::getBackMaterial, &Geometry2DCartesian::setBackMaterial,
                      "Material on the negative side of the axis along the extrusion")
        .add_property("borders", &Geometry2DCartesian_getBorders, &Space_setBorders,
                      "Dictionary specifying the type of the surroundings around the structure")
        .def("get_material", &Geometry2DCartesian::getMaterial, "Return material at given point", (py::arg("point")))
        .def("get_material", &Space_getMaterial<Geometry2DCartesian>::call, "Return material at given point", (py::arg("c0"), py::arg("c1")))
        .def("get_leafs", &Space_getLeafs<Geometry2DCartesian>, (py::arg("path")=py::object()),  "Return list of all leafs in the subtree originating from this object")
        .def("get_leafs_positions", (std::vector<Vec<2>>(Geometry2DCartesian::*)(const PathHints&)const) &Geometry2DCartesian::getLeafsPositions,
             (py::arg("path")=py::object()), "Calculate positions of all leafs")
        .def("get_leafs_bboxes", (std::vector<Box2D>(Geometry2DCartesian::*)(const PathHints&)const) &Geometry2DCartesian::getLeafsBoundingBoxes,
             (py::arg("path")=py::object()), "Calculate bounding boxes of all leafs")
        .def("get_leafs_translations", &Space_leafsAsTranslations<Geometry2DCartesian>, (py::arg("path")=py::object()), "Return list of Translation objects holding all leafs")
        .def("get_object_positions", (std::vector<Vec<2>>(Geometry2DCartesian::*)(const shared_ptr<const GeometryObject>&, const PathHints&)const) &Geometry2DCartesian::getObjectPositions,
             (py::arg("object"), py::arg("path")=py::object()), "Calculate positions of all all instances of specified object (in local coordinates)")
        .def("get_object_bboxes", (std::vector<Box2D>(Geometry2DCartesian::*)(const shared_ptr<const GeometryObject>&, const PathHints&)const) &Geometry2DCartesian::getObjectBoundingBoxes,
             (py::arg("object"), py::arg("path")=py::object()), "Calculate bounding boxes of all instances of specified object (in local coordinates)")
        .def("get_paths", &Geometry2DCartesian::getPathsAt, (py::arg("point"), py::arg("all")=false),
             "Return subtree containg paths to all leafs covering specified point")
        .def("get_paths", &Space_getPathsTo<Geometry2DCartesian>::call, "Return subtree containing paths to all leafs covering specified point", (py::arg("c0"), py::arg("c1"), py::arg("all")=false))
        .def("get_roles", &Geometry_getRolesAt<Geometry2DCartesian>, py::arg("point"), "Return roles of objects at specified point")
        .def("get_roles", &Geometry2D_getRolesAt<Geometry2DCartesian>, (py::arg("c0"), "c1"), "Return roles of objects at specified point")
        .def("has_role", &Geometry_hasRoleAt<Geometry2DCartesian>, (py::arg("role"), "point"), "Return true if the specified point has given role")
        .def("has_role", &Geometry2D_hasRoleAt<Geometry2DCartesian>, (py::arg("role"), "c0", "c1"), "Return true if the specified point has given role")
//         .def("getSubspace", py::raw_function(&Space_getSubspace<Geometry2DCartesian>, 2),
//              "Return sub- or super-space originating from provided object.\nOptionally specify 'path' to the unique instance of this object and borders of the new space")
    ;

    py::class_<Geometry2DCylindrical, shared_ptr<Geometry2DCylindrical>, py::bases<Geometry>>("Cylindrical2D",
        "Geometry in 2D cylindrical space\n\n"
        "Cylindrical2D(geometry, **borders)\n"
        "    Create a space around the two-dimensional geometry object.\n\n"
        "    'geometry' can be either a 2D geometry object or plask.geometry.Revolution.\n"
        "    'borders' is a dictionary specifying the type of the surroundings around the structure.", //TODO
        py::no_init)
        .def("__init__", raw_constructor(Geometry2DCylindrical__init__, 1))
        .add_property("item", &Geometry2DCylindrical::getChild, "GeometryObject2D at the root of the tree")
        .add_property("revolution", &Geometry2DCylindrical::getRevolution, "Revolution object at the very root of the tree")
        .add_property("bbox", &Space_childBoundingBox<Geometry2DCylindrical>, "Minimal rectangle which contains all points of the geometry object")
        .def_readwrite("default_material", &Geometry2DCylindrical::defaultMaterial, "Material of the 'empty' regions of the geometry")
        .add_property("borders", &Geometry2DCylindrical_getBorders, &Space_setBorders,
                      "Dictionary specifying the type of the surroundings around the structure")
        .def("get_material", &Geometry2DCylindrical::getMaterial, "Return material at given point", (py::arg("point")))
        .def("get_material", &Space_getMaterial<Geometry2DCylindrical>::call, "Return material at given point", (py::arg("c0"), py::arg("c1")))
        .def("get_leafs", &Space_getLeafs<Geometry2DCylindrical>, (py::arg("path")=py::object()),  "Return list of all leafs in the subtree originating from this object")
        .def("get_leafs_positions", (std::vector<Vec<2>>(Geometry2DCylindrical::*)(const PathHints&)const) &Geometry2DCylindrical::getLeafsPositions,
             (py::arg("path")=py::object()), "Calculate positions of all leafs")
        .def("get_leafs_bboxes", (std::vector<Box2D>(Geometry2DCylindrical::*)(const PathHints&)const) &Geometry2DCylindrical::getLeafsBoundingBoxes,
             (py::arg("path")=py::object()), "Calculate bounding boxes of all leafs")
        .def("get_leafs_translations", &Space_leafsAsTranslations<Geometry2DCylindrical>, (py::arg("path")=py::object()), "Return list of Translation objects holding all leafs")
        .def("get_object_positions", (std::vector<Vec<2>>(Geometry2DCylindrical::*)(const GeometryObject&, const PathHints&)const) &Geometry2DCylindrical::getObjectPositions,
             (py::arg("object"), py::arg("path")=py::object()), "Calculate positions of all all instances of specified object (in local coordinates)")
        .def("get_object_bboxes", (std::vector<Box2D>(Geometry2DCylindrical::*)(const GeometryObject&, const PathHints&)const) &Geometry2DCylindrical::getObjectBoundingBoxes,
             (py::arg("object"), py::arg("path")=py::object()), "Calculate bounding boxes of all instances of specified object (in local coordinates)")
        .def("get_paths", &Geometry2DCylindrical::getPathsAt, (py::arg("point"), py::arg("all")=false),
             "Return subtree containing paths to all leafs covering specified point")
        .def("get_paths", &Space_getPathsTo<Geometry2DCylindrical>::call, "Return subtree containing paths to all leafs covering specified point", (py::arg("c0"), py::arg("c1"), py::arg("all")=false))
        .def("get_roles", &Geometry_getRolesAt<Geometry2DCylindrical>, py::arg("point"), "Return roles of objects at specified point")
        .def("get_roles", &Geometry2D_getRolesAt<Geometry2DCylindrical>, (py::arg("c0"), "c1"), "Return roles of objects at specified point")
        .def("has_role", &Geometry_hasRoleAt<Geometry2DCylindrical>, (py::arg("role"), "point"), "Return true if the specified point has given role")
        .def("has_role", &Geometry2D_hasRoleAt<Geometry2DCylindrical>, (py::arg("role"), "c0", "c1"), "Return true if the specified point has given role")
//         .def("getSubspace", py::raw_function(&Space_getSubspace<Geometry2DCylindrical>, 2),
//              "Return sub- or super-space originating from provided object.\nOptionally specify 'path' to the unique instance of this object and borders of the new space")
    ;

    py::class_<Geometry3D, shared_ptr<Geometry3D>, py::bases<Geometry>>("Cartesian3D",
        "Geometry in 3D space\n\n"
        "Cartesian3D(geometry, **borders)\n"
        "    Create a space around the two-dimensional geometry object.\n\n"
        "    'geometry' should be either a 3D geometry object.\n"
        "    'borders' is a dictionary specifying the type of the surroundings around the structure.", //TODO
        py::no_init)
        .def("__init__", raw_constructor(Geometry3D__init__, 1))
        .add_property("item", &Geometry3D::getChild, "GeometryObject2D at the root of the tree")
        .add_property("bbox", &Space_childBoundingBox<Geometry3D>, "Minimal rectangle which contains all points of the geometry object")
        .def_readwrite("default_material", &Geometry3D::defaultMaterial, "Material of the 'empty' regions of the geometry")
        .add_property("borders", &Geometry3D_getBorders, &Space_setBorders,
                      "Dictionary specifying the type of the surroundings around the structure")
        .def("get_material", &Geometry3D::getMaterial, "Return material at given point", (py::arg("point")))
        .def("get_material", &Space_getMaterial<Geometry3D>::call, "Return material at given point", (py::arg("c0"), "c1", "c2"))
        .def("get_leafs", &Space_getLeafs<Geometry3D>, (py::arg("path")=py::object()),  "Return list of all leafs in the subtree originating from this object")
        .def("get_leafs_positions", (std::vector<Vec<3>>(Geometry3D::*)(const PathHints&)const) &Geometry3D::getLeafsPositions,
             (py::arg("path")=py::object()), "Calculate positions of all leafs")
        .def("get_leafs_bboxes", (std::vector<Box3D>(Geometry3D::*)(const PathHints&)const) &Geometry3D::getLeafsBoundingBoxes,
             (py::arg("path")=py::object()), "Calculate bounding boxes of all leafs")
        .def("get_leafs_translations", &Space_leafsAsTranslations<Geometry3D>, (py::arg("path")=py::object()), "Return list of Translation objects holding all leafs")
        .def("get_object_positions", (std::vector<Vec<3>>(Geometry3D::*)(const GeometryObject&, const PathHints&)const) &Geometry3D::getObjectPositions,
             (py::arg("object"), py::arg("path")=py::object()), "Calculate positions of all all instances of specified object (in local coordinates)")
        .def("get_object_bboxes", (std::vector<Box3D>(Geometry3D::*)(const GeometryObject&, const PathHints&)const) &Geometry3D::getObjectBoundingBoxes,
             (py::arg("object"), py::arg("path")=py::object()), "Calculate bounding boxes of all instances of specified object (in local coordinates)")
        .def("get_paths", &Geometry3D::getPathsAt, (py::arg("point"), py::arg("all")=false),
             "Return subtree containing paths to all leafs covering specified point")
        .def("get_paths", &Space_getMaterial<Geometry3D>::call, "Return subtree containing paths to all leafs covering specified point", (py::arg("c0"), py::arg("c1"), py::arg("c2"), py::arg("all")=false))
        .def("get_roles", &Geometry_getRolesAt<Geometry3D>, py::arg("point"), "Return roles of objects at specified point")
        .def("get_roles", &Geometry3D_getRolesAt, (py::arg("c0"), "c1", "c2"), "Return roles of objects at specified point")
        .def("has_role", &Geometry_hasRoleAt<Geometry3D>, (py::arg("role"), "point"), "Return true if the specified point has given role")
        .def("has_role", &Geometry3D_hasRoleAt, (py::arg("role"), "c0", "c1", "c2"), "Return true if the specified point has given role")
//         .def("getSubspace", py::raw_function(&Space_getSubspace<Geometry3D>, 2),
//              "Return sub- or super-space originating from provided object.\nOptionally specify 'path' to the unique instance of this object and borders of the new space")
    ;

    py::implicitly_convertible<shared_ptr<Geometry2DCartesian>, shared_ptr<const Geometry2DCartesian>>();
    py::implicitly_convertible<shared_ptr<Geometry2DCartesian>, shared_ptr<const GeometryD<2>>>();

    py::implicitly_convertible<shared_ptr<Geometry2DCylindrical>, shared_ptr<const Geometry2DCylindrical>>();
    py::implicitly_convertible<shared_ptr<Geometry2DCylindrical>, shared_ptr<const GeometryD<2>>>();

    py::implicitly_convertible<shared_ptr<Geometry3D>, shared_ptr<const Geometry3D>>();
    py::implicitly_convertible<shared_ptr<Geometry3D>, shared_ptr<const GeometryD<3>>>();

}


}} // namespace plask::python
