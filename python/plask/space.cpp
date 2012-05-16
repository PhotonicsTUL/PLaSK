#include "python.hpp"
#include <boost/python/stl_iterator.hpp>

#include "../util/raw_constructor.h"

#include <plask/space.h>
#include <plask/geometry/path.h>

namespace plask { namespace python {

const PathHints empty_path {};

template <typename S> struct Space_getMaterial {
    static inline shared_ptr<Material> call(const S& self, double c0, double c1) {
        return self.getMaterial(Vec<2,double>(c0, c1));
    }
};
template <> struct Space_getMaterial<Space3d> {
    static inline shared_ptr<Material> call(const Space3d& self, double c0, double c1, double c2) {
        return self.getMaterial(Vec<3,double>(c0, c1, c2));
    }
};

template <typename S>
static py::list Space_leafsAsTranslations(const S& self, const PathHints& path=0) {
    py::list result;
    auto leafs = self.getLeafs(&path);
    auto translations = self.getLeafsPositions(&path);
    auto l = leafs.begin();
    auto t = translations.begin();
    for (; l != leafs.end(); ++l, ++t) {
        result.append(make_shared<Translation<S::DIMS>>(const_pointer_cast<GeometryElementD<S::DIMS>>(static_pointer_cast<const GeometryElementD<S::DIMS>>(*l)), *t));
    }
    return result;
}

template <typename S>
static std::vector<shared_ptr<GeometryElement>> Space_getLeafs(S& self, const PathHints& path) {
    std::vector<shared_ptr<const GeometryElement>> leafs = self.getLeafs(&path);
    std::vector<shared_ptr<GeometryElement>> result;
    result.reserve(leafs.size());
    for (auto i: leafs) result.push_back(const_pointer_cast<GeometryElement>(i));
    return result;
}


static void _Space_setBorders(CalculationSpace& self, py::dict borders, std::set<std::string>& parsed, const std::string& err_msg) {
   self.setBorders(
        [&](const std::string& s) {
            std::string str = s;
            std::replace(str.begin(), str.end(), '-', '_');
            parsed.insert(str);
            return borders.has_key(str) ?
                boost::optional<std::string>( (borders[str]==py::object()) ? std::string("null") : py::extract<std::string>(borders[str]) ) :
                boost::optional<std::string>();
        },
    config.axes);

    // Test if we have any spurious borders
    py::stl_input_iterator<std::string> begin(borders), end;
    for (auto item = begin; item != end; item++)
        if (parsed.find(*item) == parsed.end())
            throw ValueError(err_msg, *item);
}


static shared_ptr<Space2dCartesian> Space2dCartesian__init__(py::tuple args, py::dict kwargs) {
    int na = py::len(args);

    shared_ptr <Space2dCartesian> space;

    if (na == 3) {
        if (kwargs.has_key("geometry")) throw TypeError("got multiple values for keyword argument 'geometry'");
        if (kwargs.has_key("length")) throw TypeError("got multiple values for keyword argument 'length'");
        shared_ptr<GeometryElementD<2>> element = py::extract<shared_ptr<GeometryElementD<2>>>(args[1]);
        double length = py::extract<double>(args[2]);
        space = make_shared<Space2dCartesian>(element, length);
    } else if (na == 2) {
        if (kwargs.has_key("geometry")) throw TypeError("got multiple values for keyword argument 'geometry'");
        try {
            shared_ptr<Extrusion> extrusion = py::extract<shared_ptr<Extrusion>>(args[1]);
            if (kwargs.has_key("length")) throw TypeError("keyword argument 'length' not allowed if 'geometry' is of type Extrusion");
            space = make_shared<Space2dCartesian>(extrusion);
        } catch (py::error_already_set) {
            PyErr_Clear();
            shared_ptr<GeometryElementD<2>> element;
            try {
                element = py::extract<shared_ptr<GeometryElementD<2>>>(args[1]);
            } catch (py::error_already_set) {
                PyErr_Clear();
                throw TypeError("'geometry' argument type must be either Extrusion or GeometryElement2D");
            }
            double length = kwargs.has_key("length")? py::extract<double>(kwargs["length"]) : INFINITY;
            space = make_shared<Space2dCartesian>(element, length);
        }
    } else if (na == 1 && kwargs.has_key("geometry")) {
        try {
            shared_ptr<Extrusion> extrusion = py::extract<shared_ptr<Extrusion>>(kwargs["geometry"]);
            if (kwargs.has_key("length")) throw TypeError("keyword argument 'length' not allowed if 'geometry' is of type Extrusion");
            space = make_shared<Space2dCartesian>(extrusion);
        } catch (py::error_already_set) {
            PyErr_Clear();
            shared_ptr<GeometryElementD<2>> element;
            try {
                element = py::extract<shared_ptr<GeometryElementD<2>>>(kwargs["geometry"]);
            } catch (py::error_already_set) {
                PyErr_Clear();
                throw TypeError("'geometry' argument type must be either Extrusion or GeometryElement2D");
            }
            double length = kwargs.has_key("length")? py::extract<double>(kwargs["length"]) : INFINITY;
            space = make_shared<Space2dCartesian>(element, length);
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

template <typename S>
static typename Primitive<S::DIMS>::Box Space_childBoundingBox(const S& self) {
    return self.getChildBoundingBox();
}

static void Space_setBorders(CalculationSpace& self, py::dict borders) {
    std::set<std::string> parsed;
    _Space_setBorders(self, borders, parsed, "unexpected border name '%s'");
}

inline static py::object _border(const CalculationSpace& self, CalculationSpace::DIRECTION direction, bool higher) {
    auto str = self.getBorder(direction, higher).str();
    return (str=="null") ? py::object() : py::object(str);
}

static py::dict Space2dCartesian_getBorders(const Space2dCartesian& self) {
    py::dict borders;
    borders["left"] = _border(self, CalculationSpace::DIRECTION_TRAN, false);
    borders["right"] = _border(self, CalculationSpace::DIRECTION_TRAN, true);
    borders["top"] = _border(self, CalculationSpace::DIRECTION_UP, true);
    borders["bottom"] = _border(self, CalculationSpace::DIRECTION_UP, false);
    return borders;
}

template <typename S>
static shared_ptr<S> Space_getSubspace(py::tuple args, py::dict kwargs) {
    const S* self = py::extract<S*>(args[0]);

    py::object arg;
    py::ssize_t n = py::len(args);
    if (n >= 2) {
        if (kwargs.has_key("element")) throw TypeError("got multiple values for keyword argument 'element'");
        arg = args[1];
    } else
        arg = kwargs["element"];
    shared_ptr<GeometryElementD<S::DIMS>> element = py::extract<shared_ptr<GeometryElementD<S::DIMS>>>(arg);

    PathHints* path = nullptr;
    if (n >= 3) {
        if (kwargs.has_key("path")) throw TypeError("got multiple values for keyword argument 'path'");
        path = py::extract<PathHints*>(args[2]);
    } else if (kwargs.has_key("path"))
        path = py::extract<PathHints*>(kwargs["path"]);

    if (n >= 4) throw TypeError("getSubspace() takes 2 or 3 non-keyword arguments (%1%) given", n);

    S* space = self->getSubspace(element, path, false);

    std::set<std::string> parsed;
    parsed.insert("element");
    parsed.insert("path");
    _Space_setBorders(*space, kwargs, parsed, "unexpected border name '%s'");

    return shared_ptr<S>(space);
}

void register_calculation_spaces() {

    py::class_<Space2dCartesian, shared_ptr<Space2dCartesian>>("Space2DCartesian",
        "Calculation space representing 2D Cartesian coordinate system\n\n"
        "Space2DCartesian(geometry, length=infty, **borders)\n"
        "    Create a space around the two-dimensional geometry element with given length.\n\n"
        "    'geometry' can be either a 2D geometry object or plask.geometry.Extrusion, in which case\n"
        "    the 'length' parameter should be skipped, as it is read directly from extrusion.\n"
        "    'borders' is a dictionary specifying the type of the surroundings around the structure.", //TODO
        py::no_init)
        .def("__init__", raw_constructor(Space2dCartesian__init__, 1))
        .add_property("child", &Space2dCartesian::getChild, "GeometryElement2D at the root of the tree")
        .add_property("extrusion", &Space2dCartesian::getExtrusion, "Extrusion object at the very root of the tree")
        .add_property("bbox", &Space_childBoundingBox<Space2dCartesian>, "Minimal rectangle which includes all points of the geometry element")
        .def_readwrite("default_material", &Space2dCartesian::defaultMaterial, "Material of the 'empty' regions of the geometry")
        .add_property("front_material", &Space2dCartesian::getFrontMaterial, &Space2dCartesian::setFrontMaterial,
                      "Material on the positive side of the axis along the extrusion")
        .add_property("back_material", &Space2dCartesian::getBackMaterial, &Space2dCartesian::setBackMaterial,
                      "Material on the negative side of the axis along the extrusion")
        .add_property("borders", &Space2dCartesian_getBorders, &Space_setBorders,
                      "Dictionary specifying the type of the surroundings around the structure")
        .def("getMaterial", &Space2dCartesian::getMaterial, "Return material at given point", (py::arg("point")))
        .def("getMaterial", &Space_getMaterial<Space2dCartesian>::call, "Return material at given point", (py::arg("c0"), py::arg("c1")))
        .def("getLeafs", &Space_getLeafs<Space2dCartesian>, (py::arg("path")=empty_path),  "Return list of all leafs in the subtree originating from this element")
        .def("getLeafsPositions", (std::vector<Vec<2>>(Space2dCartesian::*)(const PathHints&)const) &Space2dCartesian::getLeafsPositions,
             (py::arg("path")=empty_path), "Calculate positions of all leafs")
        .def("getLeafsBBoxes", (std::vector<Box2d>(Space2dCartesian::*)(const PathHints&)const) &Space2dCartesian::getLeafsBoundingBoxes,
             (py::arg("path")=empty_path), "Calculate bounding boxes of all leafs")
        .def("getLeafsAsTranslations", &Space_leafsAsTranslations<Space2dCartesian>, (py::arg("path")=empty_path), "Return list of Translation objects holding all leafs")
        .def("getSubspace", py::raw_function(&Space_getSubspace<Space2dCartesian>, 2),
             "Return sub- or super-space originating from provided object.\nOptionally specify 'path' to the unique instance of this object and borders of the new space")
    ;
}


}} // namespace plask::python
