#include "globals.h"
#include <boost/python/stl_iterator.hpp>

#include "../util/raw_constructor.h"

#include <plask/space.h>
#include <plask/geometry/path.h>

namespace plask { namespace python {

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
static py::list Space_leafsAsTranslations(const S& self, const PathHints* path=0) {
    py::list result;
    auto leafs = self.getLeafs(path);
    auto translations = self.getLeafsPositions(path);
    auto l = leafs.begin();
    auto t = translations.begin();
    for (; l != leafs.end(); ++l, ++t) {
        result.append(make_shared<Translation<S::DIMS>>(const_pointer_cast<GeometryElementD<S::DIMS>>(static_pointer_cast<const GeometryElementD<S::DIMS>>(*l)), *t));
    }
    return result;
}

template <int dim>
static std::vector<shared_ptr<GeometryElement>> Space_getLeafs(const CalculationSpaceD<dim>& self, const PathHints* path=0) {
    std::vector<shared_ptr<const GeometryElement>> leafs = self.getLeafs();
    std::vector<shared_ptr<GeometryElement>> result;
    result.reserve(leafs.size());
    for (auto i: leafs) result.push_back(const_pointer_cast<GeometryElement>(i));
    return result;
}




shared_ptr<Space2dCartesian> Space2dCartesian__init__(py::tuple args, py::dict kwargs) {
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

    static_pointer_cast<CalculationSpace>(space)->setBorders(
        [&](const std::string& s) {
            std::string str = s;
            std::replace(str.begin(), str.end(), '-', '_');
            parsed_kwargs.insert(str);
            if (kwargs.has_key(str)) return boost::optional<std::string>(py::extract<std::string>(kwargs[str]));
            return boost::optional<std::string>();
        },
    config.axes);

    // Test if we have any spurious kwargs
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto item = begin; item != end; item++)
        if (parsed_kwargs.find(*item) == parsed_kwargs.end())
            throw TypeError("__init__() got an unexpected keyword argument '%s'", *item);

    return space;
}


void register_calculation_spaces() {

    py::class_<Space2dCartesian, shared_ptr<Space2dCartesian>>("Space2DCartesian",
        "Calculation space representing 2D Cartesian coordinate system\n\n"
        "Space2DCartesian(geometry, **borders)\n"
        "    Create a space around the provided extrusion geometry object\n\n"
        "Space2DCartesian(geometry, length=infty, **borders)\n"
        "    Create a space around the two-dimensional geometry element with given length\n\n"
        "'borders' is a dictionary specifying the type of the surroundings around the structure.", //TODO
        py::no_init)
        .def("__init__", raw_constructor(Space2dCartesian__init__, 1))
        .add_property("child", &Space2dCartesian::getChild, "GeometryElement2D at the root of the tree")
        .add_property("extrusion", &Space2dCartesian::getExtrusion, "Extrusion object at the very root of the tree")
        .add_property("child_bbox", py::make_function(&Space2dCartesian::getChildBoundingBox, py::return_value_policy<py::return_by_value>()),
                      "Minimal rectangle which includes all points of the geometry element")
        .add_property("front_material", &Space2dCartesian::getFrontMaterial, &Space2dCartesian::setFrontMaterial,
                      "material on the positive side of the axis along the extrusion")
        .add_property("back_material", &Space2dCartesian::getBackMaterial, &Space2dCartesian::setBackMaterial,
                      "material on the negative side of the axis along the extrusion")
        .def("getMaterial", &Space2dCartesian::getMaterial, "Return material at given point", (py::arg("point")))
        .def("getMaterial", &Space_getMaterial<Space2dCartesian>::call, "Return material at given point", (py::arg("c0"), py::arg("c1")))
        .def("getLeafs", &Space_getLeafs<2>, (py::arg("path")=py::object()),  "Return list of all leafs in the subtree originating from this element")
        .def("getLeafsPositions", &Space2dCartesian::getLeafsPositions, (py::arg("path")=py::object()), "Calculate positions of all leafs (in local coordinates)")
        .def("getLeafsBBoxes", &Space2dCartesian::getLeafsBoundingBoxes, (py::arg("path")=py::object()), "Calculate bounding boxes of all leafs (in local coordinates)")
        .def("getLeafsAsTranslations", &Space_leafsAsTranslations<Space2dCartesian>, (py::arg("path")=py::object()), "Return list of Translation objects holding all leafs")
        //TODO getting borders
    ;

    //TODO
    //convert borders classes to str

}


}} // namespace plask::python
