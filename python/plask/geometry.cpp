#include <cmath>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include "../util/raw_constructor.h"

#include <plask/config.h>
#include <plask/exceptions.h>
#include <plask/geometry/manager.h>
#include <plask/geometry/leaf.h>
#include <plask/utils/format.h>

namespace plask { namespace python {

namespace py = boost::python;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void register_geometry_element();
void register_geometry_primitive();
void register_geometry_leafs();
void register_geometry_transform();
void register_geometry_container();


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Exc_NoSuchGeometryElement(const NoSuchGeometryElement& exc) {
    PyErr_SetString(PyExc_IndexError, exc.what());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

shared_ptr<GeometryManager> Geometry__init__(py::tuple args, py::dict kwargs) {

    std::string filename="";
    MaterialsDB* materialsDB;
    bool has_filename = false, has_materialsdb = false;

    if (py::len(args) > 3) {
        PyErr_SetString(PyExc_TypeError, format("__init__ takes at most 3 arguments (%d given)", py::len(args)).c_str());
        throw py::error_already_set();
    }
    if (py::len(args) > 1) {
        has_filename = true;
        filename = py::extract<std::string>(args[1]);
        if (py::len(args) > 2) {
                    has_materialsdb = true;
            materialsDB = py::extract<MaterialsDB*>(args[2]);
        }
    }
    py::stl_input_iterator<std::string> begin(kwargs), end;
    for (auto i = begin; i != end; ++i) {
        if (*i == "filename") {
            if (has_filename) {
                PyErr_SetString(PyExc_TypeError, "__init__() got multiple values for keyword argument 'filename'");
                throw py::error_already_set();
            } else {
                has_filename = true;
                filename = py::extract<std::string>(kwargs["filename"]);
            }
        } else if (*i == "materialsdb") {
            if (has_materialsdb) {
                PyErr_SetString(PyExc_TypeError, "__init__() got multiple values for keyword argument 'materialsdb'");
                throw py::error_already_set();
            } else {
                materialsDB = py::extract<MaterialsDB*>(kwargs["materialsdb"]);
                has_materialsdb = true;
            }
        } else {
            PyErr_SetString(PyExc_TypeError, format("__init__() got unexpected keyword argument '%s'", *i).c_str());
            throw py::error_already_set();
        }

    }

    if (!has_materialsdb) {
        py::object plask_module = py::import("plask");
        materialsDB = py::extract<MaterialsDB*>(plask_module.attr("material").attr("database"));
    }
    shared_ptr<GeometryManager> geometry(new GeometryManager());
    if (filename != "") geometry->loadFromFile(filename, *materialsDB);
    return geometry;
}

void Geometry_loadFromXMLString(GeometryManager& self, const std::string &inputXMLstr, const shared_ptr<MaterialsDB>& DB) {
    py::object plask_module = py::import("plask");
    MaterialsDB* db = DB? DB.get() : py::extract<MaterialsDB*>(plask_module.attr("material").attr("database"));
    self.loadFromXMLString(inputXMLstr, *db);
}

void Geometry_loadFromFile(GeometryManager& self, const std::string &filename, const shared_ptr<MaterialsDB>& DB) {
    py::object plask_module = py::import("plask");
    MaterialsDB* db = DB? DB.get() : py::extract<MaterialsDB*>(plask_module.attr("material").attr("database"));
    self.loadFromFile(filename, *DB);
}

shared_ptr<GeometryElement> Geometry_element(GeometryManager& self, const char* key) {
    shared_ptr<GeometryElement> result = self.getElement(key);
    if (!result) {
        PyErr_SetString(PyExc_KeyError, format("no geometry element with name '%s'", key).c_str());
        throw py::error_already_set();
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void initGeometry() {

    py::object geometry_module { py::handle<>(py::borrowed(PyImport_AddModule("plask.geometry"))) };
    py::scope().attr("geometry") = geometry_module;
    py::scope scope = geometry_module;

    scope.attr("__doc__") =
        "This module provides 2D and 3D geometry elements, necessary to describe the structure "
        "of analyzed device."; //TODO maybe more extensive description

    register_geometry_element();
    register_geometry_primitive();
    register_geometry_leafs();
    register_geometry_transform();
    register_geometry_container();

    // manager.h

    py::register_exception_translator<NoSuchGeometryElement>(&Exc_NoSuchGeometryElement);

    py::class_<GeometryManager, boost::noncopyable>("Geometry",
        "    Main geometry manager. It provides methods to read it from XML file and fetch geometry elements by name.\n\n"
        "    Geometry(filename="", materials=plask.material.database)\n"
        "        Create geometry with specified material database and optionally load it from XML file\n\n"
        "    Parameters\n"
        "    ----------\n"
        "    filename:\n"
        "        XML file with geometry specification. If left empty, empty geometry is created.\n"
        "    materialsdb:\n"
        "        Material database. If not specified, set to default database.\n"
        , py::no_init)
         .def("__init__", raw_constructor(&Geometry__init__))
         .def("element", &Geometry_element, "Get geometry element with given name", (py::arg("name")))
         .def("load", &Geometry_loadFromFile, "Load geometry from file", (py::arg("filename"), py::arg("materialsdb")=shared_ptr<MaterialsDB>()))
         .def("read", &Geometry_loadFromXMLString, "Read geometry from string", (py::arg("data"), py::arg("materialsdb")=shared_ptr<MaterialsDB>()))
    ;


}

}} // namespace plask::python
