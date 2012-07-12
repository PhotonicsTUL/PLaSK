#include "python_globals.h"
#include <numpy/arrayobject.h>

#include <plask/manager.h>

namespace plask { namespace python {

// shared_ptr<Manager> Manager__init__(py::tuple args, py::dict kwargs) {
//
//     std::string filename="";
//     MaterialsDB* materialsDB;
//     bool has_filename = false, has_materialsdb = false;
//
//     if (py::len(args) > 3) {
//         throw TypeError("__init__ takes 2 or 3 arguments (%d given)", py::len(args));
//     }
//     if (py::len(args) > 1) {
//         has_filename = true;
//         filename = py::extract<std::string>(args[1]);
//         if (py::len(args) > 2) {
//                     has_materialsdb = true;
//             materialsDB = py::extract<MaterialsDB*>(args[2]);
//         }
//     }
//     py::stl_input_iterator<std::string> begin(kwargs), end;
//     for (auto i = begin; i != end; ++i) {
//         if (*i == "filename") {
//             if (has_filename) {
//                 throw TypeError("__init__() got multiple values for keyword argument 'filename'");
//             } else {
//                 has_filename = true;
//                 filename = py::extract<std::string>(kwargs["filename"]);
//             }
//         } else if (*i == "materialsdb") {
//             if (has_materialsdb) {
//                 throw TypeError("__init__() got multiple values for keyword argument 'materialsdb'");
//             } else {
//                 materialsDB = py::extract<MaterialsDB*>(kwargs["materialsdb"]);
//                 has_materialsdb = true;
//             }
//         } else {
//             throw TypeError("__init__() got unexpected keyword argument '%s'", *i);
//         }
//
//     }
//
//     if (!has_materialsdb) {
//         py::object plask_module = py::import("plask");
//         materialsDB = &MaterialsDB::getDefault();
//     }
//     shared_ptr<Manager> geometry(new Manager());
//     if (filename != "") geometry->loadGeometryFromFile(filename, *materialsDB);
//     return geometry;
// }
//
// void Geometry_loadFromXMLString(Manager& self, const std::string &inputXMLstr, const shared_ptr<MaterialsDB>& DB) {
//     py::object plask_module = py::import("plask");
//     MaterialsDB* db = DB? DB.get() : &MaterialsDB::getDefault();
//     self.loadGeometryFromXMLString(inputXMLstr, *db);
// }
//
// void Geometry_loadFromFile(Manager& self, const std::string &filename, const shared_ptr<MaterialsDB>& DB) {
//     py::object plask_module = py::import("plask");
//     MaterialsDB* db = DB? DB.get() : &MaterialsDB::getDefault();
//     self.loadGeometryFromFile(filename, *db);
// }

void register_manager() {
//     py::class_<Manager, boost::noncopyable>("Manager",
//         "    Main geometry manager. It provides methods to read it from XML file and fetch geometry elements by name.\n\n"
//         "    GeometryReader(filename="", materials=plask.material.database)\n"
//         "        Create geometry manager with specified material database and optionally load it from XML file\n\n"
//         "    Parameters\n"
//         "    ----------\n"
//         "    filename:\n"
//         "        XML file with geometry specification. If left empty, empty geometry is created.\n"
//         "    materialsdb:\n"
//         "        Material database. If not specified, set to default database.\n"
//         , py::no_init)
//          .def("__init__", raw_constructor(&Geometry__init__))
//          .def("element", &Geometry_element, "Get geometry element with given name", (py::arg("name")))
//          .def("load", &Geometry_loadFromFile, "Load geometry from file", (py::arg("filename"), py::arg("materialsdb")=shared_ptr<MaterialsDB>()))
//          .def("read", &Geometry_loadFromXMLString, "Read geometry from string", (py::arg("data"), py::arg("materialsdb")=shared_ptr<MaterialsDB>()))
//     ;
}

}} // namespace plask::python
