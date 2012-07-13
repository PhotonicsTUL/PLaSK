#include <fstream>

#include "python_globals.h"
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <numpy/arrayobject.h>

#include <plask/manager.h>

namespace plask { namespace python {

struct PythonManager: public Manager {

    MaterialsDB* materialsDB;

    PythonManager(MaterialsDB* db=nullptr) : materialsDB(db? db : &MaterialsDB::getDefault()) {}

    void read(py::object src) {
        std::string str;
        try {
            str = py::extract<std::string>(src);
            if (str.find('<') == std::string::npos && str.find('>') == std::string::npos) { // str is not XML (a filename probably)
                std::ifstream file;
                file.open(str);
                if (!file.is_open()) throw IOError("No such file: '%1%'", str);
                file >> str;
            }
        } catch (py::error_already_set) {
            PyErr_Clear();
            try {
                str = py::extract<std::string>(src.attr("read")());
            } catch (py::error_already_set) {
                throw TypeError("argument is neither string nor a proper file-like object");
            }
        }
        loadGeometryFromXMLString(str, *materialsDB);
    }
};

template <typename T>
static py::list dict_keys(const std::map<std::string,T>& self) {
    py::list result;
    for (auto item: self) {
        result.append(item.first);
    }
    return result;
}

template <typename T>
static py::list dict_values(const std::map<std::string,T>& self) {
    py::list result;
    for (auto item: self) {
        result.append(item.second);
    }
    return result;
}

template <typename T>
static py::list dict_items(const std::map<std::string,T>& self) {
    py::list result;
    for (auto item: self) {
        result.append(py::make_tuple(item.first, item.second));
    }
    return result;
}


//         .def("__iter__")


void register_manager() {
    py::class_<PythonManager, boost::noncopyable>("Manager",
        "Main input manager. It provides methods to read the XML file and fetch geometry elements, pathes,"
        "meshes, and generators by name.\n\n"
        "GeometryReader(materials=None)\n"
        "    Create manager with specified material database (if None, use default database)\n\n",
        py::init<MaterialsDB*>(py::arg("materials")=py::object()))
        .def("read", &PythonManager::read, "Read data. source can be a filename, file, or XML string to read.", py::arg("source"))
        .def_readonly("elements", &PythonManager::namedElements, "Dictionary of all named geometry elements")
        .def_readonly("paths", &PythonManager::pathHints, "Dictionary of all named paths")
        .def_readonly("geometries", &PythonManager::geometries, "Dictionary of all named global geometries")
    ;

    py::class_<std::map<std::string, shared_ptr<GeometryElement>>, boost::noncopyable>("GeometryElementDictionary")
        .def(py::map_indexing_suite<std::map<std::string, shared_ptr<GeometryElement>>, true>())
        .def("keys", &dict_keys<shared_ptr<GeometryElement>>)
        .def("values", &dict_values<shared_ptr<GeometryElement>>)
        .def("items", &dict_items<shared_ptr<GeometryElement>>)
    ;

    py::class_<std::map<std::string, PathHints>, boost::noncopyable>("PathHintsDictionary")
        .def(py::map_indexing_suite<std::map<std::string, PathHints>>())
        .def("keys", &dict_keys<PathHints>)
        .def("values", &dict_values<PathHints>)
        .def("items", &dict_items<PathHints>)
    ;

    py::class_<std::map<std::string, shared_ptr<Geometry>>, boost::noncopyable>("GeometryDictionary")
        .def(py::map_indexing_suite<std::map<std::string, shared_ptr<Geometry>>, true>())
        .def("keys", &dict_keys<shared_ptr<Geometry>>)
        .def("values", &dict_values<shared_ptr<Geometry>>)
        .def("items", &dict_items<shared_ptr<Geometry>>)
    ;
}

}} // namespace plask::python
