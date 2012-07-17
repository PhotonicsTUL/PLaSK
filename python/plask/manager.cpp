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

template <typename T>
static py::object dict__getattr__(const std::map<std::string,T>& self, std::string key) {
    std::replace(key.begin(), key.end(), '_', ' ');
    auto found = self.find(key);
    if (found == self.end()) throw AttributeError(key);
    return py::object(found->second);
}

template <typename T>
static void dict__setattr__(std::map<std::string,T>& self, std::string key, const T& value) {
    std::replace(key.begin(), key.end(), '_', ' ');
    self[key] = value;
}


template <typename T>
static void register_manager_dict(const std::string name) {
    py::class_<std::map<std::string, T>, boost::noncopyable>(name.c_str())
        .def(py::map_indexing_suite<std::map<std::string, T>, true>())
        .def("keys", &dict_keys<T>)
        .def("values", &dict_values<T>)
        .def("items", &dict_items<T>)
        .def("__getattr__", &dict__getattr__<T>)
        .def("__setattr__", &dict__setattr__<T>)
    ;

}


void register_manager() {
    py::class_<PythonManager, boost::noncopyable> manager("Manager",
        "Main input manager. It provides methods to read the XML file and fetch geometry elements, pathes,"
        "meshes, and generators by name.\n\n"
        "GeometryReader(materials=None)\n"
        "    Create manager with specified material database (if None, use default database)\n\n",
        py::init<MaterialsDB*>(py::arg("materials")=py::object())); manager
        .def("read", &PythonManager::read, "Read data. source can be a filename, file, or XML string to read.", py::arg("source"))
        .def_readonly("elements", &PythonManager::namedElements, "Dictionary of all named geometry elements")
        .def_readonly("paths", &PythonManager::pathHints, "Dictionary of all named paths")
        .def_readonly("geometries", &PythonManager::geometries, "Dictionary of all named global geometries")
    ;
    manager.attr("el") = manager.attr("elements");
    manager.attr("pt") = manager.attr("paths");
    manager.attr("ge") = manager.attr("geometries");
    //manager.attr("ms") = manager.attr("meshes");
    //manager.attr("mg") = manager.attr("mesh_generators");

    register_manager_dict<shared_ptr<GeometryElement>>("GeometryElementDictionary");
    register_manager_dict<shared_ptr<Geometry>>("GeometryDictionary");
    register_manager_dict<PathHints>("PathHintsDictionary");
}

}} // namespace plask::python
