#include <fstream>

#include "python_globals.h"
#include <numpy/arrayobject.h>

#include <plask/manager.h>

#if PY_VERSION_HEX >= 0x03000000
#   define NEXT "__next__"
#else
#   define NEXT "next"
#endif

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
        loadFromXMLString(str, *materialsDB);
    }

    static void export_dict(py::object self, py::dict dict) {
        dict["el"] = self.attr("el");
        dict["ph"] = self.attr("ph");
        dict["ge"] = self.attr("ge");
        dict["ms"] = self.attr("ms");
        dict["mg"] = self.attr("mg");
    }
};

template <typename T> static const std::string item_name() { return ""; }
template <> const std::string item_name<shared_ptr<GeometryElement>>() { return "geometry element"; }
template <> const std::string item_name<shared_ptr<Geometry>>() { return "geometry"; }
template <> const std::string item_name<PathHints>() { return "path"; }

template <typename T>
static py::object dict__getitem__(const std::map<std::string,T>& self, std::string key) {
    auto found = self.find(key);
    if (found == self.end()) throw KeyError(key);
    return py::object(found->second);
}

template <typename T>
static void dict__setitem__(std::map<std::string,T>& self, std::string key, const T& value) {
    self[key] = value;
}

template <typename T>
static void dict__delitem__(std::map<std::string,T>& self, std::string key) {
    auto found = self.find(key);
    if (found == self.end()) throw  KeyError(key);
    self.erase(found);
}

template <typename T>
static size_t dict__len__(const std::map<std::string,T>& self) {
    return self.size();
}

template <typename T>
static bool dict__contains__(const std::map<std::string,T>& self, const std::string& key) {
    return self.find(key) != self.end();
}

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
static py::object dict__getattr__(const std::map<std::string,T>& self, const std::string& attr) {
    std::string key = attr;
    std::replace(key.begin(), key.end(), '_', ' ');
    auto found = self.find(key);
    if (found == self.end()) {
        throw AttributeError("No " + item_name<T>() + " with id '%1%'", attr);
    }
    return py::object(found->second);
}

template <typename T>
static void dict__setattr__(std::map<std::string,T>& self, const std::string& attr, const T& value) {
    std::string key = attr;
    std::replace(key.begin(), key.end(), '_', ' ');
    self[key] = value;
}

template <typename T>
static void dict__delattr__(std::map<std::string,T>& self, const std::string& attr) {
    std::string key = attr;
    std::replace(key.begin(), key.end(), '_', ' ');
    auto found = self.find(key);
    if (found == self.end()) throw AttributeError("No " + item_name<T>() + " with id '%1%'", attr);
    self.erase(found);
}

namespace detail {

    template <typename T>
    struct dict_iterator {
        const std::map<std::string,T>& dict;
        typename std::map<std::string,T>::const_iterator i;
        bool is_attr;
        static dict_iterator<T> new_iterator(const std::map<std::string,T>& d) {
            return dict_iterator<T>(d, false);
        }
        static dict_iterator<T> new_attr_iterator(const std::map<std::string,T>& d) {
            return dict_iterator<T>(d, true);
        }
        dict_iterator(const std::map<std::string,T>& d, bool attr) : dict(d), i(d.begin()), is_attr(attr) {}
        dict_iterator(const dict_iterator<T>&) = default;
        dict_iterator<T>* __iter__() { return this; }
        std::string next() {
            if (i == dict.end()) {
                PyErr_SetString(PyExc_StopIteration, "No more items.");
                boost::python::throw_error_already_set();
            }
            std::string key = (i++)->first;
            if (is_attr) std::replace(key.begin(), key.end(), '_', ' ');
            return key;
        }
    };

} // namespace detail



template <typename T>
static void register_manager_dict(const std::string name) {
    py::class_<std::map<std::string, T>, boost::noncopyable> c((name+"Dict").c_str(), ("Dictionary holding each loaded " + item_name<T>()).c_str(), py::no_init); c
        .def("__getitem__", &dict__getitem__<T>)
        // .def("__setitem__", &dict__setitem__<T>)
        // .def("__delitem__", &dict__delitem__<T>)
        .def("__len__", &dict__len__<T>)
        .def("__contains__", &dict__contains__<T>)
        .def("__iter__", &detail::dict_iterator<T>::new_iterator)
        .def("keys", &dict_keys<T>)
        .def("values", &dict_values<T>)
        .def("items", &dict_items<T>)
        .def("__getattr__", &dict__getattr__<T>)
        // .def("__setattr__", &dict__setattr__<T>)
        // .def("__delattr__", &dict__delattr__<T>)
    ;
    // This swap ensures that in case there is an element with id 'keys', 'values', or 'items' it will take precedence over corresponding method
    py::object __getattr__ = c.attr("__getattr__");
    c.attr("__getattr__") = c.attr("__getattribute__");
    c.attr("__getattribute__") = __getattr__;

    py::class_<detail::dict_iterator<T>>((name+"DictIterator").c_str(), py::no_init)
        .def("__iter__", &detail::dict_iterator<T>::__iter__, py::return_self<>())
        .def(NEXT, &detail::dict_iterator<T>::next)
    ;
}


void register_manager() {
    py::class_<PythonManager, boost::noncopyable> manager("Manager",
        "Main input manager. It provides methods to read the XML file and fetch geometry elements, pathes,"
        "meshes, and generators by name.\n\n"
        "GeometryReader(materials=None)\n"
        "    Create manager with specified material database (if None, use default database)\n\n",
        py::init<MaterialsDB*>(py::arg("materials")=py::object())); manager
        .def("read", &PythonManager::read, "Read data from source (can be a filename, file, or an XML string to read)", py::arg("source"))
        .def_readonly("elements", &PythonManager::namedElements, "Dictionary of all named geometry elements")
        .def_readonly("paths", &PythonManager::pathHints, "Dictionary of all named paths")
        .def_readonly("geometries", &PythonManager::geometries, "Dictionary of all named global geometries")
        .def_readonly("meshes", &PythonManager::meshes, "Dictionary of all named meshes")
        .def_readonly("mesh_generators", &PythonManager::generators, "Dictionary of all named mesh generators")
        .def("export", &PythonManager::export_dict, "Export loaded objects to target dictionary", py::arg("target"))
    ;
    manager.attr("el") = manager.attr("elements");
    manager.attr("ph") = manager.attr("paths");
    manager.attr("ge") = manager.attr("geometries");
    manager.attr("ms") = manager.attr("meshes");
    manager.attr("mg") = manager.attr("mesh_generators");

    register_manager_dict<shared_ptr<GeometryElement>>("GeometryElements");
    register_manager_dict<shared_ptr<Geometry>>("Geometries");
    register_manager_dict<PathHints>("PathHintses");
    register_manager_dict<shared_ptr<Mesh>>("Meshes");
    register_manager_dict<shared_ptr<MeshGenerator>>("MeshGenerators");
}

}} // namespace plask::python
