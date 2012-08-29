#include <boost/algorithm/string.hpp>

#include "python_globals.h"
#include <numpy/arrayobject.h>

#include "python_manager.h"

#if PY_VERSION_HEX >= 0x03000000
#   define NEXT "__next__"
#else
#   define NEXT "next"
#endif

namespace plask { namespace python {

void PythonManager::read(py::object src) {
    std::string str;
    try {
        str = py::extract<std::string>(src);
    } catch (py::error_already_set) {
        PyErr_Clear();
#       if PY_VERSION_HEX < 0x03000000 && !defined(__MINGW32__)
            if (!PyFile_Check(src.ptr())) throw TypeError("argument is neither string nor a proper file-like object");
            PyFileObject* pfile = (PyFileObject*)src.ptr();
            auto file = PyFile_AsFile(src.ptr());
            PyFile_IncUseCount(pfile);
            loadFromFILE(file);
            PyFile_DecUseCount(pfile);
#       else
            // TODO choose better solution if XML parser is changed from irrXML to something more robust
            if (!PyObject_HasAttrString(src.ptr(),"read")) throw TypeError("argument is neither string nor a proper file-like object");
            loadFromXMLString(py::extract<std::string>(src.attr("read")()));
#       endif
        return;
    }
    if (str.find('<') == std::string::npos && str.find('>') == std::string::npos) // str is not XML (a filename probably)
        loadFromFile(str, *materialsDB);
    else
        loadFromXMLString(str, *materialsDB);
}

shared_ptr<Solver> PythonManager::loadSolver(const std::string& category, const std::string& lib, const std::string& solver_name, const std::string& name) {
    std::string module_name = category + "." + lib;
    py::object module = py::import(module_name.c_str());
    py::object solver = module.attr(solver_name.c_str())(name);
    return py::extract<shared_ptr<Solver>>(solver);
}

void PythonManager::loadConnects(XMLReader& reader) {
    while(reader.requireTagOrEnd()) {
        if (reader.getNodeName() != "connect") throw XMLUnexpectedElementException(reader, "<connect>", reader.getNodeName());
        auto out = splitString2(reader.requireAttribute("out"), '.');
        auto in = splitString2(reader.requireAttribute("in"), '.');

        py::object py_in, py_out, provider, receiver;

        auto out_solver = solvers.find(out.first);
        if (out_solver == solvers.end()) throw ValueError("Cannot find (out) solver with name '%1%'.", out.first);
        try { py_out = py::object(out_solver->second); }
        catch (py::error_already_set) { throw TypeError("Cannot convert solver '%1%' to python object.", out.first); }

        auto in_solver = solvers.find(in.first);
        if (in_solver == solvers.end()) throw ValueError("Cannot find (in) solver with name '%1%'.", in.first);
        try { py_in = py::object(in_solver->second); }
        catch (py::error_already_set) { throw TypeError("Cannot convert solver '%1%' to python object.", in.first); }

        try { provider = py_out.attr(out.second.c_str()); }
        catch (py::error_already_set) { throw AttributeError("Solver '%1%' does not have attribute '%2%.", out.first, out.second); }

        try { receiver = py_in.attr(in.second.c_str()); }
        catch (py::error_already_set) { throw AttributeError("Solver '%1%' does not have attribute '%2%.", in.first, in.second); }

        try {
            receiver << provider;
        } catch (py::error_already_set) {
            throw TypeError("Cannot connect '%1%.%2%' to '%3%.'%4%'.", out.first, out.second, in.first, in.second);
        }

        reader.requireTagEnd();
    }
}

void PythonEvalMaterialLoadFromXML(XMLReader& reader, MaterialsDB& materialsDB);

void PythonManager::loadMaterials(XMLReader& reader, MaterialsDB& materialsDB)
{
    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "material")
            PythonEvalMaterialLoadFromXML(reader, materialsDB);
        else
            throw XMLUnexpectedElementException(reader, "<material>");
    }
}



void PythonManager::export_dict(py::object self, py::dict dict) {
    dict["ELE"] = self.attr("ele");
    dict["PTH"] = self.attr("pth");
    dict["GEO"] = self.attr("geo");
    dict["MSH"] = self.attr("msh");
    dict["MSG"] = self.attr("msg");

    PythonManager* ths = py::extract<PythonManager*>(self);

    for (auto thesolver: ths->solvers) {
        dict[thesolver.first] = py::object(thesolver.second);
    }

    if (ths->script != "") dict["__script__"] = ths->script;
}


// std::string PythonManager::removeSpaces(const std::string& source) {
//     auto line =  boost::make_split_iterator(source, boost::token_finder(boost::is_any_of("\n"), boost::token_compress_off));
//     size_t strip;
//     for (auto c = line->begin(); c != line->end(); ++c) if (!std::isspace(*c)) throw Exception("There must be a newline after <script>");
//     auto firstline = ++line;
//     auto beg = line->begin();
//     do { // Search for the first non-empty line to get initial indentation
//         strip = 0;
//         for (; beg != line->end() && (*beg == ' ' || *beg == '\t'); ++beg) {
//             if (*beg == ' ') ++strip;
//             else { strip += 8; strip -= strip % 8; } // add to closest full tab-stop
//         }
//     } while (beg == line->end());
//     std::string result;
//     line = firstline;
//     for (size_t lineno = 1; line != decltype(line)(); ++line, ++lineno) { // Indent all lines
//         size_t pos = 0;
//         for (beg = line->begin(); beg != line->end() && (pos < strip); ++beg) {
//             if (*beg == ' ') ++pos;
//             else if (*beg == '\t') { pos += 8; pos -= pos % 8; } // add to closest full tab-stop
//             else throw Exception("Line %1% in <script> section indented less than the first one", lineno);
//         }
//         result += std::string(beg, line->end());
//         result += "\n";
//     }
//     return result;
// }




template <typename T> static const std::string item_name() { return ""; }
template <> const std::string item_name<shared_ptr<GeometryElement>>() { return "geometry element"; }
template <> const std::string item_name<shared_ptr<Geometry>>() { return "geometry"; }
template <> const std::string item_name<PathHints>() { return "path"; }
template <> const std::string item_name<shared_ptr<Solver>>() { return "solver"; }

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
    py::class_<PythonManager, shared_ptr<PythonManager>, boost::noncopyable> manager("Manager",
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
        .def_readonly("solvers", &PythonManager::solvers, "Dictionary of all named solvers")
        .def_readonly("script", &PythonManager::script, "Script read from XML file")
        .def("export", &PythonManager::export_dict, "Export loaded objects to target dictionary", py::arg("target"))
    ;
    manager.attr("ele") = manager.attr("elements");
    manager.attr("pth") = manager.attr("paths");
    manager.attr("geo") = manager.attr("geometries");
    manager.attr("msh") = manager.attr("meshes");
    manager.attr("msg") = manager.attr("mesh_generators");
    manager.attr("slv") = manager.attr("solvers");

    register_manager_dict<shared_ptr<GeometryElement>>("GeometryElements");
    register_manager_dict<shared_ptr<Geometry>>("Geometries");
    register_manager_dict<PathHints>("PathHintses");
    register_manager_dict<shared_ptr<Mesh>>("Meshes");
    register_manager_dict<shared_ptr<MeshGenerator>>("MeshGenerators");
    register_manager_dict<shared_ptr<Solver>>("Solvers");
}

}} // namespace plask::python
