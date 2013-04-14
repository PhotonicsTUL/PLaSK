#include <boost/algorithm/string.hpp>
#include <cstring>
#include <fstream>

#include "python_globals.h"
#include "python_manager.h"
#include "python_provider.h"

#if PY_VERSION_HEX >= 0x03000000
#   define NEXT "__next__"
#else
#   define NEXT "next"
#endif

namespace plask { namespace python {

extern py::dict xml_globals;

struct XMLPythonDataSource: public XMLReader::DataSource {

    py::object file;

    XMLPythonDataSource(const py::object& file): file(file) {}

    virtual size_t read(char* buff, size_t buf_size) {
        size_t read = 0, len;
        do {
            py::object readobj = file.attr("read")(buf_size-read);
            const char* data = py::extract<const char*>(readobj);
            len = strlen(data);
            if (len > buf_size-read) throw CriticalException("Too much data read");
            std::copy_n(data, len, buff);
            buff += len;
            read += len;
        } while (read < buf_size && len != 0);
        return read;;
    }

    /// Empty, virtual destructor.
    virtual ~XMLPythonDataSource() {}
};

/**
 * Load data from XML
 */
void PythonManager_load(py::object self, py::object src, py::dict vars)
{
    PythonManager* manager = py::extract<PythonManager*>(self);

    XMLReader::DataSource* source;

    std::string str;
    try {
        str = py::extract<std::string>(src);
        if (str.find('<') == std::string::npos && str.find('>') == std::string::npos) // str is not XML (a filename probably)
            source = new XMLReader::StreamDataSource(new std::ifstream(str));
        else
            source = new XMLReader::StreamDataSource(new std::istringstream(str));
    } catch (py::error_already_set) {
        PyErr_Clear();
        if (!PyObject_HasAttrString(src.ptr(),"read")) throw TypeError("argument is neither string nor a proper file-like object");
        source = new XMLPythonDataSource(src);
    }
    XMLReader reader(source);

    // Variables
    manager->locals = vars.copy();
    if (!manager->locals.has_key("self")) manager->locals["self"] = self;

    reader.stringInterpreter.set(
        [manager](const std::string& str) -> double { return py::extract<double>(py::eval(py::str(str), xml_globals, manager->locals)); },
        [manager](const std::string& str) -> dcomplex { return py::extract<dcomplex>(py::eval(py::str(str), xml_globals, manager->locals)); },
        [manager](const std::string& str) -> size_t { return py::extract<size_t>(py::eval(py::str(str), xml_globals, manager->locals)); },
        [manager](const std::string& str) -> int { return py::extract<int>(py::eval(py::str(str), xml_globals, manager->locals)); },
        [manager](const std::string& str) -> short { return py::extract<short>(py::eval(py::str(str), xml_globals, manager->locals)); },
        [manager](const std::string& str) -> long { return py::extract<long>(py::eval(py::str(str), xml_globals, manager->locals)); },
        [manager](const std::string& str) -> unsigned int { return py::extract<unsigned int>(py::eval(py::str(str), xml_globals, manager->locals)); },
        [manager](const std::string& str) -> unsigned short { return py::extract<unsigned short>(py::eval(py::str(str), xml_globals, manager->locals)); },
        [manager](const std::string& str) -> unsigned long { return py::extract<unsigned long>(py::eval(py::str(str), xml_globals, manager->locals)); }
    );

    manager->loadFromReader(reader);
}


void PythonManager::loadDefines(XMLReader& reader)
{
    std::set<std::string> parsed;
    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() != "define") throw XMLUnexpectedElementException(reader, "<define>");
        std::string name = reader.requireAttribute("name");
        std::string value = reader.requireAttribute("value");
        if (!locals.has_key(name))
            locals[name] = (py::eval(py::str(value), xml_globals, locals));
        else if (parsed.find(name) != parsed.end())
            throw XMLDuplicatedElementException(reader, format("Definition of '%1%'", name));
        parsed.insert(name);
        reader.requireTagEnd();
    }
}


shared_ptr<Solver> PythonManager::loadSolver(const std::string& category, const std::string& lib, const std::string& solver_name, const std::string& name)
{
    std::string module_name = category + "." + lib;
    py::object module = py::import(module_name.c_str());
    py::object solver = module.attr(solver_name.c_str())(name);
    return py::extract<shared_ptr<Solver>>(solver);
}

void PythonManager::loadConnects(XMLReader& reader)
{

    while(reader.requireTagOrEnd()) {

        if (reader.getNodeName() != "connect" && reader.getNodeName() != "profile")
            throw XMLUnexpectedElementException(reader, "<connect> or <profile>", reader.getNodeName());

        std::string inkey = reader.requireAttribute("in");
        auto in = splitString2(inkey, '.');
        py::object solverin;

        auto in_solver = solvers.find(in.first);
        if (in_solver == solvers.end()) throw ValueError("Cannot find (in) solver with name '%1%'.", in.first);
        try { solverin = py::object(in_solver->second); }
        catch (py::error_already_set) { throw TypeError("Cannot convert solver '%1%' to python object.", in.first); }

        if (reader.getNodeName() == "connect") {

            auto out = splitString2(reader.requireAttribute("out"), '.');
            py::object solverout, provider, receiver;

            try { receiver = solverin.attr(in.second.c_str()); }
            catch (py::error_already_set) { throw AttributeError("Solver '%1%' does not have attribute '%2%.", in.first, in.second); }

            auto out_solver = solvers.find(out.first);
            if (out_solver == solvers.end()) throw ValueError("Cannot find (out) solver with name '%1%'.", out.first);
            try { solverout = py::object(out_solver->second); }
            catch (py::error_already_set) { throw TypeError("Cannot convert solver '%1%' to python object.", out.first); }

            try { provider = solverout.attr(out.second.c_str()); }
            catch (py::error_already_set) { throw AttributeError("Solver '%1%' does not have attribute '%2%.", out.first, out.second); }

            try {
                receiver.attr("connect")(provider);
            } catch (py::error_already_set) {
                throw TypeError("Cannot connect '%1%.%2%' to '%3%.'%4%'.", out.first, out.second, in.first, in.second);
            }

            reader.requireTagEnd();

        } else if (reader.getNodeName() == "profile") {

            auto defval = reader.getAttribute("default");
            py::object defaultobj;
            if (defval) defaultobj = py::eval(py::str(*defval));

            if (profiles.has_key(inkey))
                throw ValueError("There is already profile defined for receiver %1%", inkey);

            auto profile = make_shared<PythonProfile>(py::extract<const Geometry&>(solverin.attr("geometry")), defaultobj);
            profiles[inkey] = py::object(profile);

            while (reader.requireTagOrEnd("step")) {
                GeometryObject* object = requireGeometryObject(reader.requireAttribute("object")).get();
                PathHints path;
                auto hints = reader.getAttribute("path");
                if (hints) path = requirePathHints(*hints);
                py::object value = py::eval(py::str(reader.requireAttribute("value")));
                profile->places.emplace_back(PythonProfile::Place(*object, path));
                profile->values.push_back(value);
            }

            solverin.attr(in.second.c_str()) = profiles[inkey];

            reader.requireTagEnd();
        }

    }
}

void PythonEvalMaterialLoadFromXML(XMLReader& reader, MaterialsDB& materialsDB);

void PythonManager::loadMaterials(XMLReader& reader, const MaterialsSource& materialsSource)
{
    //we always use materials DB as source in python, so this cast is safe
    MaterialsDB& materialsDB = const_cast<MaterialsDB&>(*MaterialsDB::getFromSource(materialsSource));
    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "material")
            PythonEvalMaterialLoadFromXML(reader, materialsDB);
        else
            throw XMLUnexpectedElementException(reader, "<material>");
    }
}



void PythonManager::export_dict(py::object self, py::dict dict) {
    dict["OBJ"] = self.attr("obj");
    dict["PTH"] = self.attr("pth");
    dict["GEO"] = self.attr("geo");
    dict["MSH"] = self.attr("msh");
    dict["MSG"] = self.attr("msg");
    dict["DEF"] = self.attr("def");

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
template <> const std::string item_name<shared_ptr<GeometryObject>>() { return "geometry object"; }
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
    std::replace(key.begin(), key.end(), '_', '-');
    auto found = self.find(key);
    if (found == self.end()) {
        throw AttributeError("No " + item_name<T>() + " with id '%1%'", attr);
    }
    return py::object(found->second);
}

template <typename T>
static void dict__setattr__(std::map<std::string,T>& self, const std::string& attr, const T& value) {
    std::string key = attr;
    std::replace(key.begin(), key.end(), '_', '-');
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
            if (is_attr) std::replace(key.begin(), key.end(), '_', '-');
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
    // This swap ensures that in case there is an object with id 'keys', 'values', or 'items' it will take precedence over corresponding method
    py::object __getattr__ = c.attr("__getattr__");
    c.attr("__getattr__") = c.attr("__getattribute__");
    c.attr("__getattribute__") = __getattr__;
    py::delattr(py::scope(), (name+"Dict").c_str());

    py::scope scope = c;

    py::class_<detail::dict_iterator<T>>("Iterator", py::no_init)
        .def("__iter__", &detail::dict_iterator<T>::__iter__, py::return_self<>())
        .def(NEXT, &detail::dict_iterator<T>::next)
    ;
}


void register_manager() {
    py::class_<PythonManager, shared_ptr<PythonManager>, boost::noncopyable> manager("Manager",
        "Main input manager. It provides methods to read the XML file and fetch geometry objects, pathes,"
        "meshes, and generators by name.\n\n"
        "GeometryReader(materials=None)\n"
        "    Create manager with specified material database (if None, use default database)\n\n",
        py::init<MaterialsDB*>(py::arg("materials")=py::object())); manager
        .def("load", &PythonManager_load, "Load data from source (can be a filename, file, or an XML string to read)",
             (py::arg("source"), py::arg("vars")=py::dict()))
        .def_readonly("objects", &PythonManager::namedObjects, "Dictionary of all named geometry objects")
        .def_readonly("paths", &PythonManager::pathHints, "Dictionary of all named paths")
        .def_readonly("geometries", &PythonManager::geometries, "Dictionary of all named global geometries")
        .def_readonly("meshes", &PythonManager::meshes, "Dictionary of all named meshes")
        .def_readonly("mesh_generators", &PythonManager::generators, "Dictionary of all named mesh generators")
        .def_readonly("solvers", &PythonManager::solvers, "Dictionary of all named solvers")
        .def_readonly("profiles", &PythonManager::profiles, "Dictionary of constant profiles")
        .def_readonly("script", &PythonManager::script, "Script read from XML file")
        .def_readonly("scriptline", &PythonManager::scriptline, "First line of the script in the XML file")
        .def_readwrite("defines", &PythonManager::locals, "Local defines")
        .def("export", &PythonManager::export_dict, "Export loaded objects to target dictionary", py::arg("target"))
    ;
    manager.attr("obj") = manager.attr("objects");
    manager.attr("pth") = manager.attr("paths");
    manager.attr("geo") = manager.attr("geometries");
    manager.attr("msh") = manager.attr("meshes");
    manager.attr("msg") = manager.attr("mesh_generators");
    manager.attr("slv") = manager.attr("solvers");
    manager.attr("def") = manager.attr("defines");

    register_manager_dict<shared_ptr<GeometryObject>>("GeometryObjects");
    register_manager_dict<shared_ptr<Geometry>>("Geometries");
    register_manager_dict<PathHints>("PathHints");
    register_manager_dict<shared_ptr<Mesh>>("Meshes");
    register_manager_dict<shared_ptr<MeshGenerator>>("MeshGenerators");
    register_manager_dict<shared_ptr<Solver>>("Solvers");
}

}} // namespace plask::python
