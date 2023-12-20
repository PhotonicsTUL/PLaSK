/*
 * This file is part of PLaSK (https://plask.app) by Photonics Group at TUL
 * Copyright (c) 2022 Lodz University of Technology
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 */
#include <boost/algorithm/string.hpp>
#include <cstring>
#include <fstream>

#include <boost/tokenizer.hpp>

#include "plask/filters/filter.hpp"
#include "plask/manager.hpp"
#include "plask/geometry/reader.hpp"

#include "python_globals.hpp"
#include "python_manager.hpp"
#include "python_provider.hpp"
#include "geometry/geometry.hpp"

namespace plask { namespace python {

PLASK_PYTHON_API std::string getPythonExceptionMessage();

extern PLASK_PYTHON_API PyObject* pyXmlError;

class PythonXMLFilter {

    PythonManager* manager;

    // static inline bool is_first_char_in_name(char c) { return ('a' < c && c < 'z') || ('A' < c && c < 'Z') || (c == '_'); }
    // static inline bool is_char_in_name(char c) { return is_first_char_in_name(c) || ('0' < c && c < '9');  }

    //check if in[p] == in[p+1] && in[p] == in[p+2], and if so forward p by 2 positions
    static inline bool has_long_str(const std::string& in, std::string::size_type& p) {
        if (p + 2 < in.size() && in[p] == in[p+1] && in[p] == in[p+2]) {
            p += 2;
            return true;
        } else
            return false;
    }

    // move p to nearest unescaped str_terminator or end of in, support python long strings
    static inline void goto_string_end(const std::string& in, std::string::size_type& p) {
        bool long_string = has_long_str(in, p);
        const char str_terminator = in[p++];    //skip string begin
        while (p < in.size()) {
            if (in[p] == str_terminator && (!long_string || has_long_str(in, p))) break;
            if (p == '\\') {
                ++p;
                if (p == in.size()) break;
            }
            ++p;
        }
    }

  public:

    PythonXMLFilter(PythonManager* manager): manager(manager) {}

    std::string eval(std::string str) const {
        boost::algorithm::trim(str);
        try {
            return py::extract<std::string>(py::str(py_eval(str, manager->globals, manager->defs)));
        } catch (py::error_already_set&) {
            throw Exception(getPythonExceptionMessage());
        }
    }

    std::string operator()(const std::string& in) const {
        std::string result;
        result.reserve(in.size());  // we guess that output will have the similar size as input
        for (std::string::size_type pos = 0; pos < in.size(); ++pos) {
            if (in[pos] == '{') {
                ++pos;
                if (in[pos] == '{') { result += '{'; continue; }
                // find } but not inside python string, note that this code support also python long strings
                std::string::size_type close_pos = pos;
                int level = 1;
                while (close_pos < in.size() && level > 0) {
                    if (in[close_pos] == '\'' || in[close_pos] == '"') {
                        goto_string_end(in, close_pos);
                        if (close_pos == in.size()) break;  // else close_pos points to end of string
                    }
                    else if (in[close_pos] == '{') ++level;
                    else if (in[close_pos] == '}') --level;
                    ++close_pos;
                }
                if (close_pos == in.size() && level != 0)
                    throw plask::Exception("Cannot find '}' matching to '{' at position {0} in: {1}", pos-1, in);
                result += eval(in.substr(pos, close_pos-1 - pos));
                pos = close_pos-1;    // pos with '}' that will be skipped
            } else if (in[pos] == '}' && in[pos+1] == '}') {
                result += '}';
                ++pos;
            } else
                result += in[pos];
        }
        return result;
    }

};

struct XMLPythonDataSource: public XMLReader::DataSource {

    py::object file;

    XMLPythonDataSource(const py::object& file): file(file) {}

    size_t read(char* buff, size_t buf_size) override {
        size_t read = 0, len;
        do {
            py::object readobj = file.attr("read")(buf_size-read);
            const char* data;
            if (PyBytes_Check(readobj.ptr())) {
                data = PyBytes_AS_STRING(readobj.ptr());
            } else {
                data = py::extract<const char*>(readobj);
            }
            len = strlen(data);
            if (len > buf_size-read) throw CriticalException("too much data read");
            std::copy_n(data, len, buff);
            buff += len;
            read += len;
        } while (read < buf_size && len != 0);
        return read;;
    }

    /// Empty, virtual destructor.
    virtual ~XMLPythonDataSource() {}
};

inline boost::filesystem::path pyobject_to_path(py::object src) {
    Py_ssize_t size;
    wchar_t* str = PyUnicode_AsWideCharString(src.ptr(), &size);
    std::wstring wstr(str, size);
    PyMem_Free(str);
    return wstr;
}

void XMLExceptionWithCause::setPythonException() {
    PyErr_SetString(pyXmlError, what());
    PyObject *value, *type, *traceback;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    if (cause) {
        Py_INCREF(cause);
        Py_INCREF(cause);
        PyException_SetCause(value, cause);
        PyException_SetContext(value, cause);
    }
    PyErr_Restore(type, value, traceback);
}

/**
 * Load data from XML
 */
PLASK_PYTHON_API void loadXpl(py::object self, py::object src, py::dict vars, py::object filter=py::object())
{
    PythonManager* manager = py::extract<PythonManager*>(self);

    std::unique_ptr<XMLReader::DataSource> source;

    boost::filesystem::path filename;

    std::string str;
    try {
        str = py::extract<std::string>(src);
        if (str.find('<') == std::string::npos && str.find('>') == std::string::npos) { // str is not XML (a filename probably)
            boost::filesystem::path filename_tmp = pyobject_to_path(src);
            source.reset(new XMLReader::StreamDataSource(new std::ifstream(filename_tmp.string())));
            filename = std::move(filename_tmp);
        } else
            source.reset(new XMLReader::StreamDataSource(new std::istringstream(str)));
    } catch (py::error_already_set&) {
        PyErr_Clear();
        if (!PyObject_HasAttrString(src.ptr(),"read")) throw TypeError("argument is neither string nor a proper file-like object");
        try {
            py::object name_attr = src.attr("name");
            std::string name = py::extract<std::string>(name_attr);
            if (name[0] != '<') filename = pyobject_to_path(name_attr);
        } catch(...) {
            PyErr_Clear();
        }
        source.reset(new XMLPythonDataSource(src));
    }

    XMLReader reader(std::move(source));

    // Globals
    manager->globals["PTH"] = self.attr("pth");
    manager->globals["GEO"] = self.attr("geo");
    manager->globals["MSH"] = self.attr("msh");

    // Variables

    manager->overrites = py::tuple(vars.keys());
    if (vars.has_key("self"))
        throw ValueError("definition name 'self' is reserved");
    manager->defs.update(vars);

    struct SelfGuard {
        PythonManager* manager;
        SelfGuard(PythonManager* manager, const py::object& self): manager(manager) {
            manager->defs["self"] = self;
        }
        ~SelfGuard() { py::delitem(manager->defs, py::str("self")); }
    };
    SelfGuard guard(manager, self);

    // reader.setFilters(PythonXMLFilter(manager));
    reader.attributeFilter = PythonXMLFilter(manager);

    if (filter.is_none()) {
        manager->load(reader, Manager::ExternalSourcesFromFile(filename));
    } else {
        py::list sections = py::list(filter);
        auto filterfun = [sections](const std::string& section) -> bool {
            return py::extract<bool>(sections.contains(section));
        };
        manager->load(reader, Manager::ExternalSourcesFromFile(filename), filterfun);
    }

    manager->validatePositions();
}

PLASK_PYTHON_API void PythonManager_load(py::object self, py::object src, py::dict defs, py::object filter=py::object()) {
    try {
        loadXpl(self, src, defs, filter);
    } catch (XMLExceptionWithCause& err) {
        err.throwPythonException();
    }
}

PythonManager::PythonManager(bool draft): Manager(draft) {
    py::object plask = py::import("plask");
    globals = py::dict(plask.attr("__xpl_globals")).copy();
}


void PythonManager::loadDefines(XMLReader& reader)
{
    std::set<std::string> parsed;
    parsed.insert("self");
    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() != "define") throw XMLUnexpectedElementException(reader, "<define>");
        std::string name = reader.requireAttribute("name");
        std::string value = reader.requireAttribute("value");
        if (name == "self") {
            throw XMLException(reader, u8"Definition name 'self' is reserved");
        };
        if (!defs.has_key(name)) {
            try {
                defs[name] = (py_eval(value, globals, defs));
            } catch (py::error_already_set&) {
                writelog(LOG_WARNING, u8"Cannot parse XML definition '{}' (storing it as string): {}",
                         name, getPythonExceptionMessage());
                PyErr_Clear();
                defs[name] = value;
            }
        } else if (parsed.find(name) != parsed.end())
            throw XMLDuplicatedElementException(reader, format("Definition of '{0}'", name));
        parsed.insert(name);
        reader.requireTagEnd();
    }
    for (py::stl_input_iterator<std::string> key(defs), keys_end; key != keys_end; ++key) {
        try {
            if (parsed.find(*key) == parsed.end())
                writelog(LOG_WARNING, "Value '{}' is not defined in the XPL file", *key);
        } catch (py::error_already_set&) {
            PyErr_Clear();
        }
    }
    globals.update(defs);
}


shared_ptr<Solver> PythonManager::loadSolver(const std::string& category, const std::string& lib, const std::string& solver_name, const std::string& name)
{
    std::string module_name = (category == "local")? lib : category + "." + lib;
    py::object module = py::import(module_name.c_str());
    py::object solver = module.attr(solver_name.c_str())(name);
    return py::extract<shared_ptr<Solver>>(solver);
}

void PythonManager::loadConnects(XMLReader& reader)
{
    while(reader.requireTagOrEnd()) {

        if (reader.getNodeName() != "connect")
            throw XMLUnexpectedElementException(reader, "<connect>", reader.getNodeName());

        std::string inkey = reader.requireAttribute("in");

        std::pair<std::string,std::string> in;
        if (inkey.find('[') == std::string::npos) {
            in = splitString2(inkey, '.');
        } else if (inkey[inkey.length()-1] == ']') {
            in = splitString2(inkey.substr(0,inkey.length()-1), '[');
        } else
            throw XMLBadAttrException(reader, "in", inkey);

        py::object solverin, receiver;

        auto in_solver = solvers.find(in.first);
        if (in_solver == solvers.end()) throw XMLException(reader, format(u8"Cannot find (in) solver with name '{0}'.", in.first));
        try { solverin = py::object(in_solver->second); }
        catch (py::error_already_set&) {
            PyErr_Clear();
            throw XMLException(reader, format(u8"Cannot convert solver '{0}' to python object.", in.first));
        }

        if (dynamic_pointer_cast<FilterCommonBase>(in_solver->second)) {
            int points = 10;
            std::string obj, pts, pth;
            std::tie(obj, pts) = splitString2(in.second, '#');
            if (pts != "") points = reader.stringInterpreter.get<int>(pts);
            std::tie(obj, pth) = splitString2(obj, '@');
            try {
                if (pth == "")
                    receiver = solverin[py::make_tuple(geometrics[obj], points)];
                else
                    receiver = solverin[py::make_tuple(geometrics[obj], pathHints[pth], points)];
            } catch (py::error_already_set&) {
                std::string msg = getPythonExceptionMessage();
                PyErr_Clear();
                throw XMLException(reader, msg);
            } catch(std::exception& err) {
                throw XMLException(reader, err.what());
            }
        } else {
            try { receiver = solverin.attr(in.second.c_str()); }
            catch (py::error_already_set&) {
                PyErr_Clear();
                throw XMLException(reader, format("Solver '{0}' does not have attribute '{1}'.", in.first, in.second));
            }
        }

        std::string outkey = reader.requireAttribute("out");

        py::object provider;

        for (std::string key: boost::tokenizer<boost::char_separator<char>>(outkey, boost::char_separator<char>("+"))) {

            boost::algorithm::trim(key);

            auto out = splitString2(key, '.');
            py::object solverout, prov;

            auto out_solver = solvers.find(out.first);
            if (out_solver == solvers.end()) {
                throw XMLException(reader, format(u8"Cannot find (out) solver with name '{0}'.", out.first));
            }
            try { solverout = py::object(out_solver->second); }
            catch (py::error_already_set&) {
                PyErr_Clear();
                throw XMLException(reader, format(u8"Cannot convert solver '{0}' to python object.", out.first));
            }

            try { prov = solverout.attr(out.second.c_str()); }
            catch (py::error_already_set&) {
                PyErr_Clear();
                throw XMLException(reader, format(u8"Solver '{0}' does not have attribute '{1}'.", out.first, out.second));
            }

            if (provider.is_none()) provider = prov;
            else provider = provider + prov;
        }

        try {
            receiver.attr("attach")(provider);
        } catch (py::error_already_set&) {
            PyErr_Clear();
            throw XMLException(reader, format(u8"Cannot connect '{0}' to '{1}'.", outkey, inkey));
        }

        reader.requireTagEnd();
    }
}

void PythonManager::loadMaterialModule(XMLReader& reader) {
    std::string name = reader.requireAttribute("name");
    std::string file;
    try {
        if (!name.empty()) {
            py::str modname(name);
            try {
#               if (PY_VERSION_HEX >= 0x03040000)
                    py::object spec = py::import("importlib.util").attr("find_spec")(modname);
                    if (spec.is_none())
                        throw Exception(format("Cannot find materials module '{}'", name));
                    file = py::extract<std::string>(spec.attr("origin"));
#               else
                    py::object loader = !py::import("importlib").attr("find_loader")(modname);
                    if (loader.is_none())
                        throw Exception(format("Cannot find materials module '{}'", name));
                    file = py::extract<std::string>(loader.attr("get_filename")());
#               endif
            } catch (py::error_already_set) {
                PyErr_Clear();
            }
            bool reload = PyDict_Contains(PyImport_GetModuleDict(), modname.ptr());
            py::object module = py::import(modname);
            if (reload) {
                PyObject* reloaded = PyImport_ReloadModule(module.ptr());
                if (!reloaded) throw py::error_already_set();
                Py_DECREF(reloaded);
            }
        }
    } catch (py::error_already_set&) {
        PyObject *value, *type, *traceback;
        PyErr_Fetch(&type, &value, &traceback);
        PyErr_NormalizeException(&type, &value, &traceback);
        py::handle<> value_h(value), type_h(type), traceback_h(py::allow_null(traceback));
        if (traceback != NULL) PyException_SetTraceback(value, traceback);
        PyErr_Clear();
        throwErrorIfNotDraft(XMLExceptionWithCause(value, reader, file.empty()?
            format("Cannot import materials module '{}'", name) :
            format("Cannot import materials module '{}' (from '{}')", name, file)));
    } catch (std::runtime_error& err) {
        throwErrorIfNotDraft(XMLException(reader, err.what()));
    }
    reader.requireTagEnd();
}

void PythonManager::loadMaterials(XMLReader& reader)
{
    while (reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "material")
            loadMaterial(reader);
        else if (reader.getNodeName() == "library")
            loadMaterialLib(reader);
        else if (reader.getNodeName() == "module")
            loadMaterialModule(reader);
        else
            throw XMLUnexpectedElementException(reader, "<material>");
    }
    py::import("plask.material").attr("update_factories")();
}



void PythonManager::export_dict(py::object self, py::object dict) {
    dict["PTH"] = self.attr("pth");
    dict["GEO"] = self.attr("geo");
    dict["MSH"] = self.attr("msh");
    dict["DEF"] = self.attr("defs");

    dict["__overrites__"] = self.attr("overrites");

    PythonManager* ths = py::extract<PythonManager*>(self);

    for (auto thesolver: ths->solvers) {
        dict[thesolver.first] = py::object(thesolver.second);
    }

    if (ths->script != "") dict["__script__"] = ths->script;
}

void PythonManager::loadScript(XMLReader &reader) {
    AssignWithBackup<XMLReader::Filter> backup(reader.contentFilter);   // do not filter script content
    unsigned line = reader.getLineNr();
    Manager::loadScript(reader);
    removeIndent(script, line);
}



template <typename T> static const std::string item_name() { return ""; }
template <> const std::string item_name<shared_ptr<GeometryObject>>() { return "geometry object"; }
template <> const std::string item_name<PathHints>() { return "path"; }
template <> const std::string item_name<shared_ptr<Solver>>() { return "solver"; }

template <typename T>
static py::object dict__getitem__(const Manager::Map<T>& self, std::string key) {
    auto found = self.find(key);
    if (found == self.end()) throw KeyError(key);
    return py::object(found->second);
}

template <typename T>
static T& dict_get_or_create(Manager::Map<T>& self, std::string key) {
    return self[key];
}

template <typename T>
static void dict__setitem__(Manager::Map<T>& self, std::string key, const T& value) {
    self[key] = value;
}

template <typename T>
static void dict__delitem__(Manager::Map<T>& self, std::string key) {
    auto found = self.find(key);
    if (found == self.end()) throw  KeyError(key);
    self.erase(found);
}

template <typename T>
static size_t dict__len__(const Manager::Map<T>& self) {
    return self.size();
}

template <typename T>
static bool dict__contains__(const Manager::Map<T>& self, const std::string& key) {
    return self.find(key) != self.end();
}

template <typename T>
static py::list dict_keys(const Manager::Map<T>& self) {
    py::list result;
    for (auto item: self) {
        result.append(item.first);
    }
    return result;
}

template <typename T>
static py::list dict_values(const Manager::Map<T>& self) {
    py::list result;
    for (auto item: self) {
        result.append(item.second);
    }
    return result;
}

template <typename T>
static py::list dict_items(const Manager::Map<T>& self) {
    py::list result;
    for (auto item: self) {
        result.append(py::make_tuple(item.first, item.second));
    }
    return result;
}

template <typename T>
static py::object dict__getattr__(const Manager::Map<T>& self, const std::string& attr) {
    std::string key = attr;
    auto found = self.find(key);
    if (found == self.end()) {
        PyErr_SetString(PyExc_AttributeError, format("No " + item_name<T>() + " with id '{0}'", attr).c_str());
        boost::python::throw_error_already_set();
    }
    return py::object(found->second);
}

template <typename T>
static void dict__setattr__(Manager::Map<T>& self, const std::string& attr, const T& value) {
    std::string key = attr;
    self[key] = value;
}

template <typename T>
static void dict__delattr__(Manager::Map<T>& self, const std::string& attr) {
    std::string key = attr;
    auto found = self.find(key);
    if (found == self.end()) throw AttributeError("no " + item_name<T>() + " with id '{0}'", attr);
    self.erase(found);
}

namespace detail {

    template <typename T>
    struct dict_iterator {
        const Manager::Map<T>& dict;
        typename Manager::Map<T>::const_iterator i;
        bool is_attr;
        static dict_iterator<T> new_iterator(const Manager::Map<T>& d) {
            return dict_iterator<T>(d, false);
        }
        static dict_iterator<T> new_attr_iterator(const Manager::Map<T>& d) {
            return dict_iterator<T>(d, true);
        }
        dict_iterator(const Manager::Map<T>& d, bool attr) : dict(d), i(d.begin()), is_attr(attr) {}
        dict_iterator(const dict_iterator<T>&) = default;
        dict_iterator<T>* __iter__() { return this; }
        std::string next() {
            if (i == dict.end()) {
                PyErr_SetString(PyExc_StopIteration, u8"No more items.");
                boost::python::throw_error_already_set();
            }
            std::string key = (i++)->first;
            if (is_attr) std::replace(key.begin(), key.end(), '_', '-');
            return key;
        }
    };

} // namespace detail

static py::list Manager_errors(const Manager& self) {
    py::list result;
    for (auto err: self.errors) {
        std::string msg = err.second;
        py::object line = (err.first != -1)? py::object(err.first) : py::object();
        result.append(py::make_tuple(msg, line));
    }
    return result;
}

template <typename T>
static inline void register_manager_dict_getters(py::class_<Manager::Map<T>, boost::noncopyable>& c) {
    c.def("__getitem__", &dict__getitem__<T>).def("__getattr__", &dict__getattr__<T>);
}

template <>
inline void register_manager_dict_getters(py::class_<Manager::Map<PathHints>, boost::noncopyable>& c) {
    c.def("__getitem__", &dict_get_or_create<PathHints>, py::return_internal_reference<>());
    c.attr("__getattr__") = c.attr("__getitem__");
}

template <typename T>
static void register_manager_dict(const std::string name) {
    py::class_<Manager::Map<T>, boost::noncopyable> c((name+"Dict").c_str(), (u8"Dictionary holding each loaded " + item_name<T>()).c_str(), py::no_init); c
        .def("__setitem__", &dict__setitem__<T>)
        .def("__delitem__", &dict__delitem__<T>)
        .def("__len__", &dict__len__<T>)
        .def("__contains__", &dict__contains__<T>)
        .def("__iter__", &detail::dict_iterator<T>::new_iterator)
        .def("keys", &dict_keys<T>)
        .def("values", &dict_values<T>)
        .def("items", &dict_items<T>)
        .def("clear", &Manager::Map< T>::clear, u8"Remove all elements from the dictionary.")
        .def("__setattr__", &dict__setattr__<T>)
        .def("__delattr__", &dict__delattr__<T>)
    ;
    register_manager_dict_getters<T>(c);

    py::delattr(py::scope(), (name+"Dict").c_str());

    py::scope scope = c;
    (void) scope;   // don't warn about unused variable scope

    py::class_<detail::dict_iterator<T>>("_Iterator", py::no_init)
        .def("__iter__", &detail::dict_iterator<T>::__iter__, py::return_self<>())
        .def("__next__", &detail::dict_iterator<T>::next)
    ;
}

struct ManagerRoots {
    Manager& manager;
    ManagerRoots(Manager& manager): manager(manager) {}
    shared_ptr<Geometry> getitem(int i) const {
        if (i < 0) i += int(manager.roots.size());
        if (i < 0 || std::size_t(i) >= manager.roots.size()) throw IndexError(u8"geometry roots index out of range");
        return manager.roots[i];
    }
    size_t len() const { return manager.roots.size(); }
    static ManagerRoots init(PythonManager& manager) { return ManagerRoots(manager); }
    void clear() { manager.roots.clear(); }
};


static void registerXplPythonGeometryObjectChanger(const std::string& name, py::object changer_class) {
    GeometryReader::registerChangerReader(name,
        [changer_class](GeometryReader& reader) -> GeometryObject::Changer* {
            auto level0 = reader.source.getLevel() - 1;
            py::object changer = changer_class(boost::ref(reader.source), boost::ref(reader.manager));
            auto current_level = reader.source.getLevel() - size_t(reader.source.getNodeType() == XMLReader::NODE_ELEMENT_END);
            for (size_t i = current_level; i > level0; --i) reader.source.requireTagEnd();
            return new PythonGeometryObjectChanger(changer);
        }
    );
}


void register_manager() {
    py::class_<Manager, shared_ptr<Manager>, boost::noncopyable>
    manager("Manager", u8"Main input manager.\n", py::no_init); manager
        .def_readonly("pth", &Manager::pathHints, u8"Dictionary of all named paths.")
        .def_readonly("geo", &Manager::geometrics, u8"Dictionary of all named geometries and geometry objects.")
        .def_readonly("msh", &Manager::meshes, u8"Dictionary of all named meshes and generators.")
        .def_readonly("solvers", &Manager::solvers, u8"Dictionary of all named solvers.")
        // .def_readonly("profiles", &Manager::profiles, u8"Dictionary of constant profiles")
        .def_readonly("script", &Manager::script, u8"Script read from XML file.")
        .def_readwrite("draft", &Manager::draft,
                       u8"Flag indicating draft mode. If True then dummy material is created if the proper\n"
                       u8"one cannot be found in the database. Also some objects do not need to have all\n"
                       u8"the attributes set, which are then filled with some reasonable defaults."
                       u8"Otherwise an exception is raised.")
        .add_property("errors", &Manager_errors, u8"List of errors that occurred during loading in draft mode.")
        .def_readonly("_scriptline", &Manager::scriptline, "First line of the script.")
        .add_property("_roots", py::make_function(ManagerRoots::init, py::with_custodian_and_ward_postcall<0,1>()),
                      u8"Root geometries.")
    ;
    manager.attr("msg") = manager.attr("msh");  // TODO: Remove in the future

    py::class_<PythonManager, shared_ptr<PythonManager>, py::bases<Manager>, boost::noncopyable>("Manager",
        u8"Main input manager.\n\n"

        u8"Object of this class provides methods to read the XML file and fetch geometry\n"
        u8"objects, paths, meshes, and generators by name. It also allows to access\n"
        u8"solvers defined in the XPL file.\n\n"

        u8"Some global PLaSK function like :func:`~plask.loadxpl` or :func:`~plask.runxpl`\n"
        u8"create a default manager and use it to load the data from XPL into the global\n"
        u8"namespace.\n\n"

        u8"Manager(materials=None, draft=False)\n\n"

        u8"Args:\n"
        u8"    draft (bool): If *True* then partially incomplete XML is accepted\n"
        u8"                  (e.g. non-existent materials are allowed).\n",

        py::init<bool>(py::arg("draft")=false))
        .def("load", &PythonManager_load,
             u8"Load data from source.\n\n"
             u8"Args:\n"
             u8"    source (string or file): File to read.\n"
             u8"        The value of this argument can be either a file name, an open file\n"
             u8"        object, or an XML string to read.\n"
             u8"    defs (dict): Dictionary of user-defined variables (which string keys).\n"
             u8"        The values of this dictionary overrides the ones given in the\n"
             u8"        :xml:tag:`<defines>` section of the XPL file.\n"
             u8"    sections (list): List of section to read.\n"
             u8"        If this parameter is given, only the listed sections of the XPL file are\n"
             u8"        read and the other ones are skipped.\n",
             (py::arg("source"), py::arg("defs")=py::dict(), py::arg("sections")=py::object()))
        .def_readonly("defs", &PythonManager::defs,
                       u8"Local defines.\n\n"
                       u8"This is a combination of the values specified in the :xml:tag:`<defines>`\n"
                       u8"section of the XPL file and the ones specified by the user in the\n"
                       u8":meth:`~plask.Manager.load` method.\n"
                      )
        .def_readonly("overrites", &PythonManager::overrites,
                      u8"Overridden local defines.\n\n"
                      u8"This is a list of local defines that have been overridden in a ``plask`` command\n"
                      u8"line or specified as a ``vars`` argument to the :meth:`~plask.Manager.load`\n"
                      u8"method.\n"
                     )
        .def("export", &PythonManager::export_dict,
             u8"Export loaded objects into a target dictionary.\n\n"
             u8"All the loaded solvers are exported with keys equal to their names and the other objects\n"
             u8"under the following keys:\n\n"
             u8"* geometries and geometry objects (:attr:`~plask.Manager.geo`): ``GEO``,\n\n"
             u8"* paths to geometry objects (:attr:`~plask.Manager.pth`): ``PTH``,\n\n"
             u8"* meshes and generators (:attr:`~plask.Manager.msh`): ``MSH``,\n\n"
             u8"* custom defines (:attr:`~plask.Manager.defs`): ``DEF``.\n",
             py::arg("target"))
        .def_readonly("globals", &PythonManager::globals,
                      u8"Global variables.\n\n"
                      u8"This is a dictionary of global variables that are available in the XPL file.\n"
                      u8"By default it contains the following variables:\n\n"
                      u8"* ``PTH``: dictionary of all named paths,\n\n"
                      u8"* ``GEO``: dictionary of all named geometries and geometry objects,\n\n"
                      u8"* ``MSH``: dictionary of all named meshes and generators,\n\n"
                      u8"* ``DEF``: dictionary of all local defines,\n\n"
                      u8"* ``self``: the manager object itself.\n"
                     )
    ;

    register_manager_dict<shared_ptr<GeometryObject>>("GeometryObjects");
    register_manager_dict<PathHints>("PathHints");
    register_manager_dict<shared_ptr<MeshBase>>("Meshes");
    register_manager_dict<shared_ptr<Solver>>("Solvers");

    py::scope scope(manager);
    (void) scope;   // don't warn about unused variable scope
    py::class_<ManagerRoots>("_Roots", py::no_init)
        .def("__getitem__", &ManagerRoots::getitem)
        .def("__len__", &ManagerRoots::len)
        .def("clear",  &ManagerRoots::clear)
    ;

    py::def("_register_geometry_changer", &registerXplPythonGeometryObjectChanger, (py::arg("name"), "changer_class"));
}

}} // namespace plask::python
