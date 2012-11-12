#include <fstream>

#include "manager.h"

#include "utils/stl.h"
#include "geometry/reader.h"

#include "utils/dynlib/manager.h"

#include "utils/system.h"

namespace plask {

void Manager::ExternalSourcesFromFile::operator ()(Manager &manager, const MaterialsSource& materialsSource, const std::string &url, const std::string &section) {
    boost::filesystem::path url_path(url);
    if (url_path.is_relative()) {
        if (originalFileName.empty())
            throw Exception("Error while reading section \"%1%\": relative path name \"%2%\" is not supported.", section, url);
        url_path = originalFileName;
        url_path.remove_filename();
        url_path /= url;
    }
    if (hasCircularRef(url_path, section))
        throw Exception("Error while reading section \"%1%\": circular reference was detected.", section);
    XMLReader reader(url_path.string().c_str());
    manager.loadSection(reader, section, materialsSource, ExternalSourcesFromFile(url_path, section, this));
}

PathHints& Manager::requirePathHints(const std::string& path_hints_name) {
    auto result_it = pathHints.find(path_hints_name);
    if (result_it == pathHints.end()) throw NoSuchPath(path_hints_name);
    return result_it->second;
}

bool Manager::tryLoadFromExternal(XMLReader& reader, const MaterialsSource& materialsSource, const Manager::LoadFunCallbackT& load_from) {
    boost::optional<std::string> from_attr = reader.getAttribute("from");
    if (!from_attr) return false;
    load_from(*this, materialsSource, *from_attr, reader.getNodeName());
    return true;

    /*std::string section_to_load = reader.getNodeName();
    std::pair< XMLReader, std::unique_ptr<LoadFunCallbackT> > new_loader = load_from.get(*from_attr);
    load(new_loader.first, materialsSource, *new_loader.second, [&](const std::string& section_name) -> bool { return section_name == section_to_load; });
    return true;*/
}

const PathHints& Manager::requirePathHints(const std::string& path_hints_name) const {
    auto result_it = pathHints.find(path_hints_name);
    if (result_it == pathHints.end()) throw NoSuchPath(path_hints_name);
    return result_it->second;
}

shared_ptr<GeometryObject> Manager::getGeometryObject(const std::string &name) const {
    auto result_it = namedObjects.find(name);
    if (result_it == namedObjects.end()) return shared_ptr<GeometryObject>();
    // auto result = result_it->second.lock();
    // if (!result) const_cast<Manager*>(this)->namedObjects.erase(name);
    // return result;
    return result_it->second;
    /*auto result_it = namedObjects.find(name);
    return result_it != namedObjects.end() ? result_it->second.lock() : shared_ptr<GeometryObject>();*/
}

shared_ptr<GeometryObject> Manager::requireGeometryObject(const std::string &name) const {
    shared_ptr<GeometryObject> result = getGeometryObject(name);
    if (!result) throw NoSuchGeometryObject(name);
    return result;
}

shared_ptr<Geometry> Manager::getGeometry(const std::string& name) const {
    auto result_it = geometries.find(name);
    return result_it == geometries.end() ? shared_ptr<Geometry>() : result_it->second;
}

shared_ptr<Mesh> Manager::getMesh(const std::string& name) const {
    auto result_it = meshes.find(name);
    return result_it == meshes.end() ? shared_ptr<Mesh>() : result_it->second;
}

void Manager::loadFromReader(XMLReader &reader, const MaterialsDB& materialsDB, const LoadFunCallbackT& load_from_cb) {
    load(reader, materialsDB.toSource(), load_from_cb);
}

void Manager::loadFromReader(XMLReader &reader, const MaterialsSource &materialsSource, const LoadFunCallbackT& load_from_cb) {
    load(reader, materialsSource, load_from_cb);
}

void Manager::loadFromStream(std::istream* input, const MaterialsDB& materialsDB, const LoadFunCallbackT& load_from_cb) {
    XMLReader reader(input);
    loadFromReader(reader, materialsDB, load_from_cb);
}

void Manager::loadFromStream(std::istream* input, const MaterialsSource &materialsSource, const LoadFunCallbackT& load_from_cb) {
    XMLReader reader(input);
    loadFromReader(reader, materialsSource, load_from_cb);
}

void Manager::loadFromXMLString(const std::string &input_XML_str, const MaterialsDB& materialsDB, const LoadFunCallbackT& load_from_cb) {
    loadFromStream(new std::istringstream(input_XML_str), materialsDB, load_from_cb);
}

void Manager::loadFromXMLString(const std::string &input_XML_str, const MaterialsSource &materialsSource, const LoadFunCallbackT& load_from_cb) {
    loadFromStream(new std::istringstream(input_XML_str), materialsSource, load_from_cb);
}

void Manager::loadFromFile(const std::string &fileName, const MaterialsDB& materialsDB) {
    XMLReader reader(fileName.c_str());
    loadFromReader(reader, materialsDB, ExternalSourcesFromFile(fileName));
}

void Manager::loadFromFile(const std::string &fileName, const MaterialsSource &materialsSource) {
    XMLReader reader(fileName.c_str());
    loadFromReader(reader, materialsSource, ExternalSourcesFromFile(fileName));
}

void Manager::loadFromFILE(FILE* file, const MaterialsDB& materialsDB, const LoadFunCallbackT& load_from_cb) {
    XMLReader reader(file);
    loadFromReader(reader, materialsDB, load_from_cb);
}

void Manager::loadFromFILE(FILE* file, const MaterialsSource &materialsSource, const LoadFunCallbackT& load_from_cb) {
    XMLReader reader(file);
    loadFromReader(reader, materialsSource, load_from_cb);
}


void Manager::loadGeometry(GeometryReader& greader) {
    if (greader.source.getNodeType() != XMLReader::NODE_ELEMENT || greader.source.getNodeName() != std::string("geometry"))
        throw XMLUnexpectedElementException(greader.source, "<geometry>");
    GeometryReader::ReadAxisNames read_axis_tag(greader);
    while(greader.source.requireTagOrEnd())
        roots.push_back(greader.readGeometry());
}

void Manager::loadMaterials(XMLReader& reader, const MaterialsSource& materialsSource)
{
    writelog(LOG_ERROR, "Loading materials from C++ not implemented. Ignoring XML section <materials>");
    reader.gotoEndOfCurrentTag();
}

void Manager::loadGrids(XMLReader &reader)
{
    if (reader.getNodeType() != XMLReader::NODE_ELEMENT || reader.getNodeName() != std::string("grids"))
        throw XMLUnexpectedElementException(reader, "<grids>");
    while(reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "mesh") {
            std::string type = reader.requireAttribute("type");
            std::string name = reader.requireAttribute("name");
            BadId::throwIfBad("mesh", name, '-');
            if (meshes.find(name) != meshes.end() || generators.find(name) != generators.end())
                throw NamesConflictException("Mesh or mesh generator", name);
            shared_ptr<Mesh> mesh = RegisterMeshReader::getReader(type)(reader);
            meshes[name] = mesh;
        } else if (reader.getNodeName() == "generator") {
            std::string type = reader.requireAttribute("type");
            std::string method = reader.requireAttribute("method");
            std::string name = reader.requireAttribute("name");
            BadId::throwIfBad("generator", name, '-');
            std::string key = type + "." + method;
            if (meshes.find(name) != meshes.end() || generators.find(name) != generators.end())
                throw NamesConflictException("Mesh or mesh generator", name);
            shared_ptr<MeshGenerator> generator = RegisterMeshGeneratorReader::getReader(key)(reader, *this);
            generators[name] = generator;
        } else
            throw XMLUnexpectedElementException(reader, "<mesh...>, <generator...>, or </grids>");
    }
}

shared_ptr<Solver> Manager::loadSolver(const std::string &category, const std::string &lib, const std::string &solver_name, const std::string& name) {
    std::string lib_file_name = plaskSolversPath(category);
    auto found = solvers.find(name);
    if (found == solvers.end())
        throw Exception("In C++ solvers ('%1%' in this case) must be created and added to Manager::solvers manually before reading XML.", name);
    solvers.erase(found); // this is necessary so we don't have name conflicts — the solver will be added back to the map by loadSolvers
    return found->second;
}

void Manager::loadSolvers(XMLReader& reader) {
    if (reader.getNodeType() != XMLReader::NODE_ELEMENT || reader.getNodeName() != std::string("solvers"))
        throw XMLUnexpectedElementException(reader, "<solvers>");
    while (reader.requireTagOrEnd()) {
        const std::string name = reader.requireAttribute("name");
        BadId::throwIfBad("solver", name);
        boost::optional<std::string> lib = reader.getAttribute("lib");
        const std::string solver_name = reader.requireAttribute("solver");
        if (!lib) {
            auto libs = global_solver_names[reader.getNodeName()];
            if (libs.empty()) { // read lib index from file
                std::string file_name = plaskSolversPath(reader.getNodeName()) + "solvers.lst";
                try {
                    std::ifstream list_file(file_name);
                    if (list_file.is_open())
                        while (!list_file.eof()) {
                            std::string line, lib, cls;
                            list_file >> line;
                            boost::algorithm::trim(line);
                            if (line == "") continue;
                            std::tie(lib, cls) = splitString2(line, '.');
                            if (cls == "") writelog(LOG_ERROR, "Wrong format of '%1%' file", file_name);
                            else libs[cls] = lib;
                        }
                } catch (...) {}
            }
            lib.reset(libs[solver_name]);
        }
        shared_ptr<Solver> solver = loadSolver(reader.getNodeName(), *lib, solver_name, name);
        solver->loadConfiguration(reader, *this);
        if (!this->solvers.insert(std::make_pair(name, solver)).second)
            throw NamesConflictException("Solver", name);
    }
}

void Manager::loadConnects(XMLReader& reader)
{
    writelog(LOG_ERROR, "Loading interconnects only possible from Python interface. Ignoring XML section <connects>");
    reader.gotoEndOfCurrentTag();
}

void Manager::loadScript(XMLReader& reader)
{
    if (reader.getNodeType() != XMLReader::NODE_ELEMENT || reader.getNodeName() != std::string("script"))
        throw XMLUnexpectedElementException(reader, "<script>");
    scriptline = reader.getLineNr();
    script = reader.requireTextInCurrentTag();
    size_t start;
    for (start = 0; script[start] != '\n'; ++start) {
        if (!std::isspace(script[start]))
            throw XMLException(reader, "Script must begin from new line after <script>");
    }
    script = script.substr(start+1);
}


/*static inline MaterialsDB& getMaterialsDBfromSource(const Manager::MaterialsSource& materialsSource) {
    const GeometryReader::MaterialsDBSource* src = materialsSource.target<const GeometryReader::MaterialsDBSource>();
    return src ? const_cast<MaterialsDB&>(src->materialsDB) : MaterialsDB::getDefault();
}*/

void Manager::load(XMLReader& reader, const MaterialsSource& materialsSource,
                   const LoadFunCallbackT& load_from,
                   const std::function<bool(const std::string& section_name)>& section_filter)
{
    reader.requireTag(TAG_NAME_ROOT);
    reader.removeAlienNamespaceAttr();  //eventual schema decl. will be removed
    reader.requireTag();

    if (reader.getNodeName() == TAG_NAME_MATERIALS) {
        if (section_filter(TAG_NAME_MATERIALS)) {
            if (!tryLoadFromExternal(reader, materialsSource, load_from)) loadMaterials(reader, materialsSource);
        } else
            reader.gotoEndOfCurrentTag();
        if (!reader.requireTagOrEnd()) return;
    }

    if (reader.getNodeName() == TAG_NAME_GEOMETRY) {
        if (section_filter(TAG_NAME_GEOMETRY)) {
            if (!tryLoadFromExternal(reader, materialsSource, load_from)) {
                GeometryReader greader(*this, reader, materialsSource);
                loadGeometry(greader);
            }
        } else
            reader.gotoEndOfCurrentTag();
        if (!reader.requireTagOrEnd()) return;
    }

    if (reader.getNodeName() == TAG_NAME_GRIDS) {
        if (section_filter(TAG_NAME_GRIDS)) {
            if (!tryLoadFromExternal(reader, materialsSource, load_from)) loadGrids(reader);
        } else
            reader.gotoEndOfCurrentTag();
        if (!reader.requireTagOrEnd()) return;
    }

    if (reader.getNodeName() == TAG_NAME_SOVERS) {
        if (section_filter(TAG_NAME_SOVERS)) {
            if (!tryLoadFromExternal(reader, materialsSource, load_from)) loadSolvers(reader);
        } else
            reader.gotoEndOfCurrentTag();
        if (!reader.requireTagOrEnd()) return;
    }

    if (reader.getNodeName() == TAG_NAME_CONNECTS) {
        if (section_filter(TAG_NAME_CONNECTS)) {
            if (!tryLoadFromExternal(reader, materialsSource, load_from)) loadConnects(reader);
        } else
            reader.gotoEndOfCurrentTag();
        if (!reader.requireTagOrEnd()) return;
    }

    if (reader.getNodeName() == TAG_NAME_SCRIPT) {
        if (section_filter(TAG_NAME_SCRIPT)) {
            if (!tryLoadFromExternal(reader, materialsSource, load_from)) loadScript(reader);
        } else
            reader.gotoEndOfCurrentTag();
        if (!reader.requireTagOrEnd()) return;
    }
}


} // namespace plask
