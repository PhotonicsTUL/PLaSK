#include "manager.h"

#include "utils/stl.h"
#include "geometry/reader.h"

#include "utils/dynlib/manager.h"

#include "utils/system.h"

namespace plask {

PathHints& Manager::requirePathHints(const std::string& path_hints_name) {
    auto result_it = pathHints.find(path_hints_name);
    if (result_it == pathHints.end()) throw NoSuchPath(path_hints_name);
    return result_it->second;
}

template <typename MaterialsSource>
bool Manager::tryLoadFromExternal(XMLReader& reader, const MaterialsSource& materialsSource, const Manager::LoadFunCallbackT& load_from) {
    boost::optional<std::string> from_attr = reader.getAttribute("from");
    if (!from_attr) return false;
    load_from(*from_attr, reader.getNodeName());
    return true;

    /*std::string section_to_load = reader.getNodeName();
    std::pair< XMLReader, std::unique_ptr<LoadFunCallbackT> > new_loader = load_from.get(*from_attr);
    load(new_loader.first, materialsSource, *new_loader.second, [&](const std::string& section_name) -> bool { return section_name == section_to_load; });
    return true;*/
}

shared_ptr<Solver> Manager::loadSolver(const std::string &category, const std::string &lib, const std::string &solver_name, const std::string& name) {
    std::string lib_file_name = plaskSolversPath(category);
    lib_file_name += "lib";
    lib_file_name += lib;
    lib_file_name += DynamicLibrary::DEFAULT_EXTENSION;
    return shared_ptr<Solver>(
                DynamicLibraries::defaultLoad(lib_file_name)
                    .requireSymbol<solver_construct_f*>(solver_name + SOLVER_CONSTRUCT_FUNCTION_SUFFIX)(name)
                );
}

const PathHints& Manager::requirePathHints(const std::string& path_hints_name) const {
    auto result_it = pathHints.find(path_hints_name);
    if (result_it == pathHints.end()) throw NoSuchPath(path_hints_name);
    return result_it->second;
}

shared_ptr<GeometryElement> Manager::getGeometryElement(const std::string &name) const {
    auto result_it = namedElements.find(name);
    if (result_it == namedElements.end()) return shared_ptr<GeometryElement>();
    // auto result = result_it->second.lock();
    // if (!result) const_cast<Manager*>(this)->namedElements.erase(name);
    // return result;
    return result_it->second;
    /*auto result_it = namedElements.find(name);
    return result_it != namedElements.end() ? result_it->second.lock() : shared_ptr<GeometryElement>();*/
}

shared_ptr<GeometryElement> Manager::requireGeometryElement(const std::string &name) const {
    shared_ptr<GeometryElement> result = getGeometryElement(name);
    if (!result) throw NoSuchGeometryElement(name);
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

void Manager::loadFromReader(XMLReader &reader, const MaterialsDB& materialsDB) {
    load(reader, materialsDB);
}

void Manager::loadFromReader(XMLReader &reader, const GeometryReader::MaterialsSource &materialsSource) {
    load(reader, materialsSource);
}

void Manager::loadFromStream(std::istream &input, const MaterialsDB& materialsDB) {
    XMLReader reader(input);
    loadFromReader(reader, materialsDB);
}

void Manager::loadFromStream(std::istream &input, const GeometryReader::MaterialsSource &materialsSource) {
    XMLReader reader(input);
    loadFromReader(reader, materialsSource);
}

void Manager::loadFromXMLString(const std::string &input_XML_str, const MaterialsDB& materialsDB) {
    std::istringstream stream(input_XML_str);
    loadFromStream(stream, materialsDB);
}

void Manager::loadFromXMLString(const std::string &input_XML_str, const GeometryReader::MaterialsSource &materialsSource) {
    std::istringstream stream(input_XML_str);
    loadFromStream(stream, materialsSource);
}

void Manager::loadFromFile(const std::string &fileName, const MaterialsDB& materialsDB) {
    XMLReader reader(fileName.c_str());
    loadFromReader(reader, materialsDB);
}

void Manager::loadFromFile(const std::string &fileName, const GeometryReader::MaterialsSource &materialsSource) {
    XMLReader reader(fileName.c_str());
    loadFromReader(reader, materialsSource);
}

void Manager::loadFromFILE(FILE* file, const MaterialsDB& materialsDB) {
    XMLReader reader(file);
    loadFromReader(reader, materialsDB);
}

void Manager::loadFromFILE(FILE* file, const GeometryReader::MaterialsSource &materialsSource) {
    XMLReader reader(file);
    loadFromReader(reader, materialsSource);
}


void Manager::loadGeometry(GeometryReader& greader) {
    if (greader.source.getNodeType() != XMLReader::NODE_ELEMENT || greader.source.getNodeName() != std::string("geometry"))
        throw XMLUnexpectedElementException(greader.source, "<geometry>");
    GeometryReader::ReadAxisNames read_axis_tag(greader);
    while(greader.source.requireTagOrEnd())
        roots.push_back(greader.readGeometry());
}

void Manager::loadMaterials(XMLReader& reader, MaterialsDB& materialsDB)
{
    throw NotImplemented("Loading materials from C++ not implemented");
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

void Manager::loadSolvers(XMLReader& reader) {
    if (reader.getNodeType() != XMLReader::NODE_ELEMENT || reader.getNodeName() != std::string("solvers"))
        throw XMLUnexpectedElementException(reader, "<solvers>");
    while (reader.requireTagOrEnd()) {
        const std::string name = reader.requireAttribute("name");
        BadId::throwIfBad("solver", name);
        const boost::optional<std::string> lib = reader.getAttribute("lib");
        const std::string solver_name = reader.requireAttribute("solver");
        shared_ptr<Solver> solver = loadSolver(reader.getNodeName(), lib ? *lib : solver_name, solver_name, name);
        solver->loadConfiguration(reader, *this);
        if (!this->solvers.insert(std::make_pair(name, solver)).second)
            throw NamesConflictException("Solver", name);
    }
}

void Manager::loadConnects(XMLReader& reader)
{
    throw NotImplemented("Loading interconnects only possible from Python interface.");
}

void Manager::loadScript(XMLReader& reader)
{
    if (reader.getNodeType() != XMLReader::NODE_ELEMENT || reader.getNodeName() != std::string("script"))
        throw XMLUnexpectedElementException(reader, "<script>");
    script = reader.requireText();
    size_t start;
    for (start = 0; script[start] != '\n'; ++start) {
        if (!std::isspace(script[start]))
            throw XMLException(reader, "Script must begin from new line after <script>");
    }
    script = script.substr(start+1);
    reader.requireTagEnd();
}


template <typename MaterialsSource>
static inline MaterialsDB& getMaterialsDBfromSource(const MaterialsSource& materialsSource) {
    return MaterialsDB::getDefault();
}

template <>
inline MaterialsDB& getMaterialsDBfromSource<MaterialsDB>(const MaterialsDB& materialsSource) {
    return const_cast<MaterialsDB&>(materialsSource);
}

template <typename MaterialsSource>
void Manager::load(XMLReader& reader, const MaterialsSource& materialsSource,
                   const LoadFunCallbackT& load_from,
                   const std::function<bool(const std::string& section_name)>& section_filter)
{
    reader.requireTag(TAG_NAME_ROOT);
    reader.requireTag();

    if (reader.getNodeName() == TAG_NAME_MATERIALS) {
        if (section_filter(TAG_NAME_MATERIALS)) {
            if (!tryLoadFromExternal(reader, materialsSource, load_from)) loadMaterials(reader, getMaterialsDBfromSource(materialsSource));
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
