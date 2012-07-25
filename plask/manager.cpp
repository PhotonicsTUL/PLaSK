#include "manager.h"

#include "utils/stl.h"
#include "geometry/reader.h"

#include "utils/dynlib/manager.h"

namespace plask {

PathHints& Manager::requirePathHints(const std::string& path_hints_name) {
    auto result_it = pathHints.find(path_hints_name);
    if (result_it == pathHints.end()) throw NoSuchPath(path_hints_name);
    return result_it->second;
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


void Manager::loadGrids(XMLReader &reader)
{
    if (reader.getNodeType() != XMLReader::NODE_ELEMENT || reader.getNodeName() != std::string("grids"))
        throw XMLUnexpectedElementException(reader, "<grids>");
    while(reader.requireTagOrEnd()) {
        if (reader.getNodeName() == "mesh") {
            std::string type = reader.requireAttribute("type");
            std::string name = reader.requireAttribute("name");
            if (meshes.find(name) != meshes.end() || generators.find(name) != generators.end())
                throw NamesConflictException("Mesh or mesh generator", name);
            shared_ptr<Mesh> mesh = RegisterMeshReader::getReader(type)(reader);
            meshes[name] = mesh;
        } else if (reader.getNodeName() == "generator") {
            std::string type = reader.requireAttribute("type");
            std::string method = reader.requireAttribute("method");
            std::string name = reader.requireAttribute("name");
            std::string key = type + "." + method;
            if (meshes.find(name) != meshes.end() || generators.find(name) != generators.end())
                throw NamesConflictException("Mesh or mesh generator", name);
            shared_ptr<MeshGenerator> generator = RegisterMeshGeneratorReader::getReader(key)(reader, *this);
            generators[name] = generator;
        } else
            throw XMLUnexpectedElementException(reader, "<mesh...>, <generator...>, or </grids>");
    }
}

void Manager::loadSolvers(GeometryReader& greader) {
    if (greader.source.getNodeType() != XMLReader::NODE_ELEMENT || greader.source.getNodeName() != std::string("solvers"))
        throw XMLUnexpectedElementException(greader.source, "<solvers>");
    while (greader.source.requireTagOrEnd()) {
        std::string name = greader.source.requireAttribute("name");
        shared_ptr<Solver> solver(DynamicLibraries::defaultLoad(greader.source.getNodeName()).requireSymbol<solver_construct_f*>(SOLVER_CONSTRUCT_FUNCTION_NAME)());
        solver->loadConfiguration(greader);
        if (!greader.manager.solvers.insert(std::make_pair(name, solver)).second)
            throw NamesConflictException("Solver", name);
    }
}

template <typename MaterialsSource>
void Manager::load(XMLReader& reader, const MaterialsSource& materialsSource)
{
    //TODO: maybe some external tag or XML header?

    reader.requireTag();
    if (reader.getNodeName() == "geometry") {
        GeometryReader greader(*this, reader, materialsSource);
        loadGeometry(greader);
        if (!reader.read()) return;
    }

    if (reader.getNodeName() == "grids") {
        loadGrids(reader);
        if (!reader.read()) return;
    }

    if (reader.getNodeName() == "solvers") {
        GeometryReader greader(*this, reader, materialsSource);
        loadSolvers(greader);
        if (!reader.read()) return;
    }
}

} // namespace plask
