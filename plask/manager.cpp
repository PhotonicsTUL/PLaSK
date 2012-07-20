#include "manager.h"

#include "utils/stl.h"
#include "geometry/reader.h"

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

void Manager::loadFromXMLStream(std::istream &input, const MaterialsDB& materialsDB) {
    XMLReader reader(input);
    loadFromReader(reader, materialsDB);
}

void Manager::loadFromXMLStream(std::istream &input, const GeometryReader::MaterialsSource &materialsSource) {
    XMLReader reader(input);
    loadFromReader(reader, materialsSource);
}

void Manager::loadFromXMLString(const std::string &input_XML_str, const MaterialsDB& materialsDB) {
    std::istringstream stream(input_XML_str);
    loadFromXMLStream(stream, materialsDB);
}

void Manager::loadFromXMLString(const std::string &input_XML_str, const GeometryReader::MaterialsSource &materialsSource) {
    std::istringstream stream(input_XML_str);
    loadFromXMLStream(stream, materialsSource);
}

//TODO skip geometry elements ends (why?)
void Manager::loadFromFile(const std::string &fileName, const MaterialsDB& materialsDB) {
    XMLReader reader(fileName.c_str());
    loadFromReader(reader, materialsDB);
}

void Manager::loadFromFile(const std::string &fileName, const GeometryReader::MaterialsSource &materialsSource) {
    XMLReader reader(fileName.c_str());
    loadFromReader(reader, materialsSource);
}


void Manager::loadGeometry(GeometryReader& greader) {
    if (greader.source.getNodeType() != XMLReader::NODE_ELEMENT || greader.source.getNodeName() != std::string("geometry"))
        throw XMLUnexpectedElementException("<geometry> tag");
    GeometryReader::ReadAxisNames read_axis_tag(greader);
    while(greader.source.read()) {
        switch (greader.source.getNodeType()) {
            case XMLReader::NODE_ELEMENT_END:
                if (greader.source.getNodeName() != std::string("geometry"))
                    throw XMLUnexpectedElementException("end of \"geometry\" tag");
                return;  //end of geometry
            case XMLReader::NODE_ELEMENT:
                roots.push_back(greader.readGeometry());
                break;
            case XMLReader::NODE_COMMENT:
                break;   //just ignore
            default:
                throw XMLUnexpectedElementException("begin of geometry element tag or </geometry>");
        }
    }
    throw XMLUnexpectedEndException();
}


void Manager::loadGrids(XMLReader &reader)
{
    if (reader.getNodeType() != XMLReader::NODE_ELEMENT || reader.getNodeName() != std::string("grids"))
        throw XMLUnexpectedElementException("<grids> tag");
    while(reader.read()) {
        switch (reader.getNodeType()) {
            case XMLReader::NODE_ELEMENT_END:
                if (reader.getNodeName() != std::string("grids"))
                    throw XMLUnexpectedElementException("end of \"grids\" tag");
                return;  //end of grids
            case XMLReader::NODE_ELEMENT:
                if (reader.getNodeName() == "mesh") {
                    std::string type = reader.requireAttribute("type");
                    std::string name = reader.requireAttribute("name");
                    if (meshes.find(name) != meshes.end() || generators.find(name) != generators.end())
                        throw NotUniqueElementException("Duplicated mesh name");
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
                    throw XMLUnexpectedElementException("<mesh...>, <generator...>, or </grids>");
                break;
            case XMLReader::NODE_COMMENT:
                break;   //just ignore
            default:
                throw XMLUnexpectedElementException("<mesh...>, <generator...>, or </grids>");
        }
    }
    throw XMLUnexpectedEndException();
}

void Manager::loadModules(XMLReader &reader)
{
    /*TODO
    - read module name
    - load module library from file
    - call module = createModule(...) from library
    - call module.loadConfiguration(reader)
    - add module to modules map
    */
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

    if (reader.getNodeName() == "modules") {
        loadModules(reader);
        if (!reader.read()) return;
    }
}

} // namespace plask
