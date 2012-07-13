#include "manager.h"

#include "utils/stl.h"
#include "geometry/reader.h"

namespace plask {

PathHints& Manager::requirePathHints(const std::string& path_hints_name) {
    auto result_it = pathHints.find(path_hints_name);
    if (result_it == pathHints.end()) throw Exception("No such path hints: %1%", path_hints_name);
    return result_it->second;
}

const PathHints& Manager::requirePathHints(const std::string& path_hints_name) const {
    auto result_it = pathHints.find(path_hints_name);
    if (result_it == pathHints.end()) throw Exception("No such path hints: %1%", path_hints_name);
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

void Manager::loadGeometryFromReader(GeometryReader& reader) {
    if (reader.source.getNodeType() != XMLReader::NODE_ELEMENT || reader.source.getNodeName() != std::string("geometry"))
        throw XMLUnexpectedElementException("<geometry> tag");
    GeometryReader::ReadAxisNames read_axis_tag(reader);
    while(reader.source.read()) {
        switch (reader.source.getNodeType()) {
            case XMLReader::NODE_ELEMENT_END:
                if (reader.source.getNodeName() != std::string("geometry"))
                    throw XMLUnexpectedElementException("end of \"geometry\" tag");
                return;  //end of geometry
            case XMLReader::NODE_ELEMENT:
                //roots.push_back(reader.readElement());
                reader.readGeometry();
                break;
            case XMLReader::NODE_COMMENT:
                break;   //just ignore
            default:
                throw XMLUnexpectedElementException("begin of geometry element tag or </geometry>");
        }
    }
    throw XMLUnexpectedEndException();
}

void Manager::loadGeometryFromReader(XMLReader &XMLreader, const MaterialsDB& materialsDB) {
    GeometryReader reader(*this, XMLreader, materialsDB);
    loadGeometryFromReader(reader);
}

void Manager::loadGeometryFromReader(XMLReader &XMLreader, const GeometryReader::MaterialsSource &materialsSource) {
    GeometryReader reader(*this, XMLreader, materialsSource);
    loadGeometryFromReader(reader);
}

void Manager::loadGeometryFromXMLStream(std::istream &input, const MaterialsDB& materialsDB) {
    XMLReader reader(input);
    reader.requireNext();
    loadGeometryFromReader(reader, materialsDB);
}

void Manager::loadGeometryFromXMLStream(std::istream &input, const GeometryReader::MaterialsSource &materialsSource) {
    XMLReader reader(input);
    reader.requireNext();
    loadGeometryFromReader(reader, materialsSource);
}

void Manager::loadGeometryFromXMLString(const std::string &input_XML_str, const MaterialsDB& materialsDB) {
    std::istringstream stream(input_XML_str);
    loadGeometryFromXMLStream(stream, materialsDB);
}

void Manager::loadGeometryFromXMLString(const std::string &input_XML_str, const GeometryReader::MaterialsSource &materialsSource) {
    std::istringstream stream(input_XML_str);
    loadGeometryFromXMLStream(stream, materialsSource);
}

//TODO skip geometry elements ends
void Manager::loadGeometryFromFile(const std::string &fileName, const MaterialsDB& materialsDB) {
    XMLReader reader(fileName.c_str());
    reader.requireNext();
    loadGeometryFromReader(reader, materialsDB);
}

void Manager::loadGeometryFromFile(const std::string &fileName, const GeometryReader::MaterialsSource &materialsSource) {
    XMLReader reader(fileName.c_str());
    reader.requireNext();
    loadGeometryFromReader(reader, materialsSource);
}

}	// namespace plask
