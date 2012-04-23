#include "manager.h"

#include "../utils/stl.h"
#include "reader.h"

namespace plask {

PathHints& GeometryManager::requirePathHints(const std::string& path_hints_name) {
    auto result_it = pathHints.find(path_hints_name);
    if (result_it == pathHints.end()) throw Exception("No such path hints: %1%", path_hints_name);
    return result_it->second;
}

const PathHints& GeometryManager::requirePathHints(const std::string& path_hints_name) const {
    auto result_it = pathHints.find(path_hints_name);
    if (result_it == pathHints.end()) throw Exception("No such path hints: %1%", path_hints_name);
    return result_it->second;
}

shared_ptr<GeometryElement> GeometryManager::getElement(const std::string &name) const {
    auto result_it = namedElements.find(name);
    if (result_it == namedElements.end()) return shared_ptr<GeometryElement>();
    auto result = result_it->second.lock();
    if (!result) const_cast<GeometryManager*>(this)->namedElements.erase(name);
    return result;

    /*auto result_it = namedElements.find(name);
    return result_it != namedElements.end() ? result_it->second.lock() : shared_ptr<GeometryElement>();*/
}

shared_ptr<GeometryElement> GeometryManager::requireElement(const std::string &name) const {
    shared_ptr<GeometryElement> result = getElement(name);
    if (!result) throw NoSuchGeometryElement(name);
    return result;
}

shared_ptr<CalculationSpace> GeometryManager::getCalculationSpace(const std::string& name) const {
    auto result_it = calculationSpaces.find(name);
    return result_it == calculationSpaces.end() ? shared_ptr<CalculationSpace>() : result_it->second;
}

void GeometryManager::loadGeometryFromReader(GeometryReader& reader, const MaterialsDB& materialsDB) {
    if (reader.source.getNodeType() != irr::io::EXN_ELEMENT || reader.source.getNodeName() != std::string("geometry"))
        throw XMLUnexpectedElementException("<geometry> tag");
    GeometryReader::ReadAxisNames read_axis_tag(reader);
    while(reader.source.read()) {
        switch (reader.source.getNodeType()) {
            case irr::io::EXN_ELEMENT_END:
                if (reader.source.getNodeName() != std::string("geometry"))
                    throw XMLUnexpectedElementException("end of \"geometry\" tag");
                return;  //end of geometry
            case irr::io::EXN_ELEMENT:
                roots.push_back(reader.readElement());
                break;
            case irr::io::EXN_COMMENT:
                break;   //just ignore
            default:
                throw XMLUnexpectedElementException("begin of geometry element tag or </geometry>");
        }
    }
    throw XMLUnexpectedEndException();
}

void GeometryManager::loadSpacesFromReader(GeometryReader& reader) {
    if (reader.source.getNodeType() != irr::io::EXN_ELEMENT || reader.source.getNodeName() != std::string("spaces"))
        return; //space are optional
    GeometryReader::ReadAxisNames read_axis_tag(reader);
    while(reader.source.read()) {
        switch (reader.source.getNodeType()) {
            case irr::io::EXN_ELEMENT_END:
                if (reader.source.getNodeName() != std::string("spaces"))
                    throw XMLUnexpectedElementException("end of \"spaces\" tag");
                return;  //end of spaces
            case irr::io::EXN_ELEMENT:
                //roots.push_back(reader.readSpace());
                break;
            case irr::io::EXN_COMMENT:
                break;   //just ignore
            default:
                throw XMLUnexpectedElementException("begin of space element tag or </spaces>");
        }
    }
    throw XMLUnexpectedEndException();
}

void GeometryManager::loadFromReader(XMLReader &XMLreader, const MaterialsDB& materialsDB) {
    GeometryReader reader(*this, XMLreader, materialsDB);
    loadGeometryFromReader(reader, materialsDB);
}

void GeometryManager::loadFromXMLStream(std::istream &input, const MaterialsDB& materialsDB) {
    XML::StreamReaderCallback cb(input);
    std::unique_ptr< XMLReader > reader(irr::io::createIrrXMLReader(&cb));
    XML::requireNext(*reader);
    loadFromReader(*reader, materialsDB);
}

void GeometryManager::loadFromXMLString(const std::string &input_XML_str, const MaterialsDB& materialsDB) {
    std::istringstream stream(input_XML_str);
    loadFromXMLStream(stream, materialsDB);
}

//TODO skip geometry elements ends
void GeometryManager::loadFromFile(const std::string &fileName, const MaterialsDB& materialsDB) {
    std::unique_ptr< XMLReader > reader(irr::io::createIrrXMLReader(fileName.c_str()));
    if (reader == nullptr) throw Exception("Can't read from file \"%1%\".", fileName);
    XML::requireNext(*reader);
    loadFromReader(*reader, materialsDB);
}

}	// namespace plask
