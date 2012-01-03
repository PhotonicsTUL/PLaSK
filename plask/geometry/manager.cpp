#include "manager.h"

#include "../utils/stl.h"

namespace plask {

std::map<std::string, GeometryManager::element_read_f*>& GeometryManager::elementReaders() {
    static std::map<std::string, GeometryManager::element_read_f*> result;
    return result;
}

GeometryManager::GeometryManager(MaterialsDB& materialsDB): materialsDB(materialsDB) {
}

GeometryManager::~GeometryManager() {
    for (GeometryElement* e: elements) delete e;
}

GeometryElement *GeometryManager::getElement(const std::string &name) {
    return map_find(namedElements, name);
}

GeometryElement& GeometryManager::requireElement(const std::string &name) {
    GeometryElement* result = getElement(name);
    if (result == nullptr) throw NoSuchGeometryElement(name);
    return *result;
}

void GeometryManager::registerElementReader(const std::string &tag_name, element_read_f *reader) {
    elementReaders()[tag_name] = reader;
}

GeometryElement& GeometryManager::readElement(XMLReader &source) {
    std::string nodeName = source.getNodeName();
    if (nodeName == "ref")
        return requireElement(XML::requireAttr(source, "name"));
    auto reader_it = elementReaders().find(nodeName);
    if (reader_it == elementReaders().end())
        throw NoSuchGeometryElementType(nodeName);
    const char* name_exists = source.getAttributeValue("name");    //must be call before reader call (reader function can change XMLReader)
    std::string name = name_exists ? name_exists : "";     //reader can also delete name_exists, we need copy
    GeometryElement* new_element = reader_it->second(*this, source);
    elements.insert(new_element);   //first this, to ensure that memory will be freed
    if (name_exists) {
        if (!namedElements.insert(std::map<std::string, GeometryElement*>::value_type(name, new_element)).second)
            throw GeometryElementNamesConflictException(name);
    }
    return *new_element;
}

GeometryElement& GeometryManager::readExactlyOneChild(XMLReader& source) {
    XML::requireTag(source);
    GeometryElement& result = readElement(source);
    XML::requireTagEnd(source);
    return result;
}

void GeometryManager::loadFromReader(XMLReader &reader) {
    if (reader.getNodeType() != irr::io::EXN_ELEMENT || reader.getNodeName() != std::string("geometry"))
        throw XMLUnexpectedElementException("<geometry> tag");   
    while(reader.read()) {
        switch (reader.getNodeType()) {
            case irr::io::EXN_ELEMENT_END: return;  //end of geometry
            case irr::io::EXN_ELEMENT: readElement(reader); break;
            case irr::io::EXN_COMMENT: break;   //just ignore
            default: throw XMLUnexpectedElementException("begin of geometry element tag or </geometry>");  
        }
    }
    throw XMLUnexpectedEndException();
}

void GeometryManager::loadFromXMLStream(std::istream &input) {
    XML::StreamReaderCallback cb(input);
    std::unique_ptr< XMLReader > reader(irr::io::createIrrXMLReader(&cb));
    XML::requireNext(*reader);
    loadFromReader(*reader);
}

void GeometryManager::loadFromXMLString(const std::string &input_XML_str) {
    std::istringstream stream(input_XML_str);
    loadFromXMLStream(stream);
}

//TODO skip geometry elements ends
void GeometryManager::loadFromFile(const std::string &fileName) {
    std::unique_ptr< XMLReader > reader(irr::io::createIrrXMLReader(fileName.c_str()));
    XML::requireNext(*reader);
    loadFromReader(*reader);
}

}	// namespace plask
