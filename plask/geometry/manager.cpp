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
    std::string name = name_exists;     //reader can also delete name_exists, we need copy
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

}	// namespace plask
