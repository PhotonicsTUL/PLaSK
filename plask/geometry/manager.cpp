#include "manager.h"

#include "../utils/stl.h"

namespace plask {

std::map<std::string, GeometryManager::element_read_f*> GeometryManager::elementReaders;

GeometryManager::GeometryManager() {
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
    elementReaders[tag_name] = reader;
}

std::string requireAttr(XMLReader &source, const char* attr_name) {
    const char* result = source.getAttributeValue(attr_name);
    if (result == nullptr)
        throw NoAttrException(source.getNodeName(), attr_name);
    return result;
}

GeometryElement& GeometryManager::readElement(XMLReader &source) {
    std::string nodeName = source.getNodeName();
    if (nodeName == "ref")
        return requireElement(requireAttr(source, "name"));
    auto reader_it = elementReaders.find(nodeName);
    if (reader_it == elementReaders.end())
        throw NoSuchGeometryElementType(nodeName);
    GeometryElement* new_element = reader_it->second(*this, source);
    elements.insert(new_element);   //first this to ensure that memory will be freed
    const char* name = source.getAttributeValue("name");
    if (name) {
        if (!namedElements.insert(std::map<std::string, GeometryElement*>::value_type(name, new_element)).second)
            throw GeometryElementNamesConflictException(name);
    }
    return *new_element;
}

}	// namespace plask
