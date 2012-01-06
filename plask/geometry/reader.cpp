#include "reader.h"

#include "manager.h"

namespace plask {

std::map<std::string, GeometryReader::element_read_f*>& GeometryReader::elementReaders() {
    static std::map<std::string, GeometryReader::element_read_f*> result;
    return result;
}

void GeometryReader::registerElementReader(const std::string &tag_name, element_read_f *reader) {
    elementReaders()[tag_name] = reader;
}

plask::GeometryReader::GeometryReader(plask::GeometryManager &manager, plask::XMLReader &source)
    : manager(manager), source(source)
{
}

GeometryElement& GeometryReader::readElement() {
    std::string nodeName = source.getNodeName();
    if (nodeName == "ref")
        return manager.requireElement(XML::requireAttr(source, "name"));
    auto reader_it = elementReaders().find(nodeName);
    if (reader_it == elementReaders().end())
        throw NoSuchGeometryElementType(nodeName);
    const char* name_exists = source.getAttributeValue("name");    //must be call before reader call (reader function can change XMLReader)
    std::string name = name_exists ? name_exists : "";     //reader can also delete name_exists, we need copy
    GeometryElement* new_element = reader_it->second(*this);
    manager.elements.insert(new_element);   //first this, to ensure that memory will be freed
    if (name_exists) {
        if (!manager.namedElements.insert(std::map<std::string, GeometryElement*>::value_type(name, new_element)).second)
            throw GeometryElementNamesConflictException(name);
    }
    return *new_element;
}

GeometryElement& GeometryReader::readExactlyOneChild() {
    XML::requireTag(source);
    GeometryElement& result = readElement();
    XML::requireTagEnd(source);
    return result;
}

}   // namespace plask
