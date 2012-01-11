#include "reader.h"

#include "manager.h"
#include "../utils/string.h"

namespace plask {

AxisNames::AxisNames(const std::string &c0_name, const std::string &c1_name, const std::string &c2_name)
    : byIndex{c0_name, c1_name, c2_name} {}

std::size_t AxisNames::operator [](const std::string &name) const {
    if (byIndex[0] == name) return 0;
    if (byIndex[1] == name) return 1;
    if (byIndex[2] == name) return 2;
    return 3;  
}

const AxisNames& AxisNames::Register::get(const std::string &name) const {
    auto i = axisNames.find(removedChars(name, ",._ \t"));
    if (i == axisNames.end())
        throw NoSuchAxisNames(name);
    return i->second;    
}

std::map<std::string, GeometryReader::element_read_f*>& GeometryReader::elementReaders() {
    static std::map<std::string, GeometryReader::element_read_f*> result;
    return result;
}

void GeometryReader::registerElementReader(const std::string &tag_name, element_read_f *reader) {
    elementReaders()[tag_name] = reader;
}

AxisNames::Register GeometryReader::axisNamesRegister(
        //c0, c1, c2, axis names:
        AxisNames::Register
        ("x", "y", "z", "yz", "se", "zup")
        ("z", "x", "y", "xy", "ee", "yup")
        ("r", "phi", "z", "rz", "rad")
        ("lon", "tran", "up")
);

GeometryReader::ReadAxisNames::ReadAxisNames(GeometryReader &reader)
    : reader(reader), old(reader.axisNames) {
    const char* axis = reader.source.getAttributeValue("axis");
    if (axis) reader.axisNames = &axisNamesRegister.get(axis);
}

plask::GeometryReader::GeometryReader(plask::GeometryManager &manager, plask::XMLReader &source)
    : manager(manager), source(source)
{
    axisNames = &axisNamesRegister.get("lon, tran, up");
}

GeometryElement& GeometryReader::readElement() {
    std::string nodeName = source.getNodeName();
    if (nodeName == "ref")
        return manager.requireElement(XML::requireAttr(source, "name"));
    ReadAxisNames axis_reader(*this);   //try set up new axis names, store old, and restore old on end of block
    auto reader_it = elementReaders().find(nodeName);
    if (reader_it == elementReaders().end())
        throw NoSuchGeometryElementType(nodeName);
    boost::optional<std::string> name = XML::getAttribute(source, "name");
    GeometryElement* new_element = reader_it->second(*this);
    manager.elements.insert(new_element);   //first this, to ensure that memory will be freed
    if (name) {
        if (!manager.namedElements.insert(std::map<std::string, GeometryElement*>::value_type(*name, new_element)).second)
            throw GeometryElementNamesConflictException(*name);
    }
    return *new_element;
}

GeometryElement& GeometryReader::readExactlyOneChild() {
    XML::requireTag(source);
    std::string tag_name = source.getNodeName();
    GeometryElement& result = readElement();
    XML::requireTagEnd(source, tag_name);
    return result;
}

}   // namespace plask
