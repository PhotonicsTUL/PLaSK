#include "reader.h"

#include "../manager.h"

namespace plask {

std::map<std::string, GeometryReader::element_read_f*>& GeometryReader::elementReaders() {
    static std::map<std::string, GeometryReader::element_read_f*> result;
    return result;
}

void GeometryReader::registerElementReader(const std::string &tag_name, element_read_f *reader) {
    elementReaders()[tag_name] = reader;
}


GeometryReader::ReadAxisNames::ReadAxisNames(GeometryReader &reader)
    : reader(reader), old(reader.axisNames) {
    const char* axis = reader.source.getAttributeValueC("axes");
    if (axis) reader.axisNames = &AxisNames::axisNamesRegister.get(axis);
}

GeometryReader::SetExpectedSuffix::SetExpectedSuffix(GeometryReader &reader, const char* new_expected_suffix)
    : reader(reader), old(reader.expectedSuffix) {
    reader.expectedSuffix = new_expected_suffix;
}

plask::GeometryReader::GeometryReader(plask::Manager &manager, plask::XMLReader &source, const MaterialsDB& materialsDB)
    : expectedSuffix(0), manager(manager), source(source),
      materialSource([&materialsDB](const std::string& mat_name) { return materialsDB.get(mat_name); })
{
    axisNames = &AxisNames::axisNamesRegister.get("lon, tran, up");
}

GeometryReader::GeometryReader(Manager &manager, XMLReader &source, const GeometryReader::MaterialsSource &materialsSource)
    : expectedSuffix(0), manager(manager), source(source), materialSource(materialsSource)
{
    axisNames = &AxisNames::axisNamesRegister.get("lon, tran, up");
}

shared_ptr<GeometryElement> GeometryReader::readElement() {
    std::string nodeName = source.getNodeName();
    if (nodeName == "ref") {
        shared_ptr<GeometryElement> result = manager.requireGeometryElement(source.requireAttribute("name"));
        source.requireTagEnd("ref");
        return result;
    }
    ReadAxisNames axis_reader(*this);   //try set up new axis names, store old, and restore old on end of block
    auto reader_it = elementReaders().find(nodeName);
    if (reader_it == elementReaders().end()) {
        if (expectedSuffix == 0)
            throw NoSuchGeometryElementType(nodeName);
        reader_it = elementReaders().find(nodeName + expectedSuffix);
        if (reader_it == elementReaders().end())
            throw NoSuchGeometryElementType(nodeName + "[" + expectedSuffix + "]");
    }
    boost::optional<std::string> name = source.getAttribute("name");    //read name
    if (name) BadId::throwIfBad("geometry element", *name, ' ');
    shared_ptr<GeometryElement> new_element = reader_it->second(*this); //and rest (but while reading this subtree, name is not registred yet)
    if (name) { //if have name, register it (add it to map of names)
        if (!manager.namedElements.insert(std::map<std::string, shared_ptr<GeometryElement> >::value_type(*name, new_element)).second)
            throw GeometryElementNamesConflictException(*name);
    }
    return new_element;
}

shared_ptr<GeometryElement> GeometryReader::readExactlyOneChild() {
    std::string parent_tag = source.getNodeName();
    source.requireTag();
    shared_ptr<GeometryElement> result = readElement();
    source.requireTagEnd(parent_tag);
    return result;
}

shared_ptr<Geometry> GeometryReader::readGeometry() {
    ReadAxisNames axis_reader(*this);   //try set up new axis names, store old, and restore old on end of block
    std::string nodeName = source.getNodeName();
    boost::optional<std::string> name = source.getAttribute("name");
    if (name) BadId::throwIfBad("geometry", *name, ' ');
//    std::string src = source.requireAttribute("over");
    shared_ptr<Geometry> result;
    if (nodeName == "2d" || nodeName == "cartesian2d") {    //TODO register with space names(?)
        SetExpectedSuffix suffixSetter(*this, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
        boost::optional<double> l = source.getAttribute<double>("length");
        if (l) {
            result = make_shared<Geometry2DCartesian>(readExactlyOneChild<GeometryElementD<2>>(), *l);
        } else {
            auto child = readExactlyOneChild<GeometryElement>();
            auto extrusion = dynamic_pointer_cast<Extrusion>(child);
            if (extrusion) {
                result = make_shared<Geometry2DCartesian>(extrusion);
            } else {
                result = make_shared<Geometry2DCartesian>(dynamic_pointer_cast<GeometryElementD<2>>(child), INFINITY);
                if (!result) throw UnexpectedGeometryElementTypeException();
            }
        }
    } else
    if (nodeName == "cylindrical") {
        SetExpectedSuffix suffixSetter(*this, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
        result = make_shared<Geometry2DCylindrical>(readExactlyOneChild<GeometryElementD<2>>());
    } else
        throw XMLUnexpectedElementException("space tag (cartesian or cylindrical)");

    result->setBorders([&](const std::string& s) { return source.getAttribute(s); }, *axisNames );

    if (name) manager.geometries[*name] = result;
    return result;
}

}   // namespace plask
