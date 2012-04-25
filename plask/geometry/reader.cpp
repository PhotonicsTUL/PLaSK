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
    const char* axis = reader.source.getAttributeValueC("axes");
    if (axis) reader.axisNames = &axisNamesRegister.get(axis);
}

GeometryReader::SetExpectedSuffix::SetExpectedSuffix(GeometryReader &reader, const char* new_expected_suffix)
    : reader(reader), old(reader.expectedSuffix) {
    reader.expectedSuffix = new_expected_suffix;
}

plask::GeometryReader::GeometryReader(plask::GeometryManager &manager, plask::XMLReader &source, const MaterialsDB& materialsDB)
    : manager(manager), source(source), expectedSuffix(0), materialsDB(materialsDB)
{
    axisNames = &axisNamesRegister.get("lon, tran, up");
}

shared_ptr<GeometryElement> GeometryReader::readElement() {
    std::string nodeName = source.getNodeName();
    if (nodeName == "ref")
        return manager.requireElement(source.requireAttribute("name"));
    ReadAxisNames axis_reader(*this);   //try set up new axis names, store old, and restore old on end of block
    auto reader_it = elementReaders().find(nodeName);
    if (reader_it == elementReaders().end()) {
        if (expectedSuffix == 0)
            throw NoSuchGeometryElementType(nodeName);
        reader_it = elementReaders().find(nodeName + expectedSuffix);
        if (reader_it == elementReaders().end())
            throw NoSuchGeometryElementType(nodeName + "[" + expectedSuffix + "]");
    }
    boost::optional<std::string> name = source.getAttribute("name");
    shared_ptr<GeometryElement> new_element = reader_it->second(*this);
    //manager.elements.insert(new_element);   //first this, to ensure that memory will be freed
    if (name) {
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

shared_ptr<CalculationSpace> GeometryReader::readCalculationSpace() {
    std::string nodeName = source.getNodeName();
    std::string name = source.requireAttribute("name");
    std::string src = source.requireAttribute("over");
    shared_ptr<CalculationSpace> result;
    if (nodeName == "cartesian") {    //TODO register with space names(?)
        boost::optional<double> l = source.getAttribute<double>("length");
        result = l ?
            make_shared<Space2dCartesian>(manager.requireElement< GeometryElementD<2> >(src), *l) :
            make_shared<Space2dCartesian>(manager.requireElement< Extrusion >(src));
    } else
    if (nodeName == "cylindrical") {
        result = make_shared<Space2dCylindrical>(manager.requireElement< GeometryElementD<2> >(src));
    } else
        throw XMLUnexpectedElementException("space tag (cartesian or cylindrical)");

    boost::optional<std::string> v;
    v = source.getAttribute("borders");
    if (v) result->setAllBorders(*border::Strategy::fromStrUnique(*v));
    v = source.getAttribute("planar");
    if (v) result->setPlanarBorders(*border::Strategy::fromStrUnique(*v));
    for (int dir_nr = 0; dir_nr < 3; ++dir_nr) {
        std::string axis_name = getAxisName(dir_nr);
        if (v) result->setBorders(plask::Primitive<3>::DIRECTION(dir_nr), *border::Strategy::fromStrUnique(*v));
        v = source.getAttribute(axis_name);
        if (v) result->setBorders(plask::Primitive<3>::DIRECTION(dir_nr), *border::Strategy::fromStrUnique(*v));
        v = source.getAttribute(axis_name + "-lo");
        if (v) result->setBorder(plask::Primitive<3>::DIRECTION(dir_nr), false, *border::Strategy::fromStrUnique(*v));
        v = source.getAttribute(axis_name + "-hi");
        if (v) result->setBorder(plask::Primitive<3>::DIRECTION(dir_nr), true, *border::Strategy::fromStrUnique(*v));
    }

    manager.calculationSpaces[name] = result;
    return result;
}

}   // namespace plask
