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

inline bool isAutoName(const std::string& name) { return !name.empty() && name[0] == '#'; }

shared_ptr<GeometryElement> GeometryReader::readElement() {
    std::string nodeName = source.getNodeName();

    if (nodeName == "ref") {
        shared_ptr<GeometryElement> result = requireElementWithName(source.requireAttribute("name"));
        source.requireTagEnd();
        return result;
    }

    boost::optional<std::string> name = source.getAttribute("name");    //read name
    if (name && !isAutoName(*name))
        BadId::throwIfBad("geometry element", *name, ' ');

    shared_ptr<GeometryElement> new_element;    //new element will be constructed

    if (nodeName == "copy") {   //TODO(?) move code of copy to virtual method of manager to allow custom support for it in GUI
        shared_ptr<GeometryElement> from = requireElementWithName(source.requireAttribute("from"));
        GeometryElement::CompositeChanger changers;
        while (source.requireTagOrEnd()) {
            const std::string operation_name = source.getNodeName();
            if (operation_name == "replace") {
                shared_ptr<GeometryElement> op_from = requireElementWithName(source.requireAttribute("element"));
                shared_ptr<GeometryElement> to;
                boost::optional<std::string> to_name = source.getAttribute("to");
                if (to_name) {
                    to = requireElementWithName(*to_name);
                } else {
                    source.requireTag();
                    to = readElement();
                }
                //TODO read translation
                changers.append(new GeometryElement::ReplaceChanger(op_from, to, vec(0.0, 0.0, 0.0)));
                source.requireTagEnd();
            } else if (operation_name == "toblock") {
                shared_ptr<GeometryElement> op_from = requireElementWithName(source.requireAttribute("element"));
                shared_ptr<Material> blockMaterial = getMaterial(source.requireAttribute("material"));
                changers.append(new GeometryElement::ToBlockChanger(op_from, blockMaterial));
                source.requireTagEnd();
            } else {
                throw Exception("\"%1%\" is not proper name of copy operation and so it is not alowed in <copy> tag.", operation_name);
            }
        }
        new_element = const_pointer_cast<GeometryElement>(from->changedVersion(changers));
    } else {
        ReadAxisNames axis_reader(*this);   //try set up new axis names, store old, and restore old on end of block
        auto reader_it = elementReaders().find(nodeName);
        if (reader_it == elementReaders().end()) {
            if (expectedSuffix == 0)
                throw NoSuchGeometryElementType(nodeName);
            reader_it = elementReaders().find(nodeName + expectedSuffix);
            if (reader_it == elementReaders().end())
                throw NoSuchGeometryElementType(nodeName + "[" + expectedSuffix + "]");
        }
        new_element = reader_it->second(*this); //and rest (but while reading this subtree, name is not registred yet)
    }

    if (name) { //if have name, register it (add it to map of names)
        if (isAutoName(*name)) {
            if (!autoNamedElements.insert(std::map<std::string, shared_ptr<GeometryElement> >::value_type(*name, new_element)).second)
                throw NamesConflictException("Auto-named geometry element", *name);
        } else {    //normal name
            if (!manager.namedElements.insert(std::map<std::string, shared_ptr<GeometryElement> >::value_type(*name, new_element)).second)
                throw NamesConflictException("Geometry element", *name);
        }
    }
    return new_element;
}

shared_ptr<GeometryElement> GeometryReader::readExactlyOneChild() {
    source.requireTag();
    shared_ptr<GeometryElement> result = readElement();
    source.requireTagEnd();
    return result;
}

shared_ptr<Geometry> GeometryReader::readGeometry() {
    ReadAxisNames axis_reader(*this);   //try set up new axis names, store old, and restore old on end of block
    std::string nodeName = source.getNodeName();
    boost::optional<std::string> name = source.getAttribute("name");
    if (name) BadId::throwIfBad("geometry", *name, ' ');
//    std::string src = source.requireAttribute("over");
    // TODO read subspaces from XML
    shared_ptr<Geometry> result;
    if (nodeName == "cartesian2d") {
        SetExpectedSuffix suffixSetter(*this, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
        boost::optional<double> l = source.getAttribute<double>("length");
        shared_ptr<Geometry2DCartesian> cartesian2d = make_shared<Geometry2DCartesian>();   //result with oryginal type
        result = cartesian2d;
        result->setBorders([&](const std::string& s) { return source.getAttribute(s); }, *axisNames );
        if (l) {
            cartesian2d->setExtrusion(make_shared<Extrusion>(readExactlyOneChild<GeometryElementD<2>>(), *l));
        } else {
            auto child = readExactlyOneChild<GeometryElement>();
            auto child_as_extrusion = dynamic_pointer_cast<Extrusion>(child);
            if (child_as_extrusion) {
                cartesian2d->setExtrusion(child_as_extrusion);
            } else {
                auto child_as_2d = dynamic_pointer_cast<GeometryElementD<2>>(child);
                if (!child_as_2d) throw UnexpectedGeometryElementTypeException();
                cartesian2d->setExtrusion(make_shared<Extrusion>(child_as_2d, INFINITY));
            }
        }
    } else if (nodeName == "cylindrical" || nodeName == "cylindrical2d") {
        SetExpectedSuffix suffixSetter(*this, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_2D);
        result = make_shared<Geometry2DCylindrical>();
        result->setBorders([&](const std::string& s) { return source.getAttribute(s); }, *axisNames );
        static_pointer_cast<Geometry2DCylindrical>(result)->
            setRevolution(make_shared<Revolution>(readExactlyOneChild<GeometryElementD<2>>()));
    } else if (nodeName == "3d" || nodeName == "cartesian3d") {
        SetExpectedSuffix suffixSetter(*this, PLASK_GEOMETRY_TYPE_NAME_SUFFIX_3D);
        result = make_shared<Geometry3D>();
        result->setBorders([&](const std::string& s) { return source.getAttribute(s); }, *axisNames );
        static_pointer_cast<Geometry3D>(result)->setChildUnsafe(
            readExactlyOneChild<GeometryElementD<3>>());
    } else
        throw XMLUnexpectedElementException(source, "geometry tag (<cartesian2d>, <cartesian3d>, or <cylindrical>)");

    if (name) manager.geometries[*name] = result;
    return result;
}

shared_ptr<GeometryElement> GeometryReader::requireElementWithName(const std::string &name) const {
    if (isAutoName(name)) {
        auto it = autoNamedElements.find(name);
        if (it == autoNamedElements.end()) throw NoSuchGeometryElement(name);
        return it->second;
    } else
        return manager.requireGeometryElement(name);
}

}   // namespace plask
