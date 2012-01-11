#include "manager.h"

#include "../utils/stl.h"
#include "reader.h"

namespace plask {



GeometryManager::GeometryManager(MaterialsDB& materialsDB): materialsDB(materialsDB) {
}

GeometryManager::~GeometryManager() {
    //for (GeometryElement* e: elements) delete e;
}

shared_ptr<GeometryElement> GeometryManager::getElement(const std::string &name) {
    return map_find(namedElements, name, shared_ptr<GeometryElement>());
}

shared_ptr<GeometryElement> GeometryManager::requireElement(const std::string &name) {
    shared_ptr<GeometryElement> result = getElement(name);
    if (result == nullptr) throw NoSuchGeometryElement(name);
    return result;
}

//TODO move to reader (?)
void GeometryManager::loadFromReader(XMLReader &XMLreader) {
    GeometryReader reader(*this, XMLreader);
    if (XMLreader.getNodeType() != irr::io::EXN_ELEMENT || XMLreader.getNodeName() != std::string("geometry"))
        throw XMLUnexpectedElementException("<geometry> tag");
    GeometryReader::ReadAxisNames read_axis_tag(reader);
    while(XMLreader.read()) {
        switch (XMLreader.getNodeType()) {
            case irr::io::EXN_ELEMENT_END:
                //if (XMLreader.getNodeName() != std::string("geometry"))
                //    throw XMLUnexpectedElementException("end of \"geometry\" tag");
                return;  //end of geometry
            case irr::io::EXN_ELEMENT:
                reader.readElement();
                break;
            case irr::io::EXN_COMMENT:
                break;   //just ignore
            default:
                throw XMLUnexpectedElementException("begin of geometry element tag or </geometry>");  
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
