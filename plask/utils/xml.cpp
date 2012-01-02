#include "xml.h"

#include "../exceptions.h"

namespace plask { namespace XML {

std::string requireAttr(XMLReader &source, const char* attr_name) {
    const char* result = source.getAttributeValue(attr_name);
    if (result == nullptr)
        throw XMLNoAttrException(source.getNodeName(), attr_name);
    return result;
}

void requireNext(XMLReader& reader) {
    do {
        if (!reader.read())
            throw XMLUnexpectedEndException();
    } while (reader.getNodeType() == irr::io::EXN_COMMENT); 
}

void requireTag(XMLReader& reader) {
    requireNext(reader);
    if (reader.getNodeType() != irr::io::EXN_ELEMENT)
        throw XMLUnexpectedElementException("begin of tag");    
}

void requireTagEnd(XMLReader& reader) {
    requireNext(reader);
    if (reader.getNodeType() != irr::io::EXN_ELEMENT_END)
        throw XMLUnexpectedElementException("end of tag");
}

bool skipComments(XMLReader& reader) {
    bool result;
    while (reader.getNodeType() == irr::io::EXN_COMMENT && (result = reader.read()));
    return result;
}


} } // namespace plask::XML
