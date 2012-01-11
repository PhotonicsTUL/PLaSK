#include "xml.h"

#include "../exceptions.h"

namespace plask { namespace XML {

boost::optional<std::string> getAttribute(XMLReader& reader, const char* name) {
    const char* v = reader.getAttributeValue(name);
    return v != nullptr ? boost::optional<std::string>(v) : boost::optional<std::string>();
}
    
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

void requireTagEnd(XMLReader& reader, const std::string& tag) {
    if (reader.getNodeType() == irr::io::EXN_ELEMENT && reader.isEmptyElement())
        return;
    requireNext(reader);
    if (reader.getNodeType() != irr::io::EXN_ELEMENT_END || reader.getNodeName() != tag)
        throw XMLUnexpectedElementException("end of tag \"" +tag + "\"");
}

bool skipComments(XMLReader& reader) {
    bool result;
    while (reader.getNodeType() == irr::io::EXN_COMMENT && (result = reader.read()));
    return result;
}

StreamReaderCallback::StreamReaderCallback(std::istream &input)
: input(input) {
    std::streampos beg = input.tellg();
    input.seekg(0, std::ios::end);
    std::streampos end = input.tellg();
    input.seekg(beg);
    size = end - beg;
}

int StreamReaderCallback::read(void *buffer, int sizeToRead) {
    input.read((char*) buffer, sizeToRead);
    return input.gcount();
}


} } // namespace plask::XML
