#include "xml.h"

#include "../exceptions.h"

namespace plask {

///Allow to read XML from standard C++ input stream (std::istream).
struct StreamReaderCallback: public irr::io::IFileReadCallBack {

    ///Stream to read from.
    std::istream& input;

    ///Size of stream, number of bytes to read.
    int size;

    /**
     * @param input stream to read from
     */
    StreamReaderCallback(std::istream& input);

    virtual int read(void* buffer, int sizeToRead);

    virtual int getSize() { return size; }

};

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

XMLReader::XMLReader(const char* file_name): irrReader(irr::io::createIrrXMLReader(file_name)), currentNodeType(NODE_NONE) {
    if (irrReader == nullptr) throw Exception("Can't read from file \"%1%\"", file_name);
}

XMLReader::XMLReader(std::istream& input)
    :currentNodeType(NODE_NONE)
{
    StreamReaderCallback cb(input);
    irrReader = irr::io::createIrrXMLReader(&cb);
}

bool XMLReader::read() {
    if (currentNodeType == NODE_ELEMENT && irrReader->isEmptyElement())
        currentNodeType = NODE_ELEMENT_END;
    else {
        if (!irrReader->read()) return false;
        currentNodeType = NodeType(irrReader->getNodeType());
    }
    return true;
}

boost::optional<std::string> XMLReader::getAttribute(const char* name) const {
    const char* v = getAttributeValueC(name);
    return v != nullptr ? boost::optional<std::string>(v) : boost::optional<std::string>();
}

std::string XMLReader::requireAttribute(const char* attr_name) const {
    const char* result = getAttributeValueC(attr_name);
    if (result == nullptr)
        throw XMLNoAttrException(getNodeName(), attr_name);
    return result;
}

void XMLReader::requireNext() {
    do {
        if (!read()) throw XMLUnexpectedEndException();
    } while (getNodeType() == NODE_COMMENT);
}

void XMLReader::requireTag() {
    requireNext();
    if (getNodeType() != NODE_ELEMENT)
        throw XMLUnexpectedElementException("begin of tag");
}

void XMLReader::requireTagEnd(const std::string& tag) {
    requireNext();
    if (getNodeType() != NODE_ELEMENT_END || getNodeName() != tag)
        throw XMLUnexpectedElementException("end of tag \"" +tag + "\"");
}

/*void requireTagEndOrEmptyTag(XMLReader& reader, const std::string& tag) {
    if (reader.getNodeType() == irr::io::EXN_ELEMENT && reader.isEmptyElement())
        return;
    requireTagEnd(reader, tag);
}*/

bool XMLReader::skipComments() {
    bool result = true;
    while (getNodeType() == NODE_COMMENT && (result = read()));
    return result;
}

} // namespace plask
