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
    if (currentNodeType == NODE_ELEMENT) {
        if (std::size_t(getAttributeCount()) != read_attribiutes.size()) {
            std::string attr;
            for (int i = 0; i < irrReader->getAttributeCount(); ++i)
                if (read_attribiutes.find(irrReader->getAttributeName(i)) == read_attribiutes.end()) {
                    if (!attr.empty()) attr += ", ";
                    attr += irrReader->getAttributeName(i);
                }
            throw Exception("Following attributes are unexpected in XML tag \"%1%\": %2%", getNodeName(), attr);
        }
        read_attribiutes.clear();
    }

    if (currentNodeType == NODE_ELEMENT && irrReader->isEmptyElement())
        currentNodeType = NODE_ELEMENT_END;
    else {
        if (!irrReader->read()) {
            if (!path.empty()) throw std::runtime_error("unexpected end of XML input, some tags was not closed");
            return false;
        }
        currentNodeType = NodeType(irrReader->getNodeType());
    }

    switch (currentNodeType) {
        case NODE_ELEMENT:
            path.push_back(getNodeName());
            break;

        case NODE_ELEMENT_END:
            if (path.empty())
                throw Exception("unexpected closing of XML tag \"%1%\"", getNodeName());
            if (path.back() != getNodeName())
                throw Exception("expected closing of %1% tag, but obtained closing of \"%2%\" tag", path.back(), getNodeName());
            path.pop_back();

        default:    //just for compiler warning
            ;
    }

    return true;
}

const char *XMLReader::getAttributeValueC(const std::string &name) const {
     const char * result = irrReader->getAttributeValue(name.c_str());
     if (result) const_cast<std::unordered_set<std::string>&>(read_attribiutes).insert(name);
     return result;
}

boost::optional<std::string> XMLReader::getAttribute(const std::string& name) const {
    const char* v = getAttributeValueC(name);
    return v != nullptr ? boost::optional<std::string>(v) : boost::optional<std::string>();
}

std::string XMLReader::requireAttribute(const std::string& attr_name) const {
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

void XMLReader::requireTagEnd() {
    requireNext();
    if (getNodeType() != NODE_ELEMENT_END)
        throw XMLUnexpectedElementException("end of tag \"" +path.back() + "\"");
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
