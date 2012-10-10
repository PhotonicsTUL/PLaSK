#include "reader.h"

namespace plask {

/// Allow to read XML from standard C++ input stream (std::istream).
struct StreamReaderCallback: public irr::io::IFileReadCallBack {

    /// Stream to read from.
    std::istream& input;

    /// Size of stream, number of bytes to read.
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

XMLReader::XMLReader(const char* file_name):
    irrReader(irr::io::createIrrXMLReader(file_name)), currentNodeType(NODE_NONE), check_if_all_attributes_were_read(true)
{
    if (irrReader == nullptr) throw XMLException("Can't read from file \"" + std::string(file_name) +"\"");
}

XMLReader::XMLReader(std::istream& input):
    currentNodeType(NODE_NONE), check_if_all_attributes_were_read(true)
{
    StreamReaderCallback cb(input);
    irrReader = irr::io::createIrrXMLReader(&cb);
}

XMLReader::XMLReader(FILE* file):
    irrReader(irr::io::createIrrXMLReader(file)), currentNodeType(NODE_NONE), check_if_all_attributes_were_read(true)
{
    if (irrReader == nullptr) throw XMLException("Can't read from file");
}

#if (__cplusplus >= 201103L) || defined(__GXX_EXPERIMENTAL_CXX0X__)
XMLReader::XMLReader(XMLReader &&to_move)
    : irrReader(to_move.irrReader),
      currentNodeType(to_move.currentNodeType),
      path(std::move(to_move.path)),
      read_attributes(std::move(to_move.read_attributes)),
      check_if_all_attributes_were_read(to_move.check_if_all_attributes_were_read)
{
    to_move.irrReader = 0;
}

XMLReader &XMLReader::operator=(XMLReader &&to_move)
{
    swap(to_move);
    return *this;
}
#endif

void XMLReader::swap(XMLReader &to_swap)
{
    std::swap(irrReader, to_swap.irrReader);
    std::swap(currentNodeType, to_swap.currentNodeType);
    std::swap(path, to_swap.path);
    std::swap(read_attributes, to_swap.read_attributes);
    std::swap(check_if_all_attributes_were_read, to_swap.check_if_all_attributes_were_read);
}

bool XMLReader::read() {
    if (currentNodeType == NODE_ELEMENT) {
        if (check_if_all_attributes_were_read && (std::size_t(getAttributeCount()) != read_attributes.size())) {
            std::string attr;
            for (int i = 0; i < irrReader->getAttributeCount(); ++i)
                if (read_attributes.find(irrReader->getAttributeName(i)) == read_attributes.end()) {
                    if (!attr.empty()) attr += ", ";
                    attr += irrReader->getAttributeName(i);
                }
            throw XMLUnexpectedAttrException(*this, attr);
        }
        read_attributes.clear();
    }
    check_if_all_attributes_were_read = true;

    if (currentNodeType == NODE_ELEMENT && irrReader->isEmptyElement())
        currentNodeType = NODE_ELEMENT_END;
    else {
        if (!irrReader->read()) {
            if (!path.empty()) throw XMLUnexpectedEndException(*this);
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
                throw XMLUnexpectedElementException(*this, "opening of a new tag or end of file", "closing of tag <" + getNodeName() +">");
            if (path.back() != getNodeName())
                throw XMLUnexpectedElementException(*this, "closing of tag <" + path.back() +">", "end of tag <" + getNodeName() +">");
            path.pop_back();

        default:    //just for compiler warning
            ;
    }

    return true;
}

const char *XMLReader::getAttributeValueC(const std::string &name) const {
     const char * result = irrReader->getAttributeValue(name.c_str());
     if (result) const_cast<std::unordered_set<std::string>&>(read_attributes).insert(name);
     return result;
}

std::string XMLReader::getTextContent() const {
    if (getNodeType() != NODE_TEXT)
        throw XMLUnexpectedElementException(*this, "text");
    return getNodeDataC();
}

boost::optional<std::string> XMLReader::getAttribute(const std::string& name) const {
    const char* v = getAttributeValueC(name);
    return v != nullptr ? boost::optional<std::string>(v) : boost::optional<std::string>();
}

std::string XMLReader::requireAttribute(const std::string& attr_name) const {
    const char* result = getAttributeValueC(attr_name);
    if (result == nullptr)
        throw XMLNoAttrException(*this, attr_name);
    return result;
}

void XMLReader::requireNext() {
    do {
        if (!read()) throw XMLUnexpectedEndException(*this);
    } while (getNodeType() == NODE_COMMENT);
}

void XMLReader::requireTag() {
    requireNext();
    if (getNodeType() != NODE_ELEMENT)
        throw XMLUnexpectedElementException(*this, "begin of a new tag");
}

void XMLReader::requireTag(const std::string& name) {
    requireNext();
    if (getNodeType() != NODE_ELEMENT || getNodeName() != name)
        throw XMLUnexpectedElementException(*this, "begin of tag <" + name + ">");
}

bool XMLReader::requireTagOrEnd() {
    requireNext();
    if (getNodeType() != NODE_ELEMENT && getNodeType() != NODE_ELEMENT_END)
        throw XMLUnexpectedElementException(*this, "begin of a new tag or </" + path.back() + ">");
    return getNodeType() == NODE_ELEMENT;
}

bool XMLReader::requireTagOrEnd(const std::string& name) {
    if (requireTagOrEnd()) {
        if (getNodeName() != name)
            throw XMLUnexpectedElementException(*this, "begin of tag <" + name + ">");
        return true;
    } else
        return false;
}

void XMLReader::requireTagEnd() {
    requireNext();
    if (getNodeType() != NODE_ELEMENT_END)
        throw XMLUnexpectedElementException(*this, "</" + path.back() + ">");
}

/*void requireTagEndOrEmptyTag(XMLReader& reader, const std::string& tag) {
    if (reader.getNodeType() == irr::io::EXN_ELEMENT && reader.isEmptyElement())
        return;
    requireTagEnd(reader, tag);
}*/

std::string XMLReader::requireText() {
    requireNext();
    return getTextContent();
}

std::string XMLReader::requireTextUntilEnd() {
    std::string text;
    requireNext();
    while (getNodeType() == NODE_TEXT) {
        text += getTextContent();
        requireNext();
    }
    if (text == "")
        throw XMLUnexpectedElementException(*this, "text");
    if (getNodeType() != NODE_ELEMENT_END)
        throw XMLUnexpectedElementException(*this, "</" + path.back() + ">");
    return text;
}


bool XMLReader::skipComments() {
    bool result = true;
    while (getNodeType() == NODE_COMMENT && (result = read()));
    return result;
}

bool XMLReader::gotoNextOnLevel(std::size_t required_level, NodeType required_type)
{
    ignoreAllAttributes();
    while (read()) {
        if (getLevel() == required_level && getNodeType() == required_type)
            return true;
        ignoreAllAttributes();
    }
    return false;
}

bool XMLReader::gotoNextTagOnCurrentLevel() {
    return gotoNextOnLevel(getLevel());
}

void XMLReader::gotoEndOfCurrentTag() {
    gotoNextOnLevel(getLevel(), NODE_ELEMENT_END);
}


} // namespace plask
