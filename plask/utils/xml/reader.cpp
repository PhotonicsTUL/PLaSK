#include "reader.h"
#include "expat.h"

#include "../../math.h"

#include <fstream>

namespace plask {

std::size_t XMLReader::StreamDataSource::read(char* buff, std::size_t buf_size) {
    input->read(buff, buf_size);
    if (input->bad())
        throw XMLException("XML reader: Can't read input data from C++ stream.");
    return input->gcount();
}

std::size_t XMLReader::CFileDataSource::read(char* buff, std::size_t buf_size) {
    std::size_t read = fread(buff, sizeof(char), buf_size, desc);
    if (read != buf_size && ferror(desc)) {
        throw XMLException("XML reader: Can't read input data from C file.");
    }
    return read;
}


void XMLReader::startTag(void *data, const char *element, const char **attribute) {
    State& state = reinterpret_cast<XMLReader*>(data)->appendState(NODE_ELEMENT, element);
    for (int i = 0; attribute[i]; i += 2) {
        state.attributes[attribute[i]] = attribute[i+1];
    }
}

void XMLReader::endTag(void *data, const char *element) {
    reinterpret_cast<XMLReader*>(data)->appendState(NODE_ELEMENT_END, element);
}

void XMLReader::characterData(void* data, const XML_Char *string, int string_len) {
    reinterpret_cast<XMLReader*>(data)->appendState(NODE_TEXT, std::string(string, string_len));
}

XMLReader::State& XMLReader::appendState(NodeType type, const std::string& text) {
    if (!states.empty() && states.back().type == NODE_TEXT) {
        if (type == NODE_TEXT) {
            states.back().text.append(text);
            return states.back();
        } else
            if (states.back().hasWhiteText())
                states.pop_back();
    }
    states.emplace_back(type, XML_GetCurrentLineNumber(parser), XML_GetCurrentColumnNumber(parser), text);
    return states.back();
}

bool XMLReader::readSome() {
    constexpr std::size_t buff_size = 1024 * 8;
    char buff[buff_size];
    std::size_t read = source->read(buff, buff_size);
    bool has_more = buff_size == read;
    if (XML_Parse(parser, buff, read, !has_more) == XML_STATUS_ERROR) {
        throw XMLException("XML line " +
                           boost::lexical_cast<std::string>(XML_GetCurrentLineNumber(parser)) + ": parse error: "
                           + XML_ErrorString(XML_GetErrorCode(parser)));
    }
    return has_more;
}

void XMLReader::initParser() {
    parser = XML_ParserCreateNS(NULL, ' ');
    XML_SetUserData(parser, this);
    XML_SetElementHandler(parser, &XMLReader::startTag, &XMLReader::endTag);
    XML_SetCharacterDataHandler(parser, &XMLReader::characterData);
}

XMLReader::XMLReader(DataSource* source):
    source(source), stringInterpreter(&XMLReader::strToBool, &parse_complex<double>), check_if_all_attributes_were_read(true)
{
    initParser();
}

XMLReader::XMLReader(std::istream *istream):
    source(new StreamDataSource(istream)), stringInterpreter(&XMLReader::strToBool, &parse_complex<double>), check_if_all_attributes_were_read(true)
{
    initParser();
}

XMLReader::XMLReader(const char* file_name):
    source(new StreamDataSource(new std::ifstream(file_name))), stringInterpreter(&XMLReader::strToBool, &parse_complex<double>), check_if_all_attributes_were_read(true)
{
    initParser();
}

XMLReader::XMLReader(FILE* file):
    source(new CFileDataSource(file)), stringInterpreter(&XMLReader::strToBool, &parse_complex<double>), check_if_all_attributes_were_read(true)
{
    initParser();
}

#if (__cplusplus >= 201103L) || defined(__GXX_EXPERIMENTAL_CXX0X__)
XMLReader::XMLReader(XMLReader &&to_move)
    : source(std::move(to_move.source)),
      states(std::move(to_move.states)),
      parser(to_move.parser),
      path(std::move(to_move.path)),
      read_attributes(std::move(to_move.read_attributes)),
      stringInterpreter(std::move(to_move.stringInterpreter)),
      check_if_all_attributes_were_read(to_move.check_if_all_attributes_were_read)
{
    to_move.parser = 0;
}

XMLReader &XMLReader::operator=(XMLReader &&to_move)
{
    swap(to_move);
    return *this;
}
#endif

XMLReader::~XMLReader() {
    XML_ParserFree(this->parser);
    delete source;
}

void XMLReader::swap(XMLReader &to_swap)
{
    std::swap(source, to_swap.source);
    std::swap(states, to_swap.states);
    std::swap(parser, to_swap.parser);
    std::swap(path, to_swap.path);
    std::swap(read_attributes, to_swap.read_attributes);
    std::swap(stringInterpreter, to_swap.stringInterpreter);
    std::swap(check_if_all_attributes_were_read, to_swap.check_if_all_attributes_were_read);
}

bool XMLReader::read() {
    if (!states.empty() && getNodeType() == NODE_ELEMENT) {
        if (check_if_all_attributes_were_read && (std::size_t(getAttributeCount()) != read_attributes.size())) {
            std::string attr;
            for (const std::pair<const std::string, std::string>& a: getCurrent().attributes)
                if (read_attributes.find(a.first) == read_attributes.end()) {
                    if (!attr.empty()) attr += ", ";
                    attr += a.first;
                }
            throw XMLUnexpectedAttrException(*this, attr);
        }
        read_attributes.clear();
    }
    check_if_all_attributes_were_read = true;

    if (!states.empty()) {
        if (getCurrent().type == NODE_ELEMENT_END) path.pop_back();
        states.pop_front();
    }

    while (!hasCurrent() && readSome())
        ;

    if (hasCurrent()) {
        if (getCurrent().type == NODE_ELEMENT) path.push_back(getCurrent().text);
        return true;
    } else
        return false;
}

void XMLReader::removeAlienNamespaceAttr() {
    if (getNodeType() != NODE_ELEMENT)
        throw XMLUnexpectedElementException(*this, "element");
    auto iter = states.front().attributes.begin();
    while (iter != states.front().attributes.end()) {
        if (iter->first.find(' ') != std::string::npos) //not in default NS?
            states.front().attributes.erase(iter++);
        else
            ++iter;
    }
}

std::string XMLReader::getNodeName() const {
    NodeType n = getNodeType();
    if (n != NODE_ELEMENT && n != NODE_ELEMENT_END)
        throw XMLUnexpectedElementException(*this, "element or end of element");
    return getCurrent().text;
}

std::string XMLReader::getTextContent() const {
    if (getNodeType() != NODE_TEXT)
        throw XMLUnexpectedElementException(*this, "text");
    if (contentFilter) {
        try {
            return contentFilter(getCurrent().text);
        } catch (const std::exception& e) {
            throw XMLException("XML line " + boost::lexical_cast<std::string>(this->getCurrent().lineNr) + 
                               ": Bad parsed expression", e.what());
        }
    } else
        return getCurrent().text;
}

boost::optional<std::string> XMLReader::getAttribute(const std::string& name) const {
    auto res_it = this->getCurrent().attributes.find(name);
    if (res_it == this->getCurrent().attributes.end())
        return boost::optional<std::string>();
    const_cast<std::set<std::string>&>(read_attributes).insert(name);
    if (attributeFilter) {
        try {
            return attributeFilter(res_it->second);
        } catch (const std::exception& e) {
            throw XMLException("XML line " + boost::lexical_cast<std::string>(this->getCurrent().lineNr) + 
                               " in <" + this->getCurrent().text + "> attribute '" + name +
                               "': Bad parsed expression", e.what());
        }
    } else
        return res_it->second;
}

std::string XMLReader::requireAttribute(const std::string& attr_name) const {
    boost::optional<std::string> result = getAttribute(attr_name);
    if (!result) throw XMLNoAttrException(*this, attr_name);
    return *result;
}

void XMLReader::requireNext() {
    if (!read()) throw XMLUnexpectedEndException(*this);
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
    if (getNodeType() == NODE_ELEMENT) {
        path.pop_back();
        throw XMLUnexpectedElementException(*this, "</" + path.back() + ">");
    } else if (getNodeType() != NODE_ELEMENT_END)
        throw XMLUnexpectedElementException(*this, "</" + path.back() + ">");
}

std::string XMLReader::requireText() {
    requireNext();
    return getTextContent();
}

std::string XMLReader::requireTextInCurrentTag() {
    std::string t = requireText();
    requireTagEnd();
    return t;
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
