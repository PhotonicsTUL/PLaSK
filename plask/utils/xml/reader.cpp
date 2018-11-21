#include "reader.h"
#include <expat.h>

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
    constexpr int buff_size = 1024 * 8;
    char buff[buff_size];
    int read = int(source->read(buff, buff_size));
    bool has_more = buff_size == read;
    if (XML_Parse(parser, buff, read, !has_more) == XML_STATUS_ERROR) {
        auto error_code = XML_GetErrorCode(parser);
        if (error_code != XML_ERROR_FINISHED) {
            auto line = XML_GetCurrentLineNumber(parser);
            throw XMLException("XML line " +
                               boost::lexical_cast<std::string>(line) + ": parse error: "
                               + XML_ErrorString(error_code), int(line));
        }
    }
    return has_more;
}

void XMLReader::initParser() {
    parser = XML_ParserCreateNS(NULL, ' ');
    XML_SetUserData(parser, this);
    XML_SetElementHandler(parser, &XMLReader::startTag, &XMLReader::endTag);
    XML_SetCharacterDataHandler(parser, &XMLReader::characterData);
}

XMLReader::NodeType XMLReader::ensureNodeTypeIs(int required_types, const char *new_tag_name) const
{
    NodeType result = this->getNodeType();
    if (((required_types & result) == 0) ||
        (new_tag_name && (result == NODE_ELEMENT) && (getNodeName() != new_tag_name)))
    {
        std::string msg;
        if (required_types & NODE_ELEMENT) {
            if (new_tag_name) {
                msg += "begining of tag ";
                msg += new_tag_name;
            } else
                msg += "begining of a new tag";
        }
        if (required_types & NODE_ELEMENT_END) {
            if (!msg.empty()) msg += " or ";
            if (result == NODE_ELEMENT) {
                assert(path.size() >= 2);
                msg += "</" + path[path.size()-2] + ">";
            } else
                msg += "</" + path.back() + ">";
        }
        if (required_types & NODE_TEXT) {
            if (!msg.empty()) msg += " or ";
            msg += "content of <" + path.back() + "> tag";
        }
        throwUnexpectedElementException(msg);
    }
    return result;
}

XMLReader::XMLReader(std::unique_ptr<DataSource> &&source):
    source(std::move(source)), stringInterpreter(&XMLReader::strToBool, &parse_complex<double>), check_if_all_attributes_were_read(true)
{
    initParser();
}

XMLReader::XMLReader(std::unique_ptr<std::istream> &&istream):
    source(new StreamDataSource(std::move(istream))), stringInterpreter(&XMLReader::strToBool, &parse_complex<double>), check_if_all_attributes_were_read(true)
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

bool XMLReader::next() {
    if (!states.empty()) {
        if (getNodeType() == NODE_ELEMENT) {
            if (check_if_all_attributes_were_read && (getAttributeCount() != read_attributes.size())) {
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
        else if (getCurrent().type == NODE_ELEMENT_END)
            path.pop_back();
        states.pop_front();
    }
    check_if_all_attributes_were_read = true;

    while (!hasCurrent() && readSome())
        ;

    if (hasCurrent()) {
        if (getCurrent().type == NODE_ELEMENT) path.push_back(getCurrent().text);
        return true;
    } else
        return false;
}

const std::map<std::string, std::string> XMLReader::getAttributes() const {
    ensureHasCurrent();
    this->ignoreAllAttributes();
    if (attributeFilter) {
        std::map<std::string, std::string> parsed;
        for (const auto& attr: getCurrent().attributes) {
            try {
                parsed[attr.first] = attributeFilter(attr.second);
            } catch (const std::exception& e) {
                unsigned line = this->getCurrent().lineNr;
                throw XMLException("XML line " + boost::lexical_cast<std::string>(line) +
                                " in <" + this->getCurrent().text + "> attribute '" + attr.first +
                                "': Bad parsed expression", e.what(), int(line));
            }
        }
        return parsed;
    } else {
        return getCurrent().attributes;
    }
}

void XMLReader::removeAlienNamespaceAttr() {
    if (getNodeType() != NODE_ELEMENT)
        throwUnexpectedElementException("element");
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
    if (n == NODE_TEXT)
        return path.back();
    //if (n == NODE_ELEMENT || n == NODE_ELEMENT_END)
    //    return getCurrent().text;
    //throw XMLUnexpectedElementException(*this, "element or end of element");
    return getCurrent().text;
}

std::string XMLReader::getTextContent() const {
    if (getNodeType() != NODE_TEXT) {
        if (getNodeType() == NODE_ELEMENT_END)
            return "";
        else
            throw XMLUnexpectedElementException(*this, "text");
    }
    if (contentFilter) {
        try {
            return contentFilter(getCurrent().text);
        } catch (const std::exception& e) {
            unsigned line = this->getCurrent().lineNr;
            throw XMLException("XML line " + boost::lexical_cast<std::string>(line) +
                               ": Bad parsed expression", e.what(), int(line));
        }
    } else
        return getCurrent().text;
}

plask::optional<std::string> XMLReader::getAttribute(const std::string& name) const {
    auto res_it = this->getCurrent().attributes.find(name);
    if (res_it == this->getCurrent().attributes.end())
        return plask::optional<std::string>();
    read_attributes.insert(name);   //TODO should this be thread-safe?
    if (attributeFilter) {
        try {
            return attributeFilter(res_it->second);
        } catch (const std::exception& e) {
            unsigned line = this->getCurrent().lineNr;
            throw XMLException("XML line " + boost::lexical_cast<std::string>(line) +
                               " in <" + this->getCurrent().text + "> attribute '" + name +
                               "': Bad parsed expression", e.what(), int(line));
        }
    } else
        return res_it->second;
}

std::string XMLReader::requireAttribute(const std::string& attr_name) const {
    plask::optional<std::string> result = getAttribute(attr_name);
    if (!result) throw XMLNoAttrException(*this, attr_name);
    return *result;
}

void XMLReader::requireNext() {
    if (!next()) throwUnexpectedEndException();
}

XMLReader::NodeType XMLReader::requireNext(int required_types, const char *new_tag_name) {
    requireNext();
    return ensureNodeTypeIs(required_types, new_tag_name);
}

void XMLReader::requireTag() {
    requireNext(NODE_ELEMENT);
}

void XMLReader::requireTag(const std::string& name) {
    requireNext(NODE_ELEMENT, name.c_str());
}

bool XMLReader::requireTagOrEnd() {
    return requireNext(NODE_ELEMENT | NODE_ELEMENT_END) == NODE_ELEMENT;
}

bool XMLReader::requireTagOrEnd(const std::string& name) {
    return requireNext(NODE_ELEMENT | NODE_ELEMENT_END, name.c_str()) == NODE_ELEMENT;
}

void XMLReader::requireTagEnd() {
    requireNext(NODE_ELEMENT_END);
}

std::string XMLReader::requireText() {
    requireNext();
    return getTextContent();
}

std::string XMLReader::requireTextInCurrentTag() {
    std::string t = requireText();
    if (t.length() != 0) requireTagEnd();
    return t;
}

bool XMLReader::gotoNextOnLevel(std::size_t required_level, NodeType required_type) {
    ignoreAllAttributes();
    while (next()) {
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
