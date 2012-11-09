#include "exceptions.h"
#include "reader.h"

namespace plask {

XMLException::XMLException(const XMLReader& reader, const std::string& msg):
    std::runtime_error("In " +
        ( (reader.getNodeType() == XMLReader::NODE_ELEMENT)? "<" + reader.getNodeName() + ">" :
          (reader.getNodeType() == XMLReader::NODE_ELEMENT_END)? "</" + reader.getNodeName() + ">" :
          "\"" + reader.getNodeName() + "\"" )
        + ": " + msg) {}

XMLException::XMLException(const std::string& where, const std::string& msg):
    std::runtime_error("In " + where + ": " + msg) {}

XMLException::XMLException(const std::string& msg): std::runtime_error(msg) {}

XMLUnexpectedElementException::XMLUnexpectedElementException(const XMLReader& reader, const std::string& what_is_expected):
    XMLException(reader, "expected " + what_is_expected + ", got " + (
                    reader.getNodeType() == XMLReader::NODE_ELEMENT ?     ("<"+reader.getNodeName()+">") :
                    reader.getNodeType() == XMLReader::NODE_ELEMENT_END ? ("</"+reader.getNodeName()+">") :
                    "text"
                 ) + " instead") {}


} // namespace plask
