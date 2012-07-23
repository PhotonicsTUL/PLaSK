#include "exceptions.h"
#include "reader.h"

namespace plask {

XMLException::XMLException(const XMLReader& reader, const std::string& msg):
    std::runtime_error("XML error in " +
        ( (reader.getNodeType() == XMLReader::NODE_ELEMENT)? "<" + reader.getNodeName() + ">" :
          (reader.getNodeType() == XMLReader::NODE_ELEMENT_END)? "</" + reader.getNodeName() + ">" :
          "\"" + reader.getNodeName() + "\"" )
        + ": " + msg) {}

XMLException::XMLException(const std::string& where, const std::string& msg):
    std::runtime_error("XML error in " + where + ": " + msg) {}

XMLException::XMLException(const std::string& msg): std::runtime_error("XML error: " + msg) {}

} // namespace plask