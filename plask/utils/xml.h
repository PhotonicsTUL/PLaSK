#ifndef PLASK__XML_H
#define PLASK__XML_H

#include <irrxml/irrXML.h>

#include <string>
#include <boost/lexical_cast.hpp>

namespace plask {

typedef irr::io::IrrXMLReader XMLReader; 

namespace XML {

template <typename T>
inline T getAttribiute(XMLReader& reader, const char* name, T&& default_value) {
    const char* attr_str = reader.getAttributeValue(name);
    if (attr_str == nullptr) return default_value;
    return boost::lexical_cast<T>(attr_str);
}

std::string requireAttr(XMLReader &source, const char* attr_name);

template <typename T>
inline T requireAttr(XMLReader& reader, const char* name) {
    return boost::lexical_cast<T>(requireAttr(reader, name));
}

/**
 * Call reader.read(), one or more time (skeep comments).
 * @throw XMLUnexpectedEndException if there is no next element
 */
void requireNext(XMLReader& reader);

void requireTag(XMLReader& reader);

void requireTagEnd(XMLReader& reader);

}

}

#endif // PLASK__XML_H
